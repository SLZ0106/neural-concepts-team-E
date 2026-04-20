#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate language models on the synthetic economic uncertainty dataset.
This script reads the output of generate_synthetic_pairs.py, prompts the target
model to classify the statements as HIGH or LOW uncertainty, and calculates
performance metrics (Accuracy and Confusion Matrix).

This version uses NNsight / NDIF instead of raw transformers for model
loading and generation.  To run on NDIF's remote GPUs, pass --remote.
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from nnsight import LanguageModel
from transformers import BitsAndBytesConfig, set_seed

# ====================================================================
# Prompt definitions (aligned with activation_patching_highlow.py)
# ====================================================================

SYNTHETIC_UNCERTAINTY_DEFINITION = (
    "Economic uncertainty is the variance or spread of future business conditions "
    "(e.g. revenue, demand, or the broader economy). High uncertainty means business "
    "conditions cannot be predicted with reasonable confidence. Low uncertainty means "
    "the direction or approximate size of future business conditions are clear.\n\n"
    "KEY PRINCIPLE - UNCERTAINTY IS NOT SENTIMENT: "
    "Sentiment reflects the expected value of future business conditions. "
    "Uncertainty reflects the variance around that expected value.\n"
    '- "We expect a 10% drop in sales due to tariffs" -> negative sentiment, low uncertainty\n'
    '- "Sales could drop 5% or 30% depending on how tariffs develop" -> negative sentiment, high uncertainty'
)

SYNTHETIC_FEW_SHOT_DEMOS = [
    {
        "high": (
            "Point, we do have the business plan or targets for 2026 "
            "prepared last year. We are currently on the process of evaluating "
            "it because of the geopolitical issues that we are encountering and "
            "we are expecting project delays due to logistical supply uncertainties, "
            "which we have no control."
        ),
        "no": (
            "We have evaluated our business plan or targets for 2026 to reflect "
            "the confirmed impact of current geopolitical issues. We are now guiding "
            "to a six-month delay across all major projects due to the structural "
            "breakdown in logistical supply chains."
        ),
    },
    {
        "high": (
            "Having said that, the uncertain duration and future potential impacts "
            "of the government shutdown creates a lack of clear visibility into our cash "
            "forecast for the remainder of the year. We are taking prudent actions to conserve "
            "cash and liquidity. If a resolution can be reached in the near term, we would "
            "expect to be able to achieve the forecast that I just discussed. However, in the "
            "event of a protracted shutdown, it is unclear how and when our cash flow will be "
            "impacted despite our careful efforts to diligently manage cash."
        ),
        "no": (
            "In terms of impact from shut down, no major impact from shutdown and "
            "that was reflected just due to the strong year-over-year growth in us "
            "exceeding the top end of our guidance range on sales, but then also the "
            "$3.3 billion of cash in Q4, creating that 26%."
        ),
    },
]

LABELS = ["HIGH", "LOW"]

# ---------------- IO & Data Parsing ----------------

def load_synthetic_dataset(path: str) -> List[Dict[str, Any]]:
    """
    Parses the flat JSON array output from generate_synthetic_pairs.py
    and splits each pair into two independent evaluation items.
    """
    with open(path, "r", encoding="utf-8-sig") as f:
        raw_pairs = json.load(f)

    eval_items = []
    for row in raw_pairs:
        pair_id = row.get("pair_id", "unknown")
        topic = row.get("topic", "unknown")

        if "high" in row:
            eval_items.append({
                "id": f"{pair_id}_HIGH",
                "pair_id": pair_id,
                "topic": topic,
                "statement": row["high"],
                "label": "HIGH"
            })
        low_stmt = row.get("low", row.get("no"))
        if low_stmt is not None:
            eval_items.append({
                "id": f"{pair_id}_LOW",
                "pair_id": pair_id,
                "topic": topic,
                "statement": low_stmt,
                "label": "LOW"
            })
    return eval_items


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------------- Utilities ----------------

def normalize_label(lbl: Any) -> Optional[str]:
    if not isinstance(lbl, str):
        return None
    s = lbl.strip().upper()
    if s in LABELS:
        return s
    return None


def extract_label_from_text(text: str) -> Optional[str]:
    """
    Extract HIGH/LOW from free-form output, prioritizing standalone labels.
    """
    t = (text or "").strip().upper()
    if t in LABELS:
        return t
    m = re.search(r"\b(HIGH|LOW)\b", t)
    if m:
        return m.group(1)
    return None


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Safely extract JSON by finding the first complete brace block.
    """
    t = (text or "").strip()
    if not t:
        return None

    start = t.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start:i + 1])
                except Exception:
                    return None
    return None

# ---------------- Prompt Builder ----------------

def build_prompt(statement: str, zero_shot: bool = False) -> str:
    prompt = (
        "Determine whether the following economic statement contains high or low uncertainty.\n\n"
        "Definition of Economic Uncertainty:\n"
        f"{SYNTHETIC_UNCERTAINTY_DEFINITION}\n\n"
        "Respond with exactly one word: HIGH or LOW.\n\n"
    )

    if not zero_shot:
        for demo in SYNTHETIC_FEW_SHOT_DEMOS:
            prompt += f"Statement: {demo['no']}\nLabel: LOW\n\n"
            prompt += f"Statement: {demo['high']}\nLabel: HIGH\n\n"

    prompt += f"Statement: {statement.strip()}\nLabel:"
    return prompt

# ---------------- NNsight Model Wrapper ----------------

def load_model(model_id: str, device_map: str, dtype: str, quant: str, remote: bool):
    """
    Load a HuggingFace language model via NNsight's LanguageModel wrapper.
    When *remote* is True the model is initialised with meta-tensors only
    (config + tokenizer are downloaded, weights are NOT).  Actual inference
    happens on NDIF's GPUs.
    When *remote* is False the weights are downloaded and dispatched locally,
    respecting device_map / dtype / quantisation settings.
    """
    if remote:
        # Meta-tensor init: downloads only config & tokenizer, no weights.
        # dispatch defaults to False, so no local GPU memory is used.
        model = LanguageModel(model_id)
        return model

    # ---- local execution ----
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]

    kwargs: Dict[str, Any] = dict(device_map=device_map, dispatch=True)

    if quant == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quant == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    else:
        kwargs["torch_dtype"] = torch_dtype

    model = LanguageModel(model_id, **kwargs)
    return model


def _prepare_input_ids(model: LanguageModel, full_prompt: str):
    """
    Tokenise *full_prompt*, applying a chat template when the tokenizer has one.
    Returns the input_ids tensor and the prompt length (in tokens).
    """
    tok = model.tokenizer
    use_chat = (
        hasattr(tok, "apply_chat_template")
        and getattr(tok, "chat_template", None) is not None
    )

    if use_chat:
        messages = [{"role": "user", "content": full_prompt}]
        inputs = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = inputs["input_ids"]
    else:
        inputs = tok(full_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

    return input_ids, input_ids.shape[1]


def _run_generate(
    model: LanguageModel,
    input_ids,
    prompt_len: int,
    max_new_tokens: int,
    temperature: float,
    remote: bool,
) -> str:
    """
    Run NNsight's .generate() and return the decoded new-token string.
    """
    tok = model.tokenizer

    eos_id = tok.eos_token_id
    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        eos_token_id=eos_id,
        pad_token_id=eos_id,
    )
    if temperature > 0:
        gen_kwargs["temperature"] = float(temperature)
    if remote:
        gen_kwargs["remote"] = True

    with model.generate(input_ids, **gen_kwargs) as tracer:
        output = model.generator.output.save()

    # output is the full sequence (prompt + generated).  Slice off prompt.
    generated_ids = output[0][prompt_len:]
    raw = tok.decode(generated_ids, skip_special_tokens=True).strip()
    return raw


def generate_json_label(
    model: LanguageModel,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    remote: bool,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Generate a label for *prompt* using NNsight, with one retry on parse failure.
    """
    prefix = "IMPORTANT OUTPUT CONSTRAINTS:\n- Return exactly one word: HIGH or LOW.\n\n"
    full_prompt = prefix + prompt

    input_ids, prompt_len = _prepare_input_ids(model, full_prompt)
    raw = _run_generate(model, input_ids, prompt_len, max_new_tokens, temperature, remote)

    # ---------- parse ----------
    parsed = extract_first_json_object(raw)
    if parsed is None:
        label = extract_label_from_text(raw)
        if label is not None:
            parsed = {"classification": label}

    # ---------- retry ----------
    if parsed is None:
        retry_prompt = full_prompt + "\nRETRY. Return ONLY one word: HIGH or LOW."
        input_ids, prompt_len = _prepare_input_ids(model, retry_prompt)
        raw = _run_generate(model, input_ids, prompt_len, max_new_tokens, temperature, remote)

        parsed = extract_first_json_object(raw)
        if parsed is None:
            label = extract_label_from_text(raw)
            if label is not None:
                parsed = {"classification": label}

    return raw, parsed

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate a model on the synthetic uncertainty dataset (NNsight / NDIF)."
    )
    ap.add_argument("--data_json", type=str, required=True, help="Path to synthetic_pairs_generated.json")
    ap.add_argument("--out_dir", type=str, default="out_synthetic_eval", help="Directory to save results")
    ap.add_argument("--models", type=str, default="Qwen/Qwen2.5-7B-Instruct-1M",
                     help="Comma-separated list of HF model IDs")
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--quant", type=str, default="none", choices=["none", "8bit", "4bit"])
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--zero_shot", action="store_true",
                     help="If set, removes the few-shot examples from the prompt.")
    ap.add_argument("--remote", action="store_true",
                     help="Run inference on NDIF's remote GPUs instead of locally.")
    args = ap.parse_args()

    set_seed(42)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading data from {args.data_json}...")
    try:
        data = load_synthetic_dataset(args.data_json)
        print(f"Loaded {len(data)} evaluation items (split from pairs).")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    models: Dict[str, LanguageModel] = {}
    for mid in model_ids:
        print(f"Loading model: {mid}  (remote={args.remote})...")
        models[mid] = load_model(mid, args.device_map, args.dtype, args.quant, args.remote)

    preds_all_rows: List[Dict[str, Any]] = []
    eval_gold: Dict[str, List[str]] = {mid: [] for mid in model_ids}
    eval_pred: Dict[str, List[str]] = {mid: [] for mid in model_ids}

    for i, item in enumerate(data, 1):
        item_id = item["id"]
        gold = normalize_label(item["label"])
        print(f"Processing {i}/{len(data)}: {item_id} (Gold: {gold})")

        prompt = build_prompt(item["statement"], args.zero_shot)

        for mid in model_ids:
            model = models[mid]
            raw, parsed = generate_json_label(
                model, prompt, args.max_new_tokens, args.temperature, args.remote
            )

            pred_label, reasoning, parse_ok = None, None, False
            if isinstance(parsed, dict):
                pred_label = normalize_label(parsed.get("classification"))
                reasoning = parsed.get("reasoning")
                if pred_label is not None:
                    parse_ok = True

            preds_all_rows.append({
                "id": item_id,
                "pair_id": item["pair_id"],
                "topic": item["topic"],
                "model_id": mid,
                "gold_label": gold,
                "pred_label": pred_label,
                "reasoning": reasoning,
                "parse_ok": parse_ok,
                "raw_text": raw,
            })

            if gold and parse_ok:
                eval_gold[mid].append(gold)
                eval_pred[mid].append(pred_label)

    preds_path = os.path.join(args.out_dir, "preds_all.jsonl")
    write_jsonl(preds_path, preds_all_rows)
    print(f"\n[OK] Detailed predictions saved to: {preds_path}")

    eval_report: Dict[str, Any] = {
        "_meta": {
            "dataset": args.data_json,
            "total_items": len(data),
        },
        "models": {},
    }

    for mid in model_ids:
        golds = eval_gold[mid]
        preds = eval_pred[mid]
        n = len(golds)
        acc = sum(1 for g, p in zip(golds, preds) if g == p) / n if n > 0 else 0.0

        cm: Dict[str, Dict[str, int]] = {
            "HIGH": {"HIGH": 0, "LOW": 0},
            "LOW": {"HIGH": 0, "LOW": 0},
        }
        for g, p in zip(golds, preds):
            if g in cm and p in cm[g]:
                cm[g][p] += 1

        eval_report["models"][mid] = {
            "accuracy": round(acc, 4),
            "total_evaluated_successfully": n,
            "confusion_matrix": cm,
            "metrics_breakdown": {
                "HIGH_precision": (
                    cm["HIGH"]["HIGH"] / (cm["HIGH"]["HIGH"] + cm["LOW"]["HIGH"])
                    if (cm["HIGH"]["HIGH"] + cm["LOW"]["HIGH"]) > 0
                    else 0
                ),
                "HIGH_recall": (
                    cm["HIGH"]["HIGH"] / (cm["HIGH"]["HIGH"] + cm["HIGH"]["LOW"])
                    if (cm["HIGH"]["HIGH"] + cm["HIGH"]["LOW"]) > 0
                    else 0
                ),
            },
        }

    report_path = os.path.join(args.out_dir, "performance_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, ensure_ascii=False, indent=2)

    print(f"[OK] Performance report saved to: {report_path}")
    print("\nEvaluation Complete!")


if __name__ == "__main__":
    main()