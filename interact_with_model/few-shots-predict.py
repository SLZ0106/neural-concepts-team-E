#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed

LABELS = ["NO_UNCERTAINTY", "INTERMEDIATE_UNCERTAINTY", "HIGH_UNCERTAINTY"]

# ---------- IO ----------

def load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list at {path}, got: {type(data)}")
    return data

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------- Prompt + parsing ----------

DEF_BLOCK = (
    "DEFINITION OF UNCERTAINTY:\n"
    "- Uncertainty measures second-moment uncertainty: lack of visibility, conditionality, inability to estimate, "
    "or wide range of possible outcomes.\n"
    "- Do not treat positive/negative sentiment or clear numeric guidance as uncertainty by itself.\n"
    "- Focus on whether the speaker expresses confidence in their knowledge/predictions vs. acknowledges limitations.\n"
    "- Base your decision primarily on the ANSWER (the question can contain speculation).\n"
)

def clip(s: str, n: int) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[: n - 12] + " ...[TRUNCATED]"

def build_fewshot_block(
    examples: List[Dict[str, Any]],
    max_q_chars: int = 1800,
    max_a_chars: int = 1800,
) -> str:
    blocks = []
    for i, ex in enumerate(examples, 1):
        q = clip(ex.get("question", ""), max_q_chars)
        a = clip(ex.get("answer", ""), max_a_chars)
        lbl = ex.get("label", "")
        blocks.append(
            f"Example {i}\n"
            f"Q: {q}\n"
            f"A: {a}\n"
            f"Label: {lbl}\n"
        )
    return "\n".join(blocks).strip()

def build_prompt(
    fewshot_block: str,
    query_item: Dict[str, Any],
    prompt_variant: str,
    max_q_chars: int = 1800,
    max_a_chars: int = 1800,
) -> str:
    """
    prompt_variant:
      - "with_def": includes explicit definition block
      - "no_def": no explicit definition block
    """

    header = (
        "You are an expert financial analyst specializing in analyzing earnings call transcripts.\n\n"
        "TASK: Classify the uncertainty level in the following Question-Answer pair from an earnings call.\n\n"
    )

    if prompt_variant == "with_def":
        header += DEF_BLOCK + "\n"

    header += (
        "CLASSIFICATION LABELS:\n"
        "- NO_UNCERTAINTY: Speaker provides clear, definitive information with confidence. No hedging about visibility or outcomes.\n"
        "- INTERMEDIATE_UNCERTAINTY: Some hedging language, mild conditionality, or moderate unknowns, but still provides substantive guidance.\n"
        "- HIGH_UNCERTAINTY: Explicit lack of visibility, inability to estimate, dependence on unknown factors, or wide range of possible outcomes.\n\n"
        "INSTRUCTIONS:\n"
        "1. Analyze the ANSWER for indicators of second-moment uncertainty.\n"
        "2. Provide brief reasoning (max 3 bullet points).\n"
        "3. Output strict JSON only.\n\n"
        "REQUIRED OUTPUT FORMAT (JSON only; no markdown):\n"
        '{\n'
        '  "reasoning": "<max 3 bullet points>",\n'
        '  "classification": "NO_UNCERTAINTY|INTERMEDIATE_UNCERTAINTY|HIGH_UNCERTAINTY"\n'
        '}\n\n'
        "FEW-SHOT EXAMPLES:\n"
    )

    q = clip(query_item.get("question", ""), max_q_chars)
    a = clip(query_item.get("answer", ""), max_a_chars)

    query = (
        "\nNOW CLASSIFY THIS Q&A:\n"
        f"<question>\n{q}\n</question>\n\n"
        f"<answer>\n{a}\n</answer>\n\n"
        "JSON:\n"
    )

    return header + fewshot_block + query

def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()

    # Direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Extract first {...}
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    blob = m.group(0)
    try:
        obj = json.loads(blob)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None

def normalize_label(lbl: Any) -> Optional[str]:
    if not isinstance(lbl, str):
        return None
    lbl = lbl.strip().upper()
    return lbl if lbl in LABELS else None

# ---------- Few-shot selection (GLOBAL, FIXED) ----------

def select_global_demos_stratified(
    labeled: List[Dict[str, Any]],
    n_per_class: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Select a single global demo set: n_per_class per label.
    This demo set is reused across:
      - all models
      - all prompt variants
      - all query items
    """
    rng = random.Random(seed)
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ex in labeled:
        lbl = ex.get("label")
        if lbl in LABELS:
            by_label[lbl].append(ex)

    demos: List[Dict[str, Any]] = []
    for lbl in LABELS:
        pool = list(by_label.get(lbl, []))
        rng.shuffle(pool)
        if len(pool) < n_per_class:
            raise ValueError(f"Not enough labeled samples for {lbl}: have {len(pool)}, need {n_per_class}")
        demos.extend(pool[:n_per_class])

    # Fix order deterministically (stable across runs)
    # Sort by label then id to reduce variance; this also makes your prompt consistent.
    demos.sort(key=lambda x: (x.get("label", ""), int(x.get("id", 10**18))))
    return demos

# ---------- Model wrapper ----------

def load_model(model_id: str, device_map: str, dtype: str, quant: str):
    qcfg = None
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]

    if quant == "8bit":
        qcfg = BitsAndBytesConfig(load_in_8bit=True)
        torch_dtype = None
    elif quant == "4bit":
        qcfg = BitsAndBytesConfig(load_in_4bit=True)
        torch_dtype = None

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=torch_dtype,
        quantization_config=qcfg,
    )
    model.eval()
    return model, tok

@torch.no_grad()
def generate_json_label(
    model,
    tok,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    # Try chat template if tokenizer has it
    use_chat = hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None) is not None

    if use_chat:
        messages = [{"role": "user", "content": prompt}]
        inputs = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=top_p,
            return_dict_in_generate=True,
        )
        seq = out.sequences[0]
        gen_ids = seq[prompt_len:]
        raw = tok.decode(gen_ids, skip_special_tokens=True).strip()
    else:
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=top_p,
            return_dict_in_generate=True,
        )
        seq = out.sequences[0]
        gen_ids = seq[prompt_len:]
        raw = tok.decode(gen_ids, skip_special_tokens=True).strip()

    parsed = extract_first_json_object(raw)
    return raw, parsed

# ---------- Eval + disagreement ----------

def confusion_matrix(golds: List[str], preds: List[str]) -> Dict[str, Dict[str, int]]:
    cm = {g: {p: 0 for p in LABELS} for g in LABELS}
    for g, p in zip(golds, preds):
        if g in LABELS and p in LABELS:
            cm[g][p] += 1
    return cm

def all_equal(xs: List[str]) -> bool:
    return len(set(xs)) <= 1

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="out_fewshot")
    ap.add_argument("--models", type=str, default="google/gemma-2-9b-it,Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct")

    ap.add_argument("--n_per_class", type=int, default=5, help="few-shot examples per class (global demos)")
    ap.add_argument("--seed", type=int, default=21)

    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--quant", type=str, default="none", choices=["none", "8bit", "4bit"])

    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument("--limit", type=int, default=0, help="debug: only run first N samples (0 = all)")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    data = load_json_list(args.data_json)

    # Labeled pool
    labeled = [x for x in data if x.get("label") in LABELS]
    if len(labeled) == 0:
        raise ValueError("No labeled examples found. Expected some with label in {NO,INTERMEDIATE,HIGH}.")

    # Global fixed demos
    demos = select_global_demos_stratified(labeled, args.n_per_class, args.seed)
    demo_ids = [int(x["id"]) for x in demos if "id" in x]
    demo_id_set = set(demo_ids)
    fewshot_block = build_fewshot_block(demos)

    # Load models
    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    models = {}
    for mid in model_ids:
        model, tok = load_model(mid, args.device_map, args.dtype, args.quant)
        models[mid] = (model, tok)

    run_items = data if args.limit <= 0 else data[: args.limit]

    prompt_variants = ["with_def", "no_def"]

    # Per-variant outputs
    for pv in prompt_variants:
        preds_all_rows: List[Dict[str, Any]] = []
        by_item: Dict[int, Dict[str, Any]] = {}

        # Eval buffers on gold-labeled subset EXCLUDING demos
        eval_gold: Dict[str, List[str]] = {mid: [] for mid in model_ids}
        eval_pred: Dict[str, List[str]] = {mid: [] for mid in model_ids}

        for item in run_items:
            item_id = item.get("id")
            if item_id is None:
                continue
            item_id = int(item_id)

            gold = item.get("label") if item.get("label") in LABELS else None

            # container for disagreements
            if item_id not in by_item:
                by_item[item_id] = {
                    "id": item_id,
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "gold_label": gold,
                    "prompt_variant": pv,
                    "demo_ids": demo_ids,
                    "predictions": {},  # model_id -> {pred_label, reasoning, parse_ok}
                }

            prompt = build_prompt(fewshot_block, item, prompt_variant=pv)

            for mid in model_ids:
                model, tok = models[mid]
                raw, parsed = generate_json_label(
                    model=model,
                    tok=tok,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

                pred_label = None
                reasoning = None
                parse_ok = False

                if isinstance(parsed, dict):
                    pred_label = normalize_label(parsed.get("classification"))
                    reasoning = parsed.get("reasoning")
                    if pred_label in LABELS and isinstance(reasoning, str) and len(reasoning.strip()) > 0:
                        parse_ok = True

                rec = {
                    "id": item_id,
                    "model_id": mid,
                    "prompt_variant": pv,
                    "seed": args.seed,
                    "n_per_class": args.n_per_class,
                    "demo_ids": demo_ids,
                    "gold_label": gold,
                    "pred_label": pred_label,
                    "reasoning": reasoning,
                    "parse_ok": parse_ok,
                    "raw_text": raw,
                }
                preds_all_rows.append(rec)

                by_item[item_id]["predictions"][mid] = {
                    "pred_label": pred_label,
                    "reasoning": reasoning,
                    "parse_ok": parse_ok,
                }

                # Eval: only gold-labeled AND not a demo AND has valid pred
                if gold is not None and (item_id not in demo_id_set) and (pred_label in LABELS):
                    eval_gold[mid].append(gold)
                    eval_pred[mid].append(pred_label)

        # Write preds_all
        preds_path = os.path.join(args.out_dir, f"preds_all_{pv}.jsonl")
        write_jsonl(preds_path, preds_all_rows)

        # Disagreements
        disagreements: List[Dict[str, Any]] = []
        for item_id, blob in by_item.items():
            # Collect per-model labels (None if missing/parse fail)
            labels_here = []
            for mid in model_ids:
                pl = blob["predictions"].get(mid, {}).get("pred_label")
                labels_here.append(pl if pl in LABELS else None)

            valid = [x for x in labels_here if x is not None]
            disagree = (len(valid) < len(model_ids)) or (len(valid) >= 2 and not all_equal(valid))
            if disagree:
                disagreements.append(blob)

        disag_path = os.path.join(args.out_dir, f"disagreements_{pv}.jsonl")
        write_jsonl(disag_path, disagreements)

        # Eval report
        n_gold_total = len(labeled)
        n_demos = len(demos)
        # In full run (limit=0), this should be 69-15=54 given your dataset counts.
        n_eval_expected = len([x for x in labeled if int(x["id"]) not in demo_id_set])

        eval_report = {
            "_meta": {
                "prompt_variant": pv,
                "seed": args.seed,
                "n_per_class": args.n_per_class,
                "demo_ids": demo_ids,
                "n_gold_total": n_gold_total,
                "n_demos": n_demos,
                "n_eval_expected_excluding_demos": n_eval_expected,
                "models": model_ids,
            },
            "per_model": {}
        }

        for mid in model_ids:
            golds = eval_gold[mid]
            preds = eval_pred[mid]
            n = len(golds)
            acc = sum(1 for g, p in zip(golds, preds) if g == p) / n if n > 0 else None
            eval_report["per_model"][mid] = {
                "n_eval": n,
                "accuracy": acc,
                "confusion_matrix": confusion_matrix(golds, preds) if n > 0 else None,
            }

        eval_path = os.path.join(args.out_dir, f"eval_labeled_{pv}.json")
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_report, f, ensure_ascii=False, indent=2)

        print(f"[OK] wrote: {preds_path}")
        print(f"[OK] wrote: {disag_path}  (n={len(disagreements)})")
        print(f"[OK] wrote: {eval_path}")

if __name__ == "__main__":
    main()
