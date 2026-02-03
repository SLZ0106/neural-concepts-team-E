#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import re
from collections import defaultdict, Counter
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

def build_fewshot_prompt(
    examples: List[Dict[str, Any]],
    query_item: Dict[str, Any],
    max_q_chars: int = 1800,
    max_a_chars: int = 1800,
) -> str:
    """
    Few-shot prompt: show K labeled Q/A examples, then ask model to label a new Q/A.
    Output must be strict JSON only.
    """

    def clip(s: str, n: int) -> str:
        s = (s or "").strip()
        if len(s) <= n:
            return s
        return s[: n - 12] + " ...[TRUNCATED]"

    header = (
        "You are labeling economic uncertainty in earnings-call Q&A.\n"
        "Class labels:\n"
        f"- {LABELS[0]}: clear answers / plans / explanations; no lack of visibility.\n"
        f"- {LABELS[1]}: some limited visibility or mild conditionality; not strongly uncertain.\n"
        f"- {LABELS[2]}: explicit uncertainty (too early to tell, cannot estimate, depends on policy/macro, unknown impact).\n\n"
        "Important:\n"
        "- Do NOT confuse positive/negative sentiment with uncertainty.\n"
        "- Uncertainty is about lack of visibility / conditional outcomes / inability to estimate.\n"
        "- Use BOTH question and answer.\n\n"
        "Return STRICT JSON only (no markdown, no extra text):\n"
        '{ "label": one of ["NO_UNCERTAINTY","INTERMEDIATE_UNCERTAINTY","HIGH_UNCERTAINTY"], '
        '"rationale": "1-3 concise sentences explaining the key cues in the Q/A" }\n\n'
        "Labeled examples:\n"
    )

    ex_blocks = []
    for i, ex in enumerate(examples, 1):
        q = clip(ex.get("question", ""), max_q_chars)
        a = clip(ex.get("answer", ""), max_a_chars)
        lbl = ex.get("label", "")
        ex_blocks.append(
            f"Example {i}\n"
            f"Q: {q}\n"
            f"A: {a}\n"
            f"Label: {lbl}\n"
        )

    q2 = clip(query_item.get("question", ""), max_q_chars)
    a2 = clip(query_item.get("answer", ""), max_a_chars)

    query = (
        "\nNow label this new Q/A:\n"
        f"Q: {q2}\n"
        f"A: {a2}\n"
        "JSON:\n"
    )
    return header + "\n".join(ex_blocks) + query

def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort: parse strict JSON object from model output.
    """
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
    if lbl in LABELS:
        return lbl
    return None

# ---------- Few-shot selection ----------

def stratified_shots(
    labeled: List[Dict[str, Any]],
    n_per_class: int,
    seed: int,
    exclude_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Pick n_per_class examples per label from labeled pool.
    If exclude_id matches a labeled item, do leave-one-out (avoid trivial copying).
    """
    rng = random.Random(seed)

    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ex in labeled:
        if exclude_id is not None and ex.get("id") == exclude_id:
            continue
        lbl = ex.get("label")
        if lbl in LABELS:
            by_label[lbl].append(ex)

    shots: List[Dict[str, Any]] = []
    for lbl in LABELS:
        pool = by_label.get(lbl, [])
        rng.shuffle(pool)
        shots.extend(pool[: min(n_per_class, len(pool))])

    rng.shuffle(shots)
    return shots

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
    """
    Generate output (raw text) and parsed JSON dict.
    Uses chat_template if available; otherwise uses plain text prompting.
    """

    # Try chat template if tokenizer has it
    use_chat = hasattr(tok, "apply_chat_template") and tok.chat_template is not None

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

    ap.add_argument("--n_per_class", type=int, default=5, help="few-shot examples per class")
    ap.add_argument("--seed", type=int, default=21)

    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--quant", type=str, default="none", choices=["none", "8bit", "4bit"])

    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument("--limit", type=int, default=0, help="debug: only run first N samples (0 = all)")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    data = load_json_list(args.data_json)

    # Labeled pool for shots
    labeled = [x for x in data if x.get("label") in LABELS]
    if len(labeled) == 0:
        raise ValueError("No labeled examples found. Expected ~50 with label in {NO,INTERMEDIATE,HIGH}.")

    # Load models
    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    models = {}
    for mid in model_ids:
        model, tok = load_model(mid, args.device_map, args.dtype, args.quant)
        models[mid] = (model, tok)

    preds_all_rows: List[Dict[str, Any]] = []
    # For disagreement aggregation
    by_item: Dict[int, Dict[str, Any]] = {}  # id -> {base fields + per-model}

    # Eval buffers on gold-labeled subset
    eval_gold: Dict[str, List[str]] = {mid: [] for mid in model_ids}
    eval_pred: Dict[str, List[str]] = {mid: [] for mid in model_ids}

    run_items = data if args.limit <= 0 else data[: args.limit]

    for item in run_items:
        item_id = item.get("id")
        if item_id is None:
            continue
        gold = item.get("label") if item.get("label") in LABELS else None

        # Make per-item container
        if item_id not in by_item:
            by_item[item_id] = {
                "id": item_id,
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "gold_label": gold,
                "predictions": {},  # model_id -> {label, rationale, parse_ok}
            }

        # Select few-shot examples (leave-one-out if this is labeled)
        shots = stratified_shots(
            labeled=labeled,
            n_per_class=args.n_per_class,
            seed=args.seed,
            exclude_id=item_id if gold is not None else None,
        )

        prompt = build_fewshot_prompt(shots, item)

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
            rationale = None
            parse_ok = False

            if isinstance(parsed, dict):
                pred_label = normalize_label(parsed.get("label"))
                rationale = parsed.get("rationale")
                if pred_label in LABELS and isinstance(rationale, str) and len(rationale.strip()) > 0:
                    parse_ok = True
                else:
                    # still store partial
                    parse_ok = False

            # If parse failed, keep raw for debugging; label stays None
            rec = {
                "id": item_id,
                "model_id": mid,
                "gold_label": gold,
                "pred_label": pred_label,
                "rationale": rationale,
                "parse_ok": parse_ok,
                "raw_text": raw,
            }
            preds_all_rows.append(rec)

            # Store for disagreement set
            by_item[item_id]["predictions"][mid] = {
                "pred_label": pred_label,
                "rationale": rationale,
                "parse_ok": parse_ok,
            }

            # Update eval if gold exists and we got a valid pred_label
            if gold is not None and pred_label in LABELS:
                eval_gold[mid].append(gold)
                eval_pred[mid].append(pred_label)

    # Write preds_all.jsonl
    preds_path = os.path.join(args.out_dir, "preds_all.jsonl")
    write_jsonl(preds_path, preds_all_rows)

    # Disagreements: at least two models with different valid labels
    disagreements: List[Dict[str, Any]] = []
    for item_id, blob in by_item.items():
        labels_here = []
        for mid in model_ids:
            pl = blob["predictions"].get(mid, {}).get("pred_label")
            if pl in LABELS:
                labels_here.append(pl)
            else:
                # treat parse failure / missing label as disagreement-worthy
                labels_here.append(None)

        # If any None OR not all equal among non-None -> disagreement
        valid = [x for x in labels_here if x is not None]
        disagree = (len(valid) < len(model_ids)) or (len(valid) >= 2 and not all_equal(valid))
        if disagree:
            disagreements.append(blob)

    disag_path = os.path.join(args.out_dir, "disagreements.jsonl")
    write_jsonl(disag_path, disagreements)

    # Eval report on labeled subset
    eval_report = {}
    for mid in model_ids:
        golds = eval_gold[mid]
        preds = eval_pred[mid]
        n = len(golds)
        acc = sum(1 for g, p in zip(golds, preds) if g == p) / n if n > 0 else None
        eval_report[mid] = {
            "n_eval": n,
            "accuracy": acc,
            "confusion_matrix": confusion_matrix(golds, preds) if n > 0 else None,
        }

    eval_path = os.path.join(args.out_dir, "eval_labeled.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote: {preds_path}")
    print(f"[OK] wrote: {disag_path}  (n={len(disagreements)})")
    print(f"[OK] wrote: {eval_path}")

if __name__ == "__main__":
    main()
