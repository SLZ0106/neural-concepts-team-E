#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate language models on the synthetic economic uncertainty dataset.
This script reads the output of generate_synthetic_pairs.py, prompts the target
model to classify the statements as HIGH or LOW uncertainty, and calculates
performance metrics (Accuracy and Confusion Matrix).
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed

# ====================================================================
# EXACT PROMPT DEFINITIONS (Aligned with generate_synthetic_pairs.py)
# ====================================================================

DEF_BLOCK = """\
## Definition of Economic Uncertainty
Economic uncertainty is the variance or spread of future business conditions (e.g. revenue, demand, or the broader economy). High uncertainty means business conditions cannot be predicted with reasonable confidence. Low uncertainty means the direction or approximate size of future business conditions are clear.

KEY PRINCIPLE - UNCERTAINTY IS NOT SENTIMENT:
Sentiment reflects the expected value of future business conditions. Uncertainty reflects the variance around that expected value.
- "We expect a 10% drop in sales due to tariffs" -> negative sentiment, LOW uncertainty
- "Sales could drop 5% or 30% depending on how tariffs develop" -> negative sentiment, HIGH uncertainty\
"""

LABELS = ["HIGH", "LOW"]

INSTRUCTION_BLOCK = """\
CLASSIFICATION LABELS:
- Allowed labels: HIGH | LOW

OUTPUT FORMAT (JSON only, no markdown):
{
  "reasoning": "<1-2 sentences explaining variance assessment>",
  "classification": "HIGH|LOW"
}
"""

FEW_SHOT_BLOCK = """\
## Few-Shot Examples

Example 1:
<statement>
Point, we do have the business plan or targets for 2026 prepared last year. We are currently on the process of evaluating it because of the geopolitical issues that we are encountering and we are expecting project delays due to logistical supply uncertainties, which we have no control.
</statement>
Output:
{"reasoning": "The speaker explicitly mentions unpredictable project delays and acknowledges having no control due to ongoing logistical issues.", "classification": "HIGH"}

Example 2:
<statement>
We have evaluated our business plan or targets for 2026 to reflect the confirmed impact of current geopolitical issues. We are now guiding to a six-month delay across all major projects due to the structural breakdown in logistical supply chains.
</statement>
Output:
{"reasoning": "Despite the negative sentiment of a delay, the duration is definitively bounded at exactly six months.", "classification": "LOW"}

Example 3:
<statement>
So as we enter -- potentially enter a period where inflation is lower or higher. We'll manage the commodities as they come through. We'll focus on the lowest prices we can focus on. We had 6,200 rollbacks in Walmart U.S. this quarter, up about 23% from a year ago. So we'll just continue to focus on low prices.
</statement>
Output:
{"reasoning": "The speaker notes inflation could be 'lower or higher' indicating a lack of clear foresight on macroeconomic conditions.", "classification": "HIGH"}

Example 4:
<statement>
We worked hard to mitigate grocery inflation as tariff-related costs lifted prices across many categories. We're seeing share gains in GM and in fashion. We've had several quarters in a row of mid-single-digit sales growth.
</statement>
Output:
{"reasoning": "The speaker confidently reports historical performance and current share gains without hedging about future variance.", "classification": "LOW"}
"""

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
        
        # 提取 high 不确定性的句子
        if "high" in row:
            eval_items.append({
                "id": f"{pair_id}_HIGH",
                "pair_id": pair_id,
                "topic": topic,
                "statement": row["high"],
                "label": "HIGH"
            })
        # 提取 low 不确定性的句子（兼容旧字段 no）
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

def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Safely extract JSON by finding the first complete brace block.
    """
    t = (text or "").strip()
    if not t:
        return None

    # Find the first opening brace
    start = t.find("{")
    if start == -1:
        return None
        
    # Match the closing brace
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start:i+1])
                except Exception:
                    return None
    return None

# ---------------- Prompt Builder ----------------

def build_prompt(statement: str, zero_shot: bool = False) -> str:
    header = "You are an expert in economics and financial communications. Your task is to classify the following economic statement based on its level of uncertainty.\n\n"
    
    query = (
        "\n## Task\n"
        "Classify the following statement based on the definitions and examples above.\n"
        f"<statement>\n{statement.strip()}\n</statement>\n\n"
        "JSON:\n"
    )
    
    # 👇 动态拼接 Prompt
    prompt = header + DEF_BLOCK + "\n\n" + INSTRUCTION_BLOCK + "\n"
    
    # 如果不是 zero_shot，才把 examples 拼进去
    if not zero_shot:
        prompt += FEW_SHOT_BLOCK + "\n"
        
    prompt += query
    
    return prompt

# ---------------- Model Wrapper ----------------

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
        model_id, device_map=device_map, torch_dtype=torch_dtype, quantization_config=qcfg
    )
    model.eval()
    return model, tok

@torch.no_grad()
def generate_json_label(model, tok, prompt: str, max_new_tokens: int, temperature: float) -> Tuple[str, Optional[Dict[str, Any]]]:
    use_chat = hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None) is not None
    
    prefix = (
        "IMPORTANT OUTPUT CONSTRAINTS:\n"
        "- Output ONLY one JSON object.\n"
        "- JSON must start with '{' and end with '}'.\n\n"
    )
    
    bad_words_ids = []
    if tok.pad_token_id is not None:
        bad_words_ids.append([int(tok.pad_token_id)])
    if [0] not in bad_words_ids:
        bad_words_ids.append([0])

    eos_id = tok.eos_token_id
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        eos_token_id=eos_id,
        pad_token_id=eos_id,
        bad_words_ids=bad_words_ids,
        return_dict_in_generate=True,
    )
    if temperature > 0:
        gen_kwargs["temperature"] = float(temperature)

    full_prompt = prefix + prompt

    if use_chat:
        messages = [{"role": "user", "content": full_prompt}]
        inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    else:
        inputs = tok(full_prompt, return_tensors="pt").to(model.device)

    out = model.generate(**inputs, **gen_kwargs)
    
    prompt_len = inputs["input_ids"].shape[1]
    raw = tok.decode(out.sequences[0][prompt_len:], skip_special_tokens=True).strip()
    
    parsed = extract_first_json_object(raw)
    
    # Retry mechanism
    if parsed is None:
        repair_prompt = full_prompt + "\nRETRY. Return ONLY valid JSON format."
        if use_chat:
            messages = [{"role": "user", "content": repair_prompt}]
            inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        else:
            inputs = tok(repair_prompt, return_tensors="pt").to(model.device)
            
        out2 = model.generate(**inputs, **gen_kwargs)
        raw = tok.decode(out2.sequences[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        parsed = extract_first_json_object(raw)

    return raw, parsed

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate a model on the synthetic uncertainty dataset.")
    ap.add_argument("--data_json", type=str, required=True, help="Path to synthetic_pairs_generated.json")
    ap.add_argument("--out_dir", type=str, default="out_synthetic_eval", help="Directory to save results")
    ap.add_argument("--models", type=str, default="google/gemma-2-9b-it", help="Comma-separated list of HF model IDs")
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--quant", type=str, default="none", choices=["none", "8bit", "4bit"])
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--zero_shot", action="store_true", help="If set, removes the few-shot examples from the prompt.")
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
    models = {}
    for mid in model_ids:
        print(f"Loading model: {mid}...")
        models[mid] = load_model(mid, args.device_map, args.dtype, args.quant)

    preds_all_rows = []
    eval_gold = {mid: [] for mid in model_ids}
    eval_pred = {mid: [] for mid in model_ids}

    for i, item in enumerate(data, 1):
        item_id = item["id"]
        gold = normalize_label(item["label"])
        print(f"Processing {i}/{len(data)}: {item_id} (Gold: {gold})")

        prompt = build_prompt(item["statement"], args.zero_shot)

        for mid in model_ids:
            model, tok = models[mid]
            raw, parsed = generate_json_label(model, tok, prompt, args.max_new_tokens, args.temperature)

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

    eval_report = {
        "_meta": {
            "dataset": args.data_json,
            "total_items": len(data)
        },
        "models": {}
    }
    
    for mid in model_ids:
        golds = eval_gold[mid]
        preds = eval_pred[mid]
        n = len(golds)
        acc = sum(1 for g, p in zip(golds, preds) if g == p) / n if n > 0 else 0.0
        
        cm = {
            "HIGH": {"HIGH": 0, "LOW": 0}, 
            "LOW": {"HIGH": 0, "LOW": 0}
        }
        for g, p in zip(golds, preds):
            if g in cm and p in cm[g]: 
                cm[g][p] += 1
                
        eval_report["models"][mid] = {
            "accuracy": round(acc, 4),
            "total_evaluated_successfully": n,
            "confusion_matrix": cm,
            "metrics_breakdown": {
                "HIGH_precision": cm["HIGH"]["HIGH"] / (cm["HIGH"]["HIGH"] + cm["LOW"]["HIGH"]) if (cm["HIGH"]["HIGH"] + cm["LOW"]["HIGH"]) > 0 else 0,
                "HIGH_recall": cm["HIGH"]["HIGH"] / (cm["HIGH"]["HIGH"] + cm["HIGH"]["LOW"]) if (cm["HIGH"]["HIGH"] + cm["HIGH"]["LOW"]) > 0 else 0,
            }
        }

    report_path = os.path.join(args.out_dir, "performance_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, ensure_ascii=False, indent=2)

    print(f"[OK] Performance report saved to: {report_path}")
    print("\nEvaluation Complete! 🎉")

if __name__ == "__main__":
    main()