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

# ============================================================
# Assumed to be defined ABOVE this section :
DEF_BLOCK = (
    "DEFINITION OF UNCERTAINTY:\n\n"
    
    "Uncertainty measures the VARIANCE or SPREAD of possible outcomes, "
    "not the expected value of outcomes.\n\n"
    
    "EXAMPLES:\n"
    "UNCERTAINTY (variance):\n"
    "- 'Revenue could be anywhere from $50M to $200M' → wide range\n"
    "- 'It depends on whether the regulation passes' → binary outcomes far apart\n"
    "- 'Roll two dice' → 11 possible sums with different probabilities\n\n"
    
    "NO UNCERTAINTY (zero variance):\n"
    "- 'Revenue will be $100M' → single outcome\n"
    "- 'We will lose $50M due to the tariff' → bad but certain\n"
    "- '2 + 2 = 4' → deterministic\n\n"
    
    "KEY PRINCIPLE NOT ABOUT SENTIMENT:\n"
    "- 'We expect difficult market conditions' → negative sentiment, but if the difficulty is certain, this is LOW uncertainty\n"
    "- 'Sales will definitely drop 20%' → bad news but NO uncertainty\n"
)

# ====================================================================
# LABELS BLOCK (3-way classification)
# ================================================================
LABELS_BLOCK_3WAY = (
    "CLASSIFICATION LABELS:\n"
    "- NO_UNCERTAINTY: Single clear outcome, definitive answer, or precise prediction\n"
    "- INTERMEDIATE_UNCERTAINTY: Some range of outcomes or mild conditionality, but substantially bounded\n"
    "- HIGH_UNCERTAINTY: Wide range of outcomes, outcome depends on unknowns, or no ability to estimate\n\n"
    
    "OUTPUT FORMAT (JSON only, no markdown):\n"
    '{\n'
    '  "reasoning": "<1-2 sentences explaining variance assessment>",\n'
    '  "classification": "NO_UNCERTAINTY|INTERMEDIATE_UNCERTAINTY|HIGH_UNCERTAINTY"\n'
    '}\n'
)



# ======================================================================
# LABELS BLOCK (binary - simpler for interpretability)
#======================================================================
LABELS_BLOCK_BINARY = (
    "CLASSIFICATION LABELS:\n"
    "- CERTAIN: Single outcome or precise prediction with narrow range\n"
    "- UNCERTAIN: Wide range of outcomes, conditionality, or inability to estimate\n\n"
    
    "OUTPUT FORMAT (JSON only, no markdown):\n"
    '{\n'
    '  "reasoning": "<1-2 sentences explaining variance assessment>",\n'
    '  "classification": "CERTAIN|UNCERTAIN"\n'
    '}\n'
)

def labels_only_block(label_mode: str) -> str:
    labs = allowed_labels(label_mode)
    allowed = "|".join(labs)

    return (
        "CLASSIFICATION LABELS:\n"
        f"- Allowed labels (no descriptions): {allowed}\n\n"
        "OUTPUT FORMAT (JSON only, no markdown):\n"
        "{\n"
        '  "reasoning": "<1-2 sentences explaining variance assessment>",\n'
        f'  "classification": "{allowed}"\n'
        "}\n"
    )


# ============================================================

LABELS_3WAY = ["NO_UNCERTAINTY", "INTERMEDIATE_UNCERTAINTY", "HIGH_UNCERTAINTY"]
LABELS_BINARY = ["CERTAIN", "UNCERTAIN"]

EXPERIMENTS = {
    "no_def_no_label":  {"use_def": False, "use_label_desc": False},
    "def_no_label":     {"use_def": True,  "use_label_desc": False},
    "no_def_label":     {"use_def": False, "use_label_desc": True},
    "def_label":        {"use_def": True,  "use_label_desc": True},
}


# ---------------- IO ----------------

def load_json_list(path: str) -> List[Dict[str, Any]]:
    """
    Accepts:
      1) JSON array: [ {...}, {...}, ... ]
      2) JSONL: each line is a JSON object {...}
    """
    with open(path, "r", encoding="utf-8-sig") as f:
        txt = f.read().strip()

    if not txt:
        raise ValueError(f"Input file is empty: {path}")

    # Try JSON array
    try:
        obj = json.loads(txt)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass

    # Fallback JSONL
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {ln} of {path}: {e}") from None

    if not rows:
        raise ValueError(f"No valid JSON objects found in {path}")
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------- Utilities ----------------

def clip(s: str, n: int) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[: n - 12] + " ...[TRUNCATED]"


def allowed_labels(label_mode: str) -> List[str]:
    if label_mode == "3way":
        return LABELS_3WAY
    if label_mode == "binary":
        return LABELS_BINARY
    raise ValueError(f"Unknown label_mode: {label_mode}")


def normalize_label(lbl: Any, label_mode: str) -> Optional[str]:
    if not isinstance(lbl, str):
        return None
    s = lbl.strip().upper()

    if label_mode == "3way":
        return s if s in set(LABELS_3WAY) else None

    if label_mode == "binary":
        # accept already-binary
        if s in set(LABELS_BINARY):
            return s
        # map 3-way -> binary
        if s == "NO_UNCERTAINTY":
            return "CERTAIN"
        if s in ("INTERMEDIATE_UNCERTAINTY", "HIGH_UNCERTAINTY"):
            return "UNCERTAIN"
        return None

    raise ValueError(f"Unknown label_mode: {label_mode}")



def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    t = (text or "").strip()
    if not t:
        return None

    # Prefer fenced JSON if exists
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        blob = m.group(1).strip()
        try:
            obj = json.loads(blob)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    # Direct parse
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Brace scan for first balanced object
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
                blob = t[start:i+1]
                try:
                    obj = json.loads(blob)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None



# ---------------- Few-shot block ----------------

def build_fewshot_block(
    examples: List[Dict[str, Any]],
    label_mode: str,
    max_q_chars: int = 1800,
    max_a_chars: int = 1800,
) -> str:
    """
    Few-shot examples include:
      - question, answer
      - an OUTPUT JSON with the gold classification (more stable than "Label: X")
    """
    blocks = []
    for i, ex in enumerate(examples, 1):
        q = clip(ex.get("question", ""), max_q_chars)
        a = clip(ex.get("answer", ""), max_a_chars)

        gold = ex.get("label")
        gold_norm = normalize_label(gold, label_mode)
        if gold_norm is None:
            # If your demo pool is guaranteed clean, this should not happen.
            gold_norm = (gold or "").strip().upper()

        blocks.append(
            f"Example {i}\n"
            f"<question>\n{q}\n</question>\n"
            f"<answer>\n{a}\n</answer>\n"
            f'Output:\n{{"reasoning":"- (example)","classification":"{gold_norm}"}}\n'
        )
    return "\n".join(blocks).strip()


def build_prompt(
    fewshot_block: str,
    query_item: Dict[str, Any],
    *,
    use_def: bool,
    use_label_desc: bool,
    label_mode: str,
    max_q_chars: int = 1800,
    max_a_chars: int = 1800,
) -> str:
    header = (
        "You are an expert financial analyst specializing in analyzing earnings call transcripts.\n\n"
        "TASK: Classify the uncertainty level in the following Question-Answer pair from an earnings call.\n\n"
    )

    if use_def:
        header += DEF_BLOCK + "\n\n"

    if use_label_desc:
        header += (LABELS_BLOCK_3WAY if label_mode == "3way" else LABELS_BLOCK_BINARY) + "\n"
    else:
        header += labels_only_block(label_mode) + "\n"

    header += (
        "\nINSTRUCTIONS:\n"
        "1) Base your decision primarily on the ANSWER (the question can contain speculation).\n"
        "2) Provide 1-2 sentences of reasoning.\n"
        "3) Output strict JSON only (no markdown, no extra text).\n\n"
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


def select_global_demos_stratified(
    labeled: List[Dict[str, Any]],
    n_per_class: int,
    seed: int,
    label_mode: str,
) -> List[Dict[str, Any]]:
    """
    Select ONE global demo set (reused for all models + all experiments):
      - n_per_class per gold label (stratified)
      - deterministic given seed
    """
    rng = random.Random(seed)
    allowed = allowed_labels(label_mode)

    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ex in labeled:
        gold = normalize_label(ex.get("label"), label_mode)
        if gold is not None:
            by_label[gold].append(ex)

    demos: List[Dict[str, Any]] = []
    for lbl in allowed:
        pool = list(by_label.get(lbl, []))
        rng.shuffle(pool)
        if len(pool) < n_per_class:
            raise ValueError(f"Not enough labeled samples for {lbl}: have {len(pool)}, need {n_per_class}")
        demos.extend(pool[:n_per_class])

    # Stable order -> stable prompt
    def _safe_int(x: Any) -> int:
        try:
            return int(x)
        except Exception:
            return 10**18

    demos.sort(key=lambda x: (normalize_label(x.get("label"), label_mode) or "", _safe_int(x.get("id"))))
    return demos


# ---------------- Model wrapper ----------------

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
    use_chat = hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None) is not None

    prefix = (
        "IMPORTANT OUTPUT CONSTRAINTS:\n"
        "- Output ONLY one JSON object.\n"
        "- No markdown, no code fences.\n"
        "- No extra text before/after JSON.\n"
        "- JSON must start with '{' and end with '}'.\n\n"
    )

    # ---- Hard fix for Gemma-style PAD spam ----
    # Many tokenizers use pad_token_id=0. The model can still *generate* 0 unless we ban it.
    bad_words_ids = []
    if tok.pad_token_id is not None:
        bad_words_ids.append([int(tok.pad_token_id)])
    # Extra safety: also ban 0 if it's not already banned (common pad id)
    if [0] not in bad_words_ids:
        bad_words_ids.append([0])

    def _gen_kwargs(force_greedy: bool = False):
        eos_id = tok.eos_token_id
        kw = dict(
            max_new_tokens=max_new_tokens,
            min_new_tokens=8,
            do_sample=(False if force_greedy else (temperature > 0)),
            eos_token_id=eos_id,
            pad_token_id=eos_id,              # padding uses EOS (fine)
            bad_words_ids=bad_words_ids,
            return_dict_in_generate=True,
        )
        if (not force_greedy) and (temperature > 0):
            kw["temperature"] = float(temperature)
            kw["top_p"] = float(top_p)
        return kw

    def _clean_text(s: str) -> str:
        s = (s or "").strip()
        # strip any leading role artifacts some templates leak
        s = re.sub(r"^\s*(user|assistant|model)\s*\n+", "", s, flags=re.IGNORECASE).strip()
        return s

    def _run_one(_prompt: str, force_greedy: bool = False) -> str:
        full_prompt = prefix + _prompt

        if use_chat:
            messages = [{"role": "user", "content": full_prompt}]
            inputs = tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            out = model.generate(**inputs, **_gen_kwargs(force_greedy=force_greedy))

            # Decode full sequence + prompt sequence WITHOUT skipping special tokens
            full = tok.decode(out.sequences[0], skip_special_tokens=False)
            prompt_text = tok.decode(inputs["input_ids"][0], skip_special_tokens=False)

            if full.startswith(prompt_text):
                raw = full[len(prompt_text):]
            else:
                tail = prompt_text[-200:] if len(prompt_text) > 200 else prompt_text
                j = full.rfind(tail)
                raw = full[j + len(tail):] if j != -1 else full

            # IMPORTANT: do NOT re-tokenize. Just clean.
            raw = _clean_text(raw)

            # Debug
            prompt_len = int(inputs["input_ids"].shape[1])
            gen_ids = out.sequences[0][prompt_len:]
            print(
                "DEBUG pad_id=", tok.pad_token_id,
                "eos_id=", tok.eos_token_id,
                "DEBUG gen_len=", int(gen_ids.numel()),
                "first_gen_id=", int(gen_ids[0]) if gen_ids.numel() > 0 else None,
            )
            return raw

        # non-chat
        inputs = tok(full_prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, **_gen_kwargs(force_greedy=force_greedy))

        full = tok.decode(out.sequences[0], skip_special_tokens=False)
        prompt_text = tok.decode(inputs["input_ids"][0], skip_special_tokens=False)

        if full.startswith(prompt_text):
            raw = full[len(prompt_text):]
        else:
            tail = prompt_text[-200:] if len(prompt_text) > 200 else prompt_text
            j = full.rfind(tail)
            raw = full[j + len(tail):] if j != -1 else full

        raw = _clean_text(raw)
        return raw

    raw = _run_one(prompt, force_greedy=False)
    parsed = extract_first_json_object(raw)

    # Retry only if empty/unparsable. Force greedy.
    if (not raw) or (parsed is None):
        repair = (
            "RETRY. Return ONLY one JSON object exactly like:\n"
            '{"reasoning":"...","classification":"..."}\n'
            "No other text.\n\n"
        )
        raw2 = _run_one(repair + prompt, force_greedy=True)
        parsed2 = extract_first_json_object(raw2)
        if raw2 and (parsed2 is not None):
            return raw2, parsed2

    return raw, parsed




# ---------------- Eval + disagreements ----------------

def confusion_matrix(golds: List[str], preds: List[str], label_mode: str) -> Dict[str, Dict[str, int]]:
    labs = allowed_labels(label_mode)
    cm = {g: {p: 0 for p in labs} for g in labs}
    for g, p in zip(golds, preds):
        if g in cm and p in cm[g]:
            cm[g][p] += 1
    return cm


def all_equal(xs: List[str]) -> bool:
    return len(set(xs)) <= 1


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="out_fewshot")

    ap.add_argument(
        "--models",
        type=str,
        default="google/gemma-2-9b-it,Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct",
    )

    ap.add_argument("--label_mode", type=str, default="3way", choices=["3way", "binary"])
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

    # labeled pool (for demo selection + eval)
    labeled = [x for x in data if normalize_label(x.get("label"), args.label_mode) is not None]
    if len(labeled) == 0:
        raise ValueError(f"No labeled examples found for label_mode={args.label_mode}.")

    demos = select_global_demos_stratified(
        labeled=labeled,
        n_per_class=args.n_per_class,
        seed=args.seed,
        label_mode=args.label_mode,
    )
    demo_ids = []
    for x in demos:
        try:
            demo_ids.append(int(x["id"]))
        except Exception:
            raise ValueError("All demo examples must have integer-like 'id' field.")
    demo_id_set = set(demo_ids)

    fewshot_block = build_fewshot_block(demos, label_mode=args.label_mode)

    # models
    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    models: Dict[str, Tuple[Any, Any]] = {}
    for mid in model_ids:
        model, tok = load_model(mid, args.device_map, args.dtype, args.quant)
        models[mid] = (model, tok)

    run_items = data if args.limit <= 0 else data[: args.limit]

    # Run all 4 experiments
    for exp_name, cfg in EXPERIMENTS.items():
        preds_all_rows: List[Dict[str, Any]] = []
        by_item: Dict[int, Dict[str, Any]] = {}

        eval_gold: Dict[str, List[str]] = {mid: [] for mid in model_ids}
        eval_pred: Dict[str, List[str]] = {mid: [] for mid in model_ids}

        for item in run_items:
            item_id_raw = item.get("id", None)
            if item_id_raw is None:
                continue
            try:
                item_id = int(item_id_raw)
            except Exception:
                continue

            gold = normalize_label(item.get("label"), args.label_mode)

            if item_id not in by_item:
                by_item[item_id] = {
                    "id": item_id,
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "gold_label": gold,
                    "experiment": exp_name,
                    "label_mode": args.label_mode,
                    "demo_ids": demo_ids,
                    "predictions": {},
                }

            prompt = build_prompt(
                fewshot_block,
                item,
                use_def=cfg["use_def"],
                use_label_desc=cfg["use_label_desc"],
                label_mode=args.label_mode,
            )

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
                    pred_label = normalize_label(parsed.get("classification"), args.label_mode)
                    reasoning = parsed.get("reasoning")
                    if pred_label is not None and isinstance(reasoning, str) and reasoning.strip():
                        parse_ok = True

                rec = {
                    "id": item_id,
                    "model_id": mid,
                    "experiment": exp_name,
                    "label_mode": args.label_mode,
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

                # eval: only on gold-labeled AND exclude demos AND require valid pred
                if gold is not None and (item_id not in demo_id_set) and (pred_label is not None):
                    eval_gold[mid].append(gold)
                    eval_pred[mid].append(pred_label)

        # Write preds_all
        preds_path = os.path.join(args.out_dir, f"preds_all_{exp_name}.jsonl")
        write_jsonl(preds_path, preds_all_rows)

        # Disagreements: any parse fail OR valid labels differ across models
        disagreements: List[Dict[str, Any]] = []
        labs = allowed_labels(args.label_mode)

        for item_id, blob in by_item.items():
            labels_here = []
            any_missing = False
            for mid in model_ids:
                pl = blob["predictions"].get(mid, {}).get("pred_label")
                if pl not in labs:
                    any_missing = True
                    labels_here.append(None)
                else:
                    labels_here.append(pl)

            valid = [x for x in labels_here if x is not None]
            disagree = any_missing or (len(valid) >= 2 and not all_equal(valid))
            if disagree:
                disagreements.append(blob)

        disag_path = os.path.join(args.out_dir, f"disagreements_{exp_name}.jsonl")
        write_jsonl(disag_path, disagreements)

        # Eval report
        n_gold_total = len(labeled)
        n_demos = len(demos)
        n_eval_expected = len([x for x in labeled if int(x["id"]) not in demo_id_set])

        eval_report = {
            "_meta": {
                "experiment": exp_name,
                "label_mode": args.label_mode,
                "seed": args.seed,
                "n_per_class": args.n_per_class,
                "demo_ids": demo_ids,
                "n_gold_total": n_gold_total,
                "n_demos": n_demos,
                "n_eval_expected_excluding_demos": n_eval_expected,
                "models": model_ids,
            },
            "per_model": {},
        }

        for mid in model_ids:
            golds = eval_gold[mid]
            preds = eval_pred[mid]
            n = len(golds)
            acc = sum(1 for g, p in zip(golds, preds) if g == p) / n if n > 0 else None
            eval_report["per_model"][mid] = {
                "n_eval": n,
                "accuracy": acc,
                "confusion_matrix": confusion_matrix(golds, preds, args.label_mode) if n > 0 else None,
            }

        eval_path = os.path.join(args.out_dir, f"eval_labeled_{exp_name}.json")
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_report, f, ensure_ascii=False, indent=2)

        print(f"[OK] wrote: {preds_path}")
        print(f"[OK] wrote: {disag_path}  (n={len(disagreements)})")
        print(f"[OK] wrote: {eval_path}")


if __name__ == "__main__":
    main()
