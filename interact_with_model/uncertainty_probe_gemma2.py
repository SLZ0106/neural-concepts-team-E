#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed,
)

HEDGE_MARKERS = [
    "uncertain", "uncertainty", "may", "might", "could", "possibly", "likely",
    "we think", "we believe", "we expect", "hard to", "difficult to", "not sure",
    "unknown", "cannot predict", "unable to", "visibility", "headwind",
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="google/gemma-2-9b-it")
    p.add_argument("--input_jsonl", type=str, required=True)
    p.add_argument("--output_jsonl", type=str, required=True)

    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--quant", type=str, default="none", choices=["none", "8bit", "4bit"])

    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--save_hidden", action="store_true")
    p.add_argument("--hidden_dir", type=str, default="out_hidden")
    p.add_argument("--save_token_logprobs", action="store_true")

    p.add_argument("--qa_regex", type=str, default=r"\bQ&A\b|\bQuestions and Answers\b")
    return p.parse_args()

def get_dtype(dtype_str: str):
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float16":
        return torch.float16
    return torch.float32

def load_model_and_tokenizer(model_id: str, device_map: str, dtype: torch.dtype, quant: str):
    quant_cfg = None
    if quant == "8bit":
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
    elif quant == "4bit":
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=None if quant_cfg is not None else dtype,
        quantization_config=quant_cfg,
    )
    model.eval()
    return model, tokenizer

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def naive_split_prepared_vs_qa(text: str, qa_regex: str) -> Tuple[str, str]:
    m = re.search(qa_regex, text, flags=re.IGNORECASE)
    if not m:
        return text, ""
    idx = m.start()
    return text[:idx].strip(), text[idx:].strip()

def build_messages(transcript: str) -> List[Dict[str, str]]:
    """
    Minimal JSON output:
      - uncertainty_score in [0,1]
      - summary (1-2 sentences)
      - evidence: 1-3 short exact quotes (<= 25 words each)
    """
    instruction = f"""
You are an economics research assistant analyzing uncertainty language in corporate earnings calls.

Return a JSON object ONLY (no markdown, no extra text) with this schema:
{{
  "uncertainty_score": float, 
  "summary": string,
  "evidence": [string, ...]
}}

Definition:
- uncertainty_score measures second-moment uncertainty (lack of visibility, conditionality, inability to estimate, wide range of outcomes).
- Do NOT treat positive/negative sentiment or clear numeric guidance as uncertainty by itself.

Rules:
- uncertainty_score must be between 0 and 1.
- summary must be 1-2 sentences.
- evidence must contain 1-3 short exact quotes copied verbatim from the transcript (<= 25 words each).

Transcript:
{transcript}
""".strip()
    return [{"role": "user", "content": instruction}]

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    # Try direct JSON parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Best-effort: extract first {...} block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception:
        return None

def validate_min_schema(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Ensure schema keys exist and types are reasonable.
    If invalid, return None so caller can record parse failure.
    """
    if not isinstance(obj, dict):
        return None
    if "uncertainty_score" not in obj or "summary" not in obj or "evidence" not in obj:
        return None
    try:
        score = float(obj["uncertainty_score"])
    except Exception:
        return None
    if not (0.0 <= score <= 1.0):
        return None
    if not isinstance(obj["summary"], str):
        return None
    if not isinstance(obj["evidence"], list) or not all(isinstance(x, str) for x in obj["evidence"]):
        return None

    # Normalize (e.g., clamp score just in case)
    obj["uncertainty_score"] = float(max(0.0, min(1.0, score)))
    obj["summary"] = obj["summary"].strip()
    obj["evidence"] = [e.strip() for e in obj["evidence"] if e.strip()]
    return obj

@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    save_token_logprobs: bool = False,
    save_hidden: bool = False,
) -> Dict[str, Any]:
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=top_p,
        return_dict_in_generate=True,
        output_scores=save_token_logprobs,
    )

    hidden_pack = None
    if save_hidden:
        out_fwd = model(**inputs, output_hidden_states=True, use_cache=False)
        last_hidden = out_fwd.hidden_states[-1].detach().cpu()
        hidden_pack = {
            "prompt_last_hidden": last_hidden,
            "prompt_input_ids": inputs["input_ids"].detach().cpu(),
            "prompt_attention_mask": inputs["attention_mask"].detach().cpu(),
        }

    out = model.generate(**inputs, **gen_kwargs)
    seq = out.sequences[0]
    gen_ids = seq[prompt_len:]
    raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    parsed = extract_json_from_text(raw_text)
    parsed_min = validate_min_schema(parsed) if parsed is not None else None

    out_tokens = int(len(gen_ids))
    prompt_tokens = int(prompt_len)
    text_lower = raw_text.lower()
    hedge_hits = {m: text_lower.count(m) for m in HEDGE_MARKERS if m in text_lower}
    is_brief = (out_tokens < 120)

    result: Dict[str, Any] = {
        "raw_text": raw_text,
        "parsed_json": parsed_min,
        "stats": {
            "prompt_tokens": prompt_tokens,
            "output_tokens": out_tokens,
            "brevity_ratio": float(out_tokens / max(prompt_tokens, 1)),
            "hedge_hits": hedge_hits,
            "is_brief_heuristic": bool(is_brief),
        },
    }

    if save_token_logprobs and out.scores is not None:
        token_logprobs = []
        for step_logits, tok_id in zip(out.scores, gen_ids):
            logp = torch.log_softmax(step_logits[0], dim=-1)[tok_id].item()
            token_logprobs.append(float(logp))
        result["token_logprobs"] = {
            "mean_logprob": float(sum(token_logprobs) / max(len(token_logprobs), 1)),
            "per_token_logprob": token_logprobs[:200],
        }

    if hidden_pack is not None:
        result["hidden_states"] = {
            "prompt_last_hidden_shape": list(hidden_pack["prompt_last_hidden"].shape),
        }
        result["_hidden_pack"] = hidden_pack

    return result

def main():
    args = parse_args()
    set_seed(args.seed)

    dtype = get_dtype(args.dtype)
    model, tokenizer = load_model_and_tokenizer(args.model_id, args.device_map, dtype, args.quant)

    if args.save_hidden:
        os.makedirs(args.hidden_dir, exist_ok=True)

    with open(args.output_jsonl, "w", encoding="utf-8") as out_f:
        for ex in iter_jsonl(args.input_jsonl):
            ex_id = ex.get("id", None)
            text_full = ex["text"]

            prepared = ex.get("prepared_remarks", None)
            qa = ex.get("qa_text", None)
            if prepared is None and qa is None:
                prepared, qa = naive_split_prepared_vs_qa(text_full, args.qa_regex)

            variants = {
                "full": text_full,
                "no_qa": prepared if prepared else text_full,
                "qa_only": qa if qa else "",
            }

            for variant_name, variant_text in variants.items():
                if not variant_text.strip():
                    continue

                messages = build_messages(variant_text)
                pred = generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    messages=messages,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    save_token_logprobs=args.save_token_logprobs,
                    save_hidden=args.save_hidden,
                )

                record = {
                    "id": ex_id,
                    "variant": variant_name,
                    "model_id": args.model_id,
                    "raw_text": pred["raw_text"],
                    "parsed_json": pred["parsed_json"],  # only the minimal schema or None
                    "stats": pred["stats"],
                }
                if args.save_token_logprobs and "token_logprobs" in pred:
                    record["token_logprobs"] = pred["token_logprobs"]

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

                if args.save_hidden and "_hidden_pack" in pred and ex_id is not None:
                    save_path = os.path.join(args.hidden_dir, f"{ex_id}_{variant_name}.pt")
                    torch.save(pred["_hidden_pack"], save_path)

if __name__ == "__main__":
    main()
