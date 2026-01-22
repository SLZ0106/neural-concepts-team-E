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

    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--save_hidden", action="store_true")
    p.add_argument("--hidden_dir", type=str, default="out_hidden")
    p.add_argument("--save_token_logprobs", action="store_true")

    # Optional: naive split if qa_text not provided
    p.add_argument("--qa_regex", type=str, default=r"\bQ&A\b|\bQuestions and Answers\b", help="regex marker for Q&A section")
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

def naive_split_prepared_vs_qa(text: str, qa_regex: str) -> Tuple[str, str]:
    m = re.search(qa_regex, text, flags=re.IGNORECASE)
    if not m:
        return text, ""
    idx = m.start()
    return text[:idx].strip(), text[idx:].strip()

def build_messages(transcript: str) -> List[Dict[str, str]]:
    """
    Gemma chat template can be applied via tokenizer.apply_chat_template.
    We'll put all instruction in a single user message for reliability.
    """
    instruction = f"""
You are an economics research assistant analyzing corporate earnings call language.

Task:
1) Define and separate:
   - First-moment information (expected mean / sentiment / level effects)
   - Second-moment information (uncertainty / volatility / dispersion / lack of visibility)
2) Extract sources of uncertainty mentioned (e.g., inflation/costs, demand, supply chain, regulation, FX, geopolitics, monetary policy).
3) Provide evidence spans: short exact quotes (<= 25 words each) from the transcript for each category.
4) Output strictly valid JSON only (no markdown, no extra text).

JSON schema:
{{
  "first_moment": {{
    "score_0_1": float,
    "summary": string,
    "evidence": [string, ...]
  }},
  "second_moment_uncertainty": {{
    "score_0_1": float,
    "summary": string,
    "evidence": [string, ...],
    "sources": [{{"source": string, "score_0_1": float, "evidence": [string, ...]}}, ...]
  }},
  "omission_or_brevity": {{
    "is_brief": bool,
    "why_brief": string
  }}
}}

Transcript:
{transcript}
""".strip()

    return [{"role": "user", "content": instruction}]

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort JSON extraction if model emits extra tokens.
    """
    # Try direct load
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try find first {...} block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception:
        return None

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
    """
    Returns:
      - raw_text
      - parsed_json (best-effort)
      - stats: lengths, hedge markers
      - optional: token_logprobs, hidden_states (last layer)
    """
    # Ensure chat template is applied as recommended by HF model card. :contentReference[oaicite:2]{index=2}
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

    # Optional hidden states: we capture prompt forward pass hidden states
    hidden_pack = None
    if save_hidden:
        out_fwd = model(**inputs, output_hidden_states=True, use_cache=False)
        # Save only last layer hidden states for prompt tokens
        last_hidden = out_fwd.hidden_states[-1].detach().cpu()  # (1, seq, dim)
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

    # Stats: brevity + hedge markers
    out_tokens = len(gen_ids)
    prompt_tokens = prompt_len
    text_lower = raw_text.lower()
    hedge_hits = {m: text_lower.count(m) for m in HEDGE_MARKERS if m in text_lower}
    is_brief = (out_tokens < 120)  # heuristic threshold; tune in your study

    result: Dict[str, Any] = {
        "raw_text": raw_text,
        "parsed_json": parsed,
        "stats": {
            "prompt_tokens": int(prompt_tokens),
            "output_tokens": int(out_tokens),
            "brevity_ratio": float(out_tokens / max(prompt_tokens, 1)),
            "hedge_hits": hedge_hits,
            "is_brief_heuristic": bool(is_brief),
        },
    }

    if save_token_logprobs and out.scores is not None:
        # out.scores: list of (vocab,) logits per generated step
        # Compute logprob of the actually chosen token at each step
        token_logprobs = []
        for step_logits, tok_id in zip(out.scores, gen_ids):
            logp = torch.log_softmax(step_logits[0], dim=-1)[tok_id].item()
            token_logprobs.append(float(logp))
        result["token_logprobs"] = {
            "mean_logprob": float(sum(token_logprobs) / max(len(token_logprobs), 1)),
            "per_token_logprob": token_logprobs[:200],  # truncate to keep files manageable
        }

    if hidden_pack is not None:
        result["hidden_states"] = {
            # Store shapes only in json; tensors should be saved separately by caller
            "prompt_last_hidden_shape": list(hidden_pack["prompt_last_hidden"].shape),
        }
        result["_hidden_pack"] = hidden_pack  # internal use by caller for saving tensors

    return result

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

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
                    "parsed_json": pred["parsed_json"],
                    "stats": pred["stats"],
                }
                if args.save_token_logprobs and "token_logprobs" in pred:
                    record["token_logprobs"] = pred["token_logprobs"]

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

                # Save hidden tensors (optional)
                if args.save_hidden and "_hidden_pack" in pred and ex_id is not None:
                    pack = pred["_hidden_pack"]
                    save_path = os.path.join(args.hidden_dir, f"{ex_id}_{variant_name}.pt")
                    torch.save(pack, save_path)

if __name__ == "__main__":
    main()
