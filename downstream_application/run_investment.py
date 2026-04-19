"""
Investment-allocation task with optional activation steering.

For each CFO statement in sentences_with_context.json, prompt the model to
split $1000 between US Treasuries and the company's stock. Runs N times per
statement to capture sampling variance.

Optionally steers the residual stream at a chosen layer by adding
`alpha * direction` at every forward pass (prefill + each decoding step),
using a PyTorch forward hook registered on the target layer.

Example (no steering):
    python run_investment.py \
        --data data/sentences_with_context.json \
        --out_dir results/ \
        --alphas 0 \
        --n_runs 10

Example (alpha sweep):
    python run_investment.py \
        --data data/sentences_with_context.json \
        --direction /projects/frink/wang.xil/concepts_E/script_outputs/synthetic_direction/direction.npy \
        --layer 12 \
        --alphas="-4,-2,0,2,4" \
        --out_dir results/ \
        --n_runs 10
"""

import argparse
import json
import os
import re
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_HEADER = """You are a financial analyst. Here is recent information about {company} ({ticker}):

  Sector:              {sector}
  3-month return:      {return_3m}%
  30-day volatility:   {vol_30d}% (annualized)
  Beta:                {beta}
  Market cap:          ${market_cap_b}B"""

PROMPT_STATEMENT_BLOCK = """


The company's CFO made the following statement during their most recent earnings call:

  "{sentence}\""""

# PROMPT_FOOTER = """
#
#
#
# Now, you have $1000 to split between two investment options:
# 1. US Treasuries -- low risk and low return.
# 2. The stock for the company {company} ({ticker}) -- higher risk but potentially higher return.
#
# Based on {basis}, how much do you put in each investment option? Explain with 2-3 sentences and then give your final answer as a JSON object on its own line, using EXACTLY this schema (integer dollar amounts that sum to 1000, no extra keys, no comments, no units):
#
# {{"treasuries": <int>, "stock": <int>}}"""

PROMPT_FOOTER = """



You have $1000 to split between US Treasuries (low risk) and {company} ({ticker}) stock (higher risk).

Based on {basis}, how many dollars (0–1000) do you put in the stock? The rest goes to Treasuries. Answer with one sentence of reasoning followed by a single integer on the last line."""


def format_value(v):
    if v is None:
        return "N/A"
    if isinstance(v, float) and np.isnan(v):
        return "N/A"
    return str(v)


def build_prompt(record, include_statement=True):
    fc = record["financial_context"]
    header = PROMPT_HEADER.format(
        company=record["company"],
        ticker=record["ticker"],
        sector=fc.get("sector") or "N/A",
        return_3m=format_value(fc.get("return_3m")),
        vol_30d=format_value(fc.get("vol_30d")),
        beta=format_value(fc.get("beta")),
        market_cap_b=format_value(fc.get("market_cap_b")),
    )
    if include_statement:
        middle = PROMPT_STATEMENT_BLOCK.format(sentence=record["sentence"])
        basis = "both the recent financial data and this statement"
    else:
        middle = ""
        basis = "the recent financial data"
    footer = PROMPT_FOOTER.format(
        company=record["company"],
        ticker=record["ticker"],
        basis=basis,
    )
    return header + middle + footer


_INTEGER_RE = re.compile(r"\b(\d+)\b")


def parse_allocation(text):
    """Extract stock allocation from the model's response.

    The prompt asks for a single integer (dollars in stock, 0-1000) on the
    last line. We take the last integer in [0, 1000] found in the response.

    Returns (parsed_dict_or_None, treasuries_amount, stock_amount).
    """
    matches = _INTEGER_RE.findall(text)
    for m in reversed(matches):
        v = int(m)
        if 0 <= v <= 1000:
            stock = float(v)
            treasuries = 1000.0 - stock
            return {"stock": v, "treasuries": int(treasuries)}, treasuries, stock
    return None, None, None


def generate_once(model, tokenizer, prompt, max_new_tokens, temperature, top_p,
                  layer, direction_tensor, alpha, steer_prefill=False):
    """Run one sampled generation, optionally steering at every forward pass.

    When steering is active, a forward hook on model.model.layers[layer] adds
    alpha * direction to the residual stream output at every forward pass
    (prefill + each decoding step).

    Two-step tokenization matches activation_patching_highlow.apply_template:
    apply_chat_template(tokenize=False) → string, then tokenizer.encode()
    → list[int], to avoid the tokenizers.Encoding dtype issue.
    """
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    ids = tokenizer.encode(formatted, add_special_tokens=False)
    input_ids = torch.tensor([ids], device=model.device)
    prompt_len = input_ids.shape[1]

    gen_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )

    steering = direction_tensor is not None and alpha != 0.0
    handle = None

    if steering:
        shift = alpha * direction_tensor  # (hidden,) — broadcast over batch & seq
        prefill_tokens = [0]
        decode_tokens = [0]

        def _hook(module, inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            seq_len = hidden.shape[1]
            is_decode = seq_len == 1
            if is_decode or steer_prefill:
                hidden = hidden + shift
                if is_decode:
                    decode_tokens[0] += 1        # always 1 per decoding step
                else:
                    prefill_tokens[0] += seq_len  # full prompt length in one pass
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        handle = model.model.layers[layer].register_forward_hook(_hook)

    try:
        with torch.inference_mode():
            out = model.generate(**gen_kwargs)
    finally:
        if handle is not None:
            handle.remove()
            print(f"    [steer] layer {layer} | prefill={prefill_tokens[0]} tokens steered, "
                  f"decode={decode_tokens[0]} tokens steered "
                  f"(max_new_tokens={gen_kwargs['max_new_tokens']})")

    gen_ids = out[0, prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def alpha_tag(alpha):
    """Stable, filename-safe string for an alpha value."""
    a = float(alpha)
    return str(int(a)) if a.is_integer() else str(a)


def main():
    parser = argparse.ArgumentParser(description="Investment allocation task with optional activation steering")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--data", required=True, help="sentences_with_context.json")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory. One file per alpha is written: "
                             "investment_alpha{a}_layer{L}[_nostmt].json")
    parser.add_argument("--n_runs", type=int, default=10,
                        help="Number of generations per statement")
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)

    # Steering arguments
    parser.add_argument("--direction", default=None,
                        help="Path to .npy steering direction (unit-norm). Omit to skip steering.")
    parser.add_argument("--layer", type=int, default=12,
                        help="Layer index to steer at (matches extract_direction.py default)")
    parser.add_argument("--alphas", default="0",
                        help="Comma-separated list of scaling factors to sweep. "
                             "E.g. '-4,-2,0,2,4'. alpha=0 runs without steering.")

    # Filtering
    parser.add_argument("--subset", default=None,
                        help="If set, only run records whose 'subset' matches this value")
    parser.add_argument("--limit", type=int, default=None,
                        help="Debug: only process the first N records")

    # Prompt variants
    parser.add_argument("--no-statement", dest="no_statement", action="store_true",
                        help="Omit the CFO earnings-call statement; use only financial context.")
    parser.add_argument("--steer_prefill", action="store_true",
                        help="Also steer prefill tokens (default: decode steps only).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each raw model response to stdout.")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    alphas = [float(a) for a in args.alphas.split(",") if a.strip() != ""]
    if not alphas:
        raise ValueError("--alphas must contain at least one value")
    print(f"Alpha sweep: {alphas}")

    # -- Load data -------------------------------------------------------------
    with open(args.data) as f:
        records = json.load(f)
    n_raw = len(records)
    records = [r for r in records if r.get("financial_context") is not None]
    n_dropped = n_raw - len(records)
    if n_dropped:
        print(f"Dropped {n_dropped} records with null financial_context")
    if args.subset:
        records = [r for r in records if r.get("subset") == args.subset]
    if args.limit:
        records = records[:args.limit]
    print(f"Loaded {len(records)} records from {args.data}")

    # -- Load model ------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading {args.model} on {device} (dtype={dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=device
    )
    model.eval()
    print(f"Model loaded. Layers: {model.config.num_hidden_layers}")

    # -- Direction (loaded once; shared across all non-zero alphas) ------------
    direction_tensor = None
    direction_norm = None
    needs_direction = any(a != 0.0 for a in alphas)
    if needs_direction:
        if args.direction is None:
            raise ValueError("Non-zero alpha requested but --direction was not provided")
        direction_np = np.load(args.direction)
        direction_norm = float(np.linalg.norm(direction_np))
        print(f"Loaded direction: shape={direction_np.shape}, norm={direction_norm:.6f}")
        direction_tensor = torch.tensor(direction_np, dtype=dtype, device=device)
    else:
        print("All alphas are 0 — no direction loaded")

    # -- Sweep over alphas -----------------------------------------------------
    nostmt_suffix = "_nostmt" if args.no_statement else ""
    t0_total = time.time()

    for alpha in alphas:
        steering_active = alpha != 0.0 and direction_tensor is not None
        out_name = f"investment_alpha{alpha_tag(alpha)}_layer{args.layer}{nostmt_suffix}.json"
        out_path = os.path.join(args.out_dir, out_name)

        config = {
            "model": args.model,
            "data": args.data,
            "n_runs": args.n_runs,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "direction": args.direction,
            "direction_norm": direction_norm,
            "layer": args.layer if steering_active else None,
            "alpha": alpha,
            "steering_active": steering_active,
            "subset": args.subset,
            "n_records": len(records),
            "include_statement": not args.no_statement,
        }

        print(f"\n{'='*60}")
        print(f"Alpha = {alpha}   ({'steering' if steering_active else 'baseline'})")
        print(f"Output: {out_path}")
        print(f"{'='*60}")

        results = []
        t0 = time.time()

        for ri, record in enumerate(records):
            prompt = build_prompt(record, include_statement=not args.no_statement)
            runs = []
            for run_idx in range(args.n_runs):
                response = generate_once(
                    model, tokenizer, prompt,
                    args.max_new_tokens, args.temperature, args.top_p,
                    args.layer, direction_tensor, alpha,
                    steer_prefill=args.steer_prefill,
                )
                if args.verbose:
                    print(f"  [run {run_idx}] {response}\n")
                parsed, treasuries, stock = parse_allocation(response)
                runs.append({
                    "run_idx": run_idx,
                    "response": response,
                    "parsed": parsed,
                    "treasuries": treasuries,
                    "stock": stock,
                    "parse_ok": parsed is not None,
                })

            t_vals = [r["treasuries"] for r in runs if r["treasuries"] is not None]
            s_vals = [r["stock"] for r in runs if r["stock"] is not None]

            entry = {
                "id": record["id"],
                "ticker": record["ticker"],
                "company": record["company"],
                "sentence": record["sentence"],
                "subset": record.get("subset"),
                "financial_context": record.get("financial_context"),
                "call_time": record.get("call_time"),
                "runs": runs,
                "n_parsed": len(t_vals),
                "mean_treasuries": float(np.mean(t_vals)) if t_vals else None,
                "mean_stock": float(np.mean(s_vals)) if s_vals else None,
                "std_treasuries": float(np.std(t_vals)) if len(t_vals) > 1 else None,
                "std_stock": float(np.std(s_vals)) if len(s_vals) > 1 else None,
            }
            results.append(entry)

            elapsed = time.time() - t0
            mean_t = f"{entry['mean_treasuries']:.0f}" if entry['mean_treasuries'] is not None else "N/A"
            mean_s = f"{entry['mean_stock']:.0f}" if entry['mean_stock'] is not None else "N/A"
            print(f"[a={alpha} {ri+1}/{len(records)}] {record['ticker']:<8} "
                  f"parsed={entry['n_parsed']}/{args.n_runs}  "
                  f"mean_treasuries={mean_t}  mean_stock={mean_s}  "
                  f"elapsed={elapsed:.1f}s")

            if (ri + 1) % 5 == 0:
                with open(out_path, "w") as f:
                    json.dump({"config": config, "results": results}, f, indent=2)

        with open(out_path, "w") as f:
            json.dump({"config": config, "results": results}, f, indent=2)
        print(f"Saved -> {out_path}   (alpha={alpha}, took {time.time() - t0:.1f}s)")

    print(f"\nSweep complete. Total time: {time.time() - t0_total:.1f}s")


if __name__ == "__main__":
    main()
