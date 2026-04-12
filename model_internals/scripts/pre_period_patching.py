"""
Patch 19 pre-period positions (excluding period) at layer 12 and compare with
period-only and combined patching. Runs on 100 pairs for both datasets.

Usage:
    conda activate ./econ_env
    python run_pre_period_patch.py --dataset synthetic
    python run_pre_period_patch.py --dataset templated
"""

import sys, os, json, argparse
import torch, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from activation_patching_highlow import (
    build_messages, build_messages_synthetic, apply_template,
    find_last_period_pos, get_label_token_ids, compute_metric, load_pairs,
)
from nnsight import LanguageModel

PRE_PERIOD_TOKENS = 20
LAYER = 12
METRIC = "prob_diff"


def run_pre_period_patching(pair, model, tokenizer, low_id, high_id,
                            dataset, demo_certain=None, demo_uncertain=None):
    if dataset == "synthetic":
        clean_msgs = build_messages_synthetic(pair["uncertain"])
        corrupt_msgs = build_messages_synthetic(pair["certain"])
    else:
        clean_msgs = build_messages(demo_certain, demo_uncertain, pair["uncertain"])
        corrupt_msgs = build_messages(demo_certain, demo_uncertain, pair["certain"])

    clean_ids = apply_template(clean_msgs, tokenizer, tokenize=True)
    corrupt_ids = apply_template(corrupt_msgs, tokenizer, tokenize=True)
    clean_input = torch.tensor([clean_ids])
    corrupt_input = torch.tensor([corrupt_ids])

    clean_period = find_last_period_pos(clean_ids, tokenizer)
    corrupt_period = find_last_period_pos(corrupt_ids, tokenizer)
    clean_start = max(0, clean_period - PRE_PERIOD_TOKENS)
    corrupt_start = max(0, corrupt_period - PRE_PERIOD_TOKENS)

    n_pos = min(len(clean_ids) - clean_start, len(corrupt_ids) - corrupt_start)
    period_offset = min(clean_period - clean_start, corrupt_period - corrupt_start)

    # Collect clean activations at layer 12
    clean_acts = {}
    with model.trace(clean_input):
        for offset in range(n_pos):
            pos = clean_start + offset
            clean_acts[offset] = (
                model.model.layers[LAYER].output[0][pos, :].save()
            )
        clean_logits = model.output.logits[0, -1, :].save()
    clean_base = compute_metric(clean_logits, high_id, low_id, METRIC)

    with model.trace(corrupt_input):
        corrupt_logits = model.output.logits[0, -1, :].save()
    corrupt_base = compute_metric(corrupt_logits, high_id, low_id, METRIC)

    results = {}

    # Config 1: period only
    with model.trace(corrupt_input):
        cpos = corrupt_start + period_offset
        model.model.layers[LAYER].output[0][cpos, :] = clean_acts[period_offset]
        patched_logits = model.output.logits[0, -1, :].save()
    results["period_only"] = compute_metric(patched_logits, high_id, low_id, METRIC)

    # Config 2: 19 pre-period positions (exclude period)
    pre_offsets = list(range(max(0, period_offset - 19), period_offset))
    with model.trace(corrupt_input):
        for off in pre_offsets:
            cpos = corrupt_start + off
            model.model.layers[LAYER].output[0][cpos, :] = clean_acts[off]
        patched_logits = model.output.logits[0, -1, :].save()
    results["pre_period_19"] = compute_metric(patched_logits, high_id, low_id, METRIC)

    # Config 3: 19 pre-period + period
    all_offsets = list(range(max(0, period_offset - 19), period_offset + 1))
    with model.trace(corrupt_input):
        for off in all_offsets:
            cpos = corrupt_start + off
            model.model.layers[LAYER].output[0][cpos, :] = clean_acts[off]
        patched_logits = model.output.logits[0, -1, :].save()
    results["pre_period_19_plus_period"] = compute_metric(patched_logits, high_id, low_id, METRIC)

    return results, clean_base, corrupt_base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["synthetic", "templated"], required=True)
    parser.add_argument("--n_pairs", type=int, default=100)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    args = parser.parse_args()

    if args.dataset == "synthetic":
        pairs_path = "neural-concepts-team-E/synthetic_data/API/synthetic_pairs_200.json"
    else:
        pairs_path = "neural-concepts-team-E/synthetic_data/templated/templated_pairs_200.json"

    pairs = load_pairs(pairs_path)
    if args.dataset == "templated":
        demo = pairs[0]
        test_pairs = pairs[1 : 1 + args.n_pairs]
        dk = dict(demo_certain=demo["certain"], demo_uncertain=demo["uncertain"])
    else:
        test_pairs = pairs[: args.n_pairs]
        dk = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LanguageModel(args.model, device_map=device, dispatch=True)
    tokenizer = model.tokenizer
    low_id, high_id = get_label_token_ids(tokenizer)

    configs = ["period_only", "pre_period_19", "pre_period_19_plus_period"]
    all_results = {k: [] for k in configs}
    clean_bases, corrupt_bases = [], []

    print(f"Dataset: {args.dataset}, Pairs: {len(test_pairs)}, Layer: {LAYER}")
    for i, pair in enumerate(test_pairs):
        res, cb, ub = run_pre_period_patching(
            pair, model, tokenizer, low_id, high_id,
            dataset=args.dataset, **dk,
        )
        for k in configs:
            all_results[k].append(res[k])
        clean_bases.append(cb)
        corrupt_bases.append(ub)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(test_pairs)}")

    mean_clean = np.mean(clean_bases)
    mean_corrupt = np.mean(corrupt_bases)
    gap = mean_clean - mean_corrupt

    summary = {
        "dataset": args.dataset,
        "layer": LAYER,
        "n_pairs": len(test_pairs),
        "mean_clean_baseline": float(mean_clean),
        "mean_corrupt_baseline": float(mean_corrupt),
    }
    print(f"\nClean: {mean_clean:.4f}, Corrupt: {mean_corrupt:.4f}, Gap: {gap:.4f}")
    for k in configs:
        m = float(np.mean(all_results[k]))
        r = (m - mean_corrupt) / gap if gap != 0 else 0
        summary[k] = {"mean_metric": m, "recovery": float(r)}
        print(f"  {k:30s}: metric={m:+.4f}  recovery={r:.1%}")

    out_path = f"script_outputs/advanced_analysis/multi_position_patching/{args.dataset}/pre_period_patching.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
