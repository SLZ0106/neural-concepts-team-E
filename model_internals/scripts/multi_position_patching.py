"""
Multi-position activation patching: patch K consecutive token positions
simultaneously (ending at the period token) and measure how the patching
effect scales with K.

For each K in a configurable list, patches K consecutive positions ending
at the final period token of the test statement, across all layers.
Produces scaling curves, heatmaps, and recovery fraction plots.
"""

import sys
import os
import json
import argparse
import time

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nnsight import LanguageModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from activation_patching_highlow import (
    build_messages, build_messages_synthetic, apply_template,
    find_last_period_pos, get_label_token_ids, compute_metric, load_pairs,
    SYNTHETIC_FEW_SHOT_DEMOS,
)

PRE_PERIOD_TOKENS = 20


def run_multi_patching_for_pair(
    pair, model, tokenizer, n_layers, low_id, high_id,
    k_values, dataset="templated", demo_certain=None, demo_uncertain=None,
    metric="logit_diff",
):
    """
    Patch K consecutive positions ending at the period token, for each K in k_values.

    Returns:
        patched_by_k: dict mapping K -> np.array of shape (n_layers,)
        clean_baseline: float
        corrupt_baseline: float
        period_offset: int (offset of period in the patching window)
    """
    # 1. Build clean/corrupt messages and tokenize
    if dataset == "synthetic":
        clean_msgs   = build_messages_synthetic(pair["uncertain"])
        corrupt_msgs = build_messages_synthetic(pair["certain"])
    else:
        clean_msgs   = build_messages(demo_certain, demo_uncertain, pair["uncertain"])
        corrupt_msgs = build_messages(demo_certain, demo_uncertain, pair["certain"])

    clean_ids   = apply_template(clean_msgs,   tokenizer, tokenize=True)
    corrupt_ids = apply_template(corrupt_msgs, tokenizer, tokenize=True)

    clean_input   = torch.tensor([clean_ids])
    corrupt_input = torch.tensor([corrupt_ids])

    # 2. Find period positions
    clean_period_pos   = find_last_period_pos(clean_ids,   tokenizer)
    corrupt_period_pos = find_last_period_pos(corrupt_ids, tokenizer)

    # 3. Compute start positions
    clean_start_pos   = max(0, clean_period_pos   - PRE_PERIOD_TOKENS)
    corrupt_start_pos = max(0, corrupt_period_pos - PRE_PERIOD_TOKENS)

    # 4. Number of positions from start to last token
    clean_n_positions   = len(clean_ids)   - clean_start_pos
    corrupt_n_positions = len(corrupt_ids) - corrupt_start_pos
    n_positions = min(clean_n_positions, corrupt_n_positions)

    # 5. Period offset within the patching window
    period_offset = min(
        clean_period_pos   - clean_start_pos,
        corrupt_period_pos - corrupt_start_pos,
    )

    # 6. Collect clean activations at ALL positions, ALL layers in a single trace
    clean_acts = {}
    with model.trace(clean_input):
        for l in range(n_layers):
            for offset in range(n_positions):
                pos = clean_start_pos + offset
                clean_acts[(l, offset)] = (
                    model.model.layers[l].output[0][pos, :].save()
                )
        clean_logits = model.output.logits[0, -1, :].save()

    clean_baseline = compute_metric(clean_logits, high_id, low_id, metric)

    # 7. Corrupt baseline (no patching)
    with model.trace(corrupt_input):
        corrupt_logits = model.output.logits[0, -1, :].save()

    corrupt_baseline = compute_metric(corrupt_logits, high_id, low_id, metric)

    # 8. For each K, patch K consecutive positions ending at period_offset
    patched_by_k = {}
    for K in k_values:
        # K consecutive positions ending at period_offset
        start_offset = max(0, period_offset - K + 1)
        end_offset = period_offset + 1  # exclusive
        patch_offsets = list(range(start_offset, end_offset))

        patched_by_k[K] = np.zeros(n_layers)
        for l in range(n_layers):
            with model.trace(corrupt_input):
                for offset in patch_offsets:
                    corrupt_pos = corrupt_start_pos + offset
                    model.model.layers[l].output[0][corrupt_pos, :] = (
                        clean_acts[(l, offset)]
                    )
                patched_logits = model.output.logits[0, -1, :].save()

            patched_by_k[K][l] = compute_metric(
                patched_logits, high_id, low_id, metric
            )

    return patched_by_k, clean_baseline, corrupt_baseline, period_offset


def main():
    parser = argparse.ArgumentParser(
        description="Multi-position activation patching: effect vs. number of patched positions"
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HuggingFace model name or local path")
    parser.add_argument("--pairs", default="neural-concepts-team-E/synthetic_data/API/synthetic_pairs_200.json")
    parser.add_argument("--n_pairs", type=int, default=100,
                        help="Number of contrastive pairs to average over")
    parser.add_argument("--dataset", choices=["templated", "synthetic"], default="templated",
                        help="templated: 1 demo pair from dataset. synthetic: fixed few-shot demos.")
    parser.add_argument("--metric", choices=["logit_diff", "prob_diff"], default="prob_diff",
                        help="logit_diff: logit(HIGH)-logit(LOW). prob_diff: P(HIGH)-P(LOW).")
    parser.add_argument("--k_values", default="1,2,3,5,10,15,20",
                        help="Comma-separated list of K values (number of positions to patch)")
    parser.add_argument("--out_dir", default="script_outputs/advanced_analysis/multi_position_patching",
                        help="Output directory for plots and results")
    args = parser.parse_args()

    # ── Load pairs ────────────────────────────────────────────────────────────
    pairs = load_pairs(args.pairs)
    print(f"Loaded {len(pairs)} contrastive pairs.")
    print(f"Dataset mode: {args.dataset}")

    if args.dataset == "templated":
        demo_pair      = pairs[0]
        demo_certain   = demo_pair["certain"]
        demo_uncertain = demo_pair["uncertain"]
        test_pairs     = pairs[1 : 1 + args.n_pairs]
        print(f"Demo pair: {demo_pair['pair_id']}  |  Test pairs: {len(test_pairs)}")
    else:
        demo_certain   = None
        demo_uncertain = None
        test_pairs     = pairs[:args.n_pairs]
        print(f"Using fixed few-shot demos  |  Test pairs: {len(test_pairs)}")

    # ── Load model ────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device} ...")
    model = LanguageModel(args.model, device_map=device, dispatch=True)
    tokenizer = model.tokenizer
    n_layers  = model.config.num_hidden_layers
    print(f"Model loaded. Layers: {n_layers}")

    low_id, high_id = get_label_token_ids(tokenizer)
    print(f"Label token IDs — LOW: {low_id}  HIGH: {high_id}")

    # ── Parse K values ────────────────────────────────────────────────────────
    k_values = [int(k.strip()) for k in args.k_values.split(",")]
    print(f"K values: {k_values}")

    # ── Run patching ──────────────────────────────────────────────────────────
    all_results_by_k = {K: [] for K in k_values}
    clean_baselines   = []
    corrupt_baselines = []

    total_start = time.time()
    for i, pair in enumerate(test_pairs):
        pair_start = time.time()
        print(f"\nPair {i+1}/{len(test_pairs)}  (pair_id={pair['pair_id']})")

        patched_by_k, cb, ub, period_offset = run_multi_patching_for_pair(
            pair, model, tokenizer, n_layers, low_id, high_id,
            k_values=k_values,
            dataset=args.dataset,
            demo_certain=demo_certain,
            demo_uncertain=demo_uncertain,
            metric=args.metric,
        )

        clean_baselines.append(cb)
        corrupt_baselines.append(ub)
        for K in k_values:
            all_results_by_k[K].append(patched_by_k[K])

        pair_elapsed = time.time() - pair_start
        print(f"  clean baseline: {cb:.4f}  |  corrupt baseline: {ub:.4f}")
        print(f"  period_offset: {period_offset}  |  elapsed: {pair_elapsed:.1f}s")
        for K in k_values:
            print(f"    K={K}: range [{patched_by_k[K].min():.4f}, {patched_by_k[K].max():.4f}]")

    total_elapsed = time.time() - total_start
    print(f"\nTotal time: {total_elapsed:.1f}s")

    # ── Average across pairs ──────────────────────────────────────────────────
    mean_results = {}
    for K in k_values:
        mean_results[K] = np.stack(all_results_by_k[K]).mean(axis=0)  # (n_layers,)

    mean_clean_base   = np.mean(clean_baselines)
    mean_corrupt_base = np.mean(corrupt_baselines)

    print(f"\nMean clean baseline:   {mean_clean_base:.4f}")
    print(f"Mean corrupt baseline: {mean_corrupt_base:.4f}")

    # ── Create output directory ───────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)

    metric_label = (
        "logit(HIGH) − logit(LOW)" if args.metric == "logit_diff"
        else "P(HIGH) − P(LOW)"
    )
    model_short = args.model.split("/")[-1]

    # ── Plot 1: Scaling curve (metric vs K for selected layers) ───────────────
    # Select ~6 layers spread across the model
    if n_layers >= 30:
        selected_layers = [5, 10, 15, 20, 25, 30]
    elif n_layers >= 20:
        selected_layers = [3, 7, 11, 15, 19, n_layers - 1]
    else:
        step = max(1, n_layers // 6)
        selected_layers = list(range(0, n_layers, step))[:6]
    # Clamp to valid range
    selected_layers = [l for l in selected_layers if l < n_layers]

    fig, ax = plt.subplots(figsize=(10, 6))
    for layer in selected_layers:
        ys = [mean_results[K][layer] for K in k_values]
        ax.plot(k_values, ys, marker="o", label=f"Layer {layer}")

    ax.axhline(mean_clean_base, color="green", linestyle="--", linewidth=1.5,
               label=f"Clean baseline ({mean_clean_base:.3f})")
    ax.axhline(mean_corrupt_base, color="red", linestyle="--", linewidth=1.5,
               label=f"Corrupt baseline ({mean_corrupt_base:.3f})")

    ax.set_xlabel("K (number of patched positions)")
    ax.set_ylabel(metric_label)
    ax.set_title(
        f"Multi-Position Patching: Effect vs. Number of Patched Positions\n"
        f"(averaged over {len(test_pairs)} pairs, model={model_short})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    scaling_path = os.path.join(args.out_dir, "scaling_by_layer.png")
    fig.savefig(scaling_path, dpi=150)
    print(f"Saved scaling plot -> {scaling_path}")
    plt.close(fig)

    # ── Plot 2: Heatmap (K values x layers) ──────────────────────────────────
    heatmap_data = np.array([mean_results[K] for K in k_values])  # (len(k_values), n_layers)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(heatmap_data, aspect="auto", cmap="RdBu_r",
                   origin="lower", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, label=metric_label)

    ax.set_xlabel("Layer")
    ax.set_ylabel("K (number of patched positions)")
    ax.set_xticks(range(0, n_layers, max(1, n_layers // 16)))
    ax.set_yticks(range(len(k_values)))
    ax.set_yticklabels([str(K) for K in k_values])
    ax.set_title(
        f"Patching Effect: K Positions × Layer\n"
        f"(averaged over {len(test_pairs)} pairs, model={model_short})"
    )
    fig.tight_layout()
    heatmap_path = os.path.join(args.out_dir, "heatmap_k_vs_layer.png")
    fig.savefig(heatmap_path, dpi=150)
    print(f"Saved heatmap -> {heatmap_path}")
    plt.close(fig)

    # ── Plot 3: Recovery fraction ─────────────────────────────────────────────
    baseline_gap = mean_clean_base - mean_corrupt_base
    fig, ax = plt.subplots(figsize=(10, 6))

    for layer in selected_layers:
        recovery = [
            (mean_results[K][layer] - mean_corrupt_base) / baseline_gap
            if baseline_gap != 0 else 0.0
            for K in k_values
        ]
        ax.plot(k_values, recovery, marker="o", label=f"Layer {layer}")

    ax.axhline(1.0, color="green", linestyle="--", linewidth=1.5,
               label="Full recovery (1.0)")
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.5,
               label="No recovery (0.0)")

    ax.set_xlabel("K (number of patched positions)")
    ax.set_ylabel("Recovery fraction")
    ax.set_title(
        f"Multi-Position Patching: Recovery Fraction vs. K\n"
        f"(averaged over {len(test_pairs)} pairs, model={model_short})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    recovery_path = os.path.join(args.out_dir, "recovery_fraction.png")
    fig.savefig(recovery_path, dpi=150)
    print(f"Saved recovery plot -> {recovery_path}")
    plt.close(fig)

    # ── Save results JSON ─────────────────────────────────────────────────────
    results = {
        "k_values": k_values,
        "mean_results": {str(K): mean_results[K].tolist() for K in k_values},
        "mean_clean_baseline": float(mean_clean_base),
        "mean_corrupt_baseline": float(mean_corrupt_base),
        "args": vars(args),
    }
    results_path = os.path.join(args.out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results -> {results_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
