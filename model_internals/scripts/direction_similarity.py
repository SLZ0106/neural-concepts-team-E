"""
Direction similarity analysis across token positions.

Extracts the uncertainty direction (mean difference between uncertain and
certain activations) at each token position in a window around the period
token, then computes pairwise cosine similarity between positions.
"""

import sys, os, json, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from activation_patching_highlow import (
    build_messages, build_messages_synthetic, apply_template,
    find_last_period_pos, get_label_token_ids, load_pairs,
    SYNTHETIC_FEW_SHOT_DEMOS,
)
from nnsight import LanguageModel

PRE_PERIOD_TOKENS = 20


# ── Pair loading for templated flat format ───────────────────────────────────

def load_flat_pairs(path: str):
    """
    Load contrastive pairs from the flat format used by the templated dataset.

    Flat entries have: id, statement, uncertainty ("high"/"no"), pair_id, role.
    Groups by pair_id and returns list of dicts:
        {"pair_id": int, "certain": str, "uncertain": str}
    """
    with open(path) as f:
        data = json.load(f)

    grouped = {}
    for entry in data:
        pid = entry["pair_id"]
        if pid not in grouped:
            grouped[pid] = {}
        if entry["uncertainty"] == "high":
            grouped[pid]["uncertain"] = entry["statement"]
        else:
            grouped[pid]["certain"] = entry["statement"]
        grouped[pid]["pair_id"] = pid

    pairs = []
    for pid in sorted(grouped):
        p = grouped[pid]
        if "uncertain" in p and "certain" in p:
            pairs.append(p)
    return pairs


# ── Core extraction ──────────────────────────────────────────────────────────

def extract_activations_multi_pos(pairs, model, tokenizer, layer, pre_period=20,
                                   dataset="templated", demo_certain=None, demo_uncertain=None):
    """
    Extract activations at multiple token positions for each pair.

    Window: pre_period tokens before period through len(ids)-2 (exclude final token).

    Returns:
        uncertain_acts: np.array shape (n_pairs, n_positions, hidden_dim)
        certain_acts: np.array shape (n_pairs, n_positions, hidden_dim)
        offset_labels: list of int, relative offsets from period (e.g., [-20, -19, ..., 0, 1, 2, ...])
        period_index: int, index into offset_labels where offset == 0
    """
    all_uncertain = []
    all_certain = []
    all_pre_counts = []
    all_post_counts = []

    for i, pair in enumerate(pairs):
        pair_acts = {}
        for label_key in ("uncertain", "certain"):
            stmt = pair[label_key]
            if dataset == "synthetic":
                msgs = build_messages_synthetic(stmt)
            else:
                msgs = build_messages(demo_certain, demo_uncertain, stmt)

            ids = apply_template(msgs, tokenizer, tokenize=True)
            period_pos = find_last_period_pos(ids, tokenizer)
            start_pos = max(0, period_pos - pre_period)
            end_pos = len(ids) - 2  # exclude the very last token
            n_positions = end_pos - start_pos + 1

            input_tensor = torch.tensor([ids])
            saved = []
            with model.trace(input_tensor):
                for offset in range(n_positions):
                    pos = start_pos + offset
                    saved.append(model.model.layers[layer].output[0][pos, :].save())

            acts_for_pair = [s.detach().cpu().float().numpy() for s in saved]
            pair_acts[label_key] = {
                "acts": acts_for_pair,
                "pre_count": period_pos - start_pos,
                "post_count": end_pos - period_pos,
            }

        all_uncertain.append(pair_acts["uncertain"])
        all_certain.append(pair_acts["certain"])
        all_pre_counts.append(min(pair_acts["uncertain"]["pre_count"],
                                   pair_acts["certain"]["pre_count"]))
        all_post_counts.append(min(pair_acts["uncertain"]["post_count"],
                                    pair_acts["certain"]["post_count"]))

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Extracted pair {i+1}/{len(pairs)}")

    # Find common window across all pairs
    min_pre = min(all_pre_counts)
    min_post = min(all_post_counts)
    n_positions = min_pre + 1 + min_post
    print(f"  Common window: {min_pre} pre-period + period + {min_post} post-period = {n_positions} positions")

    # Slice each pair to common window centered on period
    uncertain_stacked = []
    certain_stacked = []
    for i in range(len(pairs)):
        for label_data, out_list in [(all_uncertain[i], uncertain_stacked),
                                      (all_certain[i], certain_stacked)]:
            p = label_data["pre_count"]  # local period index
            sliced = label_data["acts"][p - min_pre : p + min_post + 1]
            out_list.append(np.stack(sliced))  # (n_positions, hidden_dim)

    uncertain_acts = np.stack(uncertain_stacked)  # (n_pairs, n_positions, hidden_dim)
    certain_acts = np.stack(certain_stacked)

    offset_labels = list(range(-min_pre, min_post + 1))
    period_index = min_pre

    return uncertain_acts, certain_acts, offset_labels, period_index


# ── Direction computation ────────────────────────────────────────────────────

def compute_directions(uncertain_acts, certain_acts):
    """
    Compute per-position uncertainty direction.

    Args:
        uncertain_acts: (n_pairs, n_positions, hidden_dim)
        certain_acts: (n_pairs, n_positions, hidden_dim)

    Returns:
        directions: (n_positions, hidden_dim), unit-normalized per-position directions
        norms: (n_positions,), pre-normalization norms (diagnostic)
    """
    diffs = uncertain_acts - certain_acts  # (n_pairs, n_positions, hidden_dim)
    mean_diffs = diffs.mean(axis=0)  # (n_positions, hidden_dim)
    norms = np.linalg.norm(mean_diffs, axis=1)  # (n_positions,)
    directions = mean_diffs / norms[:, None]  # normalize each row
    return directions, norms


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_similarity_heatmap(similarity, offset_labels, period_index, title, save_path):
    """Plot a cosine similarity heatmap with offset tick labels."""
    n = len(offset_labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(similarity, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # Tick labeling — show every Nth label to avoid crowding
    step = max(1, n // 20)
    tick_positions = list(range(0, n, step))
    # Ensure period position is included
    if period_index not in tick_positions:
        tick_positions.append(period_index)
        tick_positions.sort()

    tick_labels = []
    for pos in tick_positions:
        lbl = str(offset_labels[pos])
        if pos == period_index:
            lbl = "0(.)"
        tick_labels.append(lbl)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=8)

    ax.set_xlabel("Token offset from period")
    ax.set_ylabel("Token offset from period")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cosine Similarity")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_cross_period_similarity(synthetic_dirs, templated_dirs,
                                  syn_offsets, tmpl_offsets,
                                  syn_period_idx, tmpl_period_idx,
                                  save_path):
    """
    Two subplots:
      Left:  synthetic period direction vs all templated positions
      Right: templated period direction vs all synthetic positions
    """
    # Synthetic period direction vs all templated positions
    syn_period_dir = synthetic_dirs[syn_period_idx]  # (hidden_dim,)
    sim_syn_to_tmpl = templated_dirs @ syn_period_dir  # (n_tmpl_positions,)

    # Templated period direction vs all synthetic positions
    tmpl_period_dir = templated_dirs[tmpl_period_idx]
    sim_tmpl_to_syn = synthetic_dirs @ tmpl_period_dir  # (n_syn_positions,)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(tmpl_offsets, sim_syn_to_tmpl, marker=".", markersize=3)
    ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.7)
    ax1.set_xlabel("Templated token offset")
    ax1.set_ylabel("Cosine similarity")
    ax1.set_title("Synthetic period dir vs templated positions")
    ax1.set_ylim(-1, 1)
    ax1.grid(True, alpha=0.3)

    ax2.plot(syn_offsets, sim_tmpl_to_syn, marker=".", markersize=3)
    ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Synthetic token offset")
    ax2.set_ylabel("Cosine similarity")
    ax2.set_title("Templated period dir vs synthetic positions")
    ax2.set_ylim(-1, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Direction similarity analysis across token positions."
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--pre_period", type=int, default=20)
    parser.add_argument("--n_train", type=int, default=100)
    parser.add_argument(
        "--synthetic_pairs",
        default="neural-concepts-team-E/synthetic_data/API/synthetic_pairs_200.json",
    )
    parser.add_argument(
        "--templated_pairs",
        default="neural-concepts-team-E/synthetic_data/templated/econ_uncertainty_contrastive_flat.json",
    )
    parser.add_argument(
        "--out_dir",
        default="script_outputs/advanced_analysis/direction_similarity",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load model ───────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {args.model} on {device}")
    model = LanguageModel(args.model, device_map=device, dispatch=True)
    tokenizer = model.tokenizer
    print("  Model loaded.")

    # ══════════════════════════════════════════════════════════════════════
    # Synthetic dataset
    # ══════════════════════════════════════════════════════════════════════
    print("\n=== Processing synthetic dataset ===")
    syn_pairs = load_pairs(args.synthetic_pairs)
    train_syn = syn_pairs[: args.n_train]
    print(f"  Loaded {len(syn_pairs)} pairs, using {len(train_syn)} for training.")

    print("  Extracting multi-position activations...")
    syn_uncertain, syn_certain, syn_offsets, syn_period_idx = (
        extract_activations_multi_pos(
            train_syn, model, tokenizer, args.layer,
            pre_period=args.pre_period, dataset="synthetic",
        )
    )

    print("  Computing directions...")
    syn_directions, syn_norms = compute_directions(syn_uncertain, syn_certain)
    np.save(os.path.join(args.out_dir, "synthetic_directions.npy"), syn_directions)
    print(f"  Saved synthetic_directions.npy  shape={syn_directions.shape}")

    # Within-dataset similarity
    syn_similarity = syn_directions @ syn_directions.T
    plot_similarity_heatmap(
        syn_similarity, syn_offsets, syn_period_idx,
        f"Synthetic: within-dataset direction similarity (layer {args.layer})",
        os.path.join(args.out_dir, "synthetic_similarity.png"),
    )

    # ══════════════════════════════════════════════════════════════════════
    # Templated dataset
    # ══════════════════════════════════════════════════════════════════════
    print("\n=== Processing templated dataset ===")
    tmpl_pairs = load_flat_pairs(args.templated_pairs)
    print(f"  Loaded {len(tmpl_pairs)} pairs from flat format.")

    # First pair is the demo; next n_train pairs are training
    demo_pair = tmpl_pairs[0]
    demo_certain = demo_pair["certain"]
    demo_uncertain = demo_pair["uncertain"]
    train_tmpl = tmpl_pairs[1 : 1 + args.n_train]
    print(f"  Demo pair_id={demo_pair['pair_id']}, using {len(train_tmpl)} for training.")

    print("  Extracting multi-position activations...")
    tmpl_uncertain, tmpl_certain, tmpl_offsets, tmpl_period_idx = (
        extract_activations_multi_pos(
            train_tmpl, model, tokenizer, args.layer,
            pre_period=args.pre_period, dataset="templated",
            demo_certain=demo_certain, demo_uncertain=demo_uncertain,
        )
    )

    print("  Computing directions...")
    tmpl_directions, tmpl_norms = compute_directions(tmpl_uncertain, tmpl_certain)
    np.save(os.path.join(args.out_dir, "templated_directions.npy"), tmpl_directions)
    print(f"  Saved templated_directions.npy  shape={tmpl_directions.shape}")

    # Within-dataset similarity
    tmpl_similarity = tmpl_directions @ tmpl_directions.T
    plot_similarity_heatmap(
        tmpl_similarity, tmpl_offsets, tmpl_period_idx,
        f"Templated: within-dataset direction similarity (layer {args.layer})",
        os.path.join(args.out_dir, "templated_similarity.png"),
    )

    # ══════════════════════════════════════════════════════════════════════
    # Cross-dataset analysis
    # ══════════════════════════════════════════════════════════════════════
    print("\n=== Cross-dataset analysis ===")
    cross_similarity = syn_directions @ tmpl_directions.T
    plot_similarity_heatmap(
        cross_similarity, syn_offsets, syn_period_idx,
        f"Cross-dataset direction similarity (layer {args.layer})\n"
        f"rows=synthetic offsets, cols=templated offsets",
        os.path.join(args.out_dir, "cross_dataset_similarity.png"),
    )

    plot_cross_period_similarity(
        syn_directions, tmpl_directions,
        syn_offsets, tmpl_offsets,
        syn_period_idx, tmpl_period_idx,
        os.path.join(args.out_dir, "cross_period_similarity.png"),
    )

    # ══════════════════════════════════════════════════════════════════════
    # Summary JSON
    # ══════════════════════════════════════════════════════════════════════
    period_to_period = float(
        syn_directions[syn_period_idx] @ tmpl_directions[tmpl_period_idx]
    )

    # Mean within-dataset similarity (upper triangle, excluding diagonal)
    def mean_upper_tri(mat):
        n = mat.shape[0]
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        return float(mat[mask].mean())

    summary = {
        "model": args.model,
        "layer": args.layer,
        "pre_period": args.pre_period,
        "n_train": args.n_train,
        "period_to_period_cosine": period_to_period,
        "synthetic": {
            "n_pairs": len(train_syn),
            "n_positions": len(syn_offsets),
            "offsets": syn_offsets,
            "mean_within_similarity": mean_upper_tri(syn_similarity),
            "direction_norms": syn_norms.tolist(),
        },
        "templated": {
            "n_pairs": len(train_tmpl),
            "n_positions": len(tmpl_offsets),
            "offsets": tmpl_offsets,
            "mean_within_similarity": mean_upper_tri(tmpl_similarity),
            "direction_norms": tmpl_norms.tolist(),
        },
        "cross": {
            "mean_cross_similarity": float(cross_similarity.mean()),
        },
    }

    summary_path = os.path.join(args.out_dir, "analysis_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved summary: {summary_path}")

    print(f"\n  Period-to-period cosine similarity: {period_to_period:.4f}")
    print(f"  Synthetic mean within similarity:   {summary['synthetic']['mean_within_similarity']:.4f}")
    print(f"  Templated mean within similarity:   {summary['templated']['mean_within_similarity']:.4f}")
    print(f"  Cross-dataset mean similarity:      {summary['cross']['mean_cross_similarity']:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
