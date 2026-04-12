"""
Mass-mean probe classification for economic uncertainty (HIGH/LOW labels).

Extracts a steering direction and validates classification using both
zero-boundary projection and Mahalanobis-distance (mass-mean) probe.
Steering is handled separately by steering_highlow.py.

Stages:
  1.  Extract steering direction from train activations
  1b. Fit MassMeanProbe on train activations
  2.  Validate with both projection and mass-mean classifiers
"""

import sys
import os
import json
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add parent dir so we can import from activation_patching_v2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from activation_patching_highlow import (
    build_messages,
    build_messages_synthetic,
    apply_template,
    find_last_period_pos,
    get_label_token_ids,
    load_pairs,
)
from steering_highlow import (
    extract_activations,
    compute_direction,
    validate_direction,
)

from nnsight import LanguageModel


# -- Mass-Mean Probe -----------------------------------------------------------

class MassMeanProbe:
    def __init__(self):
        self.mu_uncertain = None
        self.mu_certain = None
        self.precision = None  # Sigma^{-1}

    def fit(self, uncertain_acts, certain_acts):
        """Fit probe from labeled activations.
        Uses Ledoit-Wolf shrinkage for robust covariance estimation
        (handles rank-deficient case: ~100 samples vs 4096 dims).
        """
        self.mu_uncertain = uncertain_acts.mean(axis=0)
        self.mu_certain = certain_acts.mean(axis=0)

        # Center activations for covariance estimation
        centered_u = uncertain_acts - self.mu_uncertain
        centered_c = certain_acts - self.mu_certain
        all_centered = np.vstack([centered_u, centered_c])

        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(all_centered)
        self.precision = lw.precision_  # Already Sigma^{-1}

    def classify(self, activations):
        """Classify by argmin Mahalanobis distance to centroids.
        Returns: predictions (1=uncertain, 0=certain), dict with distances
        """
        diff_u = activations - self.mu_uncertain
        diff_c = activations - self.mu_certain

        dist_u = np.sum(diff_u @ self.precision * diff_u, axis=1)
        dist_c = np.sum(diff_c @ self.precision * diff_c, axis=1)

        preds = (dist_u < dist_c).astype(int)  # 1=uncertain (closer to uncertain centroid)
        return preds, {"dist_uncertain": dist_u, "dist_certain": dist_c}


# -- Validation ----------------------------------------------------------------

def validate_massmean(test_uncertain_acts, test_certain_acts, probe, out_dir):
    """Validate mass-mean probe on held-out test activations."""
    # Classify uncertain (should predict 1) and certain (should predict 0)
    u_preds, u_info = probe.classify(test_uncertain_acts)
    c_preds, c_info = probe.classify(test_certain_acts)

    uncertain_correct = (u_preds == 1).sum()
    certain_correct = (c_preds == 0).sum()
    total = len(u_preds) + len(c_preds)
    accuracy = (uncertain_correct + certain_correct) / total

    metrics = {
        "accuracy": float(accuracy),
        "uncertain_accuracy": float(uncertain_correct / len(u_preds)),
        "certain_accuracy": float(certain_correct / len(c_preds)),
        "n_test_pairs": len(u_preds),
        "uncertain_mean_dist_to_uncertain": float(u_info["dist_uncertain"].mean()),
        "uncertain_mean_dist_to_certain": float(u_info["dist_certain"].mean()),
        "certain_mean_dist_to_uncertain": float(c_info["dist_uncertain"].mean()),
        "certain_mean_dist_to_certain": float(c_info["dist_certain"].mean()),
    }

    print(f"\n-- Mass-Mean Probe Validation --")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Uncertain accuracy: {metrics['uncertain_accuracy']:.4f}")
    print(f"  Certain accuracy:   {metrics['certain_accuracy']:.4f}")

    # Save metrics
    with open(os.path.join(out_dir, "massmean_validation.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot histogram of dist_certain - dist_uncertain (positive = classified as uncertain)
    u_score = u_info["dist_certain"] - u_info["dist_uncertain"]  # should be positive
    c_score = c_info["dist_certain"] - c_info["dist_uncertain"]  # should be negative

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(c_score, bins=20, alpha=0.6, label="Certain (low uncertainty)", color="green")
    ax.hist(u_score, bins=20, alpha=0.6, label="Uncertain (high uncertainty)", color="red")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Decision boundary")
    ax.set_xlabel("dist(certain centroid) - dist(uncertain centroid)")
    ax.set_ylabel("Count")
    ax.set_title("Mass-Mean Probe: Mahalanobis Distance Difference")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "massmean_distance_histogram.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved histogram -> massmean_distance_histogram.png")

    return metrics


def validate_comparison(test_uncertain_acts, test_certain_acts, direction, probe, out_dir):
    """Run both projection and mass-mean validation, save combined comparison."""
    print(f"\n-- Validation Comparison: Projection vs Mass-Mean Probe --")

    proj_metrics = validate_direction(test_uncertain_acts, test_certain_acts, direction, out_dir)
    mm_metrics = validate_massmean(test_uncertain_acts, test_certain_acts, probe, out_dir)

    comparison = {
        "projection": {
            "accuracy": proj_metrics["accuracy"],
            "uncertain_accuracy": proj_metrics["uncertain_accuracy"],
            "certain_accuracy": proj_metrics["certain_accuracy"],
        },
        "massmean": {
            "accuracy": mm_metrics["accuracy"],
            "uncertain_accuracy": mm_metrics["uncertain_accuracy"],
            "certain_accuracy": mm_metrics["certain_accuracy"],
        },
    }

    with open(os.path.join(out_dir, "validation_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n  Comparison summary:")
    print(f"    Projection  -- Acc: {comparison['projection']['accuracy']:.4f}  "
          f"Uncertain: {comparison['projection']['uncertain_accuracy']:.4f}  "
          f"Certain: {comparison['projection']['certain_accuracy']:.4f}")
    print(f"    Mass-Mean   -- Acc: {comparison['massmean']['accuracy']:.4f}  "
          f"Uncertain: {comparison['massmean']['uncertain_accuracy']:.4f}  "
          f"Certain: {comparison['massmean']['certain_accuracy']:.4f}")
    print(f"  Saved -> validation_comparison.json")

    return comparison


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RepE Steering for Economic Uncertainty (HIGH/LOW labels + mass-mean probe)"
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--pairs", default="neural-concepts-team-E/synthetic_data/API/synthetic_pairs_200.json")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--method", choices=["mean", "pca"], default="mean")
    parser.add_argument("--n_train", type=int, default=100,
                        help="Number of train pairs (after demo). Matches activation patching pairs 2-101.")
    parser.add_argument("--dataset", choices=["templated", "synthetic"], default="templated",
                        help="templated: 1 demo pair from dataset + definition. "
                             "synthetic: definition + 4 fixed few-shot demos.")
    parser.add_argument("--classifier", choices=["projection", "massmean", "both"], default="both",
                        help="Which classifier(s) to use for validation.")
    parser.add_argument("--out_dir", default=".")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("RepE Steering Pipeline -- Economic Uncertainty (v3: mass-mean)")
    print("=" * 60)
    print(f"Model:       {args.model}")
    print(f"Layer:       {args.layer}")
    print(f"Method:      {args.method}")
    print(f"N train:     {args.n_train}")
    print(f"Dataset:     {args.dataset}")
    print(f"Classifier:  {args.classifier}")
    print(f"Output dir:  {args.out_dir}")

    # -- Load pairs ------------------------------------------------------------
    pairs = load_pairs(args.pairs)
    print(f"\nLoaded {len(pairs)} contrastive pairs.")

    if args.dataset == "templated":
        # Pair 1 = demo (same as activation_patching_highlow.py)
        demo_pair = pairs[0]
        demo_certain = demo_pair["certain"]
        demo_uncertain = demo_pair["uncertain"]
        # Train = pairs 2-101, Test = pairs 102-200
        train_pairs = pairs[1 : 1 + args.n_train]
        test_pairs  = pairs[1 + args.n_train :]
        print(f"Demo pair_id:  {demo_pair['pair_id']}")
    else:
        # Synthetic: demos are fixed in the prompt; no demo pair consumed
        demo_certain = None
        demo_uncertain = None
        train_pairs = pairs[:args.n_train]
        test_pairs  = pairs[args.n_train:]
        print(f"Using 4 fixed few-shot demos (no demo pair consumed)")

    print(f"Train pairs:   {len(train_pairs)}  (pair_ids {train_pairs[0]['pair_id']}-{train_pairs[-1]['pair_id']})")
    print(f"Test pairs:    {len(test_pairs)}  (pair_ids {test_pairs[0]['pair_id']}-{test_pairs[-1]['pair_id']})")

    # Save split
    split_info = {
        "demo_pair_id": demo_pair["pair_id"] if args.dataset == "templated" else None,
        "dataset": args.dataset,
        "train_pair_ids": [p["pair_id"] for p in train_pairs],
        "test_pair_ids": [p["pair_id"] for p in test_pairs],
    }
    with open(os.path.join(args.out_dir, "steering_split.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    # -- Load model ------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading model on {device}...")
    model = LanguageModel(args.model, device_map=device, dispatch=True)
    tokenizer = model.tokenizer
    n_layers = model.config.num_hidden_layers
    print(f"Model loaded. Layers: {n_layers}")

    low_id, high_id = get_label_token_ids(tokenizer)
    print(f"Label token IDs -- LOW: {low_id}  HIGH: {high_id}")

    # ==========================================================================
    # Stage 1: Extract direction from train pairs
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"Stage 1: Extract steering direction (layer {args.layer}, method={args.method})")
    print(f"  Using {len(train_pairs)} train pairs")
    print(f"{'='*60}")

    print("\nExtracting train activations...")
    train_uncertain, train_certain = extract_activations(
        train_pairs, model, tokenizer, args.layer,
        dataset=args.dataset, demo_certain=demo_certain,
        demo_uncertain=demo_uncertain,
    )
    print(f"  Train activations shape: {train_uncertain.shape}")

    direction = compute_direction(train_uncertain, train_certain, method=args.method)
    np.save(os.path.join(args.out_dir, "steering_direction.npy"), direction)
    print(f"  Direction computed and saved. Norm: {np.linalg.norm(direction):.6f}")

    # ==========================================================================
    # Stage 1b: Fit MassMeanProbe on train activations
    # ==========================================================================
    probe = MassMeanProbe()
    if args.classifier in ("massmean", "both"):
        print(f"\n{'='*60}")
        print(f"Stage 1b: Fit MassMeanProbe on train activations")
        print(f"{'='*60}")

        probe.fit(train_uncertain, train_certain)
        np.savez(
            os.path.join(args.out_dir, "massmean_probe.npz"),
            mu_uncertain=probe.mu_uncertain,
            mu_certain=probe.mu_certain,
            precision=probe.precision,
        )
        print(f"  Probe fitted and saved -> massmean_probe.npz")

    # ==========================================================================
    # Save train/test activations as .npy
    # ==========================================================================
    print(f"\nExtracting test activations...")
    test_uncertain, test_certain = extract_activations(
        test_pairs, model, tokenizer, args.layer,
        dataset=args.dataset, demo_certain=demo_certain,
        demo_uncertain=demo_uncertain,
    )

    np.save(os.path.join(args.out_dir, "train_uncertain.npy"), train_uncertain)
    np.save(os.path.join(args.out_dir, "train_certain.npy"), train_certain)
    np.save(os.path.join(args.out_dir, "test_uncertain.npy"), test_uncertain)
    np.save(os.path.join(args.out_dir, "test_certain.npy"), test_certain)
    print(f"  Saved train/test activations as .npy files")

    # ==========================================================================
    # Stage 2: Validate classification on test pairs
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"Stage 2: Validate on held-out test set ({len(test_pairs)} pairs)")
    print(f"{'='*60}")

    if args.classifier == "projection":
        validate_direction(test_uncertain, test_certain, direction, args.out_dir)
    elif args.classifier == "massmean":
        validate_massmean(test_uncertain, test_certain, probe, args.out_dir)
    else:  # both
        validate_comparison(test_uncertain, test_certain, direction, probe, args.out_dir)

    print(f"\n{'='*60}")
    print(f"Classification complete. All outputs saved to {args.out_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
