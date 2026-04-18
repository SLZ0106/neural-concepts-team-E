"""
Extract uncertainty direction from source dataset and validate on target dataset.

Supports two probe techniques:
  - projection: zero-boundary projection onto the direction vector
  - massmean: Mahalanobis-distance classification using class centroids

For same-dataset evaluation, pass the same name for --source and --target.
For cross-dataset evaluation, pass different names.

Example (same-dataset, projection):
    python extract_direction.py --source synthetic --target synthetic \
        --probe projection --out_dir results/

Example (cross-dataset, massmean):
    python extract_direction.py --source synthetic --target templated \
        --probe massmean --out_dir results/cross/
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from activation_patching_highlow import (
    build_messages,
    build_messages_synthetic,
    apply_template,
    find_last_period_pos,
    load_pairs,
)

from nnsight import LanguageModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATASET_PATHS = {
    "synthetic": os.path.join(PROJECT_ROOT, "neural-concepts-team-E/synthetic_data/API/synthetic_pairs_200.json"),
    "templated": os.path.join(PROJECT_ROOT, "neural-concepts-team-E/synthetic_data/templated/templated_pairs_200.json"),
}


# -- Activation extraction -----------------------------------------------------

def extract_activations(pairs, model, tokenizer, layer,
                        dataset="templated", demo_certain=None, demo_uncertain=None):
    """Extract layer activations at last-period token for all pairs."""
    uncertain_acts = []
    certain_acts = []

    for i, pair in enumerate(pairs):
        for label_key, act_list in [("uncertain", uncertain_acts), ("certain", certain_acts)]:
            if dataset == "synthetic":
                msgs = build_messages_synthetic(pair[label_key])
            else:
                msgs = build_messages(demo_certain, demo_uncertain, pair[label_key])
            ids = apply_template(msgs, tokenizer, tokenize=True)
            input_tensor = torch.tensor([ids])
            period_pos = find_last_period_pos(ids, tokenizer)

            with model.trace(input_tensor):
                act = model.model.layers[layer].output[0][period_pos, :].save()

            act_list.append(act.detach().cpu().float().numpy())

        if (i + 1) % 20 == 0 or (i + 1) == len(pairs):
            print(f"  Extracted {i+1}/{len(pairs)} pairs")

    return np.stack(uncertain_acts), np.stack(certain_acts)


def compute_direction(uncertain_acts, certain_acts, method="mean"):
    """Compute steering direction from paired activations."""
    diffs = uncertain_acts - certain_acts

    if method == "mean":
        direction = diffs.mean(axis=0)
    elif method == "pca":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        pca.fit(diffs)
        direction = pca.components_[0]
    else:
        raise ValueError(f"Unknown method: {method}")

    direction = direction / np.linalg.norm(direction)
    return direction


# -- Probe: simple projection --------------------------------------------------

def validate_projection(test_uncertain_acts, test_certain_acts, direction, out_dir):
    """Project test activations onto direction and evaluate classification."""
    uncertain_projs = test_uncertain_acts @ direction
    certain_projs = test_certain_acts @ direction

    uncertain_correct = (uncertain_projs > 0).sum()
    certain_correct = (certain_projs < 0).sum()
    total = len(uncertain_projs) + len(certain_projs)
    accuracy = (uncertain_correct + certain_correct) / total

    metrics = {
        "accuracy": float(accuracy),
        "uncertain_accuracy": float(uncertain_correct / len(uncertain_projs)),
        "certain_accuracy": float(certain_correct / len(certain_projs)),
        "n_test_pairs": len(uncertain_projs),
        "uncertain_mean_proj": float(uncertain_projs.mean()),
        "certain_mean_proj": float(certain_projs.mean()),
    }

    print(f"\n-- Projection Validation --")
    print(f"  Accuracy:            {metrics['accuracy']:.4f}")
    print(f"  Uncertain accuracy:  {metrics['uncertain_accuracy']:.4f}")
    print(f"  Certain accuracy:    {metrics['certain_accuracy']:.4f}")
    print(f"  Uncertain mean proj: {metrics['uncertain_mean_proj']:.4f}")
    print(f"  Certain mean proj:   {metrics['certain_mean_proj']:.4f}")

    with open(os.path.join(out_dir, "projection_validation.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(certain_projs, bins=20, alpha=0.6, label="Certain (low uncertainty)", color="green")
    ax.hist(uncertain_projs, bins=20, alpha=0.6, label="Uncertain (high uncertainty)", color="red")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Decision boundary")
    ax.set_xlabel("Projection onto steering direction")
    ax.set_ylabel("Count")
    ax.set_title("Projection of Test Activations onto Steering Direction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "projection_histogram.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved histogram -> projection_histogram.png")

    return metrics


# -- Probe: mass-mean ----------------------------------------------------------

class MassMeanProbe:
    def __init__(self):
        self.mu_uncertain = None
        self.mu_certain = None
        self.precision = None  # Sigma^{-1}

    def fit(self, uncertain_acts, certain_acts):
        """Fit probe using Ledoit-Wolf shrinkage for robust covariance estimation."""
        self.mu_uncertain = uncertain_acts.mean(axis=0)
        self.mu_certain = certain_acts.mean(axis=0)

        centered_u = uncertain_acts - self.mu_uncertain
        centered_c = certain_acts - self.mu_certain
        all_centered = np.vstack([centered_u, centered_c])

        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(all_centered)
        self.precision = lw.precision_

    def classify(self, activations):
        """Classify by argmin Mahalanobis distance to centroids.
        Returns: predictions (1=uncertain, 0=certain), dict with distances
        """
        diff_u = activations - self.mu_uncertain
        diff_c = activations - self.mu_certain

        dist_u = np.sum(diff_u @ self.precision * diff_u, axis=1)
        dist_c = np.sum(diff_c @ self.precision * diff_c, axis=1)

        preds = (dist_u < dist_c).astype(int)
        return preds, {"dist_uncertain": dist_u, "dist_certain": dist_c}


def validate_massmean(test_uncertain_acts, test_certain_acts, probe, out_dir):
    """Validate mass-mean probe on held-out test activations."""
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

    with open(os.path.join(out_dir, "massmean_validation.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    u_score = u_info["dist_certain"] - u_info["dist_uncertain"]
    c_score = c_info["dist_certain"] - c_info["dist_uncertain"]

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
    fig.savefig(os.path.join(out_dir, "massmean_histogram.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved histogram -> massmean_histogram.png")

    return metrics


# -- Pair splitting ------------------------------------------------------------

def split_pairs(pairs, dataset, n_train):
    """Split pairs into demo + train + test based on dataset type.

    For templated: pair 0 is the demo, next n_train are train, rest are test.
    For synthetic: demos are fixed in the prompt, first n_train are train, rest are test.
    """
    if dataset == "templated":
        demo_pair = pairs[0]
        demo_certain = demo_pair["certain"]
        demo_uncertain = demo_pair["uncertain"]
        train_pairs = pairs[1 : 1 + n_train]
        test_pairs = pairs[1 + n_train :]
    else:
        demo_certain = None
        demo_uncertain = None
        train_pairs = pairs[:n_train]
        test_pairs = pairs[n_train:]
    return demo_certain, demo_uncertain, train_pairs, test_pairs


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract uncertainty direction from source dataset and validate on target dataset"
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--source", choices=["templated", "synthetic"], required=True,
                        help="Source dataset (train activations extracted from here)")
    parser.add_argument("--target", choices=["templated", "synthetic"], required=True,
                        help="Target dataset (test activations extracted from here)")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--method", choices=["mean", "pca"], default="mean",
                        help="How to compute direction from activation diffs")
    parser.add_argument("--probe", choices=["projection", "massmean", "both"], default="both",
                        help="Probe technique for validation")
    parser.add_argument("--n_train", type=int, default=100,
                        help="Number of train pairs from source dataset")
    parser.add_argument("--out_dir", default=".")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    source_path = DATASET_PATHS[args.source]
    target_path = DATASET_PATHS[args.target]
    cross_dataset = args.source != args.target

    print("=" * 60)
    print("Extract Direction & Validate")
    print("=" * 60)
    print(f"Model:          {args.model}")
    print(f"Source:         {args.source}  ({source_path})")
    print(f"Target:         {args.target}  ({target_path})")
    print(f"Cross-dataset:  {cross_dataset}")
    print(f"Layer:          {args.layer}")
    print(f"Method:         {args.method}")
    print(f"Probe:          {args.probe}")
    print(f"N train:        {args.n_train}")
    print(f"Output dir:     {args.out_dir}")

    # -- Load source pairs and split -------------------------------------------
    source_pairs = load_pairs(source_path)
    print(f"\nLoaded {len(source_pairs)} source pairs.")

    src_demo_certain, src_demo_uncertain, train_pairs, source_test_pairs = \
        split_pairs(source_pairs, args.source, args.n_train)

    if args.source == "templated":
        print(f"Source demo pair_id: {source_pairs[0]['pair_id']}")
    else:
        print(f"Source: using fixed few-shot demos (no demo pair consumed)")
    print(f"Train pairs: {len(train_pairs)}  "
          f"(pair_ids {train_pairs[0]['pair_id']}-{train_pairs[-1]['pair_id']})")

    # -- Load target pairs and split -------------------------------------------
    if cross_dataset:
        target_pairs_all = load_pairs(target_path)
        print(f"Loaded {len(target_pairs_all)} target pairs.")
        tgt_demo_certain, tgt_demo_uncertain, _, test_pairs = \
            split_pairs(target_pairs_all, args.target, args.n_train)
        if args.target == "templated":
            print(f"Target demo pair_id: {target_pairs_all[0]['pair_id']}")
        else:
            print(f"Target: using fixed few-shot demos (no demo pair consumed)")
    else:
        test_pairs = source_test_pairs
        tgt_demo_certain = src_demo_certain
        tgt_demo_uncertain = src_demo_uncertain

    print(f"Test pairs:  {len(test_pairs)}  "
          f"(pair_ids {test_pairs[0]['pair_id']}-{test_pairs[-1]['pair_id']})")

    # -- Save split info -------------------------------------------------------
    split_info = {
        "source": args.source,
        "target": args.target,
        "cross_dataset": cross_dataset,
        "train_pair_ids": [p["pair_id"] for p in train_pairs],
        "test_pair_ids": [p["pair_id"] for p in test_pairs],
    }
    with open(os.path.join(args.out_dir, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    # -- Load model ------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading model on {device}...")
    model = LanguageModel(args.model, device_map=device, dispatch=True)
    tokenizer = model.tokenizer
    print(f"Model loaded. Layers: {model.config.num_hidden_layers}")

    # ==========================================================================
    # Stage 1: Extract direction from source train pairs
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"Stage 1: Extract direction (layer {args.layer}, method={args.method})")
    print(f"  Using {len(train_pairs)} train pairs")
    print(f"{'='*60}")

    print("\nExtracting source train activations...")
    train_uncertain, train_certain = extract_activations(
        train_pairs, model, tokenizer, args.layer,
        dataset=args.source,
        demo_certain=src_demo_certain,
        demo_uncertain=src_demo_uncertain,
    )
    print(f"  Train activations shape: {train_uncertain.shape}")

    direction = compute_direction(train_uncertain, train_certain, method=args.method)
    np.save(os.path.join(args.out_dir, "direction.npy"), direction)
    print(f"  Direction saved. Norm: {np.linalg.norm(direction):.6f}")

    # Save train activations
    np.save(os.path.join(args.out_dir, "train_uncertain.npy"), train_uncertain)
    np.save(os.path.join(args.out_dir, "train_certain.npy"), train_certain)

    # Fit mass-mean probe if requested
    probe = None
    if args.probe in ("massmean", "both"):
        print(f"\nFitting MassMeanProbe on train activations...")
        probe = MassMeanProbe()
        probe.fit(train_uncertain, train_certain)
        np.savez(
            os.path.join(args.out_dir, "massmean_probe.npz"),
            mu_uncertain=probe.mu_uncertain,
            mu_certain=probe.mu_certain,
            precision=probe.precision,
        )
        print(f"  Probe saved -> massmean_probe.npz")

    # ==========================================================================
    # Stage 2: Validate on target test pairs
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"Stage 2: Validate on test set ({len(test_pairs)} pairs)")
    print(f"{'='*60}")

    print("\nExtracting target test activations...")
    test_uncertain, test_certain = extract_activations(
        test_pairs, model, tokenizer, args.layer,
        dataset=args.target,
        demo_certain=tgt_demo_certain,
        demo_uncertain=tgt_demo_uncertain,
    )

    # Save test activations
    np.save(os.path.join(args.out_dir, "test_uncertain.npy"), test_uncertain)
    np.save(os.path.join(args.out_dir, "test_certain.npy"), test_certain)
    print(f"  Saved train/test activations as .npy files")

    if args.probe == "projection":
        validate_projection(test_uncertain, test_certain, direction, args.out_dir)
    elif args.probe == "massmean":
        validate_massmean(test_uncertain, test_certain, probe, args.out_dir)
    else:  # both
        proj_metrics = validate_projection(test_uncertain, test_certain, direction, args.out_dir)
        mm_metrics = validate_massmean(test_uncertain, test_certain, probe, args.out_dir)

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
        with open(os.path.join(args.out_dir, "validation_comparison.json"), "w") as f:
            json.dump(comparison, f, indent=2)

        print(f"\n  Comparison:")
        print(f"    Projection -- Acc: {comparison['projection']['accuracy']:.4f}  "
              f"Uncertain: {comparison['projection']['uncertain_accuracy']:.4f}  "
              f"Certain: {comparison['projection']['certain_accuracy']:.4f}")
        print(f"    Mass-Mean  -- Acc: {comparison['massmean']['accuracy']:.4f}  "
              f"Uncertain: {comparison['massmean']['uncertain_accuracy']:.4f}  "
              f"Certain: {comparison['massmean']['certain_accuracy']:.4f}")

    print(f"\n{'='*60}")
    print(f"Complete. Outputs saved to {args.out_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
