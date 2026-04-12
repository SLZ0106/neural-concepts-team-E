"""
Cross-dataset mass-mean probe classification for economic uncertainty.

Validates three classification modes when the direction is trained on one
dataset and applied to another:
  A. Transfer source-fitted MassMeanProbe directly to target test data
  B. Refit MassMeanProbe on a small calibration set from the target domain
  Baseline. Zero-boundary projection using source direction

Steering is handled separately by steering_cross_dataset_highlow.py.
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
    get_label_token_ids,
    load_pairs,
)
from steering_highlow import (
    extract_activations,
    validate_direction,
)
from concepts_E.scripts.massmean_highlow import MassMeanProbe, validate_massmean, validate_comparison

from nnsight import LanguageModel


def main():
    parser = argparse.ArgumentParser(
        description="Cross-dataset steering with mass-mean probe transfer (v3)"
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--direction", required=True,
                        help="Path to .npy steering direction (from source dataset)")
    parser.add_argument("--source_activations_dir", required=True,
                        help="Path to directory with saved .npy activations from source dataset "
                             "(train_uncertain.npy, train_certain.npy)")
    parser.add_argument("--pairs", required=True,
                        help="Path to contrastive pairs JSON (target dataset to evaluate on)")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--n_train", type=int, default=100,
                        help="Number of pairs to use for validation (mirrors train split of target dataset)")
    parser.add_argument("--n_calibration", type=int, default=20,
                        help="Number of target pairs to use for probe refitting")
    parser.add_argument("--dataset", choices=["templated", "synthetic"], required=True,
                        help="Prompt format for the TARGET dataset being evaluated on")
    parser.add_argument("--out_dir", default=".")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("Cross-Dataset Steering with Mass-Mean Probe (v3)")
    print("=" * 60)
    print(f"Model:              {args.model}")
    print(f"Direction from:     {args.direction}")
    print(f"Source activations: {args.source_activations_dir}")
    print(f"Target pairs:       {args.pairs}")
    print(f"Target dataset:     {args.dataset}")
    print(f"Layer:              {args.layer}")
    print(f"N calibration:      {args.n_calibration}")
    print(f"Output dir:         {args.out_dir}")

    # -- Load steering direction from source dataset ---------------------------
    direction = np.load(args.direction)
    print(f"\nLoaded steering direction: shape={direction.shape}, norm={np.linalg.norm(direction):.6f}")

    # -- Load source activations and fit source probe --------------------------
    src_dir = args.source_activations_dir
    source_uncertain = np.load(os.path.join(src_dir, "train_uncertain.npy"))
    source_certain = np.load(os.path.join(src_dir, "train_certain.npy"))
    print(f"Loaded source activations: uncertain={source_uncertain.shape}, certain={source_certain.shape}")

    print("\nFitting MassMeanProbe on source activations...")
    source_probe = MassMeanProbe()
    source_probe.fit(source_uncertain, source_certain)
    print("  Source probe fitted.")

    # -- Load target pairs -----------------------------------------------------
    pairs = load_pairs(args.pairs)
    print(f"\nLoaded {len(pairs)} contrastive pairs from target dataset.")

    if args.dataset == "templated":
        demo_pair = pairs[0]
        demo_certain = demo_pair["certain"]
        demo_uncertain = demo_pair["uncertain"]
        remaining_pairs = pairs[1:]
        print(f"Demo pair_id:  {demo_pair['pair_id']}")
    else:
        demo_certain = None
        demo_uncertain = None
        remaining_pairs = pairs
        print(f"Using 4 fixed few-shot demos (no demo pair consumed)")

    # Split: calibration + test
    calibration_pairs = remaining_pairs[:args.n_calibration]
    test_pairs = remaining_pairs[args.n_calibration:]

    print(f"Calibration pairs: {len(calibration_pairs)}  "
          f"(pair_ids {calibration_pairs[0]['pair_id']}-{calibration_pairs[-1]['pair_id']})")
    print(f"Test pairs:        {len(test_pairs)}  "
          f"(pair_ids {test_pairs[0]['pair_id']}-{test_pairs[-1]['pair_id']})")

    # Save config
    config = {
        "direction_source": args.direction,
        "source_activations_dir": args.source_activations_dir,
        "target_pairs": args.pairs,
        "target_dataset": args.dataset,
        "layer": args.layer,
        "n_calibration": args.n_calibration,
        "calibration_pair_ids": [p["pair_id"] for p in calibration_pairs],
        "test_pair_ids": [p["pair_id"] for p in test_pairs],
    }
    with open(os.path.join(args.out_dir, "cross_steering_v3_config.json"), "w") as f:
        json.dump(config, f, indent=2)

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
    # Extract activations from target dataset
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"Extracting calibration activations ({len(calibration_pairs)} pairs)")
    print(f"{'='*60}")

    cal_uncertain, cal_certain = extract_activations(
        calibration_pairs, model, tokenizer, args.layer,
        dataset=args.dataset, demo_certain=demo_certain,
        demo_uncertain=demo_uncertain,
    )
    print(f"  Calibration activations shape: {cal_uncertain.shape}")

    print(f"\n{'='*60}")
    print(f"Extracting test activations ({len(test_pairs)} pairs)")
    print(f"{'='*60}")

    test_uncertain, test_certain = extract_activations(
        test_pairs, model, tokenizer, args.layer,
        dataset=args.dataset, demo_certain=demo_certain,
        demo_uncertain=demo_uncertain,
    )
    print(f"  Test activations shape: {test_uncertain.shape}")

    # ==========================================================================
    # Mode A: Transfer source probe directly to target test data
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"Mode A: Transfer source probe to target test data")
    print(f"{'='*60}")

    transfer_u_preds, transfer_u_info = source_probe.classify(test_uncertain)
    transfer_c_preds, transfer_c_info = source_probe.classify(test_certain)

    transfer_uncertain_acc = float((transfer_u_preds == 1).sum() / len(transfer_u_preds))
    transfer_certain_acc = float((transfer_c_preds == 0).sum() / len(transfer_c_preds))
    transfer_acc = float(((transfer_u_preds == 1).sum() + (transfer_c_preds == 0).sum())
                         / (len(transfer_u_preds) + len(transfer_c_preds)))

    transfer_metrics = {
        "accuracy": transfer_acc,
        "uncertain_accuracy": transfer_uncertain_acc,
        "certain_accuracy": transfer_certain_acc,
        "n_test_pairs": len(test_uncertain),
    }
    print(f"  Accuracy:           {transfer_acc:.4f}")
    print(f"  Uncertain accuracy: {transfer_uncertain_acc:.4f}")
    print(f"  Certain accuracy:   {transfer_certain_acc:.4f}")

    # Histogram for transfer
    u_score_transfer = transfer_u_info["dist_certain"] - transfer_u_info["dist_uncertain"]
    c_score_transfer = transfer_c_info["dist_certain"] - transfer_c_info["dist_uncertain"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(c_score_transfer, bins=20, alpha=0.6, label="Certain (low uncertainty)", color="green")
    ax.hist(u_score_transfer, bins=20, alpha=0.6, label="Uncertain (high uncertainty)", color="red")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Decision boundary")
    ax.set_xlabel("dist(certain centroid) - dist(uncertain centroid)")
    ax.set_ylabel("Count")
    ax.set_title("Mode A: Source Probe Transfer to Target")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "massmean_transfer_histogram.png"), dpi=150)
    plt.close(fig)

    # ==========================================================================
    # Mode B: Refit probe on target calibration data
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"Mode B: Refit MassMeanProbe on {len(calibration_pairs)} calibration pairs")
    print(f"{'='*60}")

    refit_probe = MassMeanProbe()
    refit_probe.fit(cal_uncertain, cal_certain)
    print(f"  Refit probe fitted on calibration data.")

    refit_u_preds, refit_u_info = refit_probe.classify(test_uncertain)
    refit_c_preds, refit_c_info = refit_probe.classify(test_certain)

    refit_uncertain_acc = float((refit_u_preds == 1).sum() / len(refit_u_preds))
    refit_certain_acc = float((refit_c_preds == 0).sum() / len(refit_c_preds))
    refit_acc = float(((refit_u_preds == 1).sum() + (refit_c_preds == 0).sum())
                      / (len(refit_u_preds) + len(refit_c_preds)))

    refit_metrics = {
        "accuracy": refit_acc,
        "uncertain_accuracy": refit_uncertain_acc,
        "certain_accuracy": refit_certain_acc,
        "n_test_pairs": len(test_uncertain),
        "n_calibration_pairs": len(calibration_pairs),
    }
    print(f"  Accuracy:           {refit_acc:.4f}")
    print(f"  Uncertain accuracy: {refit_uncertain_acc:.4f}")
    print(f"  Certain accuracy:   {refit_certain_acc:.4f}")

    # Histogram for refit
    u_score_refit = refit_u_info["dist_certain"] - refit_u_info["dist_uncertain"]
    c_score_refit = refit_c_info["dist_certain"] - refit_c_info["dist_uncertain"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(c_score_refit, bins=20, alpha=0.6, label="Certain (low uncertainty)", color="green")
    ax.hist(u_score_refit, bins=20, alpha=0.6, label="Uncertain (high uncertainty)", color="red")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Decision boundary")
    ax.set_xlabel("dist(certain centroid) - dist(uncertain centroid)")
    ax.set_ylabel("Count")
    ax.set_title("Mode B: Refit Probe on Target Calibration Data")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "massmean_refit_histogram.png"), dpi=150)
    plt.close(fig)

    # ==========================================================================
    # Baseline: Zero-boundary projection with source direction
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"Baseline: Zero-boundary projection on target test data")
    print(f"{'='*60}")

    uncertain_projs = test_uncertain @ direction
    certain_projs = test_certain @ direction

    proj_uncertain_acc = float((uncertain_projs > 0).sum() / len(uncertain_projs))
    proj_certain_acc = float((certain_projs < 0).sum() / len(certain_projs))
    proj_acc = float(((uncertain_projs > 0).sum() + (certain_projs < 0).sum())
                     / (len(uncertain_projs) + len(certain_projs)))

    proj_metrics = {
        "accuracy": proj_acc,
        "uncertain_accuracy": proj_uncertain_acc,
        "certain_accuracy": proj_certain_acc,
        "n_test_pairs": len(test_uncertain),
        "uncertain_mean_proj": float(uncertain_projs.mean()),
        "certain_mean_proj": float(certain_projs.mean()),
    }
    print(f"  Accuracy:           {proj_acc:.4f}")
    print(f"  Uncertain accuracy: {proj_uncertain_acc:.4f}")
    print(f"  Certain accuracy:   {proj_certain_acc:.4f}")

    # Histogram for projection baseline
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(certain_projs, bins=20, alpha=0.6, label="Certain (low uncertainty)", color="green")
    ax.hist(uncertain_projs, bins=20, alpha=0.6, label="Uncertain (high uncertainty)", color="red")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Decision boundary")
    ax.set_xlabel("Projection onto steering direction")
    ax.set_ylabel("Count")
    ax.set_title("Baseline: Projection with Source Direction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "cross_projection_histogram.png"), dpi=150)
    plt.close(fig)

    # ==========================================================================
    # Save combined comparison
    # ==========================================================================
    comparison = {
        "projection_baseline": proj_metrics,
        "massmean_transfer": transfer_metrics,
        "massmean_refit": refit_metrics,
    }
    with open(os.path.join(args.out_dir, "cross_validation_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Cross-dataset validation comparison:")
    print(f"  Projection baseline: {proj_acc:.4f}")
    print(f"  Mass-mean transfer:  {transfer_acc:.4f}")
    print(f"  Mass-mean refit:     {refit_acc:.4f}")
    print(f"  Saved -> cross_validation_comparison.json")

    print(f"\n{'='*60}")
    print(f"Cross-dataset classification complete. Outputs saved to {args.out_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
