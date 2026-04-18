"""
Run steering intervention on a target dataset using a pre-computed direction.

Sweeps across alpha values and evaluates the effect of adding the uncertainty
direction vector to model activations during inference. Evaluates on both
train and test splits of the target dataset and generates plots.

The direction file is produced by extract_direction.py.

Example:
    python run_intervention.py \
        --direction results/direction.npy \
        --dataset synthetic \
        --steer_mode period \
        --alphas "-5,-4,-3,-2,-1,0,1,2,3,4,5" \
        --out_dir results/intervention/
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

from nnsight import LanguageModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATASET_PATHS = {
    "synthetic": os.path.join(PROJECT_ROOT, "neural-concepts-team-E/synthetic_data/API/synthetic_pairs_200.json"),
    "templated": os.path.join(PROJECT_ROOT, "neural-concepts-team-E/synthetic_data/templated/templated_pairs_200.json"),
}


# -- Steering ------------------------------------------------------------------

def steer_and_evaluate(pairs, model, tokenizer,
                       layer, direction_tensor, alphas, steer_mode,
                       low_id, high_id,
                       dataset="templated", demo_certain=None, demo_uncertain=None):
    """Sweep alpha values and evaluate steered model."""
    all_alpha_results = []

    for alpha in alphas:
        print(f"\n  Alpha = {alpha:.1f}")
        results_for_alpha = []

        for i, pair in enumerate(pairs):
            for label_key, true_label in [("certain", "LOW"), ("uncertain", "HIGH")]:
                if dataset == "synthetic":
                    msgs = build_messages_synthetic(pair[label_key])
                else:
                    msgs = build_messages(demo_certain, demo_uncertain, pair[label_key])
                ids = apply_template(msgs, tokenizer, tokenize=True)
                input_tensor = torch.tensor([ids])
                period_pos = find_last_period_pos(ids, tokenizer)

                with model.trace(input_tensor):
                    if steer_mode == "period":
                        model.model.layers[layer].output[0][period_pos, :] += (
                            alpha * direction_tensor
                        )
                    elif steer_mode == "last":
                        model.model.layers[layer].output[0][-1, :] += (
                            alpha * direction_tensor
                        )
                    elif steer_mode == "all":
                        model.model.layers[layer].output[0][:, :] += (
                            alpha * direction_tensor
                        )
                    logits = model.output.logits[0, -1, :].save()

                logit_high = logits[high_id].item()
                logit_low = logits[low_id].item()

                top_token_id = logits.argmax().item()
                top_token_str = tokenizer.decode([top_token_id]).strip()

                top_upper = top_token_str.upper()
                if top_upper == "HIGH":
                    pred_label = "HIGH"
                elif top_upper == "LOW":
                    pred_label = "LOW"
                else:
                    pred_label = "OTHER"

                results_for_alpha.append({
                    "pair_id": pair["pair_id"],
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "decoded_token": top_token_str,
                    "decoded_token_id": top_token_id,
                    "logit_high": logit_high,
                    "logit_low": logit_low,
                    "logit_diff": logit_high - logit_low,
                })

        # Compute summary for this alpha
        correct = sum(r["true_label"] == r["pred_label"] for r in results_for_alpha)
        total = len(results_for_alpha)
        accuracy = correct / total

        high_samples = [r for r in results_for_alpha if r["true_label"] == "HIGH"]
        low_samples = [r for r in results_for_alpha if r["true_label"] == "LOW"]
        high_acc = sum(r["pred_label"] == "HIGH" for r in high_samples) / len(high_samples) if high_samples else 0
        low_acc = sum(r["pred_label"] == "LOW" for r in low_samples) / len(low_samples) if low_samples else 0
        mean_logit_diff = np.mean([r["logit_diff"] for r in results_for_alpha])

        other_count = sum(r["pred_label"] == "OTHER" for r in results_for_alpha)
        other_pct = other_count / total
        other_tokens = set(
            r["decoded_token"] for r in results_for_alpha if r["pred_label"] == "OTHER"
        )

        alpha_summary = {
            "alpha": alpha,
            "accuracy": accuracy,
            "uncertain_accuracy": high_acc,
            "certain_accuracy": low_acc,
            "mean_logit_diff": float(mean_logit_diff),
            "other_count": other_count,
            "other_pct": float(other_pct),
            "n_samples": total,
            "samples": results_for_alpha,
        }
        all_alpha_results.append(alpha_summary)

        other_info = f"  |  OTHER: {other_count}/{total} ({other_pct:.1%})" if other_count else ""
        other_tok_info = f" tokens={other_tokens}" if other_tokens else ""
        print(f"    Accuracy: {accuracy:.4f}  |  Uncertain: {high_acc:.4f}  |  "
              f"Certain: {low_acc:.4f}  |  Mean logit diff: {mean_logit_diff:.3f}"
              f"{other_info}{other_tok_info}")

    return all_alpha_results


# -- Plotting ------------------------------------------------------------------

def plot_results(alpha_results, out_dir, prefix="intervention"):
    """Generate intervention result plots."""
    alphas = [r["alpha"] for r in alpha_results]
    accuracies = [r["accuracy"] for r in alpha_results]
    uncertain_accs = [r["uncertain_accuracy"] for r in alpha_results]
    certain_accs = [r["certain_accuracy"] for r in alpha_results]
    mean_logit_diffs = [r["mean_logit_diff"] for r in alpha_results]

    split_label = prefix.replace("intervention_", "").replace("intervention", "").strip("_")
    title_suffix = f" ({split_label})" if split_label else ""

    # Plot 1: Accuracy vs Alpha
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, accuracies, marker="o", color="steelblue", linewidth=2)
    ax.set_xlabel("Steering strength (alpha)")
    ax.set_ylabel("Classification accuracy")
    ax.set_title(f"Overall Accuracy vs. Steering Strength{title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_accuracy_vs_alpha.png"), dpi=150)
    plt.close(fig)

    # Plot 2: Logit Diff vs Alpha
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, mean_logit_diffs, marker="s", color="darkorange", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Steering strength (alpha)")
    ax.set_ylabel("Mean logit diff (HIGH - LOW)")
    ax.set_title(f"Mean Logit Difference vs. Steering Strength{title_suffix}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_logitdiff_vs_alpha.png"), dpi=150)
    plt.close(fig)

    # Plot 3: Per-Class Accuracy vs Alpha
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, uncertain_accs, marker="^", color="red", linewidth=2, label="Uncertain (true=HIGH)")
    ax.plot(alphas, certain_accs, marker="v", color="green", linewidth=2, label="Certain (true=LOW)")
    ax.plot(alphas, accuracies, marker="o", color="steelblue", linewidth=2, linestyle="--", label="Overall")
    ax.set_xlabel("Steering strength (alpha)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Per-Class Accuracy vs. Steering Strength{title_suffix}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_perclass_vs_alpha.png"), dpi=150)
    plt.close(fig)

    print(f"  Saved {prefix}_*.png plots to {out_dir}/")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run steering intervention using a pre-computed uncertainty direction"
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--direction", required=True,
                        help="Path to .npy direction file (from extract_direction.py)")
    parser.add_argument("--dataset", choices=["templated", "synthetic"], required=True,
                        help="Target dataset name")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--steer_mode", choices=["period", "all", "last"], default="last")
    parser.add_argument("--alphas", default="-5,-4,-3,-2,-1,0,1,2,3,4,5")
    parser.add_argument("--n_train", type=int, default=100,
                        help="Number of pairs for train split (intervention evaluated on both splits)")
    parser.add_argument("--out_dir", default=".")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    alphas = [float(a) for a in args.alphas.split(",")]
    pairs_path = DATASET_PATHS[args.dataset]

    print("=" * 60)
    print("Steering Intervention")
    print("=" * 60)
    print(f"Model:       {args.model}")
    print(f"Direction:   {args.direction}")
    print(f"Dataset:     {args.dataset}  ({pairs_path})")
    print(f"Layer:       {args.layer}")
    print(f"Steer mode:  {args.steer_mode}")
    print(f"Alphas:      {alphas}")
    print(f"N train:     {args.n_train}")
    print(f"Output dir:  {args.out_dir}")

    # -- Load direction --------------------------------------------------------
    direction = np.load(args.direction)
    print(f"\nLoaded direction: shape={direction.shape}, norm={np.linalg.norm(direction):.6f}")

    # -- Load and split pairs --------------------------------------------------
    pairs = load_pairs(pairs_path)
    print(f"Loaded {len(pairs)} contrastive pairs.")

    if args.dataset == "templated":
        demo_pair = pairs[0]
        demo_certain = demo_pair["certain"]
        demo_uncertain = demo_pair["uncertain"]
        train_pairs = pairs[1 : 1 + args.n_train]
        test_pairs = pairs[1 + args.n_train :]
        print(f"Demo pair_id: {demo_pair['pair_id']}")
    else:
        demo_certain = None
        demo_uncertain = None
        train_pairs = pairs[:args.n_train]
        test_pairs = pairs[args.n_train:]
        print(f"Using fixed few-shot demos (no demo pair consumed)")

    print(f"Train pairs: {len(train_pairs)}  "
          f"(pair_ids {train_pairs[0]['pair_id']}-{train_pairs[-1]['pair_id']})")
    print(f"Test pairs:  {len(test_pairs)}  "
          f"(pair_ids {test_pairs[0]['pair_id']}-{test_pairs[-1]['pair_id']})")

    # Save config
    config = {
        "direction": args.direction,
        "dataset": args.dataset,
        "layer": args.layer,
        "steer_mode": args.steer_mode,
        "alphas": alphas,
        "n_train": args.n_train,
        "train_pair_ids": [p["pair_id"] for p in train_pairs],
        "test_pair_ids": [p["pair_id"] for p in test_pairs],
    }
    with open(os.path.join(args.out_dir, "intervention_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # -- Load model ------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading model on {device}...")
    model = LanguageModel(args.model, device_map=device, dispatch=True)
    tokenizer = model.tokenizer
    print(f"Model loaded. Layers: {model.config.num_hidden_layers}")

    low_id, high_id = get_label_token_ids(tokenizer)
    print(f"Label token IDs -- LOW: {low_id}  HIGH: {high_id}")

    direction_tensor = torch.tensor(direction, dtype=torch.float16, device=device)

    # ==========================================================================
    # Steer on train pairs
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"Steering on TRAIN pairs ({len(train_pairs)} pairs)")
    print(f"{'='*60}")

    train_results = steer_and_evaluate(
        train_pairs, model, tokenizer,
        args.layer, direction_tensor, alphas, args.steer_mode,
        low_id, high_id,
        dataset=args.dataset, demo_certain=demo_certain,
        demo_uncertain=demo_uncertain,
    )

    with open(os.path.join(args.out_dir, "intervention_results_train.json"), "w") as f:
        json.dump(train_results, f, indent=2)
    print(f"  Saved -> intervention_results_train.json")

    # ==========================================================================
    # Steer on test pairs
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"Steering on TEST pairs ({len(test_pairs)} pairs)")
    print(f"{'='*60}")

    test_results = steer_and_evaluate(
        test_pairs, model, tokenizer,
        args.layer, direction_tensor, alphas, args.steer_mode,
        low_id, high_id,
        dataset=args.dataset, demo_certain=demo_certain,
        demo_uncertain=demo_uncertain,
    )

    with open(os.path.join(args.out_dir, "intervention_results_test.json"), "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"  Saved -> intervention_results_test.json")

    # ==========================================================================
    # Plots
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"Generating plots")
    print(f"{'='*60}")

    plot_results(train_results, args.out_dir, prefix="intervention_train")
    plot_results(test_results, args.out_dir, prefix="intervention_test")

    print(f"\n{'='*60}")
    print(f"Intervention complete. Outputs saved to {args.out_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
