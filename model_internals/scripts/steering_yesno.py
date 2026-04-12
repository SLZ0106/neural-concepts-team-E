"""
RepE Steering Pipeline for Economic Uncertainty.

Implements 4 stages:
  1. Extract steering direction via mass mean difference or PCA
  2. Validate direction on held-out data
  3. Steer model during inference with alpha sweep
  4. Generate visualization plots

Reuses prompt utilities from activation_patching_yesno.py.
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
from sklearn.decomposition import PCA

# Add parent dir so we can import from activation_patching
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from activation_patching_yesno import (
    build_messages,
    build_messages_synthetic,
    apply_template,
    find_last_period_pos,
    get_label_token_ids,
    load_pairs,
)

from nnsight import LanguageModel


# ── Stage 1: Extract activations and compute direction ────────────────────────

def extract_activations(pairs, model, tokenizer, layer,
                        dataset="templated", demo_certain=None, demo_uncertain=None,
                        include_definition=False):
    """Extract layer activations at last-period token for all pairs."""
    uncertain_acts = []
    certain_acts = []

    for i, pair in enumerate(pairs):
        for label_key, act_list in [("uncertain", uncertain_acts), ("certain", certain_acts)]:
            if dataset == "synthetic":
                msgs = build_messages_synthetic(pair[label_key])
            else:
                msgs = build_messages(demo_certain, demo_uncertain, pair[label_key],
                                      include_definition=include_definition)
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
    diffs = uncertain_acts - certain_acts  # (N, hidden_dim)

    if method == "mean":
        direction = diffs.mean(axis=0)
    elif method == "pca":
        pca = PCA(n_components=1)
        pca.fit(diffs)
        direction = pca.components_[0]
    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize to unit length
    direction = direction / np.linalg.norm(direction)
    return direction


# ── Stage 2: Validate on held-out data ────────────────────────────────────────

def validate_direction(test_uncertain_acts, test_certain_acts, direction, out_dir):
    """Project test activations onto direction and evaluate classification."""
    # Project
    uncertain_projs = test_uncertain_acts @ direction
    certain_projs = test_certain_acts @ direction

    # Classify by sign: positive = uncertain, negative = certain
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

    print(f"\n── Stage 2: Validation Results ──")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Uncertain accuracy: {metrics['uncertain_accuracy']:.4f}")
    print(f"  Certain accuracy:   {metrics['certain_accuracy']:.4f}")
    print(f"  Uncertain mean proj: {metrics['uncertain_mean_proj']:.4f}")
    print(f"  Certain mean proj:   {metrics['certain_mean_proj']:.4f}")

    # Save metrics
    with open(os.path.join(out_dir, "steering_validation.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot overlapping histograms
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(certain_projs, bins=20, alpha=0.6, label="Certain (no uncertainty)", color="green")
    ax.hist(uncertain_projs, bins=20, alpha=0.6, label="Uncertain (high uncertainty)", color="red")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Decision boundary")
    ax.set_xlabel("Projection onto steering direction")
    ax.set_ylabel("Count")
    ax.set_title("Projection of Test Activations onto Steering Direction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "steering_projection_histogram.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved histogram → steering_projection_histogram.png")

    return metrics


# ── Stage 3: Steer model during inference ─────────────────────────────────────

def steer_and_evaluate(test_pairs, model, tokenizer,
                       layer, direction_tensor, alphas, steer_mode,
                       no_id, yes_id, out_dir,
                       dataset="templated", demo_certain=None, demo_uncertain=None,
                       include_definition=False,
                       results_filename="steering_results.json"):
    """Sweep alpha values and evaluate steered model."""
    all_alpha_results = []

    for alpha in alphas:
        print(f"\n  Alpha = {alpha:.1f}")
        results_for_alpha = []

        for i, pair in enumerate(test_pairs):
            for label_key, true_label in [("certain", "NO"), ("uncertain", "YES")]:
                if dataset == "synthetic":
                    msgs = build_messages_synthetic(pair[label_key])
                else:
                    msgs = build_messages(demo_certain, demo_uncertain, pair[label_key],
                                          include_definition=include_definition)
                ids = apply_template(msgs, tokenizer, tokenize=True)
                input_tensor = torch.tensor([ids])
                period_pos = find_last_period_pos(ids, tokenizer)

                with model.trace(input_tensor):
                    # Intervene at the specified layer
                    if steer_mode == "period":
                        model.model.layers[layer].output[0][period_pos, :] += (
                            alpha * direction_tensor
                        )
                    elif steer_mode == "all":
                        model.model.layers[layer].output[0][:, :] += (
                            alpha * direction_tensor
                        )
                    logits = model.output.logits[0, -1, :].save()

                logit_yes = logits[yes_id].item()
                logit_no = logits[no_id].item()
                pred_label = "YES" if logit_yes > logit_no else "NO"

                results_for_alpha.append({
                    "pair_id": pair["pair_id"],
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "logit_yes": logit_yes,
                    "logit_no": logit_no,
                    "logit_diff": logit_yes - logit_no,
                })

        # Compute summary for this alpha
        correct = sum(r["true_label"] == r["pred_label"] for r in results_for_alpha)
        total = len(results_for_alpha)
        accuracy = correct / total

        yes_samples = [r for r in results_for_alpha if r["true_label"] == "YES"]
        no_samples = [r for r in results_for_alpha if r["true_label"] == "NO"]
        yes_acc = sum(r["pred_label"] == "YES" for r in yes_samples) / len(yes_samples) if yes_samples else 0
        no_acc = sum(r["pred_label"] == "NO" for r in no_samples) / len(no_samples) if no_samples else 0
        mean_logit_diff = np.mean([r["logit_diff"] for r in results_for_alpha])

        alpha_summary = {
            "alpha": alpha,
            "accuracy": accuracy,
            "uncertain_accuracy": yes_acc,
            "certain_accuracy": no_acc,
            "mean_logit_diff": float(mean_logit_diff),
            "n_samples": total,
            "samples": results_for_alpha,
        }
        all_alpha_results.append(alpha_summary)
        print(f"    Accuracy: {accuracy:.4f}  |  Uncertain: {yes_acc:.4f}  |  Certain: {no_acc:.4f}  |  Mean logit diff: {mean_logit_diff:.3f}")

    # Save full results
    with open(os.path.join(out_dir, results_filename), "w") as f:
        json.dump(all_alpha_results, f, indent=2)
    print(f"\n  Saved steering results → {results_filename}")

    return all_alpha_results


# ── Stage 4: Visualization ────────────────────────────────────────────────────

def plot_steering_results(alpha_results, out_dir, prefix="steering"):
    """Generate steering result plots with configurable prefix."""
    alphas = [r["alpha"] for r in alpha_results]
    accuracies = [r["accuracy"] for r in alpha_results]
    uncertain_accs = [r["uncertain_accuracy"] for r in alpha_results]
    certain_accs = [r["certain_accuracy"] for r in alpha_results]
    mean_logit_diffs = [r["mean_logit_diff"] for r in alpha_results]

    split_label = prefix.replace("steering_", "").replace("steering", "").strip("_")
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
    ax.set_ylabel("Mean logit diff (YES - NO)")
    ax.set_title(f"Mean Logit Difference vs. Steering Strength{title_suffix}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_logitdiff_vs_alpha.png"), dpi=150)
    plt.close(fig)

    # Plot 3: Per-Class Accuracy vs Alpha
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, uncertain_accs, marker="^", color="red", linewidth=2, label="Uncertain (true=YES)")
    ax.plot(alphas, certain_accs, marker="v", color="green", linewidth=2, label="Certain (true=NO)")
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RepE Steering for Economic Uncertainty")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--pairs", default="neural-concepts-team-E/synthetic_data/API/synthetic_pairs_200.json")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--method", choices=["mean", "pca"], default="mean")
    parser.add_argument("--steer_mode", choices=["period", "all"], default="period")
    parser.add_argument("--alphas", default="-5,-4,-3,-2,-1,0,1,2,3,4,5")
    parser.add_argument("--n_train", type=int, default=100,
                        help="Number of train pairs (after demo). Matches activation patching pairs 2–101.")
    parser.add_argument("--def", dest="include_def", action="store_true", default=False)
    parser.add_argument("--dataset", choices=["templated", "synthetic"], default="templated",
                        help="templated: 1 demo pair from dataset. "
                             "synthetic: definition + 4 fixed few-shot demos.")
    parser.add_argument("--out_dir", default=".")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    alphas = [float(a) for a in args.alphas.split(",")]

    print("=" * 60)
    print("RepE Steering Pipeline — Economic Uncertainty")
    print("=" * 60)
    print(f"Model:       {args.model}")
    print(f"Layer:       {args.layer}")
    print(f"Method:      {args.method}")
    print(f"Steer mode:  {args.steer_mode}")
    print(f"Alphas:      {alphas}")
    print(f"N train:     {args.n_train}")
    print(f"Dataset:     {args.dataset}")
    print(f"Include def: {args.include_def}")
    print(f"Output dir:  {args.out_dir}")

    # ── Load pairs ────────────────────────────────────────────────────────────
    pairs = load_pairs(args.pairs)
    print(f"\nLoaded {len(pairs)} contrastive pairs.")

    if args.dataset == "templated":
        # Pair 1 = demo (same as activation_patching_yesno.py)
        demo_pair = pairs[0]
        demo_certain = demo_pair["certain"]
        demo_uncertain = demo_pair["uncertain"]
        # Train = pairs 2–101, Test = pairs 102–200
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

    print(f"Train pairs:   {len(train_pairs)}  (pair_ids {train_pairs[0]['pair_id']}–{train_pairs[-1]['pair_id']})")
    print(f"Test pairs:    {len(test_pairs)}  (pair_ids {test_pairs[0]['pair_id']}–{test_pairs[-1]['pair_id']})")

    # Save split
    split_info = {
        "demo_pair_id": demo_pair["pair_id"] if args.dataset == "templated" else None,
        "dataset": args.dataset,
        "train_pair_ids": [p["pair_id"] for p in train_pairs],
        "test_pair_ids": [p["pair_id"] for p in test_pairs],
    }
    with open(os.path.join(args.out_dir, "steering_split.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    # ── Load model ────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading model on {device}...")
    model = LanguageModel(args.model, device_map=device, dispatch=True)
    tokenizer = model.tokenizer
    n_layers = model.config.num_hidden_layers
    print(f"Model loaded. Layers: {n_layers}")

    no_id, yes_id = get_label_token_ids(tokenizer)
    print(f"Label token IDs — NO: {no_id}  YES: {yes_id}")

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 1: Extract direction from train pairs
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Stage 1: Extract steering direction (layer {args.layer}, method={args.method})")
    print(f"  Using {len(train_pairs)} train pairs")
    print(f"{'='*60}")

    print("\nExtracting train activations...")
    train_uncertain, train_certain = extract_activations(
        train_pairs, model, tokenizer, args.layer,
        dataset=args.dataset, demo_certain=demo_certain,
        demo_uncertain=demo_uncertain, include_definition=args.include_def,
    )
    print(f"  Train activations shape: {train_uncertain.shape}")

    direction = compute_direction(train_uncertain, train_certain, method=args.method)
    np.save(os.path.join(args.out_dir, "steering_direction.npy"), direction)
    print(f"  Direction computed and saved. Norm: {np.linalg.norm(direction):.6f}")

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 2: Validate classification on test pairs
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Stage 2: Validate on held-out test set ({len(test_pairs)} pairs)")
    print(f"{'='*60}")

    print("\nExtracting test activations...")
    test_uncertain, test_certain = extract_activations(
        test_pairs, model, tokenizer, args.layer,
        dataset=args.dataset, demo_certain=demo_certain,
        demo_uncertain=demo_uncertain, include_definition=args.include_def,
    )
    validation_metrics = validate_direction(test_uncertain, test_certain, direction, args.out_dir)

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 3: Steer model — run on BOTH train and test pairs separately
    # ══════════════════════════════════════════════════════════════════════════
    direction_tensor = torch.tensor(direction, dtype=torch.float16, device=device)

    print(f"\n{'='*60}")
    print(f"Stage 3a: Steer on TRAIN pairs ({len(train_pairs)} pairs, alphas={alphas})")
    print(f"{'='*60}")

    train_alpha_results = steer_and_evaluate(
        train_pairs, model, tokenizer,
        args.layer, direction_tensor, alphas, args.steer_mode,
        no_id, yes_id, args.out_dir,
        dataset=args.dataset, demo_certain=demo_certain,
        demo_uncertain=demo_uncertain, include_definition=args.include_def,
        results_filename="steering_results_train.json",
    )

    print(f"\n{'='*60}")
    print(f"Stage 3b: Steer on TEST pairs ({len(test_pairs)} pairs, alphas={alphas})")
    print(f"{'='*60}")

    test_alpha_results = steer_and_evaluate(
        test_pairs, model, tokenizer,
        args.layer, direction_tensor, alphas, args.steer_mode,
        no_id, yes_id, args.out_dir,
        dataset=args.dataset, demo_certain=demo_certain,
        demo_uncertain=demo_uncertain, include_definition=args.include_def,
        results_filename="steering_results_test.json",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 4: Visualization — separate plots for train and test
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Stage 4: Generate plots")
    print(f"{'='*60}")

    plot_steering_results(train_alpha_results, args.out_dir, prefix="steering_train")
    plot_steering_results(test_alpha_results,  args.out_dir, prefix="steering_test")

    print(f"\n{'='*60}")
    print(f"Pipeline complete. All outputs saved to {args.out_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
