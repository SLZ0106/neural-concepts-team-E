"""
Cross-dataset steering: load a steering direction computed from one dataset
and apply it (validate + steer) on pairs from a different dataset.

Reuses prompt utilities from activation_patching_highlow.py and helper functions
from steering_highlow.py.
"""

import sys
import os
import json
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

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
    steer_and_evaluate,
    plot_steering_results,
)

from nnsight import LanguageModel


def main():
    parser = argparse.ArgumentParser(
        description="Cross-dataset steering: apply one dataset's steering direction to another dataset"
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--direction", required=True,
                        help="Path to .npy steering direction (from source dataset)")
    parser.add_argument("--pairs", required=True,
                        help="Path to contrastive pairs JSON (target dataset to evaluate on)")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--steer_mode", choices=["period", "all", "last"], default="period")
    parser.add_argument("--alphas", default="-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--n_train", type=int, default=100,
                        help="Number of pairs to use for validation (mirrors train split of target dataset)")
    parser.add_argument("--dataset", choices=["templated", "synthetic"], required=True,
                        help="Prompt format for the TARGET dataset being evaluated on")
    parser.add_argument("--out_dir", default=".")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    alphas = [float(a) for a in args.alphas.split(",")]

    print("=" * 60)
    print("Cross-Dataset Steering")
    print("=" * 60)
    print(f"Model:          {args.model}")
    print(f"Direction from: {args.direction}")
    print(f"Target pairs:   {args.pairs}")
    print(f"Target dataset: {args.dataset}")
    print(f"Layer:          {args.layer}")
    print(f"Steer mode:     {args.steer_mode}")
    print(f"Alphas:         {alphas}")
    print(f"Output dir:     {args.out_dir}")

    # -- Load steering direction from source dataset ---------------------------
    direction = np.load(args.direction)
    print(f"\nLoaded steering direction: shape={direction.shape}, norm={np.linalg.norm(direction):.6f}")

    # -- Load target pairs -----------------------------------------------------
    pairs = load_pairs(args.pairs)
    print(f"Loaded {len(pairs)} contrastive pairs from target dataset.")

    if args.dataset == "templated":
        demo_pair = pairs[0]
        demo_certain = demo_pair["certain"]
        demo_uncertain = demo_pair["uncertain"]
        train_pairs = pairs[1 : 1 + args.n_train]
        test_pairs  = pairs[1 + args.n_train :]
        print(f"Demo pair_id:  {demo_pair['pair_id']}")
    else:
        demo_certain = None
        demo_uncertain = None
        train_pairs = pairs[:args.n_train]
        test_pairs  = pairs[args.n_train:]
        print(f"Using 4 fixed few-shot demos (no demo pair consumed)")

    print(f"Train pairs:   {len(train_pairs)}  (pair_ids {train_pairs[0]['pair_id']}-{train_pairs[-1]['pair_id']})")
    print(f"Test pairs:    {len(test_pairs)}  (pair_ids {test_pairs[0]['pair_id']}-{test_pairs[-1]['pair_id']})")

    # Save config
    config = {
        "direction_source": args.direction,
        "target_pairs": args.pairs,
        "target_dataset": args.dataset,
        "layer": args.layer,
        "steer_mode": args.steer_mode,
        "alphas": alphas,
        "n_train": args.n_train,
        "train_pair_ids": [p["pair_id"] for p in train_pairs],
        "test_pair_ids": [p["pair_id"] for p in test_pairs],
    }
    with open(os.path.join(args.out_dir, "cross_steering_config.json"), "w") as f:
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
    # Stage 1: Validate cross-dataset direction on target test pairs
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"Stage 1: Validate cross-dataset direction on target test pairs")
    print(f"{'='*60}")

    print("\nExtracting test activations from target dataset...")
    test_uncertain, test_certain = extract_activations(
        test_pairs, model, tokenizer, args.layer,
        dataset=args.dataset, demo_certain=demo_certain,
        demo_uncertain=demo_uncertain,
    )
    validate_direction(test_uncertain, test_certain, direction, args.out_dir)

    # ==========================================================================
    # Stage 2: Steer on train pairs
    # ==========================================================================
    direction_tensor = torch.tensor(direction, dtype=torch.float16, device=device)

    print(f"\n{'='*60}")
    print(f"Stage 2: Steer on TRAIN pairs ({len(train_pairs)} pairs)")
    print(f"{'='*60}")

    train_alpha_results = steer_and_evaluate(
        train_pairs, model, tokenizer,
        args.layer, direction_tensor, alphas, args.steer_mode,
        low_id, high_id, args.out_dir,
        dataset=args.dataset, demo_certain=demo_certain,
        demo_uncertain=demo_uncertain,
        results_filename="cross_steering_results_train.json",
    )

    # ==========================================================================
    # Stage 3: Steer on test pairs
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"Stage 3: Steer on TEST pairs ({len(test_pairs)} pairs)")
    print(f"{'='*60}")

    test_alpha_results = steer_and_evaluate(
        test_pairs, model, tokenizer,
        args.layer, direction_tensor, alphas, args.steer_mode,
        low_id, high_id, args.out_dir,
        dataset=args.dataset, demo_certain=demo_certain,
        demo_uncertain=demo_uncertain,
        results_filename="cross_steering_results_test.json",
    )

    # ==========================================================================
    # Stage 4: Visualization
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"Stage 4: Generate plots")
    print(f"{'='*60}")

    plot_steering_results(train_alpha_results, args.out_dir, prefix="cross_steering_train")
    plot_steering_results(test_alpha_results,  args.out_dir, prefix="cross_steering_test")

    print(f"\n{'='*60}")
    print(f"Cross-dataset steering complete. Outputs saved to {args.out_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
