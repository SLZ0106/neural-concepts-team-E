"""
Evaluate model on the contrastive pairs dataset.

Uses the identical 2-shot prompt and tokenization as activation_patching.py
(imported directly). The demo pair is selected by pair_id; both its certain
and uncertain statements are used as the 2-shot examples and excluded from
evaluation. All remaining pairs contribute two test samples each.

The model's response is obtained via greedy generation (max 10 new tokens).
The raw response text is checked for 'YES' / 'NO' to determine
the predicted label (YES = uncertain, NO = certain).

Output: accuracy, per-class breakdown, confusion matrix, and unparseable count.
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_patching import (
    apply_template,
    build_messages,
    load_pairs,
)


def pick_demo_pair(pairs: list[dict], demo_pair_id: int):
    """
    Return (demo_certain_stmt, demo_uncertain_stmt, test_pairs) where
    test_pairs excludes the demo pair.
    """
    demo = next((p for p in pairs if p["pair_id"] == demo_pair_id), None)
    if demo is None:
        available = sorted(p["pair_id"] for p in pairs)
        raise ValueError(
            f"pair_id {demo_pair_id} not found. Available: {available}"
        )
    test_pairs = [p for p in pairs if p["pair_id"] != demo_pair_id]
    return demo["certain"], demo["uncertain"], test_pairs


def pairs_to_samples(pairs: list[dict]) -> list[dict]:
    """
    Flatten pairs into individual test samples, each with:
      {pair_id, statement, true_label}
    Each pair contributes one CERTAIN and one UNCERTAIN sample.
    """
    samples = []
    for p in pairs:
        samples.append({
            "pair_id":   p["pair_id"],
            "statement": p["certain"],
            "true_label": "NO",
        })
        samples.append({
            "pair_id":   p["pair_id"],
            "statement": p["uncertain"],
            "true_label": "YES",
        })
    return samples


def parse_response(text: str) -> str | None:
    """
    Extract YES or NO from the model's raw response text.
    Returns None if neither label is found.
    Matches by prefix of the stripped response to avoid false positives
    (e.g. 'NO' appearing inside 'KNOW' or 'UNKNOWN').
    """
    upper = text.strip().upper()
    if upper.startswith("YES"):
        return "YES"
    if upper.startswith("NO"):
        return "NO"
    return None


@torch.inference_mode()
def run_inference(
    model,
    tokenizer,
    test_samples: list[dict],
    demo_certain: str,
    demo_uncertain: str,
    include_definition: bool = False,
    max_new_tokens: int = 10,
) -> list[dict]:
    """
    Run greedy generation on every sample. Returns a list of result dicts:
      {pair_id, statement, true_label, pred_label, response}
    pred_label is None when the response cannot be parsed.
    """
    results = []

    for i, sample in enumerate(test_samples):
        msgs      = build_messages(demo_certain, demo_uncertain, sample["statement"],
                                   include_definition=include_definition)
        ids       = apply_template(msgs, tokenizer, tokenize=True)
        input_ids = torch.tensor([ids], device=model.device)

        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        # Decode only the newly generated tokens
        new_tokens = out[0, input_ids.shape[1]:]
        response   = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        pred_label = parse_response(response)

        results.append({
            "pair_id":    sample["pair_id"],
            "statement":  sample["statement"],
            "true_label": sample["true_label"],
            "pred_label": pred_label,
            "response":   response,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(test_samples):
            parseable      = [r for r in results if r["pred_label"] is not None]
            correct_so_far = sum(r["true_label"] == r["pred_label"] for r in parseable)
            denom          = len(parseable) or 1
            print(f"  [{i+1:>3}/{len(test_samples)}]  running acc: {correct_so_far}/{denom} = {correct_so_far/denom:.3f}"
                  f"  (unparseable: {len(results) - len(parseable)})")

    return results


def print_report(results: list[dict]):
    unparseable = [r for r in results if r["pred_label"] is None]
    parseable   = [r for r in results if r["pred_label"] is not None]

    print(f"\nTotal samples : {len(results)}")
    print(f"Unparseable   : {len(unparseable)}"
          + (f"  {[r['response'] for r in unparseable]}" if unparseable else ""))

    if not parseable:
        print("No parseable responses — cannot compute accuracy.")
        return

    correct = sum(r["true_label"] == r["pred_label"] for r in parseable)
    total   = len(parseable)
    print(f"Accuracy      : {correct}/{total} = {correct/total:.4f}")

    for label in ("YES", "NO"):
        subset    = [r for r in parseable if r["true_label"] == label]
        n_correct = sum(r["pred_label"] == label for r in subset)
        print(f"  {label:>10}:  {n_correct}/{len(subset)} = {n_correct/len(subset):.4f}")

    # Confusion matrix (rows = true, cols = predicted)
    classes = ["YES", "NO"]
    idx = {c: i for i, c in enumerate(classes)}
    cm  = [[0, 0], [0, 0]]
    for r in parseable:
        cm[idx[r["true_label"]]][idx[r["pred_label"]]] += 1

    col_w = 12
    print(f"\nConfusion matrix (rows=true, cols=predicted):")
    print(f"{'':>{col_w}}" + "".join(f"{c:>{col_w}}" for c in classes))
    for i, row_label in enumerate(classes):
        print(f"{row_label:>{col_w}}" + "".join(f"{cm[i][j]:>{col_w}}" for j in range(2)))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model on contrastive econ uncertainty pairs"
    )
    parser.add_argument(
        "--model", default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--data",
        default="neural-concepts-team-E/synthetic_data/econ_uncertainty_contrastive_flat.json",
    )
    parser.add_argument(
        "--demo_pair_id", type=int, default=1,
        help="pair_id whose certain/uncertain statements serve as the 2-shot demo",
    )
    parser.add_argument(
        "--def", dest="include_def", action="store_true", default=False,
        help="Prepend the second-moment definition of uncertainty to the prompt",
    )
    parser.add_argument(
        "--out", default=None,
        help="Optional path to save per-sample results as JSON",
    )
    args = parser.parse_args()

    # ── Load pairs ────────────────────────────────────────────────────────────
    pairs = load_pairs(args.data)
    print(f"Loaded {len(pairs)} contrastive pairs from {args.data}")

    demo_certain, demo_uncertain, test_pairs = pick_demo_pair(pairs, args.demo_pair_id)
    test_samples = pairs_to_samples(test_pairs)

    print(f"Demo pair_id  : {args.demo_pair_id}")
    print(f"Demo CERTAIN  : {demo_certain[:80]}...")
    print(f"Demo UNCERTAIN: {demo_uncertain[:80]}...")
    print(f"Test pairs    : {len(test_pairs)}  →  {len(test_samples)} samples")

    # ── Load model ────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading {args.model} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model     = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    model.eval()

    print(f"Include definition: {args.include_def}")

    # Print sample prompt for verification
    sample_msgs = build_messages(demo_certain, demo_uncertain, test_samples[0]["statement"],
                                 include_definition=args.include_def)
    sample_str  = apply_template(sample_msgs, tokenizer, tokenize=False)
    print(f"\n── Sample prompt ──\n{sample_str}\n──────────────────\n")

    # ── Run inference ─────────────────────────────────────────────────────────
    print("Running inference...")
    results = run_inference(
        model, tokenizer, test_samples,
        demo_certain, demo_uncertain,
        include_definition=args.include_def,
    )

    # ── Report ────────────────────────────────────────────────────────────────
    print_report(results)

    # ── Save ──────────────────────────────────────────────────────────────────
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved per-sample results → {args.out}")


if __name__ == "__main__":
    main()
