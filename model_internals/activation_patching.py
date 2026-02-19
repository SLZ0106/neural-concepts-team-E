"""
Activation patching on economic uncertainty classification.

Prompt structure (2-shot):
  Determine whether the following economic statement contains uncertainty.
  [DEFINITION OF UNCERTAINTY ... inserted here when include_definition=True]
  If the statement contains uncertainty, respond with YES. Otherwise respond with NO.
  Statement: {certain_demo}
  Label: NO

  Statement: {uncertain_demo}
  Label: YES

  Statement: {test_stmt}
  Label:

Patching logic:
  - clean   run: test_stmt = certain statement (no uncertainty)
  - corrupt run: test_stmt = uncertain statement (high uncertainty)
  - For each layer, patch the residual stream at the test-statement's
    final period token from clean → corrupt.
  - Record logit(YES) − logit(NO) after each patch.
"""

import json
import argparse
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
from nnsight import LanguageModel


# ── Prompt ────────────────────────────────────────────────────────────────────

UNCERTAINTY_DEFINITION = (
    "DEFINITION OF UNCERTAINTY (Second-Moment):\n\n"
    "Uncertainty measures the VARIANCE or SPREAD of possible outcomes, "
    "not the expected value of outcomes."
)

UNCERTAINTY_DEFINITION_V2 = (
    "DEFINITION OF UNCERTAINTY:\n\n"
    "A statement contains uncertainty if it describes an economic outcome as "
    "CAUSALLY OPEN: the result depends on multiple unknown, competing, or "
    "uncontrollable forces, no single mechanism governs it, and even small "
    "disturbances can materially redirect the outcome. Reliable prediction is "
    "not possible because the driving forces are numerous, non-linear, or "
    "unquantifiable.\n\n"
    "A statement does NOT contain uncertainty if it describes an outcome as "
    "CAUSALLY CLOSED: a single predetermined mechanism has already been "
    "activated and will produce the result regardless of external conditions. "
    "The outcome is fixed and insulated from surrounding circumstances; "
    "external factors are explicitly rendered irrelevant."
)

USER_CONTENT_TEMPLATE = (
    "Determine whether the following economic statement contains uncertainty.\n"
    "{definition_block}"
    "Respond with exactly one word: YES or NO.\n"
    "Statement: {certain_demo}\n"
    "Label: NO\n\n"
    "Statement: {uncertain_demo}\n"
    "Label: YES\n\n"
    "Statement: {test_stmt}\n"
    "Label:"
)


def build_messages(
    certain_demo: str,
    uncertain_demo: str,
    test_stmt: str,
    include_definition: bool = False,
) -> list[dict]:
    """Return a chat-style messages list with the full prompt as the user turn.

    When include_definition=True the second-moment definition of uncertainty
    is inserted after the instruction line.
    """
    definition_block = UNCERTAINTY_DEFINITION_V2 + "\n\n" if include_definition else "\n"
    content = USER_CONTENT_TEMPLATE.format(
        definition_block=definition_block,
        certain_demo=certain_demo,
        uncertain_demo=uncertain_demo,
        test_stmt=test_stmt,
    )
    return [{"role": "user", "content": content}]


def apply_template(messages: list[dict], tokenizer, tokenize: bool = True):
    """
    Apply the model's chat template.
    tokenize=True  → returns a list[int] of token IDs.
    tokenize=False → returns the formatted string (for inspection).

    We always call apply_chat_template with tokenize=False to get the
    formatted string, then tokenize explicitly with add_special_tokens=False
    (the template string already contains all special tokens as text).
    """
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not tokenize:
        return formatted
    return tokenizer.encode(formatted, add_special_tokens=False)


# ── Token utilities ───────────────────────────────────────────────────────────

def get_label_token_ids(tokenizer):
    """
    Token IDs for 'YES' (uncertain) and 'NO' (certain), without a leading
    space. The generation prompt ends with '\\n\\n', so the model generates
    at the start of a new line where the correct tokens are the no-space
    variants, not ' YES'/' NO' which are different BPE tokens.
    """
    yes_id = tokenizer("YES", add_special_tokens=False).input_ids[0]
    no_id  = tokenizer("NO",  add_special_tokens=False).input_ids[0]
    return no_id, yes_id


def find_last_period_pos(token_ids: list[int], tokenizer) -> int:
    """
    Return the token index of the final '.' in a token ID sequence.
    Operates directly on the IDs fed to model.trace(), so there is
    no tokenization mismatch with the chat template.
    """
    for i in range(len(token_ids) - 1, -1, -1):
        if "." in tokenizer.decode([token_ids[i]]):
            return i
    raise ValueError("No '.' found in token sequence.")


# ── Pair loading ──────────────────────────────────────────────────────────────

def load_pairs(path: str):
    """
    Load flat JSON and return list of dicts:
      {"certain": <statement str>, "uncertain": <statement str>}
    Pairs where original=no, contrastive=high are straightforward;
    pairs where original=high, contrastive=no are flipped so that
    'certain' always = no-uncertainty and 'uncertain' always = high.
    """
    with open(path) as f:
        flat = json.load(f)

    pair_map = defaultdict(dict)
    for item in flat:
        pair_map[item["pair_id"]][item["role"]] = item

    pairs = []
    for pid, roles in pair_map.items():
        if "original" not in roles or "contrastive" not in roles:
            continue
        orig = roles["original"]
        cont = roles["contrastive"]
        # Identify which is certain vs uncertain by label
        by_label = {orig["uncertainty"]: orig, cont["uncertainty"]: cont}
        if "no" not in by_label or "high" not in by_label:
            continue
        pairs.append({
            "pair_id":   pid,
            "certain":   by_label["no"]["statement"],
            "uncertain": by_label["high"]["statement"],
        })
    return pairs


# ── Core patching function ────────────────────────────────────────────────────

def run_patching_for_pair(
    pair: dict,
    demo_certain: str,
    demo_uncertain: str,
    model,
    tokenizer,
    n_layers: int,
    no_id: int,
    yes_id: int,
    include_definition: bool = False,
) -> tuple[np.ndarray, float, float]:
    """
    Returns:
      patched_diffs : (n_layers,) logit(YES)-logit(NO) after patching layer l
      clean_baseline: logit diff on clean prompt (no patching)
      corrupt_baseline: logit diff on corrupt prompt (no patching)
    """
    clean_msgs   = build_messages(demo_certain, demo_uncertain, pair["uncertain"],
                                  include_definition=include_definition)
    corrupt_msgs = build_messages(demo_certain, demo_uncertain, pair["certain"],
                                  include_definition=include_definition)

    # Tokenize via chat template — returns list[int]; wrap in tensor for model.trace()
    clean_ids   = apply_template(clean_msgs,   tokenizer, tokenize=True)
    corrupt_ids = apply_template(corrupt_msgs, tokenizer, tokenize=True)

    clean_input   = torch.tensor([clean_ids])
    corrupt_input = torch.tensor([corrupt_ids])

    clean_period_pos   = find_last_period_pos(clean_ids,   tokenizer)
    corrupt_period_pos = find_last_period_pos(corrupt_ids, tokenizer)

    # ── 1. Collect clean hidden states at period position, all layers ─────────
    clean_acts = {}
    with model.trace(clean_input):
        for l in range(n_layers):
            clean_acts[l] = model.model.layers[l].output[0][clean_period_pos, :].save()
        clean_logits = model.output.logits[0, -1, :].save()

    clean_baseline = (
        clean_logits[yes_id] - clean_logits[no_id]
    ).item()

    # ── 2. Corrupt baseline (no patching) ─────────────────────────────────────
    with model.trace(corrupt_input):
        corrupt_logits = model.output.logits[0, -1, :].save()

    corrupt_baseline = (
        corrupt_logits[yes_id] - corrupt_logits[no_id]
    ).item()

    # ── 3. Patch each layer: inject clean act into corrupt run ────────────────
    patched_diffs = []
    for l in range(n_layers):
        with model.trace(corrupt_input):
            model.model.layers[l].output[0][corrupt_period_pos, :] = (
                clean_acts[l]
            )
            patched_logits = model.output.logits[0, -1, :].save()

        diff = (
            patched_logits[yes_id] - patched_logits[no_id]
        ).item()
        patched_diffs.append(diff)

    return np.array(patched_diffs), clean_baseline, corrupt_baseline


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Activation patching on economic uncertainty")
    parser.add_argument("--model",   default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HuggingFace model name or local path")
    parser.add_argument("--pairs",   default="neural-concepts-team-E/synthetic_data/econ_uncertainty_contrastive_flat.json")
    parser.add_argument("--n_pairs", type=int, default=10,
                        help="Number of contrastive pairs to average over (excl. demo pair)")
    parser.add_argument("--out",     default="patching_results.png")
    parser.add_argument(
        "--def", dest="include_def", action="store_true", default=False,
        help="Prepend the uncertainty definition to the prompt",
    )
    args = parser.parse_args()

    # ── Load pairs ────────────────────────────────────────────────────────────
    pairs = load_pairs(args.pairs)
    print(f"Loaded {len(pairs)} contrastive pairs.")

    # First pair used as fixed 2-shot demo; rest as test pairs
    demo_pair      = pairs[0]
    demo_certain   = demo_pair["certain"]
    demo_uncertain = demo_pair["uncertain"]
    test_pairs     = pairs[1 : 1 + args.n_pairs]
    print(f"Demo pair: {demo_pair['pair_id']}  |  Test pairs: {len(test_pairs)}")

    # ── Load model ────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device} ...")
    model = LanguageModel(args.model, device_map=device, dispatch=True)
    tokenizer = model.tokenizer
    n_layers  = model.config.num_hidden_layers
    print(f"Model loaded. Layers: {n_layers}")

    no_id, yes_id = get_label_token_ids(tokenizer)
    print(f"Label token IDs — NO: {no_id}  YES: {yes_id}")
    print(f"  ({tokenizer.decode([no_id])!r} / {tokenizer.decode([yes_id])!r})")

    print(f"Include definition: {args.include_def}")

    # Print a sample chat-formatted prompt for sanity check
    sample_msgs = build_messages(demo_certain, demo_uncertain, test_pairs[0]["certain"],
                                 include_definition=args.include_def)
    sample_str  = apply_template(sample_msgs, tokenizer, tokenize=False)
    print(f"\n── Sample chat-formatted prompt ──")
    print(sample_str[:])
    print("────────────────────────────────────────────────────\n")

    # ── Run patching ──────────────────────────────────────────────────────────
    all_patched   = []
    clean_baselines   = []
    corrupt_baselines = []

    for i, pair in enumerate(test_pairs):
        print(f"\nPair {i+1}/{len(test_pairs)}  (pair_id={pair['pair_id']})")
        diffs, cb, ub = run_patching_for_pair(
            pair, demo_certain, demo_uncertain,
            model, tokenizer, n_layers, no_id, yes_id,
            include_definition=args.include_def,
        )
        all_patched.append(diffs)
        clean_baselines.append(cb)
        corrupt_baselines.append(ub)
        print(f"  clean baseline: {cb:.3f}  |  corrupt baseline: {ub:.3f}")
        print(f"  patched range:  [{diffs.min():.3f}, {diffs.max():.3f}]")

    mean_patched      = np.stack(all_patched).mean(axis=0)   # (n_layers,)
    mean_clean_base   = np.mean(clean_baselines)
    mean_corrupt_base = np.mean(corrupt_baselines)

    # ── Plot ──────────────────────────────────────────────────────────────────
    layers = np.arange(n_layers)
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(layers, mean_patched, marker="o", color="steelblue",
            label="Patched (clean → corrupt at period pos)")
    ax.axhline(mean_clean_base,   color="green",  linestyle="--", linewidth=1.5,
               label=f"Clean baseline ({mean_clean_base:.2f})")
    ax.axhline(mean_corrupt_base, color="red",    linestyle="--", linewidth=1.5,
               label=f"Corrupt baseline ({mean_corrupt_base:.2f})")

    ax.set_xlabel("Layer")
    ax.set_ylabel("logit(YES) − logit(NO)")
    ax.set_title(
        f"Activation Patching at Test-Statement Period Token\n"
        f"(averaged over {len(test_pairs)} pairs, model={args.model.split('/')[-1]})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved plot → {args.out}")

    # ── Save raw arrays ───────────────────────────────────────────────────────
    np.save("patching_mean_diffs.npy", mean_patched)
    np.save("patching_all_diffs.npy",  np.stack(all_patched))
    print("Saved patching_mean_diffs.npy and patching_all_diffs.npy")

    print(f"\nSummary:")
    print(f"  clean baseline (certain test):   {mean_clean_base:.3f}")
    print(f"  corrupt baseline (uncertain test): {mean_corrupt_base:.3f}")
    print(f"  max patched diff:  layer {mean_patched.argmax()}  ({mean_patched.max():.3f})")
    print(f"  min patched diff:  layer {mean_patched.argmin()}  ({mean_patched.min():.3f})")


if __name__ == "__main__":
    main()
