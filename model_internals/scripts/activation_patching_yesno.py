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
    """Return a chat-style messages list for the TEMPLATED dataset.

    Uses a single certain/uncertain demo pair drawn from the dataset.
    When include_definition=True the V2 definition is inserted.
    """
    definition_block = UNCERTAINTY_DEFINITION_V2 + "\n\n" if include_definition else "\n"
    content = USER_CONTENT_TEMPLATE.format(
        definition_block=definition_block,
        certain_demo=certain_demo,
        uncertain_demo=uncertain_demo,
        test_stmt=test_stmt,
    )
    return [{"role": "user", "content": content}]


# ── Prompt (synthetic / API dataset) ─────────────────────────────────────────
# Matches neural-concepts-team-E/synthetic_data/generate_synthetic_pairs.py

SYNTHETIC_UNCERTAINTY_DEFINITION = (
    "Economic uncertainty is the variance or spread of future business conditions "
    "(e.g. revenue, demand, or the broader economy). High uncertainty means business "
    "conditions cannot be predicted with reasonable confidence. Low uncertainty means "
    "the direction or approximate size of future business conditions are clear.\n\n"
    "KEY PRINCIPLE - UNCERTAINTY IS NOT SENTIMENT: "
    "Sentiment reflects the expected value of future business conditions. "
    "Uncertainty reflects the variance around that expected value.\n"
    '- "We expect a 10% drop in sales due to tariffs" -> negative sentiment, low uncertainty\n'
    '- "Sales could drop 5% or 30% depending on how tariffs develop" -> negative sentiment, high uncertainty'
)

SYNTHETIC_FEW_SHOT_DEMOS = [
    {
        "high": (
            "Point, we do have the business plan or targets for 2026 "
            "prepared last year. We are currently on the process of evaluating "
            "it because of the geopolitical issues that we are encountering and "
            "we are expecting project delays due to logistical supply uncertainties, "
            "which we have no control."
        ),
        "no": (
            "We have evaluated our business plan or targets for 2026 to reflect "
            "the confirmed impact of current geopolitical issues. We are now guiding "
            "to a six-month delay across all major projects due to the structural "
            "breakdown in logistical supply chains."
        ),
    },
    {
        "high": (
            "Having said that, the uncertain duration and future potential impacts "
            "of the government shutdown creates a lack of clear visibility into our cash "
            "forecast for the remainder of the year. We are taking prudent actions to conserve "
            "cash and liquidity. If a resolution can be reached in the near term, we would "
            "expect to be able to achieve the forecast that I just discussed. However, in the "
            "event of a protracted shutdown, it is unclear how and when our cash flow will be "
            "impacted despite our careful efforts to diligently manage cash."
        ),
        "no": (
            "In terms of impact from shut down, no major impact from shutdown and "
            "that was reflected just due to the strong year-over-year growth in us "
            "exceeding the top end of our guidance range on sales, but then also the "
            "$3.3 billion of cash in Q4, creating that 26%."
        ),
    },
    # {
    #     "high": (
    #         "So as we enter -- potentially enter a period where inflation is lower or "
    #         "higher. We'll manage the commodities as they come through. We'll focus on the "
    #         "lowest prices we can focus on. We had 6,200 rollbacks in Walmart U.S. this quarter, "
    #         "up about 23% from a year ago. So we'll just continue to focus on low prices."
    #     ),
    #     "no": (
    #         "We worked hard to mitigate grocery inflation as tariff-related costs "
    #         "lifted prices across many categories. We're seeing share gains in "
    #         "GM and in fashion. We've had several quarters in a row of mid-single-digit "
    #         "sales growth."
    #     ),
    # },
    # {
    #     "high": (
    #         "Like many companies in the ad-tech space, we are dealing with the challenges "
    #         "of a dynamically changing AI landscape. We are working to expand our best-in-class "
    #         "verticalized experience for students focused on improving their outcomes. "
    #         "However, it will take time to adjust to the new opportunity and see the benefits "
    #         "in our business results."
    #     ),
    #     "no": (
    #         "These factors, the speed and scale of Google AIOs rollout and student adoption "
    #         "of generative AI products have negatively impacted our industry and our business. "
    #         "We have seen a sharp decline in overall traffic and therefore, a decline in our "
    #         "outlook on revenue. Global nonsubscriber traffic to Chegg declined year-over-year, "
    #         "8% in Q2, 19% in Q3 and we exited Q3 with trends looking even more unfavorable and "
    #         "negative 37% year-over-year for the month of October. We've taken all this into "
    #         "account and consequently, we do not expect to meet our 2025 goals of 30% adjusted "
    #         "EBITDA margin and $100 million in free cash flow."
    #     ),
    # },
]

SYNTHETIC_USER_CONTENT_TEMPLATE = (
    "Determine whether the following economic statement contains uncertainty.\n\n"
    "Definition of Economic Uncertainty:\n"
    "{definition}\n\n"
    "Respond with exactly one word: YES or NO.\n\n"
    "{demos}"
    "Statement: {test_stmt}\n"
    "Label:"
)


def build_messages_synthetic(test_stmt: str) -> list[dict]:
    """Return a chat-style messages list for the SYNTHETIC (API) dataset.

    Always includes the uncertainty definition and all 4 few-shot demos
    from generate_synthetic_pairs.py.
    """
    demos_block = ""
    for demo in SYNTHETIC_FEW_SHOT_DEMOS:
        demos_block += f"Statement: {demo['no']}\nLabel: NO\n\n"
        demos_block += f"Statement: {demo['high']}\nLabel: YES\n\n"

    content = SYNTHETIC_USER_CONTENT_TEMPLATE.format(
        definition=SYNTHETIC_UNCERTAINTY_DEFINITION,
        demos=demos_block,
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


def compute_metric(logits, yes_id: int, no_id: int, metric: str) -> float:
    """Compute the patching metric from final-position logits.

    Args:
        logits: full vocabulary logits at the last position (already .save()'d)
        yes_id: token ID for YES
        no_id:  token ID for NO
        metric: 'logit_diff' or 'prob_diff'

    Returns:
        scalar value: logit(YES)-logit(NO) or P(YES)-P(NO)
    """
    if metric == "logit_diff":
        return (logits[yes_id] - logits[no_id]).item()
    elif metric == "prob_diff":
        probs = torch.softmax(logits.float(), dim=-1)
        return (probs[yes_id] - probs[no_id]).item()
    else:
        raise ValueError(f"Unknown metric: {metric}")


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
    Load contrastive pairs JSON — list of {"pair_id", "high", "no", ...}.
    Returns list of dicts: {"pair_id": int, "certain": str, "uncertain": str}
    """
    with open(path) as f:
        data = json.load(f)
    return [
        {
            "pair_id":   item["pair_id"],
            "certain":   item["no"],
            "uncertain": item["high"],
        }
        for item in data
    ]


# ── Core patching function ────────────────────────────────────────────────────

def run_patching_for_pair(
    pair: dict,
    model,
    tokenizer,
    n_layers: int,
    no_id: int,
    yes_id: int,
    dataset: str = "templated",
    demo_certain: str = None,
    demo_uncertain: str = None,
    include_definition: bool = False,
    metric: str = "logit_diff",
) -> tuple[np.ndarray, float, float, list[str]]:
    """
    Patch at every token position from the period to the last token.

    Returns:
      patched_diffs  : (n_layers, n_positions) metric value per (layer, position)
      clean_baseline : metric value on clean prompt (no patching)
      corrupt_baseline: metric value on corrupt prompt (no patching)
      token_labels   : decoded token strings for each patched position
    """
    if dataset == "synthetic":
        clean_msgs   = build_messages_synthetic(pair["uncertain"])
        corrupt_msgs = build_messages_synthetic(pair["certain"])
    else:
        clean_msgs   = build_messages(demo_certain, demo_uncertain, pair["uncertain"],
                                      include_definition=include_definition)
        corrupt_msgs = build_messages(demo_certain, demo_uncertain, pair["certain"],
                                      include_definition=include_definition)

    # # Print the actual prompts used for patching
    # clean_str   = apply_template(clean_msgs,   tokenizer, tokenize=False)
    # corrupt_str = apply_template(corrupt_msgs, tokenizer, tokenize=False)
    # print(f"\n{'='*60}")
    # print(f"CLEAN prompt (test_stmt = uncertain):")
    # print(f"{'='*60}")
    # print(clean_str)
    # print(f"\n{'='*60}")
    # print(f"CORRUPT prompt (test_stmt = certain):")
    # print(f"{'='*60}")
    # print(corrupt_str)
    # print(f"{'='*60}\n")

    # Tokenize via chat template — returns list[int]; wrap in tensor for model.trace()
    clean_ids   = apply_template(clean_msgs,   tokenizer, tokenize=True)
    corrupt_ids = apply_template(corrupt_msgs, tokenizer, tokenize=True)

    clean_input   = torch.tensor([clean_ids])
    corrupt_input = torch.tensor([corrupt_ids])

    clean_period_pos   = find_last_period_pos(clean_ids,   tokenizer)
    corrupt_period_pos = find_last_period_pos(corrupt_ids, tokenizer)

    # Number of positions from period to last token (inclusive on both ends)
    clean_n_suffix   = len(clean_ids)   - clean_period_pos
    corrupt_n_suffix = len(corrupt_ids) - corrupt_period_pos
    n_positions = min(clean_n_suffix, corrupt_n_suffix)

    # Token labels for the positions (from the corrupt prompt for display)
    token_labels = [
        repr(tokenizer.decode([corrupt_ids[corrupt_period_pos + offset]]))
        for offset in range(n_positions)
    ]
    print(f"  Patching positions: {n_positions} tokens from period to end")
    print(f"  Token labels: {token_labels}")

    # ── 1. Collect clean hidden states at all patch positions, all layers ─────
    clean_acts = {}   # clean_acts[(layer, offset)] = saved activation
    with model.trace(clean_input):
        for l in range(n_layers):
            for offset in range(n_positions):
                pos = clean_period_pos + offset
                clean_acts[(l, offset)] = (
                    model.model.layers[l].output[0][pos, :].save()
                )
        clean_logits = model.output.logits[0, -1, :].save()

    clean_baseline = compute_metric(clean_logits, yes_id, no_id, metric)

    # ── 2. Corrupt baseline (no patching) ─────────────────────────────────────
    with model.trace(corrupt_input):
        corrupt_logits = model.output.logits[0, -1, :].save()

    corrupt_baseline = compute_metric(corrupt_logits, yes_id, no_id, metric)

    # ── 3. Patch each (layer, position): inject clean act into corrupt run ────
    patched_diffs = np.zeros((n_layers, n_positions))
    for l in range(n_layers):
        for offset in range(n_positions):
            corrupt_pos = corrupt_period_pos + offset
            with model.trace(corrupt_input):
                model.model.layers[l].output[0][corrupt_pos, :] = (
                    clean_acts[(l, offset)]
                )
                patched_logits = model.output.logits[0, -1, :].save()

            patched_diffs[l, offset] = compute_metric(
                patched_logits, yes_id, no_id, metric
            )

    return patched_diffs, clean_baseline, corrupt_baseline, token_labels


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Activation patching on economic uncertainty")
    parser.add_argument("--model",   default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HuggingFace model name or local path")
    parser.add_argument("--pairs",   default="neural-concepts-team-E/synthetic_data/API/synthetic_pairs_200.json")
    parser.add_argument("--n_pairs", type=int, default=10,
                        help="Number of contrastive pairs to average over (excl. demo pair for templated)")
    parser.add_argument("--out",     default="patching_results.png")
    parser.add_argument(
        "--def", dest="include_def", action="store_true", default=False,
        help="Prepend the uncertainty definition to the prompt (templated only)",
    )
    parser.add_argument(
        "--dataset", choices=["templated", "synthetic"], default="templated",
        help="templated: original prompt with 1 demo pair from dataset. "
             "synthetic: prompt with definition + 4 fixed few-shot demos from generate_synthetic_pairs.py",
    )
    parser.add_argument(
        "--metric", choices=["logit_diff", "prob_diff"], default="logit_diff",
        help="logit_diff: logit(YES)-logit(NO). "
             "prob_diff: P(YES)-P(NO) after softmax over full vocabulary.",
    )
    args = parser.parse_args()

    # ── Load pairs ────────────────────────────────────────────────────────────
    pairs = load_pairs(args.pairs)
    print(f"Loaded {len(pairs)} contrastive pairs.")
    print(f"Dataset mode: {args.dataset}")

    if args.dataset == "templated":
        # First pair used as fixed 2-shot demo; rest as test pairs
        demo_pair      = pairs[0]
        demo_certain   = demo_pair["certain"]
        demo_uncertain = demo_pair["uncertain"]
        test_pairs     = pairs[1 : 1 + args.n_pairs]
        print(f"Demo pair: {demo_pair['pair_id']}  |  Test pairs: {len(test_pairs)}")
    else:
        # Synthetic: demos are fixed in the prompt; all pairs available for patching
        demo_certain   = None
        demo_uncertain = None
        test_pairs     = pairs[:args.n_pairs]
        print(f"Using 4 fixed few-shot demos  |  Test pairs: {len(test_pairs)}")

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
    if args.dataset == "synthetic":
        sample_msgs = build_messages_synthetic(test_pairs[0]["certain"])
    else:
        sample_msgs = build_messages(demo_certain, demo_uncertain, test_pairs[0]["certain"],
                                     include_definition=args.include_def)
    sample_str  = apply_template(sample_msgs, tokenizer, tokenize=False)
    print(f"\n── Sample chat-formatted prompt ──")
    print(sample_str[:])
    print("────────────────────────────────────────────────────\n")

    # ── Run patching ──────────────────────────────────────────────────────────
    all_patched   = []       # list of (n_layers, n_positions) arrays
    clean_baselines   = []
    corrupt_baselines = []
    first_token_labels = None

    for i, pair in enumerate(test_pairs):
        print(f"\nPair {i+1}/{len(test_pairs)}  (pair_id={pair['pair_id']})")
        diffs, cb, ub, tok_labels = run_patching_for_pair(
            pair, model, tokenizer, n_layers, no_id, yes_id,
            dataset=args.dataset,
            demo_certain=demo_certain,
            demo_uncertain=demo_uncertain,
            include_definition=args.include_def,
            metric=args.metric,
        )
        all_patched.append(diffs)
        clean_baselines.append(cb)
        corrupt_baselines.append(ub)
        if first_token_labels is None:
            first_token_labels = tok_labels
        print(f"  clean baseline: {cb:.3f}  |  corrupt baseline: {ub:.3f}")
        print(f"  patched range:  [{diffs.min():.3f}, {diffs.max():.3f}]")

    # Truncate all pairs to the same number of positions (in case they differ)
    min_positions = min(d.shape[1] for d in all_patched)
    all_patched_trunc = [d[:, :min_positions] for d in all_patched]
    mean_patched = np.stack(all_patched_trunc).mean(axis=0)  # (n_layers, n_positions)
    mean_clean_base   = np.mean(clean_baselines)
    mean_corrupt_base = np.mean(corrupt_baselines)
    token_labels = first_token_labels[:min_positions]

    # Period-token column is index 0 (the first patched position)
    mean_patched_period = mean_patched[:, 0]  # (n_layers,)

    # ── Plot 1: Period-token line plot (original) ─────────────────────────────
    metric_label = "logit(YES) − logit(NO)" if args.metric == "logit_diff" else "P(YES) − P(NO)"
    layers = np.arange(n_layers)
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(layers, mean_patched_period, marker="o", color="steelblue",
            label="Patched (clean → corrupt at period pos)")
    ax.axhline(mean_clean_base,   color="green",  linestyle="--", linewidth=1.5,
               label=f"Clean baseline ({mean_clean_base:.2f})")
    ax.axhline(mean_corrupt_base, color="red",    linestyle="--", linewidth=1.5,
               label=f"Corrupt baseline ({mean_corrupt_base:.2f})")

    ax.set_xlabel("Layer")
    ax.set_ylabel(metric_label)
    ax.set_title(
        f"Activation Patching at Test-Statement Period Token ({args.metric})\n"
        f"(averaged over {len(test_pairs)} pairs, model={args.model.split('/')[-1]})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved period-token plot → {args.out}")
    plt.close(fig)

    # ── Plot 2: Heatmap (layers × token positions) ───────────────────────────
    out_stem = args.out.rsplit(".", 1)[0]
    heatmap_path = f"{out_stem}_heatmap.png"

    fig, ax = plt.subplots(figsize=(max(8, min_positions * 1.5), 10))
    im = ax.imshow(mean_patched, aspect="auto", cmap="RdBu_r",
                   origin="lower", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, label=metric_label)

    ax.set_xlabel("Token position (period → end)")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(min_positions))
    ax.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(0, n_layers, max(1, n_layers // 16)))
    ax.set_title(
        f"Activation Patching Heatmap: Layer × Token Position ({args.metric})\n"
        f"(averaged over {len(test_pairs)} pairs, model={args.model.split('/')[-1]})"
    )
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=150)
    print(f"Saved heatmap plot → {heatmap_path}")
    plt.close(fig)

    print(f"\nSummary:")
    print(f"  clean baseline (certain test):   {mean_clean_base:.3f}")
    print(f"  corrupt baseline (uncertain test): {mean_corrupt_base:.3f}")
    print(f"  max patched diff:  layer {mean_patched.argmax() // min_positions}, "
          f"pos {mean_patched.argmax() % min_positions}  ({mean_patched.max():.3f})")
    print(f"  min patched diff:  layer {mean_patched.argmin() // min_positions}, "
          f"pos {mean_patched.argmin() % min_positions}  ({mean_patched.min():.3f})")


if __name__ == "__main__":
    main()
