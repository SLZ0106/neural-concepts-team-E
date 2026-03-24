#!/usr/bin/env python3
"""
Generate contrastive pairs of economic statements with high/no uncertainty
using the Claude API. Outputs 100 pairs in the flat JSON format expected
by activation_patching.py and evaluate_model.py.
"""

import json
import argparse
import time
import anthropic

# ── Uncertainty definition block ──────────────────────────────────────────────
UNCERTAINTY_DEFINITION = """\
Economic uncertainty is the variance or spread of future business conditions \
(e.g. revenue, demand, or the broader economy). High uncertainty means business \
conditions cannot be predicted with reasonable confidence. Low uncertainty means \
the direction or approximate size of future business conditions are clear.\
\n
KEY PRINCIPLE - UNCERTAINTY IS NOT SENTIMENT: \
Sentiment reflects the expected value of future business conditions. \
Uncertainty reflects the variance around that expected value. \
- "We expect a 10% drop in sales due to tariffs" -> negative sentiment, low uncertainty \
- "Sales could drop 5% or 30% depending on how tariffs develop" -> negative sentiment, high uncertainty \
"""

BACKUP = """\
Economic uncertainty refers to the degree to which the future state of an \
economic variable, outcome, or policy is unpredictable or unknown. A statement \
exhibits HIGH uncertainty when the speaker explicitly acknowledges that \
outcomes are unclear, contingent on unknown factors, subject to revision, or \
fundamentally hard to forecast. A statement exhibits NO (low) uncertainty when \
the speaker conveys confidence, definiteness, or determinism about the \
economic outcome — even if the underlying reality may be complex.\
"""

# ── Five-shot example pairs ───────────────────────────────────────────────────
FEW_SHOT_PAIRS = [
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
        "topic": "Geopolitical Risk"
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
        "topic": "Government Shutdown"
    },
    {
        "high": (
            "So as we enter -- potentially enter a period where inflation is lower or "
            "higher. We'll manage the commodities as they come through. We'll focus on the "
            "lowest prices we can focus on. We had 6,200 rollbacks in Walmart U.S. this quarter, "
            "up about 23% from a year ago. So we'll just continue to focus on low prices."
        ),
        "no": (
            "We worked hard to mitigate grocery inflation as tariff-related costs "
            "lifted prices across many categories. We're seeing share gains in "
            "GM and in fashion. We've had several quarters in a row of mid-single-digit "
            "sales growth."
        ),
        "topic": "Inflation"
    },
    {
        "high": (
            "Like many companies in the ad-tech space, we are dealing with the challenges "
            "of a dynamically changing AI landscape. We are working to expand our best-in-class "
            "verticalized experience for students focused on improving their outcomes. "
            "However, it will take time to adjust to the new opportunity and see the benefits "
            "in our business results."
        ),
        "no": (
            "These factors, the speed and scale of Google AIOs rollout and student adoption "
            "of generative AI products have negatively impacted our industry and our business. "
            "We have seen a sharp decline in overall traffic and therefore, a decline in our "
            "outlook on revenue. Global nonsubscriber traffic to Chegg declined year-over-year, "
            "8% in Q2, 19% in Q3 and we exited Q3 with trends looking even more unfavorable and "
            "negative 37% year-over-year for the month of October. We've taken all this into "
            "account and consequently, we do not expect to meet our 2025 goals of 30% adjusted "
            "EBITDA margin and $100 million in free cash flow."
        ),
        "topic": "AI Acceleration"
    },
]


def build_generation_prompt(batch_start: int, batch_size: int) -> str:
    """Build the prompt that asks Claude to generate contrastive pairs."""
    # Format the few-shot examples
    examples_block = ""
    for i, pair in enumerate(FEW_SHOT_PAIRS, 1):
        examples_block += f"""
Example pair {i}:
  Topic: {pair['topic']}
  HIGH uncertainty: "{pair['high']}"
  NO uncertainty:   "{pair['no']}"
"""

    prompt = f"""\
You are an expert in economics and financial communications. Your task is to \
generate contrastive pairs of economic statements — one expressing HIGH \
uncertainty and one expressing NO uncertainty — about the same economic topic.

## Definition of Economic Uncertainty
{UNCERTAINTY_DEFINITION}

## Example Pairs
{examples_block}

## Instructions
Generate exactly {batch_size} contrastive pairs, numbered {batch_start} \
through {batch_start + batch_size - 1}. Each pair must:

1. Be about a specific economic topic. Vary the topics widely across \
macroeconomics, sector-level analysis, commodities, monetary policy, \
labor markets, etc. Topics include but not limited to tariffs, geopolitical risk, \
inflation, interest rates, AI, debt crisis, supply chain disruptions, and \
policy changes. \
2. Sound like they come from an earnings call, analyst briefing, or \
economic commentary — use realistic financial language.
3. Each statement should be 1-3 sentences long. Both statements in a pair \
should be roughly the same length — DO NOT make high-uncertainty statements \
systematically longer than low-uncertainty ones. Vary sentence structure, \
length, and punctuation across pairs.
4. The two statements in a pair should differ in UNCERTAINTY FRAMING only \
— not in the underlying economic topic.
5. Express uncertainty through DIVERSE mechanisms — not just explicit \
acknowledgements like "hard to forecast" or "genuinely unclear". Use a mix \
of: wide numerical ranges, conditional/scenario-dependent framing, hedging \
qualifiers ("could", "may", "depending on"), references to conflicting \
signals, open-ended timelines, and cautious forward guidance. Some \
statements can be explicitly uncertain, but most should convey uncertainty \
implicitly through how the speaker frames the situation.

Return your output as a JSON array of objects. Each object has:
- "pair_id": integer (the pair number)
- "topic": short label for the economic topic
- "high": the HIGH-uncertainty statement (string)
- "no": the NO-uncertainty statement (string)

Return ONLY the JSON array, no other text.
"""
    return prompt


def generate_batch(
    client: anthropic.Anthropic,
    batch_start: int,
    batch_size: int,
    model: str,
    max_retries: int = 3,
) -> list[dict]:
    """Call Claude API to generate one batch of pairs."""
    prompt = build_generation_prompt(batch_start, batch_size)

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=1.0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                text = text.rsplit("```", 1)[0]
            pairs = json.loads(text)
            if len(pairs) != batch_size:
                print(
                    f"  Warning: expected {batch_size} pairs, got {len(pairs)}. "
                    f"Retrying ({attempt + 1}/{max_retries})..."
                )
                continue
            return pairs
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            print(f"  Parse error: {e}. Retrying ({attempt + 1}/{max_retries})...")
            time.sleep(2)
        except anthropic.RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"  Rate limited. Waiting {wait}s...")
            time.sleep(wait)

    raise RuntimeError(
        f"Failed to generate batch starting at {batch_start} after {max_retries} retries"
    )



def main():
    parser = argparse.ArgumentParser(
        description="Generate contrastive economic uncertainty pairs via Claude API"
    )
    parser.add_argument(
        "--n_pairs",
        type=int,
        default=100,
        help="Total number of contrastive pairs to generate (default: 100)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=25,
        help="Pairs per API call (default: 25)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-6",
        help="Claude model to use",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="synthetic_pairs_generated.json",
        help="Output file path",
    )
    args = parser.parse_args()

    # Print a sample prompt for inspection
    sample_prompt = build_generation_prompt(1, args.batch_size)
    print("=" * 60)
    print("SAMPLE PROMPT (batch 1):")
    print("=" * 60)
    print(sample_prompt)
    print("=" * 60)

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    all_pairs = []
    n_batches = (args.n_pairs + args.batch_size - 1) // args.batch_size

    for i in range(n_batches):
        batch_start = i * args.batch_size + 1
        current_batch_size = min(args.batch_size, args.n_pairs - i * args.batch_size)
        print(
            f"Generating batch {i + 1}/{n_batches} "
            f"(pairs {batch_start}-{batch_start + current_batch_size - 1})..."
        )
        pairs = generate_batch(client, batch_start, current_batch_size, args.model)
        all_pairs.extend(pairs)
        print(f"  Got {len(pairs)} pairs.")
        if i < n_batches - 1:
            time.sleep(1)  # brief pause between batches

    # Write raw pairs
    with open(args.out, "w") as f:
        json.dump(all_pairs, f, indent=2)
    print(f"\nWrote {len(all_pairs)} pairs to {args.out}")


if __name__ == "__main__":
    main()
