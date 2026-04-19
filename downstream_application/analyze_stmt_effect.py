"""
Task 1: Compare investment_alpha0_layer12_nostmt.json vs investment_alpha0_layer12.json.

For each sample, tests whether adding the earnings-call statement shifts the model's
portfolio in the expected direction:
  - High uncertainty → statement should REDUCE stock allocation (more treasuries)
  - Low uncertainty  → statement should INCREASE stock allocation (fewer treasuries)

Reports per-sample results and group-level paired t-tests.
"""

import csv
import json
import numpy as np
from pathlib import Path
from scipy import stats

RESULTS_DIR = Path(__file__).parent / "results"
DATA_PATH   = Path(__file__).parent / "data" / "sentences_with_context.json"
OUT_CSV     = RESULTS_DIR / "stmt_effect_per_sample.csv"
OUT_TXT     = RESULTS_DIR / "stmt_effect_summary.txt"

# ── load data ──────────────────────────────────────────────────────────────────

with open(DATA_PATH) as f:
    sentences = {r["id"]: r for r in json.load(f)}

with open(RESULTS_DIR / "investment_alpha0_layer12_nostmt.json") as f:
    nostmt = json.load(f)

with open(RESULTS_DIR / "investment_alpha0_layer12.json") as f:
    stmt = json.load(f)

# index by id
nostmt_by_id = {r["id"]: r for r in nostmt["results"]}
stmt_by_id   = {r["id"]: r for r in stmt["results"]}

# ── per-sample comparison ──────────────────────────────────────────────────────

print("=" * 80)
print("PER-SAMPLE COMPARISON: no-statement vs. with-statement (alpha=0)")
print("=" * 80)
print(f"{'ID':>4}  {'Unc':>5}  {'Stock (no stmt)':>15}  {'Stock (stmt)':>12}  "
      f"{'Δ Stock':>8}  {'Expected':>9}  {'Correct':>7}")
print("-" * 80)

high_diffs, low_diffs = [], []
high_correct, low_correct = 0, 0
high_total,  low_total   = 0, 0
# store per-run arrays for paired t-tests
high_nostmt_runs, high_stmt_runs = [], []
low_nostmt_runs,  low_stmt_runs  = [], []

csv_rows = []

for sid, sentence in sentences.items():
    if sid not in nostmt_by_id or sid not in stmt_by_id:
        continue
    unc = sentence["uncertainty"]
    ns  = nostmt_by_id[sid]
    st  = stmt_by_id[sid]

    stock_no  = ns["mean_stock"]
    stock_yes = st["mean_stock"]
    delta     = stock_yes - stock_no          # positive = more stock with stmt

    # collect per-run allocations for statistical tests
    ns_runs = [r["stock"] for r in ns["runs"] if r.get("parse_ok")]
    st_runs = [r["stock"] for r in st["runs"] if r.get("parse_ok")]

    if unc == "high":
        expected   = "decrease"
        correct    = delta < 0
        high_diffs.append(delta)
        high_correct += int(correct)
        high_total   += 1
        high_nostmt_runs.extend(ns_runs)
        high_stmt_runs.extend(st_runs)
    else:  # low
        expected   = "increase"
        correct    = delta > 0
        low_diffs.append(delta)
        low_correct += int(correct)
        low_total   += 1
        low_nostmt_runs.extend(ns_runs)
        low_stmt_runs.extend(st_runs)

    marker = "yes" if correct else "no"
    csv_rows.append({
        "id": sid,
        "ticker": sentence.get("ticker", ""),
        "uncertainty": unc,
        "stock_no_stmt": stock_no,
        "stock_with_stmt": stock_yes,
        "delta_stock": delta,
        "expected_direction": expected,
        "direction_correct": marker,
    })

    print(f"{sid:>4}  {unc:>5}  {stock_no:>15.1f}  {stock_yes:>12.1f}  "
          f"{delta:>+8.1f}  {expected:>9}  {'✓' if marker == 'yes' else '✗':>7}")

# ── write per-sample CSV ───────────────────────────────────────────────────────

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
    writer.writeheader()
    writer.writerows(csv_rows)
print(f"\nPer-sample CSV saved to: {OUT_CSV}")

# ── group-level statistics ─────────────────────────────────────────────────────

summary_lines = []

def emit(line=""):
    print(line)
    summary_lines.append(line)

emit()
emit("=" * 80)
emit("GROUP-LEVEL SUMMARY")
emit("=" * 80)

for label, diffs, correct, total, ns_runs, st_runs, direction in [
    ("HIGH uncertainty", high_diffs, high_correct, high_total,
     high_nostmt_runs, high_stmt_runs, "negative Δ (less stock)"),
    ("LOW uncertainty",  low_diffs,  low_correct,  low_total,
     low_nostmt_runs,  low_stmt_runs,  "positive Δ (more stock)"),
]:
    if total == 0:
        continue
    diffs = np.array(diffs)
    emit(f"\n{label}  (n={total}, expected direction: {direction})")
    emit(f"  Accuracy (direction correct): {correct}/{total} = {correct/total:.1%}")
    emit(f"  Mean Δ stock:  {diffs.mean():+.2f}  (std {diffs.std():.2f})")
    emit(f"  Median Δ stock: {np.median(diffs):+.2f}")

    if len(ns_runs) >= 2 and len(st_runs) >= 2:
        t_stat, p_two = stats.ttest_ind(st_runs, ns_runs)
        if label.startswith("HIGH"):
            p_one = p_two / 2 if t_stat < 0 else 1 - p_two / 2
        else:
            p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
        emit(f"  t-test (stmt vs no-stmt runs): t={t_stat:.3f}, "
             f"p(two-sided)={p_two:.4f}, p(one-sided)={p_one:.4f}")

    if len(diffs) >= 2:
        if label.startswith("HIGH"):
            w_stat, p_w = stats.wilcoxon(diffs, alternative="less")
        else:
            w_stat, p_w = stats.wilcoxon(diffs, alternative="greater")
        emit(f"  Wilcoxon signed-rank (Δ stock means): W={w_stat:.1f}, p={p_w:.4f}")

emit()
emit("=" * 80)
emit("OVERALL ACCURACY")
total = high_total + low_total
correct = high_correct + low_correct
emit(f"  {correct}/{total} samples shifted in the expected direction ({correct/total:.1%})")
emit("=" * 80)

with open(OUT_TXT, "w") as f:
    f.write("\n".join(summary_lines))
print(f"\nSummary saved to: {OUT_TXT}")
