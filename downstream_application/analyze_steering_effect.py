"""
Task 2: Compare investment_alphax_layer12.json across alpha values (-4 to 4).

Plots average stock allocation vs. steering strength alpha, broken down by:
  - All samples
  - High uncertainty samples
  - Low uncertainty samples

Saves plot to results_new/steering_effect.png and prints a summary table.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
DATA_PATH   = Path(__file__).parent / "data" / "sentences_with_context.json"
OUT_PLOT_ALL   = RESULTS_DIR / "steering_effect_all.png"
OUT_PLOT_CLASS = RESULTS_DIR / "steering_effect_per_class.png"

ALPHAS = list(range(-4, 5))   # -4 … 4 (skip ±5 as unstable)

# ── load uncertainty labels ────────────────────────────────────────────────────

with open(DATA_PATH) as f:
    sentences = {r["id"]: r for r in json.load(f)}

# ── collect mean stock per alpha per sample ────────────────────────────────────

def load_alpha(alpha):
    name = f"investment_alpha{alpha}_layer12.json"
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

rows = []   # list of (alpha, sid, uncertainty, mean_stock)

for alpha in ALPHAS:
    data = load_alpha(alpha)
    if data is None:
        print(f"WARNING: missing file for alpha={alpha}")
        continue
    for r in data["results"]:
        sid = r["id"]
        unc = sentences.get(sid, {}).get("uncertainty", "unknown")
        rows.append({"alpha": alpha, "id": sid, "uncertainty": unc,
                     "mean_stock": r["mean_stock"] / 10.0})  # convert to %

# ── aggregate ──────────────────────────────────────────────────────────────────

import pandas as pd
df = pd.DataFrame(rows)

grouped = df.groupby(["alpha", "uncertainty"])["mean_stock"]
agg = grouped.agg(["mean", "sem"]).reset_index()

groups = {
    "all":  df.groupby("alpha")["mean_stock"].agg(["mean", "sem"]).reset_index(),
    "high": agg[agg["uncertainty"] == "high"].copy(),
    "low":  agg[agg["uncertainty"] == "low"].copy(),
}

# ── print summary table ────────────────────────────────────────────────────────

print("=" * 65)
print("AVERAGE STOCK ALLOCATION (out of 1000) BY ALPHA")
print("=" * 65)
print(f"{'Alpha':>6}  {'All (mean±sem)':>18}  {'High (mean±sem)':>18}  {'Low (mean±sem)':>18}")
print("-" * 65)

for alpha in ALPHAS:
    def fmt(sub):
        row = sub[sub["alpha"] == alpha]
        if row.empty:
            return "       N/A      "
        m, s = row["mean"].values[0], row["sem"].values[0]
        return f"{m:6.1f} ± {s:5.1f}"

    a_row = groups["all"][groups["all"]["alpha"] == alpha]
    all_s = f"{a_row['mean'].values[0]:.1f} ± {a_row['sem'].values[0]:.1f}" if not a_row.empty else "N/A"
    print(f"{alpha:>6}  {all_s:>18}  {fmt(groups['high']):>18}  {fmt(groups['low']):>18}")

# ── plot ───────────────────────────────────────────────────────────────────────

colors = {"high": "#d62728", "low": "#1f77b4"}
labels = {"high": "High uncertainty", "low": "Low uncertainty"}

# All samples
fig, ax = plt.subplots(figsize=(6, 5))
g = groups["all"]
ax.errorbar(g["alpha"], g["mean"], yerr=g["sem"],
            fmt="o-", color="#2ca02c", capsize=4, linewidth=2, markersize=6,
            label="All samples")
ax.axvline(0, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Steering strength α", fontsize=12)
ax.set_ylabel("Mean stock allocation (%)", fontsize=12)
ax.set_xticks(ALPHAS)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_PLOT_ALL, dpi=150)
print(f"Plot saved to: {OUT_PLOT_ALL}")
plt.close()

# High vs low uncertainty
fig, ax = plt.subplots(figsize=(6, 5))
for unc in ("high", "low"):
    g = groups[unc]
    if g.empty:
        continue
    ax.errorbar(g["alpha"], g["mean"], yerr=g["sem"],
                fmt="o-", color=colors[unc], capsize=4, linewidth=2, markersize=6,
                label=labels[unc])
ax.axvline(0, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Steering strength α", fontsize=12)
ax.set_ylabel("Mean stock allocation (%)", fontsize=12)
ax.set_xticks(ALPHAS)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_PLOT_CLASS, dpi=150)
print(f"Plot saved to: {OUT_PLOT_CLASS}")
plt.close()
