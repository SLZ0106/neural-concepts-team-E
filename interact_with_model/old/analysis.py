#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

LABELS = ["no_uncertainty", "intermediate_uncertainty", "uncertainty"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_jsonl", type=str, required=True, help="Path to preds.jsonl")
    p.add_argument("--variant", type=str, default="all",
                   help="Filter by variant: full/no_qa/qa_only/all")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Threshold for FP/FN on uncertainty_score")
    p.add_argument("--topk", type=int, default=10, help="How many extreme cases to print per label")
    p.add_argument("--output_report", type=str, default="",
                   help="Optional: write a JSON report to this path")
    return p.parse_args()

def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def infer_label(ex_id: str, record: Dict[str, Any]) -> Optional[str]:
    # Prefer explicit label if present
    lbl = record.get("label")
    if isinstance(lbl, str) and lbl in LABELS:
        return lbl

    # Infer from id; IMPORTANT: check no_uncertainty before uncertainty
    if "_no_uncertainty_" in ex_id:
        return "no_uncertainty"
    if "_intermediate_uncertainty_" in ex_id:
        return "intermediate_uncertainty"
    # careful: this must be last
    if "_uncertainty_" in ex_id:
        return "uncertainty"

    # fallback: try prefix match
    for l in LABELS:
        if l in ex_id:
            return l
    return None

def get_score(record: Dict[str, Any]) -> Optional[float]:
    pj = record.get("parsed_json")
    if not isinstance(pj, dict):
        return None
    s = pj.get("uncertainty_score")
    try:
        s = float(s)
    except Exception:
        return None
    if not (0.0 <= s <= 1.0):
        return None
    return s

def mean_std(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return m, math.sqrt(var)

def main():
    args = parse_args()

    per_label_scores: Dict[str, List[float]] = defaultdict(list)
    per_label_rows: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)  # (id, variant, score)

    n_total = 0
    n_used = 0
    n_missing_score = 0
    n_missing_label = 0

    for rec in iter_jsonl(args.pred_jsonl):
        n_total += 1
        ex_id = rec.get("id", "")
        variant = rec.get("variant", "")
        if args.variant != "all" and variant != args.variant:
            continue

        label = infer_label(ex_id, rec)
        if label is None:
            n_missing_label += 1
            continue

        score = get_score(rec)
        if score is None:
            n_missing_score += 1
            continue

        n_used += 1
        per_label_scores[label].append(score)
        per_label_rows[label].append((ex_id, variant, score))

    # Summary stats
    summary = {}
    print("\n=== Summary: mean uncertainty_score by label ===")
    for label in LABELS:
        vals = per_label_scores.get(label, [])
        m, sd = mean_std(vals)
        summary[label] = {"n": len(vals), "mean": m, "std": sd}
        print(f"{label:24s}  n={len(vals):4d}  mean={m:.4f}  std={sd:.4f}")

    print("\n=== Data coverage ===")
    print(f"records_total={n_total}")
    print(f"records_used ={n_used}  (variant filter: {args.variant})")
    print(f"missing_label={n_missing_label}")
    print(f"missing_score/parse_fail={n_missing_score}")

    # FP / FN by threshold
    thr = args.threshold
    fps: List[Tuple[str, str, float]] = []
    fns: List[Tuple[str, str, float]] = []

    for ex_id, variant, score in per_label_rows.get("no_uncertainty", []):
        if score >= thr:
            fps.append((ex_id, variant, score))
    for ex_id, variant, score in per_label_rows.get("uncertainty", []):
        if score < thr:
            fns.append((ex_id, variant, score))

    fps_sorted = sorted(fps, key=lambda x: x[2], reverse=True)
    fns_sorted = sorted(fns, key=lambda x: x[2])  # lowest first

    print(f"\n=== Potential FALSE POSITIVES (no_uncertainty but score >= {thr}) ===")
    for ex_id, variant, score in fps_sorted[: max(args.topk, 1)]:
        print(f"{score:.3f}  {variant:7s}  {ex_id}")

    print(f"\n=== Potential FALSE NEGATIVES (uncertainty but score < {thr}) ===")
    for ex_id, variant, score in fns_sorted[: max(args.topk, 1)]:
        print(f"{score:.3f}  {variant:7s}  {ex_id}")

    # Extremes per label (useful even without a fixed threshold)
    print("\n=== Extremes per label (top-k high and low) ===")
    extremes = {}
    for label in LABELS:
        rows = per_label_rows.get(label, [])
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda x: x[2])
        low = rows_sorted[: max(args.topk, 1)]
        high = list(reversed(rows_sorted[-max(args.topk, 1):]))
        extremes[label] = {"lowest": low, "highest": high}

        print(f"\n[{label}] lowest:")
        for ex_id, variant, score in low:
            print(f"{score:.3f}  {variant:7s}  {ex_id}")
        print(f"[{label}] highest:")
        for ex_id, variant, score in high:
            print(f"{score:.3f}  {variant:7s}  {ex_id}")

    # Optional JSON report
    report = {
        "pred_jsonl": args.pred_jsonl,
        "variant": args.variant,
        "threshold": thr,
        "coverage": {
            "records_total": n_total,
            "records_used": n_used,
            "missing_label": n_missing_label,
            "missing_score_or_parse_fail": n_missing_score,
        },
        "summary": summary,
        "false_positives": fps_sorted,
        "false_negatives": fns_sorted,
        "extremes": extremes,
    }

    if args.output_report:
        with open(args.output_report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nWrote report to: {args.output_report}")

if __name__ == "__main__":
    main()
