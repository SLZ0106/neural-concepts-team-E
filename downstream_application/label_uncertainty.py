"""
Task 0: Fill in missing uncertainty labels in sentences_with_context.json.
All manually-labeled 'low' samples keep their label; all others become 'high'.
"""

import json
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "sentences_with_context.json"

with open(DATA_PATH) as f:
    records = json.load(f)

changed = 0
for r in records:
    if r.get("uncertainty") != "low":
        r["uncertainty"] = "high"
        changed += 1

with open(DATA_PATH, "w") as f:
    json.dump(records, f, indent=2)

print(f"Labeled {changed} records as 'high'.")
print(f"Summary:")
from collections import Counter
counts = Counter(r["uncertainty"] for r in records)
for label, n in sorted(counts.items()):
    print(f"  {label}: {n}")
