#!/usr/bin/env python3
"""
prepare_evalset_new.py — Convert data/eval/evalset_new/*.json into the
flat format expected by evaluate_e2e.py.

Input:  data/eval/evalset_new/{YYYY-YYYY}.json
Output: data/eval/evalset_new_combined.json

Field mapping:
  Question      → question
  Answer        → answer
  Page Number   → page  (kept for reference)
  Source Context → source_context  (kept for reference)
  (filename)    → fiscal_year
"""
import json
from pathlib import Path

IN_DIR  = Path("data/eval/evalset_new")
OUT_PATH = Path("data/eval/evalset_new_combined.json")

combined = []
for f in sorted(IN_DIR.glob("*.json")):
    fy = f.stem   # e.g. "2019-2020"
    items = json.loads(f.read_text())
    for item in items:
        # Handle both capitalised and lowercase field names
        q  = item.get("Question") or item.get("question", "")
        a  = item.get("Answer")   or item.get("answer", "")
        pg = item.get("Page Number") or item.get("page_number", "")
        sc = item.get("Source Context") or item.get("source_context", "")
        combined.append({
            "question":       q,
            "answer":         a,
            "fiscal_year":    fy,
            "page":           pg,
            "source_context": sc,
        })

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps(combined, indent=2, ensure_ascii=False))
print(f"Written {len(combined)} questions → {OUT_PATH}")

# Breakdown by year
from collections import Counter
by_year = Counter(q["fiscal_year"] for q in combined)
for fy in sorted(by_year):
    print(f"  {fy}: {by_year[fy]} questions")
