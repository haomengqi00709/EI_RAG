#!/usr/bin/env python3
"""
show_failures.py — Print all questions where the LLM judge answered=false.

Run:
    python src/show_failures.py
    python src/show_failures.py --year 2023-2024
    python src/show_failures.py --type table
"""

import argparse
import json
from pathlib import Path

RESULTS_PATH = Path("data/eval/llm_judge_results.json")


def main():
    parser = argparse.ArgumentParser(description="Show failed eval questions.")
    parser.add_argument("--year",  help="Filter by fiscal year (e.g. 2023-2024)")
    parser.add_argument("--type",  help="Filter by chunk type (narrative or table)")
    args = parser.parse_args()

    if not RESULTS_PATH.exists():
        raise SystemExit(f"Results not found at {RESULTS_PATH}. Run evaluate_llm_judge.py first.")

    with open(RESULTS_PATH) as f:
        data = json.load(f)

    failures = [d for d in data["details"] if not d["answered"]]

    if args.year:
        failures = [f for f in failures if f.get("fiscal_year") == args.year]
    if args.type:
        failures = [f for f in failures if f.get("chunk_type") == args.type]

    total = len(data["details"])
    print(f"\nFailed: {len(failures)} / {total} questions\n")
    print("─" * 65)

    for i, f in enumerate(failures, 1):
        print(f"\n[{i}] {f['fiscal_year']} | {f['chunk_type']} | target: {f['chunk_id']}")
        print(f"  Q:      {f['question']}")
        print(f"  Answer: {f['ground_truth']}")
        print(f"  Reason: {f['reasoning']}")
        print(f"  Top returned:")
        for j, (cid, score) in enumerate(zip(f["top_ids"], f["scores"]), 1):
            print(f"    {j}. {cid}  (score: {score})")

    print(f"\n{'─' * 65}")
    print(f"Total failures: {len(failures)}")


if __name__ == "__main__":
    main()
