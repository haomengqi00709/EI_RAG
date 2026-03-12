#!/usr/bin/env python3
"""
evaluate_retrieval.py — Measure retrieval quality against the eval set.

For each question in data/eval/eval_set.json, retrieves the top-k chunks
and checks whether the source chunk_id appears in the results.

Metrics reported:
  Recall@1  — correct chunk was the top result
  Recall@3  — correct chunk appeared in top 3
  Recall@5  — correct chunk appeared in top 5
  Recall@10 — correct chunk appeared in top 10
  MRR       — Mean Reciprocal Rank (average of 1/rank across all questions)

Run:
    python src/evaluate_retrieval.py
    python src/evaluate_retrieval.py --top-k 10
    python src/evaluate_retrieval.py --year 2023-2024   # filter to one year
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from embed import load_embedding_model
from retrieve import load_bm25, load_chunks, load_index, retrieve_multi_query

# ── Configuration ─────────────────────────────────────────────────────────────

EVAL_PATH   = Path("data/eval/eval_set.json")
OUT_PATH    = Path("data/eval/retrieval_results.json")   # always overwritten (latest)
RUNS_DIR    = Path("data/eval/runs")                      # timestamped archive
DEFAULT_K   = 10    # retrieve this many candidates per question
RECALL_AT   = [1, 3, 5, 10]

# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(eval_set: list, vectors, manifest, chunks, bm25, model, tokenizer, top_k: int, fiscal_year: str | None) -> dict:
    hits    = {k: 0 for k in RECALL_AT}
    reciprocal_ranks = []
    misses  = []

    for item in eval_set:
        question      = item["question"]
        target_id     = item["chunk_id"]
        target_type   = item.get("chunk_type")
        target_fy     = item.get("fiscal_year")

        # Use per-question fiscal year if no global filter specified
        fy = fiscal_year or target_fy or None
        results = retrieve_multi_query(
            questions=[question],
            vectors=vectors,
            bm25=bm25,
            manifest=manifest,
            chunks=chunks,
            top_k=top_k,
            fiscal_year=fy,
            model=model,
            tokenizer=tokenizer,
            use_rerank=True,
        )

        returned_ids = [r["row"]["chunk_id"] for r in results]

        # Rank of the correct chunk (1-indexed), or None if not found
        rank = None
        for i, rid in enumerate(returned_ids, 1):
            if rid == target_id:
                rank = i
                break

        # Recall@k
        for k in RECALL_AT:
            if rank is not None and rank <= k:
                hits[k] += 1

        # MRR
        reciprocal_ranks.append(1 / rank if rank else 0)

        if rank is None or rank > 5:
            misses.append({
                "question":   question,
                "target_id":  target_id,
                "chunk_type": target_type,
                "fiscal_year": target_fy,
                "rank":       rank,
                "top5_ids":   returned_ids[:5],
            })

    n = len(eval_set)
    return {
        "n_questions": n,
        "recall": {f"@{k}": round(hits[k] / n, 3) for k in RECALL_AT},
        "mrr":    round(sum(reciprocal_ranks) / n, 3),
        "misses": misses,
    }


# ── Output formatting ─────────────────────────────────────────────────────────

def print_report(results: dict, fiscal_year: str | None):
    n = results["n_questions"]
    print(f"\n{'─' * 50}")
    print(f"Retrieval Evaluation — {n} questions"
          + (f" (filtered: {fiscal_year})" if fiscal_year else ""))
    print(f"{'─' * 50}")

    print("\nRecall (did the correct chunk appear in top-k?):")
    for label, score in results["recall"].items():
        bar = "█" * int(score * 20)
        print(f"  Recall{label:<4}  {score:.1%}  {bar}")

    print(f"\nMRR (Mean Reciprocal Rank): {results['mrr']:.3f}")
    print("  (1.0 = always top result, 0.5 = always rank 2, etc.)\n")

    misses = results["misses"]
    if misses:
        print(f"{'─' * 50}")
        print(f"Missed or low-ranked ({len(misses)} questions where correct chunk not in top 5):\n")
        for m in misses:
            rank_str = f"rank {m['rank']}" if m["rank"] else "not found"
            print(f"  [{m['fiscal_year']} | {m['chunk_type']}] {rank_str}")
            print(f"  Q: {m['question'][:80]}")
            print(f"  Target:  {m['target_id']}")
            print(f"  Top-5 returned:")
            for rid in m["top5_ids"]:
                print(f"    - {rid}")
            print()
    else:
        print("All questions: correct chunk found in top 5.")

    print(f"{'─' * 50}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality against the eval set."
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_K,
        help=f"Number of chunks to retrieve per question (default {DEFAULT_K})"
    )
    parser.add_argument(
        "--year", help="Filter retrieval to a single fiscal year (e.g. 2023-2024)"
    )
    parser.add_argument(
        "--label", default="",
        help="Short label describing this run, e.g. 'after_reindex' or 'baseline'"
    )
    args = parser.parse_args()

    if not EVAL_PATH.exists():
        raise SystemExit(f"Eval set not found at {EVAL_PATH}.\n"
                         "Run src/generate_eval.py first.")

    with open(EVAL_PATH) as f:
        eval_set = json.load(f)

    if args.year:
        eval_set = [q for q in eval_set if q.get("fiscal_year") == args.year]
        if not eval_set:
            raise SystemExit(f"No eval questions found for fiscal year {args.year}")

    print(f"Eval set: {len(eval_set)} questions")
    print("Loading model and index…")

    model, tokenizer = load_embedding_model()
    vectors, manifest = load_index()
    chunks = load_chunks()
    bm25 = load_bm25(manifest, chunks)

    print(f"Index: {len(manifest)} vectors\n")
    print("Running retrieval…")

    results = evaluate(
        eval_set=eval_set,
        vectors=vectors,
        manifest=manifest,
        chunks=chunks,
        bm25=bm25,
        model=model,
        tokenizer=tokenizer,
        top_k=args.top_k,
        fiscal_year=args.year,
    )

    print_report(results, args.year)

    # Annotate with run metadata
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    results["run_timestamp"] = ts
    results["run_label"]     = args.label

    # 1. Overwrite the fixed "latest" file
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 2. Save a timestamped copy in runs/ for comparison
    label_slug = f"_{args.label}" if args.label else ""
    run_filename = f"retrieval_{ts}{label_slug}.json"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_path = RUNS_DIR / run_filename
    with open(run_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Latest  → {OUT_PATH}")
    print(f"Archive → {run_path}")


if __name__ == "__main__":
    main()
