#!/usr/bin/env python3
"""
evaluate_llm_judge.py — LLM-as-judge retrieval evaluation.

For each question in data/eval/eval_set.json:
  1. Retrieves top-k chunks (auto-filtered to the question's own fiscal year)
  2. Asks Gemini to judge whether any retrieved chunk correctly answers the question
  3. Reports overall Answered@1/3/5 plus a per-fiscal-year breakdown

Run:
    python src/evaluate_llm_judge.py                        # all 60 questions, hybrid
    python src/evaluate_llm_judge.py --mode dense           # dense only
    python src/evaluate_llm_judge.py --year 2023-2024       # one year only
    python src/evaluate_llm_judge.py --top-k 10
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))

from embed import load_embedding_model
from retrieve import load_bm25, load_chunks, load_index, retrieve, retrieve_hybrid, rerank

# ── Configuration ─────────────────────────────────────────────────────────────

EVAL_PATH     = Path("data/eval/eval_set.json")
OUT_PATH      = Path("data/eval/llm_judge_results.json")   # always overwritten (latest)
RUNS_DIR      = Path("data/eval/runs")                      # timestamped archive
DEFAULT_K     = 5
ANSWERED_AT   = [1, 3, 5, 10]
SNIPPET_CHARS = 1500

SYSTEM_PROMPT = (
    "You are evaluating a retrieval system for Canadian government Employment "
    "Insurance (EI) policy documents. Given a question, the correct answer, and "
    "a list of retrieved text chunks, determine whether the retrieved chunks "
    "contain enough information to correctly answer the question. "
    "A chunk qualifies if it contains the specific fact, number, or policy detail "
    "in the correct answer — even if worded differently."
)

JUDGE_TEMPLATE = """\
Question: {question}

Correct answer: {ground_truth}

Retrieved chunks:
{chunks_text}

Does any retrieved chunk contain enough information to correctly answer the question?

Return ONLY valid JSON:
{{
  "answered": true or false,
  "best_chunk": 1-based index of the chunk that best answers the question (null if none),
  "reasoning": "one sentence explanation"
}}"""


# ── Query-centered snippet extraction ────────────────────────────────────────

def query_centered_snippet(text: str, query: str, window: int = SNIPPET_CHARS) -> str:
    """
    Find the position in the chunk text where query keywords cluster most
    densely, then extract a window of `window` characters centered there.
    Falls back to the start of the text if no keywords match.
    """
    # Tokenize query — drop short stopwords that add noise
    query_tokens = {
        t for t in re.sub(r"[^\w\s]", " ", query.lower()).split()
        if len(t) > 2
    }

    if not query_tokens or not text:
        return (text[:window] + ("…" if len(text) > window else ""))

    words = text.split()

    # Sliding window of ~30 words: score by how many query tokens appear
    SCORE_WINDOW = 30
    best_score   = -1
    best_word_pos = 0

    for i in range(len(words)):
        window_text = " ".join(words[i: i + SCORE_WINDOW]).lower()
        score = sum(1 for t in query_tokens if t in window_text)
        if score > best_score:
            best_score    = score
            best_word_pos = i

    # Convert best word position → character offset
    char_pos = len(" ".join(words[:best_word_pos]))

    # Extract window centered on char_pos
    half  = window // 2
    start = max(0, char_pos - half)
    end   = min(len(text), start + window)
    start = max(0, end - window)   # re-adjust if end hit the boundary

    prefix  = "…" if start > 0       else ""
    suffix  = "…" if end < len(text) else ""
    snippet = text[start:end].replace("\n", " ")

    return prefix + snippet + suffix


# ── Chunk formatting ──────────────────────────────────────────────────────────

def format_chunks_for_judge(results: list, question: str) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        row   = r["row"]
        chunk = r["chunk"]
        ctype = row["chunk_type"]
        fy    = row.get("fiscal_year", "")
        page  = row.get("page", "")
        bc    = row.get("section_breadcrumb", "")[:60]

        body    = chunk.get("text_nl") or chunk.get("text", "") if ctype == "table" else chunk.get("text", "")
        # Chunks under 2000 chars: show in full so the judge never misses a value near the end
        snippet = body if len(body) <= 2000 else query_centered_snippet(body, question)

        lines.append(f"[{i}] {fy} | {ctype} | p.{page} | {bc}\n{snippet}")
    return "\n\n".join(lines)


# ── Judge call ────────────────────────────────────────────────────────────────

def judge(model, question: str, ground_truth: str, results: list) -> dict:
    chunks_text = format_chunks_for_judge(results, question)
    prompt = JUDGE_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        chunks_text=chunks_text,
    )
    try:
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"  Judge ERROR: {e}")
        return {"answered": False, "best_chunk": None, "reasoning": f"error: {e}"}


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate(
    eval_set: list,
    vectors, manifest, chunks,
    model, embed_model, tokenizer,
    top_k: int,
    year_filter: str | None,   # if set, overrides per-question year
    mode: str = "hybrid",
    bm25=None,
    use_rerank: bool = False,
) -> dict:

    answered = {k: 0 for k in ANSWERED_AT}

    # Per-year tracking
    by_year: dict[str, dict] = defaultdict(lambda: {
        k: 0 for k in ANSWERED_AT
    })
    year_counts: dict[str, int] = defaultdict(int)

    details = []

    for i, item in enumerate(eval_set, 1):
        question     = item["question"]
        ground_truth = item["answer"]
        target_id    = item["chunk_id"]
        item_fy      = item.get("fiscal_year", "")

        # Use per-question fiscal year unless a global override is set
        fy_for_retrieval = year_filter if year_filter else item_fy

        print(f"[{i}/{len(eval_set)}] {item_fy} | {item.get('chunk_type')}")
        print(f"  Q: {question[:80]}")

        # Retrieve
        if mode == "hybrid":
            results = retrieve_hybrid(
                question=question, vectors=vectors, bm25=bm25,
                manifest=manifest, chunks=chunks, top_k=top_k,
                fiscal_year=fy_for_retrieval,
                model=embed_model, tokenizer=tokenizer,
            )
        else:
            results = retrieve(
                question=question, vectors=vectors, manifest=manifest,
                chunks=chunks, top_k=top_k,
                fiscal_year=fy_for_retrieval,
                model=embed_model, tokenizer=tokenizer,
            )

        # Optional cross-encoder rerank
        if use_rerank:
            results = rerank(question, results)

        # Judge
        verdict       = judge(model, question, ground_truth, results)
        answered_flag = verdict.get("answered", False)
        best_idx      = verdict.get("best_chunk")
        reasoning     = verdict.get("reasoning", "")

        # Overall + per-year counts
        year_counts[item_fy] += 1
        for k in ANSWERED_AT:
            if answered_flag and best_idx is not None and best_idx <= k:
                answered[k] += 1
                by_year[item_fy][k] += 1

        details.append({
            "question":     question,
            "ground_truth": ground_truth,
            "chunk_id":     target_id,
            "chunk_type":   item.get("chunk_type"),
            "fiscal_year":  item_fy,
            "answered":     answered_flag,
            "best_chunk":   best_idx,
            "reasoning":    reasoning,
            "top_ids":      [r["row"]["chunk_id"] for r in results],
            "scores":       [round(float(r["score"]), 4) for r in results],
        })

        status = f"✓ (chunk #{best_idx})" if answered_flag else "✗"
        print(f"  {status} — {reasoning[:80]}")
        time.sleep(0.5)

    n = len(eval_set)

    # Build per-year summary
    per_year = {}
    for fy in sorted(year_counts):
        nc = year_counts[fy]
        per_year[fy] = {
            "n": nc,
            "answered": {
                f"@{k}": round(by_year[fy][k] / nc, 3) for k in ANSWERED_AT
            }
        }

    return {
        "n_questions":        n,
        "top_k":              top_k,
        "mode":               mode,
        "rerank":             use_rerank,
        "year_filter":        year_filter,
        "answered":           {f"@{k}": round(answered[k] / n, 3) for k in ANSWERED_AT},
        "by_year":            per_year,
        "details":            details,
    }


# ── Output formatting ─────────────────────────────────────────────────────────

def print_report(results: dict):
    n    = results["n_questions"]
    mode = results["mode"]
    yf   = results.get("year_filter") or "auto (per question)"

    print(f"\n{'─' * 60}")
    print(f"LLM-as-Judge Evaluation  [{mode} | year: {yf}]")
    print(f"{'─' * 60}")

    print(f"\nOverall  ({n} questions):")
    for label, score in results["answered"].items():
        bar = "█" * int(score * 20)
        print(f"  Answered{label:<4}  {score:.1%}  {bar}")

    print(f"\nPer fiscal year:")
    header = f"  {'Year':<14} {'n':>3}  " + "  ".join(f"@{k}" for k in [1, 3, 5])
    print(header)
    print(f"  {'─'*14} {'─'*3}  {'─'*4}  {'─'*4}  {'─'*4}")
    for fy, data in results["by_year"].items():
        nc  = data["n"]
        a   = data["answered"]
        row = f"  {fy:<14} {nc:>3}  " + "  ".join(f"{a[f'@{k}']:.0%}" for k in [1, 3, 5])
        print(row)

    # Failure breakdown
    failures = [d for d in results["details"] if not d["answered"]]
    if failures:
        print(f"\n{'─' * 60}")
        print(f"Failed questions ({len(failures)}/{n}):\n")
        for f in failures:
            print(f"  [{f['fiscal_year']} | {f['chunk_type']}]")
            print(f"  Q: {f['question'][:80]}")
            print(f"  Reason: {f['reasoning'][:100]}")
            print()

    print(f"{'─' * 60}")
    print(f"Full results → {OUT_PATH}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-judge retrieval evaluation using Gemini."
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_K)
    parser.add_argument("--year", help="Restrict to one fiscal year (e.g. 2023-2024)")
    parser.add_argument(
        "--mode", choices=["dense", "hybrid"], default="hybrid",
        help="Retrieval mode (default: hybrid)"
    )
    parser.add_argument(
        "--label", default="",
        help="Short label describing this run, e.g. 'after_reindex' or 'with_reranker'"
    )
    parser.add_argument(
        "--rerank", action="store_true",
        help="Apply cross-encoder reranker after retrieval before judging"
    )
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY not found in .env")
    genai.configure(api_key=api_key)
    judge_model = genai.GenerativeModel(
        model_name=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        system_instruction=SYSTEM_PROMPT,
    )

    if not EVAL_PATH.exists():
        raise SystemExit(f"Eval set not found at {EVAL_PATH}. Run generate_eval.py first.")
    with open(EVAL_PATH) as f:
        eval_set = json.load(f)

    if args.year:
        eval_set = [q for q in eval_set if q.get("fiscal_year") == args.year]
        if not eval_set:
            raise SystemExit(f"No eval questions for {args.year}")

    print(f"Eval set: {len(eval_set)} questions")
    print(f"Mode: {args.mode}  |  Year filter: {args.year or 'auto (per question)'}")
    print("Loading embedding model and index…")

    embed_model, tokenizer = load_embedding_model()
    vectors, manifest      = load_index()
    chunks                 = load_chunks()
    print(f"Index: {len(manifest)} vectors\n")

    bm25 = None
    if args.mode == "hybrid":
        print("Loading BM25 index…")
        bm25 = load_bm25(manifest, chunks)

    results = evaluate(
        eval_set=eval_set,
        vectors=vectors,
        manifest=manifest,
        chunks=chunks,
        model=judge_model,
        embed_model=embed_model,
        tokenizer=tokenizer,
        top_k=args.top_k,
        year_filter=args.year,
        mode=args.mode,
        use_rerank=args.rerank,
        bm25=bm25,
    )

    print_report(results)

    # Annotate with run metadata
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    results["run_timestamp"] = ts
    results["run_label"]     = args.label

    # 1. Overwrite the fixed "latest" file (used by show_failures.py)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 2. Save a timestamped copy in runs/ for comparison
    label_slug = f"_{args.label}" if args.label else ""
    run_filename = f"judge_{ts}{label_slug}.json"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_path = RUNS_DIR / run_filename
    with open(run_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Latest  → {OUT_PATH}")
    print(f"Archive → {run_path}")


if __name__ == "__main__":
    main()
