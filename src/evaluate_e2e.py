#!/usr/bin/env python3
"""
evaluate_e2e.py — End-to-end RAG evaluation.

Goes beyond retrieval quality (Answered@k) to measure whether the full
pipeline — retrieval → generation — produces the correct final answer.

Metrics:
  Exact match    — generated answer contains the ground truth string
  Semantic match — Gemini judges whether generated answer is correct
  Faithful       — generated answer is grounded in retrieved chunks
  Abstained      — pipeline correctly abstained (no hallucination)

Generation pipeline:
  Multi-query    — 3 query variants to improve retrieval recall
  Filter-expand-reduce — filter chunks, expand adjacent if needed, reduce with full context

Run:
    python src/evaluate_e2e.py
    python src/evaluate_e2e.py --year 2023-2024
    python src/evaluate_e2e.py --top-k 5 --no-rerank
    python src/evaluate_e2e.py --no-multi-query --no-map-reduce   # baseline mode
    python src/evaluate_e2e.py --eval-set data/eval/eval_set_focused.json
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent))

from embed import load_embedding_model
from generate import generate_answer_map_reduce, generate_query_variants
from retrieve import (
    _deduplicate,
    expand_context,
    load_bm25,
    load_chunks,
    load_index,
    rerank,
    retrieve_hybrid,
    retrieve_multi_query,
    retrieve_stage3,
)

# ── Configuration ──────────────────────────────────────────────────────────────

EVAL_PATH = Path("data/eval/eval_set.json")
OUT_PATH  = Path("data/eval/e2e_results.json")
DEFAULT_K = 5

# Baseline generation (used when --no-map-reduce)
ANSWER_SYSTEM = (
    "You are an expert on Canadian Employment Insurance (EI) policy. "
    "Answer questions using ONLY the provided document excerpts. "
    "Be concise and specific. "
    "If the excerpts do not contain enough information, respond with: INSUFFICIENT_CONTEXT"
)

ANSWER_TEMPLATE = """\
Question: {question}

Document excerpts:
{context}

Answer concisely using only the excerpts."""

# Judge prompt: does not penalise answers that are MORE complete than GT
SEMANTIC_JUDGE_PROMPT = """\
Question: {question}
Correct answer: {ground_truth}
Generated answer: {generated}

Does the generated answer correctly answer the question?
- Mark correct=true if the generated answer contains the key value from the correct answer
- Additional correct context beyond the ground truth is acceptable — do not penalise for it
- Rounding differences are acceptable (e.g. "9.9%" and "10.0%" are the same)
- Equivalent phrasing is acceptable (e.g. "decrease of 73.1%" = "-73.1%", "by half" = "50%")
- Mark correct=false only if the generated answer gives a clearly wrong value or \
contradicts the correct answer

Return ONLY valid JSON: {{"correct": true or false, "reasoning": "one sentence"}}"""

FAITHFULNESS_PROMPT = """\
Answer: {answer}
Source excerpts: {context}
Is every factual claim in the answer supported by the sources?
Return ONLY valid JSON: {{"faithful": true or false}}"""


def build_context(results: list) -> str:
    """Baseline context builder (used when --no-map-reduce)."""
    parts = []
    for i, r in enumerate(results, 1):
        row  = r["row"]
        body = r["chunk"].get("text_nl") or r["chunk"].get("text", "")
        fy   = row.get("fiscal_year", "")
        page = row.get("page", "")
        bc   = row.get("section_breadcrumb", "")[:60]
        snippet = body if len(body) <= 2000 else body[:1500]
        parts.append(f"[{i}] FY {fy} | p.{page} | {bc}\n{snippet}")
        for rc in r.get("context_chunks", []):
            rbody = rc.get("text_nl") or rc.get("text", "")
            parts.append(f"    (related) {rbody[:400]}")
    return "\n\n".join(parts)


def exact_match(ground_truth: str, generated: str) -> bool:
    return ground_truth.lower().strip() in generated.lower()


def evaluate(
    eval_set, vectors, manifest, chunks, bm25,
    answer_model, judge_model,
    embed_model, tokenizer,
    top_k, year_filter, use_rerank,
    use_multi_query, use_map_reduce,
    power_search:   bool = False,
    use_stage3:     bool = False,
    auto_escalate:  bool = False,
) -> dict:

    metrics  = defaultdict(int)
    by_year: dict = defaultdict(lambda: defaultdict(int))
    year_counts: dict = defaultdict(int)
    details  = []

    for i, item in enumerate(eval_set, 1):
        question     = item["question"]
        ground_truth = item["answer"]
        item_fy      = item.get("fiscal_year", "")
        fy_filter    = year_filter or item_fy

        print(f"[{i}/{len(eval_set)}] {item_fy} | {item.get('chunk_type')}")
        print(f"  Q: {question[:80]}")

        # ── Retrieval ──────────────────────────────────────────────────────────
        retrieve_k      = top_k * 2 if (power_search or use_stage3) else top_k
        neighbor_radius = 3 if (power_search or use_stage3) else 1

        if use_stage3:
            # Stage 3: HyDE + keyword BM25 + year-relaxed dense + full-page expansion
            results = retrieve_stage3(
                question=question,
                vectors=vectors, bm25=bm25,
                manifest=manifest, chunks=chunks,
                llm_model=answer_model,
                top_k=retrieve_k, fiscal_year=fy_filter,
                model=embed_model, tokenizer=tokenizer,
            )
        elif use_multi_query:
            variants = generate_query_variants(question, answer_model, n=3)
            print(f"  Variants: {len(variants)}")
            results = retrieve_multi_query(
                questions=variants,
                vectors=vectors, bm25=bm25,
                manifest=manifest, chunks=chunks, top_k=retrieve_k,
                fiscal_year=fy_filter,
                model=embed_model, tokenizer=tokenizer,
                use_rerank=use_rerank,
            )
        else:
            results = retrieve_hybrid(
                question=question, vectors=vectors, bm25=bm25,
                manifest=manifest, chunks=chunks, top_k=retrieve_k * 3,
                fiscal_year=fy_filter,
                model=embed_model, tokenizer=tokenizer,
            )
            results = expand_context(results, chunks)
            if use_rerank:
                results = rerank(question, results)
            results = results[:retrieve_k]

        # ── Generation ─────────────────────────────────────────────────────────
        if use_map_reduce:
            gen       = generate_answer_map_reduce(
                question, results, answer_model, judge_model,
                chunks=chunks, neighbor_radius=neighbor_radius,
            )
            generated = gen["answer"]
            abstained = gen["abstained"]
            faithful  = gen["faithful"]
            mapped    = gen.get("mapped", [])
        else:
            context = build_context(results)
            try:
                resp      = answer_model.generate_content(
                    ANSWER_TEMPLATE.format(question=question, context=context)
                )
                generated = resp.text.strip()
            except Exception as e:
                generated = f"ERROR: {e}"
            abstained = "INSUFFICIENT_CONTEXT" in generated
            if abstained:
                generated = ""
            faithful  = None
            mapped    = []

            # Faithfulness for baseline mode
            if not abstained and generated:
                try:
                    f = judge_model.generate_content(
                        FAITHFULNESS_PROMPT.format(
                            answer=generated, context=build_context(results)[:3000]
                        ),
                        generation_config={"response_mime_type": "application/json"},
                    )
                    faithful = json.loads(f.text).get("faithful")
                except Exception:
                    pass

        # ── Stage 3 auto-escalation (mirrors handler.py) ───────────────────────
        if auto_escalate and abstained and use_map_reduce:
            print(f"  Abstained → escalating to Stage 3…")
            results_s3 = retrieve_stage3(
                question=question,
                vectors=vectors, bm25=bm25,
                manifest=manifest, chunks=chunks,
                llm_model=answer_model, top_k=10,
                fiscal_year=fy_filter,
                model=embed_model, tokenizer=tokenizer,
            )
            if results_s3:
                results_s3 = _deduplicate(results_s3)
                gen_s3 = generate_answer_map_reduce(
                    question, results_s3, answer_model, judge_model,
                    chunks=chunks, neighbor_radius=3,
                )
                if not gen_s3["abstained"]:
                    gen       = gen_s3
                    results   = results_s3
                    generated = gen["answer"]
                    abstained = gen["abstained"]
                    faithful  = gen["faithful"]
                    mapped    = gen.get("mapped", [])

        # ── Exact match ────────────────────────────────────────────────────────
        em = exact_match(ground_truth, generated) if not abstained else False

        # ── Semantic match via judge ───────────────────────────────────────────
        sem_correct   = False
        sem_reasoning = ""
        if not abstained:
            try:
                j = judge_model.generate_content(
                    SEMANTIC_JUDGE_PROMPT.format(
                        question=question,
                        ground_truth=ground_truth,
                        generated=generated,
                    ),
                    generation_config={"response_mime_type": "application/json"},
                )
                j_data        = json.loads(j.text)
                sem_correct   = j_data.get("correct", False)
                sem_reasoning = j_data.get("reasoning", "")
            except Exception as e:
                sem_reasoning = f"judge error: {e}"

        # ── Counters ───────────────────────────────────────────────────────────
        year_counts[item_fy] += 1
        metrics["total"] += 1
        if em:               metrics["exact_match"]    += 1
        if sem_correct:      metrics["semantic_match"] += 1
        if faithful is True: metrics["faithful"]       += 1
        if abstained:        metrics["abstained"]      += 1

        by_year[item_fy]["total"] += 1
        if em:          by_year[item_fy]["exact_match"]    += 1
        if sem_correct: by_year[item_fy]["semantic_match"] += 1

        status = "✓" if sem_correct else ("⊘" if abstained else "✗")
        print(f"  {status}  GT: {ground_truth[:50]}  |  Gen: {generated[:60]}")
        if sem_reasoning:
            print(f"     {sem_reasoning[:80]}")

        details.append({
            "question":       question,
            "ground_truth":   ground_truth,
            "fiscal_year":    item_fy,
            "chunk_type":     item.get("chunk_type"),
            "generated":      generated,
            "exact_match":    em,
            "semantic_match": sem_correct,
            "faithful":       faithful,
            "abstained":      abstained,
            "reasoning":      sem_reasoning,
            "top_ids":        [r["row"]["chunk_id"] for r in results],
            "mapped":         mapped,
        })

        time.sleep(0.3)

    n = metrics["total"]
    per_year = {}
    for fy in sorted(year_counts):
        nc = year_counts[fy]
        per_year[fy] = {
            "n":              nc,
            "exact_match":    round(by_year[fy]["exact_match"] / nc, 3),
            "semantic_match": round(by_year[fy]["semantic_match"] / nc, 3),
        }

    return {
        "n_questions":     n,
        "top_k":           top_k,
        "use_rerank":      use_rerank,
        "use_multi_query": use_multi_query,
        "use_map_reduce":  use_map_reduce,
        "year_filter":     year_filter,
        "exact_match":     round(metrics["exact_match"] / n, 3),
        "semantic_match":  round(metrics["semantic_match"] / n, 3),
        "faithful_rate":   round(metrics["faithful"] / n, 3),
        "abstention_rate": round(metrics["abstained"] / n, 3),
        "by_year":         per_year,
        "details":         details,
    }


def print_report(results: dict):
    n = results["n_questions"]
    print(f"\n{'─' * 60}")
    print(f"End-to-End Evaluation  ({n} questions)")
    print(f"  multi-query={results['use_multi_query']}  map-reduce={results['use_map_reduce']}")
    print(f"{'─' * 60}")
    print(f"  Exact match:      {results['exact_match']:.1%}")
    print(f"  Semantic match:   {results['semantic_match']:.1%}")
    print(f"  Faithful:         {results['faithful_rate']:.1%}")
    print(f"  Abstention rate:  {results['abstention_rate']:.1%}")

    print(f"\n  {'Year':<14} {'n':>3}  {'Exact':>6}  {'Semantic':>8}")
    print(f"  {'─'*14} {'─'*3}  {'─'*6}  {'─'*8}")
    for fy, d in results["by_year"].items():
        print(f"  {fy:<14} {d['n']:>3}  {d['exact_match']:>6.0%}  {d['semantic_match']:>8.0%}")

    failures = [d for d in results["details"] if not d["semantic_match"] and not d["abstained"]]
    print(f"\n  Failures: {len(failures)}/{n}")
    print(f"{'─' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="End-to-end RAG evaluation.")
    parser.add_argument("--top-k",          type=int, default=DEFAULT_K)
    parser.add_argument("--year",           help="Filter to one fiscal year")
    parser.add_argument("--no-rerank",      action="store_true", help="Skip cross-encoder reranking")
    parser.add_argument("--no-multi-query", action="store_true", help="Disable multi-query retrieval")
    parser.add_argument("--no-map-reduce",  action="store_true", help="Disable map-reduce generation")
    parser.add_argument("--power-search",   action="store_true", help="Stage 2: top_k×2 retrieval + ±3 neighbour expansion")
    parser.add_argument("--stage3",         action="store_true", help="Stage 3: HyDE + keyword BM25 + year-relaxed + full-page expansion (for all questions)")
    parser.add_argument("--escalate",       action="store_true", help="Auto-escalate to Stage 3 when Stage 1 abstains (mirrors production handler.py)")
    parser.add_argument("--eval-set",       help="Path to a custom eval set JSON (default: data/eval/eval_set.json)")
    parser.add_argument("--out",            help="Output JSON path (default: data/eval/e2e_results.json)")
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY not found in .env")
    genai.configure(api_key=api_key)

    eval_path = Path(args.eval_set) if args.eval_set else EVAL_PATH
    if not eval_path.exists():
        raise SystemExit(f"Eval set not found at {eval_path}. Run generate_eval.py first.")
    with open(eval_path) as f:
        eval_set = json.load(f)

    if args.year:
        eval_set = [q for q in eval_set if q.get("fiscal_year") == args.year]
        if not eval_set:
            raise SystemExit(f"No eval questions for {args.year}")

    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    answer_model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=ANSWER_SYSTEM,
    )
    judge_model = genai.GenerativeModel(model_name=GEMINI_MODEL)

    stage_label = "3" if args.stage3 else ("2" if args.power_search else "1")
    print(f"Eval set: {len(eval_set)} questions")
    print(f"Pipeline: stage={stage_label}  multi-query={not args.no_multi_query}  map-reduce={not args.no_map_reduce}")
    print("Loading embedding model and index…")
    em, tok = load_embedding_model()
    vecs, mfst = load_index()
    cks  = load_chunks()
    b25  = load_bm25(mfst, cks)
    print(f"Index: {len(mfst)} vectors\n")

    results = evaluate(
        eval_set=eval_set,
        vectors=vecs, manifest=mfst, chunks=cks, bm25=b25,
        answer_model=answer_model, judge_model=judge_model,
        embed_model=em, tokenizer=tok,
        top_k=args.top_k,
        year_filter=args.year,
        use_rerank=not args.no_rerank,
        use_multi_query=not args.no_multi_query,
        use_map_reduce=not args.no_map_reduce,
        power_search=args.power_search,
        use_stage3=args.stage3,
        auto_escalate=args.escalate,
    )
    results["search_stage"] = stage_label

    print_report(results)
    out_path = Path(args.out) if args.out else OUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Full results → {out_path}")


if __name__ == "__main__":
    main()
