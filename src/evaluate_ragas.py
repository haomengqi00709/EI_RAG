#!/usr/bin/env python3
"""
evaluate_ragas.py — RAGAs framework evaluation.

Measures four standard RAG quality dimensions using the ragas library:
  context_precision   — are retrieved chunks relevant to the question?
  context_recall      — do retrieved chunks cover all info needed to answer?
  answer_faithfulness — is the generated answer grounded in the chunks?
  answer_relevancy    — does the generated answer actually address the question?

Install: pip install ragas

Run:
    python src/evaluate_ragas.py
    python src/evaluate_ragas.py --year 2023-2024 --samples 20
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent))

from embed import load_embedding_model
from retrieve import (
    expand_context,
    load_bm25,
    load_chunks,
    load_index,
    rerank,
    retrieve_hybrid,
)

EVAL_PATH = Path("data/eval/eval_set.json")
OUT_PATH  = Path("data/eval/ragas_results.json")
DEFAULT_K = 5

ANSWER_SYSTEM = (
    "You are an expert on Canadian Employment Insurance (EI) policy. "
    "Answer questions using ONLY the provided document excerpts. "
    "If the excerpts do not contain enough information, say: INSUFFICIENT_CONTEXT"
)

ANSWER_TEMPLATE = """\
Question: {question}

Document excerpts:
{context}

Answer concisely using only the excerpts."""


def build_context_texts(results: list) -> list[str]:
    """Return list of chunk texts for ragas (one string per chunk)."""
    texts = []
    for r in results:
        body = r["chunk"].get("text_nl") or r["chunk"].get("text", "")
        texts.append(body[:2000])
        for rc in r.get("context_chunks", []):
            rbody = rc.get("text_nl") or rc.get("text", "")
            texts.append(rbody[:500])
    return texts


def build_context_str(results: list) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        row  = r["row"]
        body = r["chunk"].get("text_nl") or r["chunk"].get("text", "")
        fy   = row.get("fiscal_year", "")
        bc   = row.get("section_breadcrumb", "")[:60]
        parts.append(f"[{i}] FY {fy} | {bc}\n{body[:1500]}")
    return "\n\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="RAGAs evaluation.")
    parser.add_argument("--top-k",   type=int, default=DEFAULT_K)
    parser.add_argument("--year",    help="Filter to one fiscal year")
    parser.add_argument("--samples", type=int, default=None,
                        help="Limit to N random samples (for faster runs)")
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            answer_faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
    except ImportError:
        raise SystemExit(
            "ragas not installed. Run: pip install ragas datasets"
        )

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY not found in .env")
    genai.configure(api_key=api_key)
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    answer_model = genai.GenerativeModel(
        model_name=GEMINI_MODEL, system_instruction=ANSWER_SYSTEM
    )

    if not EVAL_PATH.exists():
        raise SystemExit(f"Eval set not found at {EVAL_PATH}. Run generate_eval.py first.")
    with open(EVAL_PATH) as f:
        eval_set = json.load(f)

    if args.year:
        eval_set = [q for q in eval_set if q.get("fiscal_year") == args.year]
    if args.samples:
        import random
        random.shuffle(eval_set)
        eval_set = eval_set[:args.samples]

    print(f"Evaluating {len(eval_set)} questions with RAGAs…")
    print("Loading retrieval stack…")
    em, tok   = load_embedding_model()
    vecs, mfst = load_index()
    cks       = load_chunks()
    b25       = load_bm25(mfst, cks)
    print(f"Index: {len(mfst)} vectors\n")

    ragas_rows = []

    for i, item in enumerate(eval_set, 1):
        question     = item["question"]
        ground_truth = item["answer"]
        item_fy      = item.get("fiscal_year", "")

        print(f"[{i}/{len(eval_set)}] {item_fy} — {question[:70]}")

        results = retrieve_hybrid(
            question=question, vectors=vecs, bm25=b25,
            manifest=mfst, chunks=cks, top_k=args.top_k * 3,
            fiscal_year=item_fy,
            model=em, tokenizer=tok,
        )
        results = expand_context(results, cks)
        if not args.no_rerank:
            results = rerank(question, results)
        results = results[:args.top_k]

        contexts = build_context_texts(results)
        context_str = build_context_str(results)

        try:
            resp = answer_model.generate_content(
                ANSWER_TEMPLATE.format(question=question, context=context_str)
            )
            answer = resp.text.strip()
        except Exception as e:
            answer = f"ERROR: {e}"

        if "INSUFFICIENT_CONTEXT" in answer:
            answer = ""

        ragas_rows.append({
            "question":   question,
            "answer":     answer,
            "contexts":   contexts,
            "ground_truth": ground_truth,
        })

        time.sleep(0.5)

    # Build ragas Dataset and run evaluation
    dataset = Dataset.from_list(ragas_rows)

    print("\nRunning RAGAs scoring…")
    score = ragas_evaluate(
        dataset,
        metrics=[
            context_precision,
            context_recall,
            answer_faithfulness,
            answer_relevancy,
        ],
    )

    score_dict = score.to_pandas().mean().to_dict()

    print(f"\n{'─' * 50}")
    print("RAGAs Results:")
    print(f"  Context Precision:    {score_dict.get('context_precision', 0):.3f}")
    print(f"  Context Recall:       {score_dict.get('context_recall', 0):.3f}")
    print(f"  Answer Faithfulness:  {score_dict.get('answer_faithfulness', 0):.3f}")
    print(f"  Answer Relevancy:     {score_dict.get('answer_relevancy', 0):.3f}")
    print(f"{'─' * 50}\n")

    output = {
        "n_questions":          len(eval_set),
        "top_k":                args.top_k,
        "context_precision":    score_dict.get("context_precision"),
        "context_recall":       score_dict.get("context_recall"),
        "answer_faithfulness":  score_dict.get("answer_faithfulness"),
        "answer_relevancy":     score_dict.get("answer_relevancy"),
        "rows":                 ragas_rows,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Full results → {OUT_PATH}")


if __name__ == "__main__":
    main()
