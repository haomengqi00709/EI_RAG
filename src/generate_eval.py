#!/usr/bin/env python3
"""
generate_eval.py — Generate a retrieval evaluation set using the Gemini API.

Samples chunks from data/chunked/*.jsonl (stratified by chunk type and fiscal
year), calls Gemini to generate one question + answer pair per chunk, and saves
the result to data/eval/eval_set.json.

Each entry records the source chunk_id so retrieval can be evaluated exactly:
did retrieve.py return the correct chunk for this question?

Run:
    pip install google-generativeai python-dotenv
    python src/generate_eval.py
    python src/generate_eval.py --samples-per-group 5   # more questions

Reads GOOGLE_API_KEY and GEMINI_MODEL from the .env file in the project root.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_ID          = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
SAMPLES_PER_GROUP = 3       # chunks sampled per (fiscal_year × chunk_type) cell
                            # 5 years × 2 types × 3 = 30 questions by default
MIN_CHUNK_CHARS   = 200     # skip very short chunks — not enough content for a question
CHUNK_TYPES       = ["narrative", "table"]
FISCAL_YEARS      = ["2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024"]

OUT_PATH = Path("data/eval/eval_set.json")

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are building an evaluation set for a RAG system over Canadian government "
    "Employment Insurance (EI) Monitoring and Assessment Reports. "
    "Given a chunk from one of these reports, generate ONE realistic question a "
    "policy analyst might ask, that can be answered precisely using ONLY the "
    "information in the chunk. The answer must be specific — a number, percentage, "
    "year, program name, or concrete policy detail. Avoid vague or open-ended questions."
)

USER_TEMPLATE = """\
Fiscal year: {fiscal_year}
Section: {section_breadcrumb}
Chunk type: {chunk_type}

Content:
{text}

Return ONLY valid JSON with exactly these three fields:
{{
  "question": "the question a policy analyst would ask",
  "answer": "the precise answer drawn from the content above",
  "answer_quote": "the exact sentence or phrase from the content that contains the answer"
}}"""


# ── Chunk loading & sampling ───────────────────────────────────────────────────

def load_all_chunks() -> list:
    chunks = []
    for f in sorted(Path("data/chunked").glob("*.jsonl")):
        for line in open(f):
            chunks.append(json.loads(line))
    return chunks


def stratified_sample(chunks: list, samples_per_group: int, seed: int) -> list:
    """
    Sample chunks evenly across fiscal_year × chunk_type cells.
    Ensures the eval set covers all years and both chunk types.
    """
    rng = random.Random(seed)
    selected = []
    for fy in FISCAL_YEARS:
        for ct in CHUNK_TYPES:
            group = [
                c for c in chunks
                if c["chunk_type"] == ct
                and c.get("fiscal_year") == fy
                and c.get("char_count", 0) >= MIN_CHUNK_CHARS
            ]
            if not group:
                print(f"  Warning: no chunks for {fy} / {ct}")
                continue
            pick = rng.sample(group, min(samples_per_group, len(group)))
            selected.extend(pick)
    return selected


# ── Gemini call ───────────────────────────────────────────────────────────────

def generate_qa(model, chunk: dict) -> dict | None:
    """Call Gemini to generate a Q&A pair for a single chunk. Returns None on error."""
    # For table chunks prefer text_nl — cleaner prose for the LLM than raw markdown
    if chunk["chunk_type"] == "table":
        text = chunk.get("text_nl") or chunk.get("text", "")
    else:
        text = chunk.get("text", "")

    prompt = USER_TEMPLATE.format(
        fiscal_year=chunk.get("fiscal_year", "Unknown"),
        section_breadcrumb=chunk.get("section_breadcrumb", "Unknown section"),
        chunk_type=chunk["chunk_type"],
        text=text[:3000],   # cap to stay well within token limits
    )

    try:
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
        )
        qa = json.loads(response.text)

        return {
            "chunk_id":           chunk["chunk_id"],
            "chunk_type":         chunk["chunk_type"],
            "fiscal_year":        chunk.get("fiscal_year", ""),
            "section_breadcrumb": chunk.get("section_breadcrumb", ""),
            "page":               chunk.get("page_number") or chunk.get("start_page"),
            "question":           qa.get("question", "").strip(),
            "answer":             qa.get("answer", "").strip(),
            "answer_quote":       qa.get("answer_quote", "").strip(),
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        return None


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate a RAG evaluation set using Gemini."
    )
    parser.add_argument(
        "--samples-per-group", type=int, default=SAMPLES_PER_GROUP,
        help=f"Chunks to sample per (fiscal_year × chunk_type) cell (default {SAMPLES_PER_GROUP})"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling (default 42)"
    )
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: GOOGLE_API_KEY not found.\n"
                         "Make sure it is set in the .env file at the project root.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=MODEL_ID,
        system_instruction=SYSTEM_PROMPT,
    )

    print("Loading chunks…")
    all_chunks = load_all_chunks()
    n_narrative = sum(1 for c in all_chunks if c["chunk_type"] == "narrative")
    n_table     = sum(1 for c in all_chunks if c["chunk_type"] == "table")
    print(f"  {len(all_chunks)} total chunks (narrative={n_narrative}, table={n_table})\n")

    sample = stratified_sample(all_chunks, args.samples_per_group, args.seed)
    n_groups = len(FISCAL_YEARS) * len(CHUNK_TYPES)
    print(f"Sampled {len(sample)} chunks "
          f"({args.samples_per_group} per group × {n_groups} groups)\n")

    results = []
    errors  = 0

    for i, chunk in enumerate(sample, 1):
        label = f"{chunk.get('fiscal_year')} | {chunk['chunk_type']} | {chunk['chunk_id']}"
        print(f"[{i}/{len(sample)}] {label}")

        qa = generate_qa(model, chunk)

        if qa and qa["question"] and qa["answer"]:
            results.append(qa)
            print(f"  Q: {qa['question'][:90]}")
            print(f"  A: {qa['answer'][:90]}")
        else:
            errors += 1
            print("  Skipped (empty or error)")

        time.sleep(0.5)     # gentle rate limiting

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(results)} Q&A pairs → {OUT_PATH}")
    if errors:
        print(f"Errors / skipped: {errors}")
    print(f"\nNext step: run src/evaluate_retrieval.py to measure Recall@5 "
          f"against your current retrieval pipeline.")


if __name__ == "__main__":
    main()
