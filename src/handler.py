#!/usr/bin/env python3
"""
handler.py — RunPod Serverless handler for the EI MAR RAG pipeline.

Input (job["input"]):
  question     str   — the user question (required)
  fiscal_year  str   — e.g. "2023-2024" (optional, auto-detected if omitted)
  power_search bool  — Stage 2 deep search (optional, default false)

Output:
  question, answer, abstained, faithful, search_stage, filters_detected, sources

Models and data load ONCE at worker startup and stay warm across requests.
"""

import json
import os
import sys
from pathlib import Path

# ── Volume + env setup (before any local imports) ─────────────────────────────
VOLUME  = Path(os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume"))
WORKDIR = VOLUME / "RAG_experiment2"
os.chdir(WORKDIR)

# HF model cache on volume → persists across cold starts
os.environ["HF_HOME"]            = str(VOLUME / "huggingface")
os.environ["TRANSFORMERS_CACHE"] = str(VOLUME / "huggingface")

sys.path.insert(0, str(Path(__file__).parent))

import google.generativeai as genai
import runpod

from embed import load_embedding_model
from generate import generate_answer_map_reduce, generate_query_variants
from retrieve import (
    _deduplicate,
    expand_full_page,
    load_bm25,
    load_chunks,
    load_index,
    retrieve_multi_query,
    retrieve_stage3,
)

# ── Gemini ─────────────────────────────────────────────────────────────────────
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise SystemExit("GOOGLE_API_KEY environment variable not set")
genai.configure(api_key=api_key)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
gemini = genai.GenerativeModel(model_name=GEMINI_MODEL)

# ── Query understanding ────────────────────────────────────────────────────────
UNDERSTAND_PROMPT = """\
Extract search filters from this question about Canadian Employment Insurance reports.
Return ONLY valid JSON with these fields (use null if not clearly specified):
{{
  "fiscal_year": "YYYY-YYYY format or null",
  "chunk_type": "table or narrative or null"
}}
Question: {question}"""


def understand_query(question: str) -> dict:
    try:
        r = gemini.generate_content(
            UNDERSTAND_PROMPT.format(question=question),
            generation_config={"response_mime_type": "application/json"},
        )
        return {k: v for k, v in json.loads(r.text).items() if v}
    except Exception:
        return {}


# ── Load everything once at worker startup ─────────────────────────────────────
print("Loading embedding model…")
embed_model, tokenizer = load_embedding_model()

print("Loading vector index…")
vectors, manifest = load_index()

print("Loading chunks…")
chunks = load_chunks()

print("Loading BM25 index…")
bm25 = load_bm25(manifest, chunks)

print(f"Ready — {len(manifest)} vectors indexed.\n")


# ── Handler ────────────────────────────────────────────────────────────────────
def handler(job):
    inp          = job.get("input", {})
    question     = inp.get("question", "").strip()
    fiscal_year  = inp.get("fiscal_year", "")
    power_search = bool(inp.get("power_search", False))

    if not question:
        return {"error": "No question provided"}

    # Auto-detect fiscal year if not supplied
    filters   = understand_query(question)
    fy_filter = fiscal_year or filters.get("fiscal_year", "")

    # Retrieval — Stage 1 (default) or Stage 2 (power_search)
    top_k           = 10 if power_search else 5
    neighbor_radius = 3  if power_search else 1

    variants = generate_query_variants(question, gemini, n=3)
    results  = retrieve_multi_query(
        questions=variants,
        vectors=vectors, bm25=bm25,
        manifest=manifest, chunks=chunks,
        top_k=top_k, fiscal_year=fy_filter,
        model=embed_model, tokenizer=tokenizer,
        use_rerank=True,
    )

    # Generation
    gen          = generate_answer_map_reduce(
        question, results, gemini, gemini,
        chunks=chunks, neighbor_radius=neighbor_radius,
    )
    search_stage = 2 if power_search else 1

    # Stage 3 auto-escalation when abstained
    if gen["abstained"]:
        print(f"  Stage {search_stage} abstained → escalating to Stage 3…")
        results_s3 = retrieve_stage3(
            question=question,
            vectors=vectors, bm25=bm25,
            manifest=manifest, chunks=chunks,
            llm_model=gemini, top_k=10,
            fiscal_year=fy_filter,
            model=embed_model, tokenizer=tokenizer,
        )
        if results_s3:
            results_s3 = _deduplicate(results_s3)
            gen_s3 = generate_answer_map_reduce(
                question, results_s3, gemini, gemini,
                chunks=chunks, neighbor_radius=3,
            )
            if not gen_s3["abstained"]:
                gen     = gen_s3
                results = results_s3
        search_stage = 3

    # Build sources list
    signal_by_rank = {m["rank"]: m["signal"] for m in gen.get("mapped", [])}
    sources = []
    for idx, r in enumerate(results):
        row   = r["row"]
        chunk = r["chunk"]
        body  = chunk.get("text_nl") or chunk.get("text", "")
        sources.append({
            "fiscal_year":        row.get("fiscal_year", ""),
            "chunk_type":         row["chunk_type"],
            "page":               row.get("page"),
            "section_breadcrumb": row.get("section_breadcrumb", ""),
            "table_title":        row.get("table_title", ""),
            "score":              round(float(r["score"]), 4),
            "snippet":            body[:300],
            "full_text":          chunk.get("text", ""),
            "signal":             signal_by_rank.get(idx + 1, "UNKNOWN"),
        })

    return {
        "question":         question,
        "answer":           gen["answer"],
        "abstained":        gen["abstained"],
        "faithful":         gen["faithful"],
        "search_stage":     search_stage,
        "filters_detected": filters,
        "sources":          sources,
    }


runpod.serverless.start({"handler": handler})
