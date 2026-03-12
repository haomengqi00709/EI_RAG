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
    expand_context,
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
VALID_FISCAL_YEARS = {"2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024"}

UNDERSTAND_PROMPT = """\
Extract search filters from this question about Canadian Employment Insurance reports.
Valid fiscal years are ONLY: 2019-2020, 2020-2021, 2021-2022, 2022-2023, 2023-2024.
Fiscal year abbreviations used in the reports map as follows:
FY1920 → 2019-2020, FY2021 → 2020-2021, FY2122 → 2021-2022, FY2223 → 2022-2023, FY2324 → 2023-2024.
If the question mentions a calendar year like "2019", map it to the fiscal year starting \
that year (e.g. "2019" → "2019-2020", "2020" → "2020-2021"). If unclear, use null.
Return ONLY valid JSON with these fields (use null if not clearly specified):
{{
  "fiscal_year": "one of the valid fiscal years above, or null",
  "chunk_type": "table or narrative or null"
}}
Question: {question}"""


def understand_query(question: str) -> dict:
    try:
        r = gemini.generate_content(
            UNDERSTAND_PROMPT.format(question=question),
            generation_config={"response_mime_type": "application/json", "temperature": 0},
        )
        result = {k: v for k, v in json.loads(r.text).items() if v}
        if "fiscal_year" in result and result["fiscal_year"] not in VALID_FISCAL_YEARS:
            print(f"  Invalid fiscal year '{result['fiscal_year']}' — ignoring")
            del result["fiscal_year"]
        return result
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
    image_b64    = inp.get("image", None) or None

    if not question:
        yield {"type": "result", "error": "No question provided"}
        return

    # Auto-detect fiscal year if not supplied
    yield {"type": "progress", "message": "Understanding your question\u2026"}
    filters     = understand_query(question)
    fy_filter   = fiscal_year or filters.get("fiscal_year", "")
    strict_year = bool(fiscal_year)  # skip year-relaxed in Stage 3 when caller specified a year

    # Deep search (power_search=True): run Stage 3 directly for all questions
    # Normal search: Stage 1, then auto-escalate to Stage 3 if abstained
    if power_search:
        yield {"type": "progress", "message": "Running deep search (Stage 3) \u2014 HyDE + BM25 + semantic\u2026"}
        results = retrieve_stage3(
            question=question,
            vectors=vectors, bm25=bm25,
            manifest=manifest, chunks=chunks,
            llm_model=gemini, top_k=10,
            fiscal_year=fy_filter,
            model=embed_model, tokenizer=tokenizer,
            strict_year=strict_year,
        )
        results = _deduplicate(results)
        yield {"type": "progress", "message": f"Found {len(results)} relevant chunks \u2014 generating answer\u2026"}
        gen = generate_answer_map_reduce(
            question, results, gemini, gemini,
            chunks=chunks, neighbor_radius=3,
            image_b64=image_b64,
        )
        search_stage = 3
    else:
        yield {"type": "progress", "message": "Generating search variants\u2026"}
        variants = generate_query_variants(question, gemini, n=3, image_b64=image_b64)
        yield {"type": "progress", "message": f"Searching {len(manifest):,} document chunks (Stage 1)\u2026"}
        results  = retrieve_multi_query(
            questions=variants,
            vectors=vectors, bm25=bm25,
            manifest=manifest, chunks=chunks,
            top_k=5, fiscal_year=fy_filter,
            model=embed_model, tokenizer=tokenizer,
            use_rerank=True,
        )
        results = expand_context(results, chunks)
        results = _deduplicate(results)
        results = results[:5]
        yield {"type": "progress", "message": f"Found {len(results)} relevant chunks \u2014 generating answer\u2026"}
        gen = generate_answer_map_reduce(
            question, results, gemini, gemini,
            chunks=chunks, neighbor_radius=1,
            image_b64=image_b64,
        )
        search_stage = 1

        # Stage 3 auto-escalation when Stage 1 abstains
        if gen["abstained"]:
            yield {"type": "progress", "message": "Stage 1 found insufficient context \u2014 escalating to deep search (Stage 3)\u2026"}
            results_s3 = retrieve_stage3(
                question=question,
                vectors=vectors, bm25=bm25,
                manifest=manifest, chunks=chunks,
                llm_model=gemini, top_k=10,
                fiscal_year=fy_filter,
                model=embed_model, tokenizer=tokenizer,
                strict_year=strict_year,
            )
            if results_s3:
                results_s3 = _deduplicate(results_s3)
                yield {"type": "progress", "message": f"Found {len(results_s3)} chunks from deep search \u2014 generating answer\u2026"}
                gen_s3 = generate_answer_map_reduce(
                    question, results_s3, gemini, gemini,
                    chunks=chunks, neighbor_radius=3,
                    image_b64=image_b64,
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

    yield {
        "type":             "result",
        "question":         question,
        "answer":           gen["answer"],
        "abstained":        gen["abstained"],
        "faithful":         gen["faithful"],
        "search_stage":     search_stage,
        "filters_detected": filters,
        "sources":          sources,
    }


runpod.serverless.start({"handler": handler})
