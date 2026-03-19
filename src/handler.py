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
WORKDIR = VOLUME  # data/ lives directly under the volume root
os.chdir(WORKDIR)
print(f"[DEBUG] VOLUME={VOLUME}  CWD={os.getcwd()}")
print(f"[DEBUG] Volume contents: {sorted(str(p.name) for p in VOLUME.iterdir()) if VOLUME.exists() else 'MISSING'}")
data_path = VOLUME / "data" / "embeddings"
print(f"[DEBUG] data/embeddings exists: {data_path.exists()}")

# HF model cache on volume → persists across cold starts
os.environ["HF_HOME"]            = str(VOLUME / "hf-cache")
os.environ["TRANSFORMERS_CACHE"] = str(VOLUME / "hf-cache")

sys.path.insert(0, str(Path(__file__).parent))

import google.generativeai as genai
import runpod

from embed import load_embedding_model
from generate import generate_answer_map_reduce, generate_query_variants
from retrieve import (
    _deduplicate,
    _load_qwen_reranker,
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

print("Loading reranker…")
_load_qwen_reranker()

print(f"Ready — {len(manifest)} vectors indexed.\n")


# ── Handler ────────────────────────────────────────────────────────────────────
def handler(job):
    import time
    t0 = time.time()
    def elapsed(): return f"{time.time() - t0:.1f}s"

    inp          = job.get("input", {})
    question     = inp.get("question", "").strip()
    fiscal_year  = inp.get("fiscal_year", "")
    power_search = bool(inp.get("power_search", False))
    image_b64    = inp.get("image", None) or None

    if inp.get("ping"):
        return {"pong": True}

    if not question:
        return {"error": "No question provided"}

    print(f"[{elapsed()}] understand_query start")
    filters     = understand_query(question)
    fy_filter   = fiscal_year or filters.get("fiscal_year", "")
    strict_year = bool(fiscal_year)
    print(f"[{elapsed()}] understand_query done — filters={filters}")

    if power_search:
        print(f"[{elapsed()}] retrieve_stage3 start (power_search)")
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
        print(f"[{elapsed()}] retrieve_stage3 done — {len(results)} results")
        print(f"[{elapsed()}] generate start")
        gen = generate_answer_map_reduce(
            question, results, gemini, gemini,
            chunks=chunks, neighbor_radius=3,
            image_b64=image_b64,
        )
        print(f"[{elapsed()}] generate done — abstained={gen['abstained']}")
        search_stage = 3
    else:
        print(f"[{elapsed()}] generate_query_variants start")
        variants = generate_query_variants(question, gemini, n=3, image_b64=image_b64)
        print(f"[{elapsed()}] generate_query_variants done — {len(variants)} variants")
        print(f"[{elapsed()}] retrieve_multi_query start")
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
        print(f"[{elapsed()}] retrieve_multi_query done — {len(results)} results")
        print(f"[{elapsed()}] generate start (stage 1)")
        gen = generate_answer_map_reduce(
            question, results, gemini, gemini,
            chunks=chunks, neighbor_radius=1,
            image_b64=image_b64,
        )
        print(f"[{elapsed()}] generate done — abstained={gen['abstained']}")
        search_stage = 1

        if gen["abstained"]:
            print(f"[{elapsed()}] Stage 1 abstained → escalating to Stage 3…")
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
                print(f"[{elapsed()}] stage3 retrieve done — {len(results_s3)} results")
                print(f"[{elapsed()}] generate start (stage 3)")
                gen_s3 = generate_answer_map_reduce(
                    question, results_s3, gemini, gemini,
                    chunks=chunks, neighbor_radius=3,
                    image_b64=image_b64,
                )
                print(f"[{elapsed()}] generate done (stage 3) — abstained={gen_s3['abstained']}")
                if not gen_s3["abstained"]:
                    gen     = gen_s3
                    results = results_s3
            search_stage = 3

    print(f"[{elapsed()}] total job time")

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
