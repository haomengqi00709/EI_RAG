#!/usr/bin/env python3
"""
server.py — Flask backend for the EI MAR RAG pipeline.

Endpoints:
  GET  /          — serve index.html
  POST /search    — retrieve top-k chunks (no generation)
  POST /answer    — full pipeline: query understanding → retrieval →
                    reranking → context expansion → generation with
                    citations, faithfulness check, and abstention
  POST /feedback  — log user thumbs up/down for future fine-tuning

Improvements applied:
  #8  — Query understanding Stage 1 LLM (Gemini extracts filters)
  #14 — Contextual compression before generation
  #17 — Answer with inline citations [FY | page | section]
  #18 — Faithfulness check (second LLM call verifies grounding)
  #19 — Confidence / abstention when chunks are insufficient
  #22 — User feedback loop (POST /feedback logs to JSONL)

Run:
    python src/server.py
    Opens on http://localhost:5000
"""

import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_file, send_from_directory

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))

from embed import load_embedding_model
from generate import generate_answer_map_reduce, generate_query_variants
from retrieve import (
    _deduplicate,
    _lost_in_middle_reorder,
    expand_context,
    expand_full_page,
    load_bm25,
    load_chunks,
    load_index,
    rerank,
    retrieve_hybrid,
    retrieve_multi_query,
    retrieve_stage3,
)

# ── Configuration ──────────────────────────────────────────────────────────────

SRC_DIR      = Path(__file__).parent
DOCS_DIR     = Path("Documents")
FEEDBACK_DIR = Path("data/feedback")
FEEDBACK_PATH = FEEDBACK_DIR / "feedback.jsonl"

# Fiscal year → PDF filename mapping
PDF_MAP = {
    "2019-2020": DOCS_DIR / "2019-20-EI-MAR-EN.pdf",
    "2020-2021": DOCS_DIR / "2020-2021_EI_MAR-EN.pdf",
    "2021-2022": DOCS_DIR / "2021-2022_EI_MAR-EN.pdf",
    "2022-2023": DOCS_DIR / "2022-2023-EI_MAR-EN.pdf",
    "2023-2024": DOCS_DIR / "2023-2024-EI-MAR-EN.pdf",
}

# Whether any PDFs are available locally (false on RunPod where PDFs aren't uploaded)
PDFS_AVAILABLE = any(p.exists() for p in PDF_MAP.values())

# In-memory page image cache: (fiscal_year, page_number) → PNG bytes
_page_cache: dict = {}

# ── Model setup ────────────────────────────────────────────────────────────────
# Switch between Gemini (cloud) and local Qwen via ANSWER_MODEL in .env:
#   ANSWER_MODEL=gemini   → Gemini 2.0 Flash  (default)
#   ANSWER_MODEL=local    → Qwen/Qwen3-4B-MLX-4bit  (local, no API key needed)

ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gemini").strip().lower()

if ANSWER_MODEL == "local":
    from local_model import LocalMLXModel
    LOCAL_MODEL_ID = os.getenv("LOCAL_MODEL_ID", "Qwen/Qwen3-4B-MLX-4bit")
    gemini = LocalMLXModel(model_id=LOCAL_MODEL_ID, max_tokens=768)
    print(f"Answer model: LOCAL  ({LOCAL_MODEL_ID})")
else:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY not found in .env")
    genai.configure(api_key=api_key)
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    gemini = genai.GenerativeModel(model_name=GEMINI_MODEL)
    print(f"Answer model: GEMINI ({GEMINI_MODEL})")

# ── Startup: load retrieval stack once ────────────────────────────────────────

print("Loading embedding model…")
embed_model, tokenizer = load_embedding_model()

print("Loading vector index…")
vectors, manifest = load_index()

print("Loading chunks…")
chunks = load_chunks()

print("Loading BM25 index…")
bm25 = load_bm25(manifest, chunks)

print(f"\nReady — {len(manifest)} vectors indexed.\n")

# ── Flask app ──────────────────────────────────────────────────────────────────

app = Flask(__name__)


# ── #8: Query understanding ────────────────────────────────────────────────────

VALID_FISCAL_YEARS = {"2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024"}

UNDERSTAND_PROMPT = """\
Extract search filters from this question about Canadian Employment Insurance reports.
Valid fiscal years are ONLY: 2019-2020, 2020-2021, 2021-2022, 2022-2023, 2023-2024.
If the question mentions a calendar year like "2019", map it to the fiscal year starting \
that year (e.g. "2019" → "2019-2020", "2020" → "2020-2021"). If unclear, use null.
Return ONLY valid JSON with these fields (use null if not clearly specified):
{{
  "fiscal_year": "one of the valid fiscal years above, or null",
  "chunk_type": "table or narrative or null",
  "province": "full province name or null",
  "time_period": "e.g. first half, Q3, or null"
}}

Question: {question}"""


def understand_query(question: str) -> dict:
    """
    #8: Stage 1 LLM — extract structured filters from the natural language query.
    Returns dict with fiscal_year, chunk_type, province, time_period (any may be null).
    """
    try:
        response = gemini.generate_content(
            UNDERSTAND_PROMPT.format(question=question),
            generation_config={"response_mime_type": "application/json", "temperature": 0},
        )
        filters = json.loads(response.text)
        result = {k: v for k, v in filters.items() if v}
        if "fiscal_year" in result and result["fiscal_year"] not in VALID_FISCAL_YEARS:
            print(f"  Invalid fiscal year '{result['fiscal_year']}' — ignoring")
            del result["fiscal_year"]
        return result
    except Exception as e:
        print(f"  Query understanding failed: {e}")
        return {}


# ── #14: Contextual compression ────────────────────────────────────────────────

COMPRESS_PROMPT = """\
Extract only the sentences directly relevant to the question below from the passage.
Return just the extracted sentences, nothing else. If no sentences are relevant, return "NONE".

Question: {question}
Passage: {passage}"""


def compress_chunk(question: str, passage: str) -> str:
    """
    #14: Extract only the relevant sentences from a passage.
    Falls back to first 1000 chars if the call fails.
    """
    try:
        response = gemini.generate_content(
            COMPRESS_PROMPT.format(question=question, passage=passage[:3000])
        )
        compressed = response.text.strip()
        return compressed if compressed != "NONE" else passage[:1000]
    except Exception:
        return passage[:1000]


# ── #17/#18/#19: Generation with citations, faithfulness, abstention ───────────

ANSWER_SYSTEM = (
    "You are an expert on Canadian Employment Insurance (EI) policy. "
    "Answer questions using ONLY the provided document excerpts. "
    "After your answer, cite each source as [FY YYYY-YYYY | p.N | Section]. "
    "If the excerpts do not contain enough information to answer confidently, "
    "respond with exactly: INSUFFICIENT_CONTEXT"   # #19 abstention signal
)

ANSWER_TEMPLATE = """\
Question: {question}

Document excerpts:
{context}

Answer the question using only the excerpts above. Cite your sources."""

FAITHFULNESS_PROMPT = """\
Question: {question}
Answer: {answer}
Source excerpts:
{context}

Is every factual claim in the answer directly supported by the source excerpts?
Return ONLY valid JSON: {{"faithful": true or false, "issues": "brief explanation or null"}}"""


def build_context_text(results: list, question: str) -> str:
    """Format retrieved (and optionally compressed) chunks for the generation prompt."""
    parts = []
    for i, r in enumerate(results, 1):
        row   = r["row"]
        chunk = r["chunk"]
        body  = chunk.get("text_nl") or chunk.get("text", "")
        fy    = row.get("fiscal_year", "")
        page  = row.get("page", "")
        bc    = row.get("section_breadcrumb", "")[:60]
        # Short chunks: show in full so the LLM never misses a value near the end
        snippet = body if len(body) <= 2000 else body[:1500]
        parts.append(f"[{i}] FY {fy} | p.{page} | {bc}\n{snippet}")

        # Also include related chunks (parent-child expansion #3)
        for rc in r.get("context_chunks", []):
            rbody = rc.get("text_nl") or rc.get("text", "")
            rbc   = rc.get("section_breadcrumb", "")[:60]
            parts.append(f"    (related) {rbc}\n{rbody[:600]}")

    return "\n\n".join(parts)


def generate_answer(question: str, results: list, compress: bool = False) -> dict:
    """
    Full generation pipeline:
      #17 — answer with inline citations
      #18 — faithfulness check
      #19 — abstention on INSUFFICIENT_CONTEXT
    """
    if compress:
        for r in results:
            body = r["chunk"].get("text_nl") or r["chunk"].get("text", "")
            r["chunk"]["_compressed"] = compress_chunk(question, body)

    context = build_context_text(results, question)

    # Stage 2: generate answer
    answer_model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=ANSWER_SYSTEM,
    )
    try:
        response = answer_model.generate_content(
            ANSWER_TEMPLATE.format(question=question, context=context)
        )
        answer = response.text.strip()
    except Exception as e:
        return {"answer": f"Generation error: {e}", "faithful": None, "abstained": False}

    # #19: abstention
    if "INSUFFICIENT_CONTEXT" in answer:
        return {
            "answer": "The retrieved documents do not contain enough information to answer this question confidently.",
            "faithful": None,
            "abstained": True,
        }

    # #18: faithfulness check
    faithful = None
    faithfulness_issues = None
    try:
        f_response = gemini.generate_content(
            FAITHFULNESS_PROMPT.format(
                question=question, answer=answer, context=context[:4000]
            ),
            generation_config={"response_mime_type": "application/json"},
        )
        f_data = json.loads(f_response.text)
        faithful = f_data.get("faithful")
        faithfulness_issues = f_data.get("issues")
    except Exception:
        pass

    return {
        "answer":              answer,
        "faithful":            faithful,
        "faithfulness_issues": faithfulness_issues,
        "abstained":           False,
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(SRC_DIR, "index.html")


@app.route("/search", methods=["POST"])
def search():
    """Raw retrieval — returns ranked chunks without generation."""
    data = request.get_json(force=True)
    question = (data.get("query") or "").strip()
    if not question:
        return jsonify({"error": "Empty query"}), 400

    top_k       = max(1, min(int(data.get("top_k", 5)), 20))
    fiscal_year = data.get("fiscal_year") or None
    chunk_type  = data.get("chunk_type") or None
    use_rerank  = bool(data.get("rerank", False))

    results = retrieve_hybrid(
        question=question, vectors=vectors, bm25=bm25,
        manifest=manifest, chunks=chunks, top_k=top_k,
        fiscal_year=fiscal_year, chunk_type=chunk_type,
        model=embed_model, tokenizer=tokenizer,
    )

    if use_rerank:
        results = rerank(question, results)[:top_k]

    out = []
    for r in results:
        row   = r["row"]
        chunk = r["chunk"]
        ctype = row["chunk_type"]
        snippet_src = chunk.get("text_nl") or chunk.get("text", "") if ctype == "table" else chunk.get("text", "")

        out.append({
            "score":              round(float(r["score"]), 4),
            "fiscal_year":        row.get("fiscal_year", ""),
            "chunk_type":         ctype,
            "page":               row.get("page"),
            "section_breadcrumb": row.get("section_breadcrumb", ""),
            "table_title":        row.get("table_title", ""),
            "snippet":            snippet_src[:500],
            "full_text":          chunk.get("text", ""),
            "full_nl":            chunk.get("text_nl", ""),
        })

    return jsonify({"results": out})


@app.route("/answer", methods=["POST"])
def answer():
    """
    Full RAG pipeline (v2):
      #8   query understanding → extract filters
      #MQ  multi-query → retrieve for each variant, merge, rerank
      #MAP map phase   → each chunk answered independently (parallel)
      #RED reduce phase → aggregator synthesises the best answer
    """
    data = request.get_json(force=True)
    question = (data.get("query") or "").strip()
    if not question:
        return jsonify({"error": "Empty query"}), 400

    top_k        = max(1, min(int(data.get("top_k", 5)), 20))
    use_rerank   = bool(data.get("rerank", True))
    power_search = bool(data.get("power_search", False))

    # Optional clarification from a previous turn
    clarification  = (data.get("clarification") or "").strip()
    effective_question = (
        f"{question}\n\nUser clarification: {clarification}"
        if clarification else question
    )

    # #8: Query understanding — extract fiscal_year, chunk_type, province
    filters      = understand_query(effective_question)
    user_fy      = data.get("fiscal_year")   # explicitly set by user in UI
    fiscal_year  = user_fy or filters.get("fiscal_year")
    strict_year  = bool(user_fy)             # skip year-relaxed in Stage 3 when user picked a year
    chunk_type   = data.get("chunk_type")  or filters.get("chunk_type")
    province    = filters.get("province", "")
    base_q      = f"{effective_question} {province}".strip() if province else effective_question

    # Deep search (power_search=True): run Stage 3 directly for all questions
    # Normal search: Stage 1, then auto-escalate to Stage 3 if abstained
    if power_search:
        results = retrieve_stage3(
            question=effective_question,
            vectors=vectors, bm25=bm25,
            manifest=manifest, chunks=chunks,
            llm_model=gemini, top_k=top_k * 2,
            fiscal_year=fiscal_year,
            model=embed_model, tokenizer=tokenizer,
            strict_year=strict_year,
        )
        results = _deduplicate(results)
        gen = generate_answer_map_reduce(
            effective_question, results, gemini,
            chunks=chunks, neighbor_radius=3,
        )
        search_stage = 3
    else:
        variants = generate_query_variants(base_q, gemini, n=3)
        results  = retrieve_multi_query(
            questions=variants,
            vectors=vectors, bm25=bm25,
            manifest=manifest, chunks=chunks, top_k=top_k,
            fiscal_year=fiscal_year, chunk_type=chunk_type,
            model=embed_model, tokenizer=tokenizer,
            use_rerank=use_rerank,
        )
        results = expand_context(results, chunks)
        results = _deduplicate(results)
        results = results[:top_k]
        gen = generate_answer_map_reduce(
            effective_question, results, gemini,
            chunks=chunks, neighbor_radius=1,
        )
        search_stage = 1

        # Stage 3 escalation — triggered when Stage 1 abstains
        if gen["abstained"]:
            print(f"  Stage 1 abstained → escalating to Stage 3…")
            results_s3 = retrieve_stage3(
                question=effective_question,
                vectors=vectors, bm25=bm25,
                manifest=manifest, chunks=chunks,
                llm_model=gemini, top_k=10,
                fiscal_year=fiscal_year,
                model=embed_model, tokenizer=tokenizer,
                strict_year=strict_year,
            )
            if results_s3:
                results_s3 = _deduplicate(results_s3)
                gen_s3 = generate_answer_map_reduce(
                    effective_question, results_s3, gemini,
                    chunks=chunks, neighbor_radius=3,
                )
                if not gen_s3["abstained"]:
                    gen     = gen_s3
                    results = results_s3
            search_stage = 3

    # Build source list for UI
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
            "full_text":          chunk.get("text", ""),   # raw markdown table or narrative
            "full_nl":            chunk.get("text_nl", ""),
            "signal":             signal_by_rank.get(idx + 1, "UNKNOWN"),
        })

    return jsonify({
        "question":               question,
        "answer":                 gen["answer"],
        "abstained":              gen["abstained"],
        "faithful":               gen["faithful"],
        "faithfulness_issues":    gen.get("faithfulness_issues"),
        "filters_detected":       filters,
        "variants_used":          variants,
        "sources":                sources,
        "filter_map":             [
            {"rank": m["rank"], "signal": m["signal"], "metadata": m["metadata"]}
            for m in gen.get("mapped", [])
        ],
        "clarification_needed":   gen.get("clarification_needed", False),
        "clarification_question": gen.get("clarification_question"),
        "power_search":           power_search,
        "search_stage":           search_stage,
    })


@app.route("/config")
def config():
    """Return static server capabilities so the UI can adapt."""
    return jsonify({"pdfs_available": PDFS_AVAILABLE})


@app.route("/page-image")
def page_image():
    """
    Render a single PDF page as a PNG and return it.
    Query params: fy=2022-2023  page=258  dpi=150 (optional)
    """
    fy       = request.args.get("fy", "").strip()
    page_num = request.args.get("page", type=int)
    dpi      = min(int(request.args.get("dpi", 150)), 250)

    if not fy or page_num is None:
        return jsonify({"error": "fy and page are required"}), 400

    pdf_path = PDF_MAP.get(fy)
    if not pdf_path or not pdf_path.exists():
        return jsonify({"error": f"PDF not found for fiscal year {fy}"}), 404

    cache_key = (fy, page_num, dpi)
    if cache_key not in _page_cache:
        try:
            doc  = fitz.open(str(pdf_path))
            idx  = page_num - 1   # fitz uses 0-based index
            if idx < 0 or idx >= len(doc):
                return jsonify({"error": f"Page {page_num} out of range"}), 404
            page = doc[idx]
            mat  = fitz.Matrix(dpi / 72, dpi / 72)
            pix  = page.get_pixmap(matrix=mat, alpha=False)
            _page_cache[cache_key] = pix.tobytes("png")
            doc.close()
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return send_file(
        io.BytesIO(_page_cache[cache_key]),
        mimetype="image/png",
        max_age=3600,
    )


@app.route("/feedback", methods=["POST"])
def feedback():
    """
    #22: Log user feedback for future fine-tuning.
    Expects: {query, answer, chunk_ids, rating (1=up, -1=down), comment?}
    """
    data = request.get_json(force=True)
    entry = {
        "timestamp":  datetime.utcnow().isoformat(),
        "query":      data.get("query", ""),
        "answer":     data.get("answer", ""),
        "chunk_ids":  data.get("chunk_ids", []),
        "rating":     data.get("rating"),      # 1 = thumbs up, -1 = thumbs down
        "comment":    data.get("comment", ""),
    }
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_PATH, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
