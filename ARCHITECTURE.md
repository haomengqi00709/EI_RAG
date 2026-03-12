# RAG Pipeline — Architecture Overview

This document describes the full system architecture as of 2026-02-22.
The core spine (parse → chunk → enrich → embed → retrieve) is unchanged from v1.
All improvements are layered on top without breaking existing stages.

---

## Offline Pipeline

Run once when documents change. Each stage reads from the previous stage's output.

```
PDFs  (Documents/*.pdf)
  │
  ▼
parse_pdf.py
  │  - Page extraction, table detection, narrative text extraction
  │  - Header/section detection and breadcrumb propagation
  │  - Multi-page table stitching
  │  - Content-type classification (narrative, table_heavy, toc, blank, etc.)
  │
  └──► data/extracted/*.json
         One JSON file per fiscal year. Contains pages, tables, charts, paragraphs.

  ▼
chunk.py
  │  - Produces four chunk types: narrative / table / chart / footnote
  │  - Narrative: paragraphs grouped within the same section breadcrumb
  │  - Table: one chunk per table (title + GitHub-flavoured markdown)
  │  - Chart: one chunk per chart (title + source note)
  │  - [NEW #4]  Semantic boundary detection — flushes chunk at detected
  │              sub-headers so headings always start a new chunk
  │  - [NEW #3]  Parent-child linking — each narrative chunk gets a
  │              related_chunk_ids field listing adjacent table/chart
  │              chunk IDs (±1 page), used for context expansion at query time
  │
  └──► data/chunked/*.jsonl
         One JSONL file per fiscal year. One JSON object per chunk.

  ▼
enrich_tables.py
  │  - Adds text_nl (natural language prose) to every table chunk
  │  - Uses Qwen3-4B-MLX-4bit (local, Apple Silicon)
  │  - Caches results by md5(prompt_version + table_text) — re-runs are instant
  │  - [NEW #2]  Row-enumeration prompt — forces model to list every row's
  │              values explicitly (e.g. "Women: $99, Men: $142, National: $133")
  │              instead of high-level summaries. Fixes demographic failures.
  │
  └──► data/chunked/*.jsonl  (updated in-place, text_nl added to table chunks)
       data/cache/text_nl_cache.json  (persistent cache)

  ▼
embed.py
  │  - Embeds all narrative and table chunks with Qwen3-Embedding-4B (MLX)
  │  - [NEW #1]  Contextual headers prepended before embedding:
  │                "Fiscal year: YYYY-YYYY
  │                 Section: {breadcrumb}
  │                 Table: {title}   ← tables only
  │
  │                 {chunk body}"
  │              Embedding captures WHERE the chunk lives, not just WHAT it says.
  │  - [NEW #6]  Saves per-type indexes alongside the combined index:
  │                vectors_narrative.npy + manifest_narrative.jsonl
  │                vectors_table.npy    + manifest_table.jsonl
  │              Queried separately for chunk_type-filtered searches.
  │
  └──► data/embeddings/vectors.npy          (combined, all types)
       data/embeddings/manifest.jsonl        (combined)
       data/embeddings/vectors_narrative.npy (narrative only)
       data/embeddings/vectors_table.npy     (table only)
       data/embeddings/manifest_narrative.jsonl
       data/embeddings/manifest_table.jsonl

  ▼
BM25 index  (built automatically on first retrieval if not found)
  │  - Tokenises all chunk text for keyword search
  │  - [NEW #10] Metadata tokens injected into each document:
  │              fiscal year string + section breadcrumb (repeated for weight)
  │              Province names in breadcrumbs become searchable BM25 tokens.
  │
  └──► data/embeddings/bm25_corpus.json
```

---

## Online Pipeline (per query)

Runs on every user question. Stages are modular — each can be enabled/disabled.

```
User question
  │
  ▼
Stage 1 — Query Understanding  [NEW #8]
  │  Fast Gemini call extracts structured filters:
  │    fiscal_year  → used as hard retrieval filter
  │    chunk_type   → used as hard retrieval filter
  │    province     → appended to query string for BM25 boost
  │    time_period  → logged (future: used for sub-period filtering)
  │
  ▼
Stage 2 — Hybrid Retrieval
  │
  ├── Dense retrieval (Qwen3-Embedding cosine similarity)
  │     Query text: "Instruct: ... Query: {question}"
  │     [NEW #9]  HyDE option: Gemini generates a hypothetical answer,
  │               that answer is embedded instead of the raw question.
  │               Result fused with standard dense via RRF.
  │
  ├── BM25 retrieval (keyword + metadata token matching)
  │
  └── RRF fusion (Reciprocal Rank Fusion, k=60)
        [NEW #11]  Candidate pool: max(top_k × 10, 50)
                   Previously top_k × 4. Larger pool → more for reranker.
  │
  ▼
Stage 3 — Context Expansion  [NEW #3]
  │  For each retrieved chunk that has related_chunk_ids,
  │  fetch and attach the linked table/chart chunks as context_chunks.
  │  Retrieval ranking is unchanged — only result objects are enriched.
  │  Solves: "the exact number is in the adjacent table, not the narrative."
  │
  ▼
Stage 4 — Cross-Encoder Reranking  [NEW #12]
  │  Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  │  Scores each (question, chunk_text) pair jointly.
  │  Far more accurate than bi-encoder cosine similarity for relevance ranking.
  │  Optional alternative: LLM reranker (Gemini RankGPT-style)  [NEW #13]
  │
  ▼
Stage 5 — Post-processing
  │  [NEW #15]  De-duplication: Jaccard similarity removes near-identical chunks
  │             (threshold 0.85). Common across 5 annual reports.
  │  [NEW #16]  Lost-in-the-middle ordering: most relevant chunks placed at
  │             the start and end of the context window.
  │             Pattern for 5 results [r1..r5] → [r1, r3, r5, r4, r2]
  │
  ▼
Stage 6 — Contextual Compression  [NEW #14]  (optional)
  │  Gemini extracts only the sentences directly relevant to the query
  │  from each chunk. Reduces token cost and generation noise.
  │  Enabled via compress=true in POST /answer.
  │
  ▼
Stage 7 — Answer Generation  [NEW #17 #19]
  │  Stage 2 LLM (Gemini) synthesises a final answer from the context.
  │  System prompt instructs:
  │    - Cite each source inline: [FY YYYY-YYYY | p.N | Section]
  │    - If context is insufficient, return INSUFFICIENT_CONTEXT
  │      which triggers the abstention response  [NEW #19]
  │
  ▼
Stage 8 — Faithfulness Check  [NEW #18]
  │  Second Gemini call verifies every factual claim in the answer
  │  is supported by the retrieved chunks. Returns faithful: true/false.
  │  Flags potential hallucinations in the API response.
  │
  ▼
API Response
  {
    question, answer, abstained, faithful, faithfulness_issues,
    filters_detected, sources: [{fiscal_year, chunk_type, page, section, score}]
  }
  │
  ▼
Stage 9 — Feedback Logging  [NEW #22]  (optional)
  User thumbs up/down → POST /feedback
  Logged to data/feedback/feedback.jsonl
  Future use: fine-tuning signal for embedding model and generation LLM.
```

---

## Server Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /` | Serve index.html (plain HTML/JS UI) |
| `POST /search` | Raw retrieval only — returns ranked chunks, no generation |
| `POST /answer` | Full pipeline — query understanding through faithfulness check |
| `POST /feedback` | Log user rating for future fine-tuning |

### POST /search parameters
```json
{
  "query":       "What were EI regular claims in 2023-24?",
  "top_k":       5,
  "fiscal_year": "2023-2024",   // optional override
  "chunk_type":  "table",       // optional override
  "rerank":      false          // optional: apply cross-encoder
}
```

### POST /answer parameters
```json
{
  "query":    "What were EI regular claims in 2023-24?",
  "top_k":    5,
  "rerank":   true,     // default true
  "compress": false     // default false — contextual compression
}
```

---

## Evaluation Stack

Four scripts, each measuring a different layer of quality:

| Script | What it measures | When to run |
|---|---|---|
| `evaluate_retrieval.py` | Recall@k — did the correct chunk_id appear in top-k? | After re-embedding |
| `evaluate_llm_judge.py` | Answered@k — does any retrieved chunk contain the answer? | After retrieval changes |
| `evaluate_e2e.py` | Exact match + Semantic match + Faithfulness of generated answer | After generation changes |
| `evaluate_ragas.py` | Context precision/recall + Answer faithfulness/relevancy | For structured benchmarking |

Current baseline (hybrid, top-k=5, auto year filter):
- **Answered@5: 83.3%**
- **Answered@1: 15%**  ← target for reranker improvement

---

## File Map

```
src/
  parse_pdf.py          Offline stage 1 — PDF → JSON
  chunk.py              Offline stage 2 — JSON → chunked JSONL
  enrich_tables.py      Offline stage 3 — adds text_nl to tables
  embed.py              Offline stage 4 — JSONL → vector index
  retrieve.py           Core retrieval library (dense / BM25 / hybrid / HyDE / rerank)
  server.py             Flask server — all online pipeline stages
  generate_eval.py      Generate eval question set via Gemini
  evaluate_retrieval.py Eval: Recall@k
  evaluate_llm_judge.py Eval: Answered@k (LLM judge)
  evaluate_e2e.py       Eval: End-to-end answer quality
  evaluate_ragas.py     Eval: RAGAs framework
  show_failures.py      Helper: print all answered=false from judge results

data/
  Documents/            Raw PDFs (input)
  extracted/            Parsed JSON (stage 1 output)
  chunked/              Chunked JSONL (stages 2-3 output)
  embeddings/           Vector indexes + BM25 corpus (stage 4 output)
  cache/                text_nl cache (enrich_tables.py)
  eval/                 Eval set + all evaluation results
  feedback/             User feedback log (feedback.jsonl)
```

---

## Pipeline Commands

All commands assume `source venv/bin/activate` first.

```bash
# Full offline pipeline (run in order when documents change)
python src/parse_pdf.py
python src/chunk.py
python src/enrich_tables.py
python src/embed.py --reindex

# Retrieval only (no generation)
python src/retrieve.py "What were EI regular claims in 2023-24?" --mode hybrid

# Start server (full pipeline available at localhost:5000)
python src/server.py

# Evaluation
python src/evaluate_retrieval.py
python src/evaluate_llm_judge.py
python src/evaluate_e2e.py
python src/evaluate_ragas.py --samples 20
python src/show_failures.py
```

---

## What Is Still Pending (requires GPU)

| Item | What it needs | Impact |
|---|---|---|
| Chart enrichment (#5) | Vision model (Qwen-VL / GPT-4o-mini) on RunPod | Makes charts searchable |
| Embedding fine-tuning (#7) | GPU + 200+ training pairs | Domain-specific recall boost |
