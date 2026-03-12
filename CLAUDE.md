# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval-Augmented Generation) pipeline for Employment Insurance Monitoring and Assessment Report (EI MAR) documents (2019–2024). The pipeline converts government policy PDFs into structured, embedding-ready chunks.

**Current status:** Stages 1–5 complete (parsing, chunking, table NL enrichment, embedding, retrieval). A browser UI for testing retrieval is available (`src/server.py` + `src/index.html`).

## Pipeline Stages & Commands

All commands assume the virtual environment is active: `source venv/bin/activate`

### Stage 1 — PDF Parsing
```bash
python src/parse_pdf.py                                      # all PDFs
python src/parse_pdf.py Documents/2023-2024-EI-MAR-EN.pdf   # single file
```
Input: `Documents/*.pdf` → Output: `data/extracted/*.json`

### Stage 2 — Chunking
```bash
python src/chunk.py                                          # all files
python src/chunk.py data/extracted/2023-2024-EI-MAR-EN.json # single file
```
Input: `data/extracted/*.json` → Output: `data/chunked/*.jsonl`

### Stage 3 — Table NL Enrichment
```bash
python src/enrich_tables.py                                          # all files
python src/enrich_tables.py data/chunked/2023-2024-EI-MAR-EN.jsonl  # single file
```
Input/Output: `data/chunked/*.jsonl` (adds `text_nl` field to table chunks in-place)

Requires: `Qwen/Qwen3-4B-MLX-4bit` in the HuggingFace cache. Model loads lazily on first cache miss.

**Important:** Re-running `chunk.py` wipes `text_nl` from JSONL files. Always re-run `enrich_tables.py` after re-chunking. The cache makes this instant (no model calls) if no tables changed.

### Stage 4 — Embedding
```bash
python src/embed.py             # embed all chunks, resume if interrupted
python src/embed.py --reindex   # re-embed everything from scratch
```
Input: `data/chunked/*.jsonl` → Output: `data/embeddings/vectors.npy` + `data/embeddings/manifest.jsonl`

Requires: `mlx-community/Qwen3-Embedding-4B-4bit-DWQ` (~2.3 GB, auto-downloaded to HuggingFace cache).

Embeds `narrative` and `table` chunks only. Skips `footnote` and `chart`. For table chunks, embeds `text_nl` (prose) instead of `text` (markdown). Prepends `"Section: {breadcrumb}\n\n"` to every embed text when breadcrumb is non-empty.

### Stage 5 — Retrieval
```bash
python src/retrieve.py "What were EI regular claims in 2023-24?"
python src/retrieve.py "How did sickness benefits change?" --top-k 10
python src/retrieve.py "Regular claims" --year 2023-2024
python src/retrieve.py "Benefit amounts" --type table
```
Input: `data/embeddings/vectors.npy` + `data/embeddings/manifest.jsonl` + `data/chunked/*.jsonl`

Embeds the query using the Qwen3-Embedding instruction prefix (asymmetric mode), scores all indexed chunks via dot product, applies optional filters, and prints ranked results with text snippets.

### UI — Retrieval Testing
```bash
pip install flask          # one-time
python src/server.py
```
Opens a plain HTML/JS browser UI at `http://localhost:5000`. Flask serves `src/index.html` and handles POST `/search` requests. Model, index, and chunks are loaded once at startup. Sidebar controls for fiscal year, chunk type, and top-k. Results show score, section breadcrumb, text snippet, and an expandable full-text view (tables show both `text_nl` and raw markdown).

## Current Data State

All 5 EI MAR reports (2019–2024) have been fully processed through all three stages.

**Parsing results:**
| Year | Pages | Characters | Tables | Stitched | Breadcrumb |
|---|---|---|---|---|---|
| 2019-2020 | 595 | 1,010,173 | 422 | 13 | 97% |
| 2020-2021 | 513 | 816,345 | 485 | 8 | 97% |
| 2021-2022 | 538 | 834,278 | 496 | 10 | 96% |
| 2022-2023 | 567 | 822,273 | 569 | 19 | 97% |
| 2023-2024 | 650 | 1,038,597 | 455 | 18 | 97% |

**Chunking results:** 5,124 total chunks (1,799 narrative, 2,354 table, 391 chart, 580 footnote)

**Embedding results:** 4,153 vectors embedded (1,799 narrative + 2,354 table), shape `(4153, 2560)` float32.
- L2 norms: 0.995–1.006 (properly normalised)
- 0 NaN, 0 Inf, 0 zero vectors
- Cross-year similarity validated: same section across years scores 0.89–0.91; different sections score ~0.54

- Narrative chunks: 71% carry a sentence of overlap from the previous chunk (`overlap_chars` field)
- Footnote chunks: tagged separately, excluded from vector index, kept for citation provenance
- Table chunks: 65 tiny tables (<50 chars) dropped at chunk time

**Enrichment results:** 2,354 / 2,354 table chunks have `text_nl`. 0 missing.

## Architecture

### src/parse_pdf.py
Processes each PDF in three sequential passes:
1. **Page extraction** — table detection, narrative text extraction, paragraph/header detection, content-type classification (`cover`, `toc`, `narrative`, `table_heavy`, `annex`, `blank`, etc.), temporal marker extraction.
2. **Hierarchy propagation** — carries `section_breadcrumb` (e.g. `"CHAPTER II > 2.2 EI regular benefits"`) forward across pages; enriches tables and charts with full context metadata.
3. **Multi-page table stitching** — detects and merges tables split across page boundaries.

### src/chunk.py
Produces four chunk types from each parsed JSON:
- **narrative** — consecutive paragraphs grouped within the same `section_breadcrumb`, ~1,800 char target. Skips non-content pages and filters noise paragraphs (<60 chars, table title echoes). Carries the last sentence of the previous chunk as overlap when flushing mid-section.
- **table** — one chunk per table: title + GitHub-flavoured markdown. Atomic; not split. Tables <50 chars are dropped.
- **chart** — one chunk per chart: title + source note. Metadata-only until vision extraction is added.
- **footnote** — numbered citation blocks (`^\d{1,2}\s+[A-Z]` pattern). Kept for provenance but excluded from the vector index.

Every chunk carries: `chunk_id`, `chunk_type`, `text`, `char_count`, `fiscal_year`, `source_file`, `page_number` / `start_page`+`end_page`, `section_breadcrumb`, `hierarchy`.

**chunk.py configuration constants:**
| Constant | Value | Purpose |
|---|---|---|
| `SKIP_PAGE_TYPES` | `{cover, copyright, toc, abbreviations, blank}` | Page types with no indexable content |
| `MIN_PARA_CHARS` | 60 | Drop paragraphs shorter than this (noise) |
| `NARRATIVE_TARGET_CHARS` | 1,800 | Target narrative chunk size (~450 tokens) |
| `MIN_TABLE_CHARS` | 50 | Drop tables smaller than this |
| `FOOTNOTE_PATTERN` | `^\d{1,2}\s+[A-Z]` | Regex to identify footnote paragraphs |

### src/enrich_tables.py
Adds `text_nl` to every table chunk — a natural language prose version of the table generated by `Qwen3-4B-MLX-4bit`. Both `text` (markdown) and `text_nl` are preserved so A/B comparison at embedding time is possible.

**Caching:** `data/cache/text_nl_cache.json` stores results keyed by `md5(table_text)`. Re-running after a `chunk.py` regeneration or adding new documents only calls the model for genuinely new tables.

**Tunable constants in enrich_tables.py:**
| Constant | Value | Purpose |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen3-4B-MLX-4bit` | Local MLX model |
| `MAX_TOKENS` | 512 | Max NL output length |
| `SAVE_EVERY` | 25 | Checkpoint frequency |
| `MIN_TABLE_CHARS` | 50 | Skip tables smaller than this |

### src/embed.py configuration constants
| Constant | Value | Purpose |
|---|---|---|
| `MODEL_ID` | `mlx-community/Qwen3-Embedding-4B-4bit-DWQ` | Local MLX embedding model |
| `EMBED_DIM` | 2560 | Output vector dimension |
| `MAX_SEQ_LEN` | 8192 | Model context window (tokens) |
| `SAVE_EVERY` | 50 | Checkpoint frequency |
| `EMBED_TYPES` | `{narrative, table}` | Chunk types included in the vector index |

### src/retrieve.py
Loads `vectors.npy` and `manifest.jsonl`, embeds the user query with an instruction prefix (Qwen3-Embedding asymmetric mode), scores all chunks via dot product, applies optional filters, and prints ranked results.

**Asymmetric embedding:** document chunks are embedded plain (no prefix). Queries must use the instruction prefix to land in the correct part of the vector space. `retrieve.py` handles this automatically.

**retrieve.py configuration constants:**
| Constant | Value | Purpose |
|---|---|---|
| `DEFAULT_TOP_K` | 5 | Default number of results returned |
| `SNIPPET_CHARS` | 300 | Characters of text shown in result preview |
| `QUERY_INSTRUCTION` | `"Given a Canadian government EI policy document..."` | Instruction prefix for query embedding |

### src/parse_pdf.py configuration constants
All tunable thresholds are at the top of the file:
| Constant | Value | Purpose |
|---|---|---|
| `FONT_TIER_MAJOR` | 28 pt | Major heading threshold |
| `FONT_TIER_CHAPTER` | 18 pt | Chapter subtitle threshold |
| `FONT_TIER_SECTION` | 12.5 pt | Section header threshold |
| `MIN_COLUMN_GAP_PX` | 20 px | Multi-column gap detection |
| `STITCH_BOTTOM_GAP` | 80 pt | Table stitch candidate proximity |
| `STITCH_TOP_GAP` | 95 pt | Table continuation proximity |
| `_LINE_TOLERANCE` | 3 pt | Line grouping tolerance |
| `_PARA_GAP_RATIO` | 1.6 | Paragraph gap multiplier |

These thresholds are tuned for EI MAR documents and will need adjustment for other document types.

## Chunk Schema

**narrative chunk:**
```
chunk_id, chunk_type="narrative", text, char_count,
fiscal_year, source_file, start_page, end_page,
section_breadcrumb, hierarchy,
overlap_chars (int, optional) — chars at start of text borrowed from previous chunk
```

**table chunk:**
```
chunk_id, chunk_type="table", text (markdown), text_nl (prose), char_count,
fiscal_year, source_file, page_number,
section_breadcrumb, hierarchy,
table_title, row_count, col_count,
stitched_pages (if multi-page)
```

**chart chunk:**
```
chunk_id, chunk_type="chart", text, char_count,
fiscal_year, source_file, page_number,
section_breadcrumb, hierarchy, chart_number
```

**footnote chunk:**
```
chunk_id, chunk_type="footnote", text, char_count,
fiscal_year, source_file, start_page, end_page,
section_breadcrumb, hierarchy
```
_Footnote chunks are excluded from the vector index. Embed only narrative and table chunks._

## Embedding Schema

**vectors.npy** — float32 numpy array, shape `(N, 2560)`. Each row is the L2-normalised embedding for one chunk. Row order matches `manifest.jsonl`.

**manifest.jsonl** — one JSON line per embedded chunk:
```
index, chunk_id, chunk_type, fiscal_year, source_file,
section_breadcrumb, page, embed_chars,
table_title (table chunks only)
```
`embed_chars` is the length of the string actually embedded (breadcrumb + text_nl or text), not the stored `char_count`.

## Validation

**Parsing quality:**
```python
import json, os
for f in sorted(os.listdir("data/extracted")):
    if not f.endswith('.json'): continue
    with open(os.path.join("data/extracted", f)) as fh:
        d = json.load(fh)
    empty_bc = sum(1 for p in d['pages'] if not p['section_breadcrumb'])
    bc_pct = (1 - empty_bc / d['total_pages']) * 100
    print(f"{d['fiscal_year']} | pages={d['total_pages']} | chars={d['total_characters']:,} | "
          f"tables={d['total_tables']} | stitched={d['tables_stitched']} | breadcrumb={bc_pct:.0f}%")
```
Expected: breadcrumb coverage ≥96%, tables 422–569, stitched 8–19.

**Enrichment coverage:**
```python
import json
from pathlib import Path
for f in sorted(Path('data/chunked').glob('*.jsonl')):
    chunks = [json.loads(l) for l in open(f)]
    tables = [c for c in chunks if c['chunk_type'] == 'table']
    enriched = sum(1 for c in tables if 'text_nl' in c)
    print(f"{f.stem}: {enriched}/{len(tables)} enriched")
```
Expected: enriched = total for all files (0 missing).

**Chunk type breakdown:**
```python
import json
from pathlib import Path
from collections import Counter
all_chunks = [json.loads(l) for f in Path('data/chunked').glob('*.jsonl') for l in open(f)]
print(Counter(c['chunk_type'] for c in all_chunks))
```
Expected: `{'table': 2354, 'footnote': 580, 'narrative': 1799, 'chart': 391}` (totals may shift slightly if documents are re-chunked).

**Embedding coverage:**
```python
import json, numpy as np
from pathlib import Path
manifest = [json.loads(l) for l in open('data/embeddings/manifest.jsonl')]
vectors = np.load('data/embeddings/vectors.npy')
print(f"Vectors: {vectors.shape}  Manifest: {len(manifest)} entries")
by_type = {}
for r in manifest:
    by_type[r['chunk_type']] = by_type.get(r['chunk_type'], 0) + 1
print(by_type)
```
Expected: vectors.shape = `(N, 2560)` where N = narrative + table chunk count.

## Key Documentation Files

- `PARSING_PIPELINE.md` — Technical reference for parse_pdf.py architecture, output schema, and configuration.
- `KNOWN_ISSUES.md` — Tracked limitations for parsing, chunking, and planned future improvements.
- `VALIDATION_CHECKLIST.md` — QA guide with expected value ranges.
