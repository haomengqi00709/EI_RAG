# Known Issues & Future Considerations

Tracking file for parsing and RAG pipeline issues that may need attention as we scale beyond the current EI MAR PDF demo.

---

## PDF Parsing — Multi-Column Layout

**Status**: Basic handling implemented. Edge cases remain.

| Issue | Description | Impact | Priority |
|---|---|---|---|
| Full-width headers in multi-column pages | Current detection uses font size to find headers that span both columns. If a full-width paragraph (not a header) sits above the columns, it may get split incorrectly. | Garbled text at top of some multi-column pages | Medium |
| Three or more columns | Current logic only detects a single split point (2 columns). Documents with 3+ columns will be treated as 2 columns with garbled right side. | Incorrect text extraction | Low (rare in government docs) |
| Mixed layout (single-column top, multi-column bottom) | A page might have a single-column introduction paragraph followed by two-column content. We only detect one layout per page. | Some text may be interleaved incorrectly | Medium |
| Column reading order | We assume left-to-right column order. Some layouts flow bottom-of-left to top-of-right (continuation). We don't detect this — we just concatenate left then right. | Paragraph may be split at column boundary | Low |
| Tables inside columns | If a table sits entirely within one column, `extract_tables()` should still find it via ruled lines. But a table spanning both columns in a multi-column page may be extracted incorrectly. | Table data garbled | Medium |
| False positives on wide tables | Wide data tables with spaced columns can trigger multi-column detection. Current threshold (20px gap) mitigates this, but not eliminated. Most false positives are on annex/table_heavy pages where the table extractor handles them. | Minor — table pages use `extract_tables()` anyway | Low |

---

## PDF Parsing — Tables

**Status**: Markdown conversion implemented with header cleanup. ~70% clean, ~30% have minor header issues.

| Issue | Description | Impact | Priority |
|---|---|---|---|
| Tables without ruled lines | pdfplumber relies on visible lines to detect table boundaries. Tables using only whitespace alignment (no borders) will not be detected as tables. | Table data extracted as messy text instead of structured | Medium |
| Merged cells | Cells that span multiple rows or columns are not handled well by pdfplumber. Content may appear duplicated or misaligned. | Some table data incorrect | Low |
| Nested tables | Tables within tables (rare but possible in government docs) are not supported. | Inner table data garbled | Low |
| Natural language table conversion | Converting tables to natural language sentences would improve embedding/search quality. Can be done as a batch preprocessing step using Ollama (free, ~30min/report) or Claude Haiku API (~$0.28/report). See "Future Enhancements" section below. | Better retrieval for numeric queries | Medium |
| Multi-page tables | A single table may span 3-5 pages. Currently each page is extracted independently, so the table is fragmented across page boundaries. Need to detect continuation and merge. | Incomplete table data per chunk | Medium |

---

## PDF Parsing — Charts

**Status**: Chart titles, source notes, and metadata are extracted. Actual chart data (visual content) is NOT extractable with current approach.

### What we extract now
- Chart number and title (e.g. "Chart 9 – Job vacancies and vacancy rates, Canada, Q3 2021-22 to Q4 2023-24")
- Source notes (e.g. "Source: Statistics Canada, Table 14-10-0432-01")
- Full metadata (fiscal year, page, section breadcrumb, hierarchy, temporal markers)
- `is_image: true` flag to indicate visual-only content

### What we DON'T extract
- Actual data values (bar heights, line positions, axis values)
- Legend entries
- Visual trends

### Future: Chart Data Extraction Options

| Approach | How it works | GPU needed | Cost (500 charts) | Accuracy | Effort |
|---|---|---|---|---|---|
| **Vision API (Recommended)** | Screenshot each chart page, send to Claude/GPT-4V with prompt "extract data into a table" | No | ~$1-5 total | High | Low |
| DePlot (Google Research) | ~300M param vision-language model, outputs data tables from chart images | Yes (or very slow on CPU) | Free | Medium | Medium |
| ChartOCR | Older specialized model, detects chart type + extracts axis labels and data points | Yes | Free | Medium-Low | Medium |

**Recommended implementation (Vision API approach):**
1. Use `pymupdf` or `pdfplumber` to render chart pages as images
2. Crop to the chart region (between title and source note)
3. Send to vision API (Claude Haiku with vision is cheapest at ~$0.75 for all 500 charts)
4. Prompt: "Extract all data from this chart into a Markdown table. Include axis labels, legend entries, and all data points."
5. Store extracted data table alongside existing chart metadata
6. Embed the extracted data for search

**Cost/time estimates for all 5 reports (~500 charts):**
- Claude Haiku: ~$0.75, ~15 min
- Claude Sonnet: ~$4.50, ~15 min
- GPT-4o mini: ~$1.05, ~15 min
- DePlot (GPU): Free, ~15-40 min
- DePlot (CPU only): Free, ~4-8 hours

---

## PDF Parsing — Header/Footer

| Issue | Description | Impact | Priority |
|---|---|---|---|
| Non-numeric footers | Current footer removal only strips standalone page numbers. If future docs have text footers (e.g. "Employment Insurance Report 2023-24 — Page 42"), they will remain. | Footer text pollutes chunks | Medium |
| Running headers | Some documents repeat chapter/section titles at the top of every page. We don't detect or remove these. Current EI MAR docs don't have them, but other document types might. | Duplicate header text in every chunk | Medium |

---

## PDF Parsing — Footnotes

**Status**: Not separated from body text. Left inline for now.

| Issue | Description | Impact | Priority |
|---|---|---|---|
| Footnotes mixed with body text | Footnotes (8pt font at page bottom) stay in the text field. ~31% of characters on narrative pages are footnote/chart-note sized. | Minor noise in embeddings. Footnotes are short and don't dominate chunks. | Low |
| Chart source notes mid-page | Source notes under charts (also small font) appear between narrative paragraphs, breaking text flow. | Slightly disjointed chunks | Low |
| Footnote-body linking | Superscript numbers in body text (e.g. "increased by 12%⁷") aren't linked to the corresponding footnote text at page bottom. | Can't trace citations | Low |

---

## PDF Parsing — Hierarchy Detection

| Issue | Description | Impact | Priority |
|---|---|---|---|
| Font thresholds are tuned to EI MAR | Header detection uses specific font size thresholds (28pt, 18pt, 12.5pt) learned from these PDFs. Other document types will use different fonts/sizes. | Headers missed or false positives on other doc types | High (when adding new doc types) |
| Sub-subsection headers not captured | Headers like "2.2.1", "2.2.2" at smaller fonts are not currently detected. Only top-level section headers (e.g. "2.2") are captured. | Missing granularity in hierarchy | Low |

---

## Future Document Types

| Document Type | Key Challenges | Notes |
|---|---|---|
| Excel (.xlsx) | Multiple sheets, merged cells, formulas vs values, charts | Will need `openpyxl` or `pandas`. Each sheet = separate extraction unit. |
| Word (.docx) | Embedded tables, images, headers/footers, tracked changes, styles | `python-docx` library. Styles can help with hierarchy detection. |
| Email (.eml/.msg) | Thread structure, attachments, HTML vs plain text, metadata (from/to/date/subject) | Need email parsing library. Attachments may themselves be PDF/Excel/Word. |
| Scanned PDFs | No embedded text — images of text | Requires OCR (Tesseract, AWS Textract, or similar). Completely different pipeline. |

---

## RAG-Specific Concerns (for later phases)

| Issue | Description |
|---|---|
| Cross-document entity resolution | The same concept may be described differently across years (e.g. program name changes). May need an entity/concept mapping layer. |
| Bilingual content | Some pages contain French text mixed with English. May need language detection per chunk. |

---

## Chunking — Known Gaps & Future Improvements

### High impact

| Issue | Description | Effort | Cost |
|---|---|---|---|
| ~~Tables embed poorly~~ | ~~Markdown pipe-delimited text carries almost no semantic signal for embedding models.~~ **FIXED** — `enrich_tables.py` generates `text_nl` (prose NL) for every table chunk. `embed.py` embeds `text_nl` instead of the raw markdown. | — | — |
| ~~No chunk overlap~~ | ~~When a topic spans two consecutive narrative chunks, queries about it may score poorly on both halves.~~ **FIXED** — last sentence of each narrative chunk is carried into the start of the next (same section only). `overlap_chars` field records borrowed length. | — | — |

### Medium impact

| Issue | Description | Effort | Cost |
|---|---|---|---|
| ~~Footnotes as standalone chunks~~ | ~~Footnotes were being accumulated into narrative chunks.~~ **FIXED** — footnotes are now detected via `^\d{1,2}\s+[A-Z]` pattern and tagged as `chunk_type="footnote"`. Excluded from vector index; preserved in JSONL for citation provenance. | — | — |
| Breadcrumb absent from chunk text | The `section_breadcrumb` lives in metadata only, not in the embedded text. **Handled at embed time** — `embed.py` prepends `"Section: {breadcrumb}\n\n"` to every embed string. The stored JSONL `text` field is unchanged. | — | Free |
| Large table chunks | 95 table chunks exceed 4,000 chars (max ~8,700 chars). Qwen3-Embedding's 32K token limit means no truncation occurs, but very large chunks may dilute embedding quality. Fix: split oversized tables at logical row group boundaries. | Medium | Free |

### Lower priority

| Issue | Description | Effort | Cost |
|---|---|---|---|
| Multi-granularity chunks | Produce coarser chapter-level chunks alongside the current section-level chunks, allowing the retriever to use the right granularity per query type. | Medium | Free |
| Cross-year chunk linking | Add `related_chunks` metadata pointing to the equivalent table/section in other fiscal years, enabling direct year-over-year comparison retrieval. | Medium | Free |

---

## Embedding — Potential Improvements

### High impact

**Contextual retrieval**

Some chunks lose meaning without surrounding context. A bullet-point chunk like *"• This represents a 3.2% increase compared to the previous year"* has no standalone signal — the embedding model doesn't know what "this" refers to, which report, or which metric.

Fix: before embedding each chunk, generate a 2–3 sentence LLM summary of it in the context of the full document, then prepend that summary to the embed text. The stored JSONL `text` field is unchanged; only the string sent to the embedding model gets the prepended summary.

Anthropic documented 35–49% reduction in retrieval failures using this technique.

```
# Embed text with contextual retrieval (not stored, only used at embed time):
"This passage is from the 2023-24 EI Monitoring and Assessment Report,
Highlights section. It describes labour market conditions during the year.

Section: HIGHLIGHTS

• This represents a 3.2% increase compared to the previous year..."
```

**Implementation options:**
| Approach | Cost | Time |
|---|---|---|
| Qwen3-4B-MLX-4bit (already in HF cache) | Free | ~30 min/report |
| Claude Haiku API | ~$2–5/report | ~15 min/report |

If implemented, `embed.py` would need a `--contextual` flag that loads Qwen3-4B, generates summaries, and prepends before calling `embed_one`. A cache (keyed by `chunk_id`) would prevent regeneration on re-runs.

**Recommended:** evaluate retrieval quality first. Only add contextual retrieval if ambiguous or context-dependent queries return poor results.

---

### Medium impact

**Query instruction prefix**

Qwen3-Embedding was trained with asymmetric embedding: documents are embedded as-is, but queries should use an instruction prefix to land in the correct part of the vector space.

`embed.py` correctly handles the document side (no prefix). The retrieval script must handle the query side:

```python
# Wrong — query lands in wrong part of vector space
query_vec = embed_one(model, tokenizer, "What were EI regular claims in 2023?")

# Correct — instruction-aware query embedding
QUERY_INSTRUCTION = (
    "Given a Canadian government Employment Insurance policy document, "
    "retrieve the passage that best answers the question."
)
query_text = f"Instruct: {QUERY_INSTRUCTION}\nQuery: {user_question}"
query_vec  = embed_one(model, tokenizer, query_text)
```

**Effort:** trivial — 3 lines in `src/retrieve.py`. Must be implemented before any retrieval evaluation.

---

---

## Future Roadmap

### Generation (the missing piece)
The current pipeline does retrieval only. To make it a full RAG system, a generation step is needed.

| Improvement | Description |
|---|---|
| Answer generation | Feed top-k chunks as context to an LLM (Qwen3-4B locally or Claude via API). Return a synthesised answer with inline source citations. |
| Streaming output | Stream LLM response token-by-token rather than waiting for the full answer. |
| Multi-turn conversation | Maintain conversation history so follow-up questions work correctly. |

### Retrieval quality
| Improvement | Impact | Description |
|---|---|---|
| Hybrid retrieval (BM25 + dense) | High | Combine keyword search (BM25) with semantic search. BM25 catches exact term matches that embeddings miss — specific program names, dollar figures, acronyms. |
| Cross-encoder reranking | Medium | After retrieving top-20 with embeddings, re-score with a cross-encoder that reads query + chunk together. More accurate but slower. |
| HyDE | Medium | Generate a hypothetical answer to the query, embed that, and retrieve against it. Bridges vocabulary gap between short queries and long policy passages. |
| Query expansion | Low | Automatically add EI-specific synonyms or related terms to the query before retrieval. |

### Data completeness
| Improvement | Impact | Description |
|---|---|---|
| Chart data extraction | High | 391 chart chunks are currently excluded from the index. Use Claude vision API (~$0.75 for all 500 charts) or DePlot to extract data from chart images. Full plan in the Charts section above. |
| Cross-year chunk linking | Medium | Tag equivalent sections across fiscal years so a query can explicitly surface year-over-year comparisons. |
| Multi-granularity indexing | Medium | Add chapter-level chunks alongside section-level chunks. Short or broad queries benefit from coarser granularity. |
| Bilingual support | Low | Detect French text mixed into English pages and handle separately. |

### Infrastructure
| Improvement | Impact | Description |
|---|---|---|
| Evaluation framework | High | Build a test set of 20–30 questions with known answers. Use RAGAS or manual scoring to measure retrieval quality. Without this, you cannot know whether any improvement actually helps. **Build this before optimising further.** |
| Vector database | Low now | At 4,153 vectors, numpy dot product is instant (<1 ms). Matters if corpus grows to 100k+ chunks. Chroma, Qdrant, or FAISS are drop-in options. |
| Query embedding cache | Low | Cache query vectors keyed by question text. Speeds up repeated queries in the UI. |

*Last updated: 2026-02-20*
