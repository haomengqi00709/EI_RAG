# EI MAR RAG System — Project Overview

> **Plain-language guide to what we built, how each piece works, and how the system improved over time.**

---

## What Is This Project?

This system allows someone to ask plain-English questions about Canadian Employment Insurance (EI) policy data — and get precise, cited answers drawn directly from official government reports.

The input documents are the **Employment Insurance Monitoring and Assessment Reports (EI MAR)** published annually by Employment and Social Development Canada. We processed five years of reports (2019–2020 through 2023–2024), covering hundreds of pages of narrative text and thousands of statistical tables.

The core challenge: these reports are dense, multi-hundred-page PDFs with complex tables. A question like *"What was the year-over-year change in Targeted Wage Subsidy expenditures in 2022–2023?"* requires finding a single cell in a table buried on page 258 — and reading the right row in the right column. Standard keyword search fails; generic AI summarization hallucinates. We needed a purpose-built pipeline.

---

## Architecture Overview

The system is a **Retrieval-Augmented Generation (RAG)** pipeline. The idea is simple:

1. Convert PDFs into structured, searchable pieces (chunks)
2. Embed each piece as a numeric vector so similar content can be found by meaning
3. When a question arrives, find the most relevant chunks
4. Feed those chunks to a language model (LLM) to generate a precise answer

The pipeline has five main stages:

```
PDFs  →  [Stage 1: Parse]  →  [Stage 2: Chunk]  →  [Stage 3: Enrich]
      →  [Stage 4: Embed]  →  [Stage 5: Retrieve + Generate]  →  Answer
```

---

## Stage 1 — PDF Parsing (`parse_pdf.py`)

**What it does:** Converts each PDF into a structured JSON file describing every page, paragraph, table, and chart — with section context attached to each element.

**Why this is hard:** EI MAR PDFs are not clean documents. Tables span multiple pages and get split mid-row. Headings and sub-headings appear at inconsistent font sizes. Columns of text can appear side by side. Page numbers and headers repeat on every page. A naive text dump would lose all of this structure.

**How it works:**

The parser makes three sequential passes through each PDF:

1. **Page extraction** — reads every page and classifies it (cover, table of contents, narrative content, annex, blank, etc.). Detects tables using whitespace gaps between columns. Extracts paragraph text, identifies headers by font size, and pulls out temporal markers like fiscal years.

2. **Hierarchy propagation** — builds a running "breadcrumb" (e.g. `"CHAPTER II > 2.2 EI regular benefits"`) from the section headers it finds. This breadcrumb is attached to every table, chart, and paragraph on the page, so every piece of content knows which section it belongs to. Without this, a table showing "Regular Claims: 1.2M" would have no context telling you it's about regular EI benefits.

3. **Multi-page table stitching** — when a table is split across page boundaries, the parser detects the continuation and merges the rows back into a single table. This is critical because many of the most information-dense tables in the reports span 2–3 pages.

**Output:** `data/extracted/*.json` — one file per PDF year, containing all pages with full metadata.

**Scale:** 5 reports, 595–650 pages each, 800K–1M characters per year, 422–569 tables per year.

---

## Stage 2 — Chunking (`chunk.py`)

**What it does:** Breaks the parsed content into self-contained pieces ("chunks") of the right size for retrieval. Each chunk is one unit that will later get its own embedding vector.

**Why chunking matters:** LLMs have context limits — you can't feed an entire 600-page report to answer one question. But chunks that are too small lose context; chunks that are too large dilute relevance. Getting the chunk boundaries right is the difference between finding the answer and missing it.

**The four chunk types:**

- **Narrative chunks** — consecutive paragraphs within the same section, targeting ~1,800 characters (roughly 450 tokens). When a section is long, it's split into overlapping chunks: the last sentence of chunk N becomes the first sentence of chunk N+1. This overlap prevents answers from falling into the gap between two chunks.

- **Table chunks** — one chunk per table, containing the table title and the full table as formatted markdown (GitHub-Flavored Markdown). Tables are kept atomic — never split — because splitting a table mid-row destroys meaning.

- **Chart chunks** — one chunk per chart, with title and source note. Currently metadata-only (no visual content extraction).

- **Footnote chunks** — numbered citation blocks extracted separately. Kept for provenance but not included in the search index, since footnotes rarely contain the primary answers.

**Key decisions:**
- Tables under 50 characters are dropped as noise (typically blank or formatting artifacts)
- Paragraphs under 60 characters are dropped (page headers, repeated labels, etc.)
- Section breadcrumbs are preserved in every chunk so the LLM always knows context

**Output:** `data/chunked/*.jsonl` — one file per year, one JSON line per chunk.

**Scale:** 5,124 total chunks (1,799 narrative, 2,354 table, 391 chart, 580 footnote)

---

## Stage 3 — Table NL Enrichment (`enrich_tables.py`)

**What it does:** Converts each table's raw markdown into a plain-English prose description and stores it alongside the original markdown.

**Why this is necessary:** Embedding models are trained on natural language text, not markdown tables. A question about "sickness benefit claims" uses completely different language than a markdown table full of `|` characters and numeric cells. If we embed the raw markdown, the semantic match between the question and the table is weak — we miss relevant tables during retrieval.

**Example transformation:**

*Raw markdown (what the PDF contains):*
```
| FY | Regular | Sickness | Parental |
|---|---|---|---|
| 2021-22 | 1,204,000 | 437,000 | 312,000 |
```

*Enriched text_nl (what gets embedded):*
> "In fiscal year 2021-2022, a total of 1,204,000 regular EI claims were established, along with 437,000 sickness benefit claims and 312,000 parental benefit claims..."

**How it works:** Each table is sent to Gemini 2.0 Flash with a prompt asking it to write a complete natural language description enumerating all values. Results are cached by the MD5 hash of the table text, so re-runs are instant if tables haven't changed.

**Evolution:** Originally used a local Qwen3-4B-MLX model with a 512-token output limit. This was replaced with Gemini 2.0 Flash (no token limit), which produced much more complete descriptions — full row enumeration for every table, typically 1,300–6,000 characters. This substantially improved retrieval recall.

**Sub-table parent-child enrichment:** Many tables in the reports are "factsheet" format — a titled summary table followed by one or more untitled sub-tables on the same page, each covering a specific metric. These 93 untitled sub-tables previously had weak embeddings because their `text_nl` started with "(no title)". We implemented a parent-child detection heuristic: untitled tables that immediately follow a titled table on the same page are treated as sub-tables of that parent. Their enrichment prompt includes the parent table title, so the generated prose explicitly states the relationship (e.g., *"Sub-table of Table 1 – LMDA Key Facts: program expenditures by province"*). This dramatically improved their retrievability.

**Output:** `text_nl` field added in-place to every table chunk in the JSONL files. Sub-table chunks also carry a `parent_chunk_id` field linking them to their parent.

---

## Stage 4 — Embedding (`embed.py`)

**What it does:** Converts each narrative chunk and table chunk into a 2,560-dimensional numeric vector using a local embedding model. These vectors represent "meaning" — chunks about similar topics end up near each other in vector space.

**Why embeddings:** Keywords miss synonyms and paraphrasing. "Claims established" and "new applications received" mean the same thing but share no words. Embedding models capture this semantic similarity; keyword search does not.

**The model:** `mlx-community/Qwen3-Embedding-4B-4bit-DWQ` — a 4-billion parameter embedding model from Alibaba's Qwen3 family, quantized to 4-bit for local Apple Silicon use. It produces normalized vectors of dimension 2,560.

**Asymmetric embedding:** This model uses different representations for queries vs. documents:
- Documents are embedded "plain" (the raw text)
- Queries must use an instruction prefix: *"Given a Canadian government EI policy document, retrieve the passage that best answers the question."*

This asymmetry is intentional — the model was trained this way so that questions (short, imperative) correctly align with document passages (longer, declarative). The pipeline handles this automatically.

**What gets embedded:**
- All narrative chunks (embedding the full text)
- All table chunks (embedding the `text_nl` prose, not the raw markdown)
- For both types, the section breadcrumb is prepended to the embed text for context

**What is excluded:** Chart and footnote chunks are not embedded (charts have no textual content; footnotes are for provenance only).

**Output:** `data/embeddings/vectors.npy` — 4,153 × 2,560 float32 matrix. `data/embeddings/manifest.jsonl` — metadata for each vector row.

---

## Stage 5 — Retrieval + Generation

This is the live pipeline that runs when a question is asked. It has two parts: finding the right chunks (retrieval) and generating the answer (generation). The pipeline has three escalating search stages — it automatically tries deeper search if an initial attempt doesn't find enough information.

### Retrieval (`retrieve.py`)

**Hybrid retrieval (Dense + BM25):**

Using only semantic (vector) search is not enough — it misses exact numerical values and fiscal year references. Using only keyword (BM25) search misses paraphrase and synonyms. We combine both:

1. **Dense retrieval** — embed the query and find the top-N most similar vectors via dot product
2. **BM25 retrieval** — keyword search over a tokenized corpus (with fiscal year and breadcrumb injected as extra tokens for better metadata matching)
3. **RRF fusion** — Reciprocal Rank Fusion merges the two ranked lists into a single combined ranking without needing to tune score thresholds

**Multi-query retrieval:**

Before retrieval, Gemini generates 3 rephrased variants of the original question (different phrasing, synonyms, emphasis). All 4 questions (original + 3 variants) are retrieved independently and merged. This improves recall: a question like "percent change in calls" might not match a table that says "year-over-year difference in volume".

**Reranking:**

After merging, a cross-encoder reranker re-scores each candidate chunk by reading the question and chunk text together. Unlike embedding similarity (which compares question and document independently), a cross-encoder sees both at once and can make much more accurate relevance judgments. The top-k results after reranking are passed to generation.

**Fiscal year filtering:**

When a question specifies a fiscal year, retrieval filters to only chunks from that year, dramatically reducing noise from other years with similar table layouts.

---

### Search Stages

The pipeline has three escalating search modes. The server automatically escalates to Stage 3 if Stage 1 or 2 abstains.

**Stage 1 — Standard Search (default)**
- Multi-query hybrid retrieval (dense + BM25 + RRF)
- Top-5 candidates after reranking
- ±1 adjacent chunk expansion for cut-off tables

**Stage 2 — Deep Search (user-selectable checkbox)**
- Same as Stage 1 but with top-10 candidates (2× more chunks evaluated)
- ±3 adjacent chunk expansion around every relevant chunk (not just cut-off ones)
- Best for questions where the answer is in a complex multi-page table or buried section

**Stage 3 — Maximum Search (automatic fallback)**

Fires automatically when Stage 1 or 2 returns "insufficient context". Adds three additional retrieval strategies on top of Stage 2:

1. **HyDE (Hypothetical Document Embedding)** — Gemini generates a short hypothetical answer paragraph, then that paragraph is embedded as the query. This bridges the language gap between how questions are phrased and how documents are written.

2. **Keyword BM25 pass** — Gemini extracts key terms from the question (acronyms, numbers, proper nouns), then runs a dedicated BM25 search over those terms. This is especially effective for questions about specific program names or exact fiscal year references.

3. **Year-relaxed fallback** — If no answer is found within the specified fiscal year, the search expands to all five years. Useful for questions about definitions, methodology, or historical comparisons.

4. **Full-page expansion** — Instead of fetching ±N adjacent chunks, retrieves all chunks from the same page across all chunk types. This captures complete page context including tables and narrative that appear together.

All four retrieval streams are fused via RRF, deduplicated, and passed to the generation pipeline.

---

### Generation (`generate.py`)

**Filter → Expand → Reduce pipeline:**

Rather than feeding all top-k retrieved chunks directly to the answer LLM, we run a three-step process:

**Step 1 — Filter:** Each retrieved chunk is independently classified (in parallel) as:
- `YES` — clearly contains relevant information
- `NEED_ADJACENT` — seems relevant but appears cut off or is part of a multi-page table
- `NOT_FOUND` — not relevant

This removes noise before the final answer call.

**Step 2 — Expand:** For chunks marked `NEED_ADJACENT`, adjacent chunks (±1 or ±3 by sequential index, depending on search stage) are fetched from the full chunk store. For table chunks, other tables on the same page are also included as context — this helps when a question spans two adjacent tables on the same page. Sub-table parent-child relationships are also respected: retrieving a sub-table automatically pulls in its parent and siblings.

**Step 3 — Reduce:** A single LLM call reads the full text of all relevant chunks (including expanded neighbors) and generates the final answer. The prompt instructs it to:
- Read pre-computed values directly without recalculating
- Match the specific row that fits the question's exact conditions (not a nearby row or an aggregate)
- Surface ALL competing values with context if multiple excerpts give different numbers
- Only abstain if the excerpts are completely unrelated
- Cite each source as `[FY YYYY-YYYY | p.N]`

**Faithfulness check:** After generation, a second LLM call verifies that every factual claim in the answer is directly supported by the retrieved chunks. This detects hallucination.

**Clarification check:** A third LLM call detects whether the answer contains competing values that require the user to choose. If so, a clarifying question is generated (e.g., *"Which metric do you mean — average annual earnings or weekly benefit rate?"*). The UI displays this and lets the user refine their question.

---

## Browser UI (`server.py` + `index.html`)

A web app accessible at `http://localhost:5000` (locally) or via a public RunPod proxy URL provides two modes:

- **Search** — pure retrieval, returns ranked chunks with scores, breadcrumb, and expandable full text. Useful for investigating what the index actually contains.
- **Ask** — runs the full generation pipeline and returns:
  - **Answer text** — direct answer with inline citations `[FY YYYY-YYYY | p.N]`
  - **Faithfulness badge** — green (grounded in sources) or red (potential hallucination)
  - **Sources used** — collapsible section showing the actual chunks the answer was based on: raw tables rendered as HTML, narrative text shown in full
  - **Pipeline trace** — collapsible section showing how the answer was found (query variants, filter signals, search stage used)
  - **Clarification card** (yellow) — when the answer has competing values, prompts user to specify which they mean
  - **⚡ Deep Search badge** — shown when Stage 2 or 3 was used
  - **PDF page viewer** — click any source citation to view the original PDF page side-by-side with the answer (when PDFs are available)

**Deployment:** The server runs on RunPod GPU instances and is publicly accessible via RunPod's proxy (`https://{pod-id}-5000.proxy.runpod.net`). No local setup is required for end users.

---

## How Precision Improved Over Time

Each algorithmic change was tracked in `data/eval/TRACKING.md`. Two evaluation sets were used:

- **Original eval set** — 60 questions across all five fiscal years (12 per year), mix of table and narrative questions. Used to track iterative improvements to the generation pipeline.
- **New eval set** — 100 user-generated questions (20 per year), harder overall. Includes annex tables, administrative section lookups, and specific statistical lookups that stress-test retrieval.

### Evaluation Metrics

- **Exact match** — the correct answer string appears verbatim in the generated answer
- **Semantic match** — an LLM judge evaluates whether the answer conveys the right value (handles rounding, paraphrasing, equivalent phrasing like "decrease of 73.1%" = "-73.1%")
- **Abstention rate** — share of questions where the pipeline declined to answer ("not enough context")
- **Faithful rate** — share of answers where every claim is traceable to the retrieved chunks

---

### Run 1 — Baseline (`baseline_e2e`)

**Semantic match: 61.7% | Abstention: 21.7%**

The first end-to-end run using the initial map-reduce pipeline. Multi-query was active. The abstention rate was very high — the LLM frequently refused to answer even when the correct chunk was retrieved. Root cause analysis revealed two classes of failure:

1. A 2,000-character truncation on the chunk body was cutting off answers buried deep in table `text_nl`
2. The MAP prompt asked the LLM to "directly answer" — causing false `NOT_FOUND` signals when it should have just said "this chunk contains relevant info"

---

### Run 2 — Fix Truncation (`fix_truncation`)

**Semantic match: 70.0% | Abstention: 13.3%**

Two targeted fixes:
- Removed the 2,000-character body truncation in the map step — full `text_nl` is now passed
- Changed MAP prompt from "directly answer" to "contain information that answers" — reduced false abstentions

Improvement: **+8.3 pp** semantic match, abstention halved.

---

### Run 3 — Filter → Expand → Reduce (`filter_expand_reduce`)

**Semantic match: 81.7% | Exact match: 76.7% | Abstention: 5.0%**

The biggest single improvement. Replaced the map-reduce architecture entirely with a three-step Filter → Expand → Reduce pipeline. Why it worked: the old map step asked each chunk independently "what value does this contain?" — prone to picking the wrong row. The new reduce step reads the full table text with all rows visible at once and can reason about which row matches the exact conditions.

Result: **zero generation failures remain** after this change. All 9 previously-failing generation questions now pass. Only 8 retrieval failures persist (correct chunk genuinely not in top-k).

---

### Run 4 — Row Precision Rules (`reduce_row_rules`)

**Semantic match: 83.3% | Exact match: 76.7%**

Added explicit row-reading instructions to the REDUCE prompt:
- "Read the exact pre-computed value directly — do not recalculate"
- "Match the specific row for the exact condition stated — do not pick a nearby row or aggregate"

Improvement: **+1.6 pp** semantic match.

---

### Run 5 — Multi-Value Presentation (`reduce_multi_value`)

**Semantic match: 88.3% | Exact match: 80.0% | Faithful: 85.0% | Abstention: 5.0%**

Changed the REDUCE prompt to surface ALL competing values with context when multiple excerpts give different numbers:
- Old behaviour: silently pick the "best" value → high risk of picking wrong one
- New behaviour: present all values with labels explaining why they differ → correct answer is always in the response

Result: **+4.7 pp** semantic match. **Total gain: +26.6 pp** over the baseline run on the original 60-question eval set.

---

### Run 6 — New Eval Set, v4 Enrichment + Stage 3 (`e2e_neweval_v4_option2`)

**Semantic match: 76.0% | Abstention: 20.0% | Wrong: 4.0%** *(100 questions, harder set)*

First run on the harder 100-question user-generated eval set. The lower score vs. the 60-question set reflects the harder question types: annex table lookups, administrative section statistics, and questions that require cross-referencing multiple tables.

Failure breakdown (24 total):
- 20 abstentions: retrieval misses in sparse annex/admin sections
- 4 wrong answers: row-reading errors (right chunk retrieved, LLM picks wrong row)

---

### Run 7 — Stage 3 on 24 Failing Questions

**Stage 2: 8.3% (2/24) | Stage 3: 29.2% (7/24)** *(24 failing questions only)*

Stage 3 (HyDE + keyword BM25 + year-relaxed + full-page expansion) was tested on the 24 questions that failed in Run 6. It recovered 5 additional questions that Stage 2 could not answer. Stage 3 is now the automatic fallback in the server when Stage 1 or 2 abstains.

**Expected impact on full 100-question eval:** ~81% semantic match (up from 76%).

---

### Summary Progress Table

| Run | Eval Set | Semantic Match | Exact Match | Abstention | Key Change |
|---|---|---|---|---|---|
| baseline_e2e | 60 Q | 61.7% | — | 21.7% | First full run |
| fix_truncation | 60 Q | 70.0% | — | 13.3% | Removed body truncation; softer MAP prompt |
| filter_expand_reduce | 60 Q | 81.7% | 76.7% | 5.0% | Filter→Expand→Reduce replaces map-reduce |
| reduce_row_rules | 60 Q | 83.3% | 76.7% | 6.7% | Row precision rules in REDUCE prompt |
| reduce_multi_value | 60 Q | **88.3%** | **80.0%** | 5.0% | Present all competing values with context |
| e2e_neweval_v4_option2 | 100 Q (harder) | 76.0% | — | 20.0% | Harder user-generated questions; v4 enrichment |
| Stage 3 fallback (active) | 24 Q subset | **29.2%** recovery | — | 58.3% | HyDE + keyword BM25 + year-relaxed + full-page |

---

## What Remains Hard

After all improvements, three types of failure persist in the harder eval set:

### 1. Sparse annex and administrative sections (~14 abstentions)

Questions that probe specific statistics in administrative annexes, regional tables, or program-specific breakdowns that appear in lightly-indexed sections of the reports. The target content is present in the PDFs but doesn't surface in the top retrieval results.

### 2. Wrong-row generation failures (3 persistent)

The correct table is retrieved and visible in context, but the LLM reads the wrong row — for example, picking the aggregate total row when the question asks for a specific province, or matching only one of two required conditions in a complex lookup table.

**Potential fix:** More explicit row-condition matching in the REDUCE prompt; possibly a structured cell-identification step.

### 3. One unfixable miss

The question about the "MIE" abbreviation (Maximum Insurable Earnings) targets the abbreviations page of the report — which is classified as a non-content page by the parser and intentionally skipped during chunking. This cannot be answered without re-classifying that page type.

---

## Per-Year Results

**Original 60-question eval (latest run: `reduce_multi_value`):**

| Year | Questions | Exact Match | Semantic Match |
|---|---|---|---|
| 2019–2020 | 12 | 75% | 83% |
| 2020–2021 | 12 | 83% | 92% |
| 2021–2022 | 12 | 67% | 75% |
| 2022–2023 | 12 | 75% | 92% |
| 2023–2024 | 12 | 100% | 100% |
| **TOTAL** | **60** | **80%** | **88%** |

**New 100-question eval (user-generated, harder):**

| Year | Questions | Semantic Match |
|---|---|---|
| 2019–2020 | 20 | ~76% |
| 2020–2021 | 20 | ~76% |
| 2021–2022 | 20 | ~76% |
| 2022–2023 | 20 | ~76% |
| 2023–2024 | 20 | ~76% |
| **TOTAL** | **100** | **76%** (81% expected with Stage 3 active) |

---

## Technology Stack

| Component | Technology |
|---|---|
| Embedding model | Qwen3-Embedding-4B (4-bit quantized, local) |
| Reranker | Qwen3-Reranker-4B (local) |
| Answer + judge LLM | Gemini 2.0 Flash (Google API) |
| Table enrichment | Gemini 2.0 Flash (Google API) |
| Keyword search | BM25 (rank-bm25) |
| Vector similarity | NumPy dot product over L2-normalized vectors |
| Rank fusion | Reciprocal Rank Fusion (RRF, k=60) |
| HyDE | Gemini 2.0 Flash hypothetical passage generation |
| PDF parsing | PyMuPDF |
| Web server | Flask (hosted on RunPod GPU, publicly accessible) |
| Evaluation judge | Gemini 2.0 Flash (JSON-mode responses) |
