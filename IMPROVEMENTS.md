# RAG Pipeline — Improvement Roadmap

Documented improvements organized by pipeline stage. Each item names the **general technique** (transferable to any RAG project) and describes how it applies to this EI MAR codebase specifically.

Current baseline (2026-02-21): **83.3% Answered@5**, 15% Answered@1, hybrid retrieval (dense + BM25 + RRF), top-k=5, 1500-char query-centered snippets.

---

## Stage 1 — Chunking & Enrichment

### 1. Contextual Chunk Headers
**General technique:** Prepend hierarchical context (breadcrumb, section title) to every chunk before embedding — not just stored as metadata.

**Why it helps:** The embedding captures *where* the chunk lives in the document, not just *what* it says. A chunk about "average weekly benefit rate" becomes distinguishable across the 5 fiscal years.

**Implementation:** In `embed.py`, when building the text to embed, prepend `section_breadcrumb` to the chunk body:
```python
embed_text = f"Section: {row['section_breadcrumb']}\n\n{chunk_body}"
```
Requires re-running `embed.py` (vectors.npy + manifest.jsonl).

**Expected impact:** 10–20% retrieval improvement. Well-documented in the literature.

---

### 2. Better Table text_nl (Row Enumeration)
**General technique:** Ensure NL descriptions of structured data enumerate specific values rather than summarizing abstractly.

**Why it helps:** Current text_nl likely says *"table showing benefit rates by gender and province."* BM25 needs the word "women" alongside "$99" to match a query asking for women's rates. Row enumeration gives BM25 that signal.

**Implementation:** Update the enrichment prompt in `enrich_tables.py` to instruct the model to enumerate key rows:
```
"List every row's values explicitly. Format: '[Row label]: [value]'"
```
Re-run `enrich_tables.py` — cached entries won't be recomputed unless the prompt changes the cache key.

**Fixes:** Demographic subgroup failures (women, age 24 and under).

---

### 3. Parent-Child Chunk Linking
**General technique:** Track relationships between chunks at index time so that retrieving one chunk can automatically surface related chunks (e.g. the table referenced by a narrative paragraph).

**Why it helps:** A narrative chunk often says *"as shown in Table 4.2..."* — the exact numbers are in the adjacent table chunk. If the narrative is retrieved without the table, the generation LLM sees an incomplete picture.

**Implementation (no re-chunking required):**
- In `retrieve.py`, after getting top-k results, do a secondary lookup:
  - Find all table/chart chunks sharing the same `section_breadcrumb` and overlapping `page_number` range.
  - Append them to the context passed to the generation LLM (not to the ranked results list).

**Alternatively (cleaner, requires re-chunking):**
- In `chunk.py`, when creating a narrative chunk, record adjacent table/chart chunk IDs in a `related_chunks` field.

**Fixes:** LMP-type failures where the exact number lives in an adjacent table.

---

### 4. Semantic Chunking
**General technique:** Split documents at natural semantic boundaries (topic shifts, heading changes) rather than fixed character counts.

**Why it helps:** Fixed-size chunking can split a key fact across two chunks. Semantic chunking keeps related sentences together.

**Implementation:** Replace the current character-count grouping in `chunk.py` with a semantic similarity threshold. Library: `semantic-text-splitter` (Rust-backed, fast).

**Trade-off:** Chunks become variable-length; some may be very long. Pair with a max-length cap.

---

### 5. Chart Enrichment with Vision Model
**General technique:** Use a multimodal LLM to generate text descriptions of chart/figure images so they become searchable.

**Why it helps:** Chart chunks currently have no text_nl — they are completely unsearchable by content. Charts often contain summary statistics that appear nowhere else in the document.

**Implementation:**
- Extract chart images during `parse_pdf.py` (pages classified as `table_heavy` with chart markers).
- Run a vision model (Qwen-VL, GPT-4o-mini, or Qwen Omni on RunPod) to generate `text_nl` for each chart.
- Add to `enrich_tables.py` or a new `enrich_charts.py`.

**Note:** Requires cloud GPU or a local multimodal model. See RunPod deployment notes.

---

## Stage 2 — Indexing & Embedding

### 6. Separate Dense Indexes per Chunk Type
**General technique:** Maintain separate vector indexes for different content types and fuse at retrieval time.

**Why it helps:** Narrative prose and markdown tables have very different language distributions. A single shared index forces the embedding model to reconcile both, reducing precision for each.

**Implementation:** In `embed.py`, save separate `vectors_narrative.npy` and `vectors_table.npy`. In `retrieve.py`, run dense retrieval against both and merge with RRF.

---

### 7. Fine-Tune the Embedding Model
**General technique:** Use (question, relevant chunk) pairs from the eval set as contrastive training signal to adapt the embedding model to domain-specific language.

**Why it helps:** Generic embedding models don't know EI terminology. Even 60–200 training pairs can measurably improve domain recall.

**Implementation:** Use `sentence-transformers` with `MultipleNegativesRankingLoss`. The existing `data/eval/eval_set.json` (60 pairs) is the starting dataset. Expand to 200+ pairs for better coverage.

**Trade-off:** Requires GPU time. Most impactful after other quick wins are exhausted.

---

## Stage 3 — Retrieval

### 8. Query Understanding Stage (Stage 1 LLM)
**General technique:** Pre-retrieval LLM call that extracts structured filters from the natural language query before retrieval runs.

**Why it helps:** Queries like *"How many EI claims in Newfoundland in the first half of 2021-22?"* contain three implicit filters: fiscal year, province, and time period. Without extracting these, retrieval returns national/aggregate data instead.

**Implementation:** A fast Gemini/Claude call before retrieval that returns:
```json
{
  "fiscal_year": "2021-2022",
  "province": "Newfoundland and Labrador",
  "chunk_type": "table",
  "time_period": "first half"
}
```
Apply `fiscal_year` and `chunk_type` as hard filters in `retrieve_hybrid()`. Use `province` to boost BM25 tokens or as a post-filter on `section_breadcrumb`.

**Fixes:** Geographic failures, temporal specificity failures.

---

### 9. HyDE — Hypothetical Document Embeddings
**General technique:** Instead of embedding the raw question, generate a hypothetical answer first, then embed that for retrieval.

**Why it helps:** Questions and answers live in different parts of the embedding space. A hypothetical answer *reads like a chunk* — it uses the same vocabulary and sentence structure as the documents, producing a much closer embedding match.

**Implementation:**
1. Pass the question to a fast LLM: *"Write a one-paragraph answer to this question as if from a government policy report."*
2. Embed the hypothetical answer instead of (or in addition to) the question.
3. Use RRF to fuse HyDE results with standard query embedding results.

**Expected impact:** Significant boost for factual, specific questions.

---

### 10. Metadata-Aware BM25 Corpus
**General technique:** Inject metadata tokens into the BM25 document representation so keyword search can match on structured fields.

**Why it helps:** A query mentioning "Newfoundland" should match chunks whose `section_breadcrumb` contains "Newfoundland" even if the body text uses "NL" as abbreviation.

**Implementation:** In `_chunk_to_bm25_text()` in `retrieve.py`, append metadata fields to the tokenized text:
```python
meta = f"province:{row.get('province','')} year:{row['fiscal_year']}"
return f"{body} {meta}"
```
Requires rebuilding the BM25 corpus (`data/embeddings/bm25_corpus.json`).

---

### 11. Larger Candidate Pool
**General technique:** Retrieve more candidates than needed, then rerank down to final top-k.

**Why it helps:** Current candidate pool is `max(top_k * 4, 20) = 20`. A reranker can only improve what's in the pool — if the correct chunk is at rank 21, it's unreachable.

**Implementation:** In `retrieve_hybrid()`, increase candidate multiplier from 4× to 10×:
```python
candidate_k = max(top_k * 10, 50)
```
Minimal cost since reranking happens in the next step.

---

## Stage 4 — Reranking

### 12. Cross-Encoder Reranker
**General technique:** After bi-encoder retrieval, score each (query, chunk) pair with a cross-encoder that reads both texts jointly.

**Why it helps:** Bi-encoders (current) embed query and chunk independently — they can't model fine-grained interactions. Cross-encoders see both at once and are far more accurate at judging relevance. Would push Answered@1 from 15% to ~45–55%.

**Implementation:**
```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = reranker.predict([(query, chunk["text"]) for chunk in candidates])
reranked = sorted(zip(scores, candidates), reverse=True)
```
Retrieve top 50 candidates, rerank, return top 5. Runs fast on CPU.

---

### 13. LLM-Based Reranker (RankGPT)
**General technique:** Use the generation LLM itself to rank candidates by relevance before answering.

**Why it helps:** Higher quality than a small cross-encoder, especially for domain-specific language. The LLM can reason about relevance, not just match tokens.

**Implementation:** Pass top-20 candidates to the LLM with a ranking prompt. More expensive than a cross-encoder but no additional model to deploy.

**Trade-off:** Adds latency and cost. Use cross-encoder first; upgrade to LLM reranker if needed.

---

## Stage 5 — Context Assembly

### 14. Contextual Compression
**General technique:** Before passing retrieved chunks to the generation LLM, extract only the sentences directly relevant to the query.

**Why it helps:** Current approach passes up to 1500 chars per chunk × 5 chunks = 7500 chars. Most of that is noise. Compression reduces token cost and improves answer focus.

**Implementation:** For each retrieved chunk, run a short LLM call: *"Extract only the sentences relevant to: {query}"*. Or use a lighter extractive summarizer.

**Trade-off:** Adds latency. Most valuable when generation LLM has a small context window.

---

### 15. Chunk De-duplication
**General technique:** Before passing context to the generation LLM, remove chunks that contain near-identical content.

**Why it helps:** Across 5 fiscal years, many chunks contain the same boilerplate (EI program descriptions, definitions). Duplicate context wastes tokens and can confuse the LLM.

**Implementation:** Cosine similarity check between retrieved chunks — if similarity > 0.95, drop the lower-ranked duplicate.

---

### 16. Lost-in-the-Middle Ordering
**General technique:** Place the most relevant chunks at the beginning and end of the context window, not in the middle.

**Why it helps:** Research shows LLMs perform worse when key information is buried in the middle of a long context. Simple reordering improves answer quality with zero cost.

**Implementation:** Sort retrieved chunks so rank-1 is first, rank-2 is last, rank-3 is second, etc. (interleave from both ends).

---

## Stage 6 — Generation

### 17. Answer with Citations
**General technique:** Prompt the generation LLM to cite its source chunk(s) inline.

**Why it helps:** Builds user trust and allows verification. All necessary metadata (fiscal year, page, section) is already in the chunk manifest.

**Implementation:** Add to system prompt: *"After your answer, cite the source as [FY | page X | Section]."*

---

### 18. Faithfulness Check
**General technique:** Post-generation verification that the answer is grounded in the retrieved chunks (not hallucinated).

**Why it helps:** Catches cases where the LLM generates plausible-sounding but incorrect numbers.

**Implementation:** Second LLM call: *"Is this answer supported by the provided chunks? Answer YES or NO with reasoning."* Flag low-confidence answers in the UI.

---

### 19. Confidence / Abstention
**General technique:** Instruct the LLM to say "I don't have enough information" rather than hallucinating when retrieved chunks are insufficient.

**Why it helps:** For a government policy tool, a wrong answer is worse than no answer.

**Implementation:** Add to system prompt: *"If the retrieved chunks do not contain enough information to answer confidently, say: 'The retrieved documents do not contain this information.'"*

---

## Stage 7 — Evaluation & Feedback

### 20. End-to-End Evaluation
**General technique:** Evaluate the final generated answer, not just retrieval quality.

**Why it helps:** Current eval measures whether the right chunk was retrieved. It doesn't measure whether the generation LLM produced the correct answer from that chunk.

**Implementation:** Extend `evaluate_llm_judge.py` to also pass the generated answer and check it against `ground_truth` directly.

---

### 21. RAGAs Framework
**General technique:** Standard RAG evaluation suite measuring four dimensions: context precision, context recall, answer faithfulness, answer relevance.

**Why it helps:** Gives a multi-dimensional view of where the pipeline is failing — retrieval vs. generation vs. faithfulness.

**Implementation:** `pip install ragas`. Requires an LLM judge (already have Gemini configured).

---

### 22. User Feedback Loop
**General technique:** Collect thumbs up/down signals from end users and use them to improve the pipeline over time.

**Why it helps:** Real user queries differ from eval set queries. Feedback identifies blind spots not covered by the synthetic eval set.

**Implementation:** Add feedback endpoint to `server.py`. Log (query, retrieved_chunk_ids, answer, feedback) to a file or database. Use as future fine-tuning data.

---

## Summary Table

| # | Technique | Stage | Effort | Impact | Fixes |
|---|---|---|---|---|---|
| 1 | Contextual chunk headers | Chunking | Low | High | Overall recall |
| 2 | Better text_nl (row enumeration) | Enrichment | Low | Medium | Demographic failures |
| 3 | Parent-child chunk linking | Chunking | Medium | Medium | Context boundary failures |
| 4 | Semantic chunking | Chunking | Medium | Medium | Split-fact failures |
| 5 | Chart enrichment (vision model) | Enrichment | High | Medium | Chart retrieval |
| 6 | Separate indexes per type | Indexing | Low | Low-Med | Table vs narrative precision |
| 7 | Fine-tune embedding model | Indexing | High | High | Domain-specific recall |
| 8 | Query understanding LLM | Retrieval | Medium | High | Geographic + temporal failures |
| 9 | HyDE | Retrieval | Medium | High | All factual queries |
| 10 | Metadata-aware BM25 | Retrieval | Low | Medium | Geographic failures |
| 11 | Larger candidate pool | Retrieval | Low | Low | Reranker prerequisite |
| 12 | Cross-encoder reranker | Reranking | Medium | High | @1 rate (15% → ~50%) |
| 13 | LLM-based reranker | Reranking | Medium | High | @1 rate (alternative to 12) |
| 14 | Contextual compression | Context | Medium | Medium | Token efficiency |
| 15 | Chunk de-duplication | Context | Low | Low | Noise reduction |
| 16 | Lost-in-the-middle ordering | Context | Low | Low | Generation quality |
| 17 | Answer with citations | Generation | Low | High | Trust + verifiability |
| 18 | Faithfulness check | Generation | Medium | High | Hallucination prevention |
| 19 | Confidence / abstention | Generation | Low | High | Reliability |
| 20 | End-to-end evaluation | Eval | Medium | High | Full pipeline visibility |
| 21 | RAGAs framework | Eval | Medium | Medium | Structured benchmarking |
| 22 | User feedback loop | Eval | Medium | High | Long-term improvement |

---

## Recommended Implementation Order

**Phase 1 — Quick wins (no re-indexing required):**
1, 2, 10, 15, 16, 17, 19

**Phase 2 — Core retrieval improvements (require re-indexing):**
3, 6, 8, 11, 12

**Phase 3 — Advanced retrieval:**
9, 13, 4

**Phase 4 — Generation quality:**
14, 18, 20, 21

**Phase 5 — Long-term learning:**
5, 7, 22
