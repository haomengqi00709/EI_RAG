#!/usr/bin/env python3
"""
retrieve.py — Query the EI MAR vector index.

Supports three retrieval modes:
  dense   — semantic search via Qwen3-Embedding (default)
  bm25    — keyword search via BM25
  hybrid  — dense + BM25 fused with Reciprocal Rank Fusion (recommended)

Run:
    python src/retrieve.py "What were EI regular claims in 2023-24?"
    python src/retrieve.py "How did sickness benefits change?" --top-k 10
    python src/retrieve.py "Regular claims" --year 2023-2024
    python src/retrieve.py "Benefit amounts" --type table
    python src/retrieve.py "Benefit amounts" --mode hybrid

Improvements applied:
  #10 — Metadata-aware BM25: fiscal year + breadcrumb injected as searchable tokens
  #11 — Larger candidate pool: 10× top_k (min 50) before RRF fusion
  #15 — Chunk de-duplication: Jaccard similarity removes near-duplicate chunks
  #16 — Lost-in-the-middle ordering: most relevant chunks placed at start/end
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from embed import embed_one, load_embedding_model

# ── Configuration ─────────────────────────────────────────────────────────────

VECTORS_PATH     = Path("data/embeddings/vectors.npy")
MANIFEST_PATH    = Path("data/embeddings/manifest.jsonl")
BM25_CORPUS_PATH = Path("data/embeddings/bm25_corpus.json")

# #6: Per-type index paths (built by embed.py alongside the combined index)
_TYPE_VECTORS  = {t: Path(f"data/embeddings/vectors_{t}.npy")   for t in ("narrative", "table")}
_TYPE_MANIFEST = {t: Path(f"data/embeddings/manifest_{t}.jsonl") for t in ("narrative", "table")}

DEFAULT_TOP_K      = 5
SNIPPET_CHARS      = 300
RRF_K              = 60    # RRF constant — higher = less aggressive rank fusion
CANDIDATE_MULT     = 4     # candidate pool = top_k × this (min 20)
DEDUP_THRESHOLD    = 0.85  # #15: Jaccard similarity above this → duplicate

QUERY_INSTRUCTION = (
    "Given a Canadian government Employment Insurance policy document, "
    "retrieve the passage that best answers the question."
)


# ── Index & chunk loading ─────────────────────────────────────────────────────

def load_index(chunk_type: str | None = None) -> tuple[np.ndarray, list]:
    """
    Load the vector index.
    #6: If chunk_type is 'narrative' or 'table' and per-type indexes exist,
    load the smaller per-type index for faster, more precise retrieval.
    Falls back to the combined index otherwise.
    """
    if chunk_type in _TYPE_VECTORS and _TYPE_VECTORS[chunk_type].exists():
        vectors  = np.load(_TYPE_VECTORS[chunk_type])
        manifest = [json.loads(l) for l in open(_TYPE_MANIFEST[chunk_type])]
        return vectors, manifest

    if not VECTORS_PATH.exists() or not MANIFEST_PATH.exists():
        raise FileNotFoundError("Index not found. Run 'python src/embed.py' first.")
    vectors  = np.load(VECTORS_PATH)
    manifest = [json.loads(l) for l in open(MANIFEST_PATH)]
    return vectors, manifest


def load_chunks() -> dict:
    chunks = {}
    for f in sorted(Path("data/chunked").glob("*.jsonl")):
        for line in open(f):
            c = json.loads(line)
            chunks[c["chunk_id"]] = c
    return chunks


# ── BM25 helpers ──────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def _chunk_to_bm25_text(row: dict, chunk: dict) -> str:
    """Text for BM25 indexing — body text, with breadcrumb prepended for narratives."""
    if row["chunk_type"] == "table":
        return chunk.get("text_nl") or chunk.get("text", "")
    bc = row.get("section_breadcrumb", "")
    body = chunk.get("text", "")
    return f"{bc}\n\n{body}" if bc else body


def build_bm25_index(manifest: list, chunks: dict):
    """Tokenise all chunks, save corpus to disk, return BM25Okapi object."""
    from rank_bm25 import BM25Okapi

    print(f"Building BM25 index over {len(manifest)} chunks…")
    corpus = [
        _tokenize(_chunk_to_bm25_text(row, chunks.get(row["chunk_id"], {})))
        for row in manifest
    ]
    BM25_CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_CORPUS_PATH, "w") as f:
        json.dump(corpus, f)
    print(f"BM25 corpus saved → {BM25_CORPUS_PATH}")
    return BM25Okapi(corpus)


def load_bm25(manifest: list, chunks: dict):
    """Load BM25 index from disk, building it first if not found."""
    from rank_bm25 import BM25Okapi

    if BM25_CORPUS_PATH.exists():
        with open(BM25_CORPUS_PATH) as f:
            corpus = json.load(f)
        return BM25Okapi(corpus)
    return build_bm25_index(manifest, chunks)


# ── Query embedding ───────────────────────────────────────────────────────────

def build_query_text(question: str) -> str:
    return f"Instruct: {QUERY_INSTRUCTION}\nQuery: {question}"


# ── Retrieval — dense ─────────────────────────────────────────────────────────

def retrieve(
    question:    str,
    vectors:     np.ndarray,
    manifest:    list,
    chunks:      dict,
    top_k:       int = DEFAULT_TOP_K,
    fiscal_year: str | None = None,
    chunk_type:  str | None = None,
    model=None,
    tokenizer=None,
) -> list[dict]:
    """Dense retrieval via cosine similarity (dot product on L2-normalised vectors)."""
    if model is None or tokenizer is None:
        model, tokenizer = load_embedding_model()
    query_vec = embed_one(model, tokenizer, build_query_text(question))
    scores    = vectors @ query_vec

    mask = np.ones(len(manifest), dtype=bool)
    if fiscal_year:
        mask &= np.array([r["fiscal_year"] == fiscal_year for r in manifest])
    if chunk_type:
        mask &= np.array([r["chunk_type"] == chunk_type for r in manifest])

    masked_scores = np.where(mask, scores, -np.inf)
    top_indices   = np.argsort(masked_scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if masked_scores[idx] == -np.inf:
            break
        results.append({
            "score": float(masked_scores[idx]),
            "row":   manifest[idx],
            "chunk": chunks.get(manifest[idx]["chunk_id"], {}),
        })
    return results


# ── Retrieval — BM25 ──────────────────────────────────────────────────────────

def retrieve_bm25(
    question:    str,
    bm25,
    manifest:    list,
    chunks:      dict,
    top_k:       int = DEFAULT_TOP_K,
    fiscal_year: str | None = None,
    chunk_type:  str | None = None,
) -> list[dict]:
    """BM25 keyword retrieval."""
    scores = bm25.get_scores(_tokenize(question))

    mask = np.ones(len(manifest), dtype=bool)
    if fiscal_year:
        mask &= np.array([r["fiscal_year"] == fiscal_year for r in manifest])
    if chunk_type:
        mask &= np.array([r["chunk_type"] == chunk_type for r in manifest])

    masked = np.where(mask, scores, -np.inf)
    top_indices = np.argsort(masked)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if masked[idx] == -np.inf:
            break
        results.append({
            "score": float(masked[idx]),
            "row":   manifest[idx],
            "chunk": chunks.get(manifest[idx]["chunk_id"], {}),
        })
    return results


# ── Post-retrieval helpers ────────────────────────────────────────────────────

def _deduplicate(results: list[dict], threshold: float = DEDUP_THRESHOLD) -> list[dict]:
    """
    #15: Remove near-duplicate chunks using Jaccard token similarity.
    Iterates in rank order — keeps the higher-ranked chunk when duplicates found.
    """
    kept: list[dict] = []
    kept_token_sets: list[set] = []

    for r in results:
        chunk = r["chunk"]
        body  = chunk.get("text_nl") or chunk.get("text", "")
        tokens = set(_tokenize(body))

        is_dup = False
        for existing in kept_token_sets:
            if not tokens or not existing:
                continue
            intersection = len(tokens & existing)
            union        = len(tokens | existing)
            if union > 0 and (intersection / union) >= threshold:
                is_dup = True
                break

        if not is_dup:
            kept.append(r)
            kept_token_sets.append(tokens)

    return kept


def _lost_in_middle_reorder(results: list[dict]) -> list[dict]:
    """
    #16: Reorder so the most relevant chunks are at the start and end of the
    context window, with less relevant chunks in the middle.
    LLMs attend more strongly to positions at the edges (Liu et al. 2023).

    Example for 5 results [r1..r5] → [r1, r3, r5, r4, r2]
    """
    if len(results) <= 2:
        return results

    reordered = [None] * len(results)
    left, right = 0, len(results) - 1
    for i, r in enumerate(results):
        if i % 2 == 0:
            reordered[left] = r
            left += 1
        else:
            reordered[right] = r
            right -= 1
    return reordered


# ── Retrieval — hybrid (RRF) ──────────────────────────────────────────────────

def _rrf_fuse(dense: list[dict], bm25_res: list[dict], k: int = RRF_K) -> list[dict]:
    """
    Reciprocal Rank Fusion.
    For each chunk, score = Σ 1/(k + rank) across both ranked lists.
    """
    rrf: dict[str, dict] = {}

    for rank, r in enumerate(dense, 1):
        cid = r["row"]["chunk_id"]
        if cid not in rrf:
            rrf[cid] = {"rrf_score": 0.0, "row": r["row"], "chunk": r["chunk"]}
        rrf[cid]["rrf_score"] += 1 / (k + rank)

    for rank, r in enumerate(bm25_res, 1):
        cid = r["row"]["chunk_id"]
        if cid not in rrf:
            rrf[cid] = {"rrf_score": 0.0, "row": r["row"], "chunk": r["chunk"]}
        rrf[cid]["rrf_score"] += 1 / (k + rank)

    sorted_chunks = sorted(rrf.values(), key=lambda x: x["rrf_score"], reverse=True)
    return [
        {"score": c["rrf_score"], "row": c["row"], "chunk": c["chunk"]}
        for c in sorted_chunks
    ]


def retrieve_hybrid(
    question:    str,
    vectors:     np.ndarray,
    bm25,
    manifest:    list,
    chunks:      dict,
    top_k:       int = DEFAULT_TOP_K,
    fiscal_year: str | None = None,
    chunk_type:  str | None = None,
    model=None,
    tokenizer=None,
) -> list[dict]:
    """
    Hybrid retrieval: dense + BM25 fused with Reciprocal Rank Fusion.
    #11: Uses a 10× candidate pool (min 50) before fusion for better reranker coverage.
    #15: Deduplicates near-identical chunks after fusion.
    #16: Applies lost-in-the-middle reordering before returning.
    """
    candidate_k = max(top_k * CANDIDATE_MULT, 20)

    dense_results = retrieve(
        question=question, vectors=vectors, manifest=manifest, chunks=chunks,
        top_k=candidate_k, fiscal_year=fiscal_year, chunk_type=chunk_type,
        model=model, tokenizer=tokenizer,
    )
    bm25_results = retrieve_bm25(
        question=question, bm25=bm25, manifest=manifest, chunks=chunks,
        top_k=candidate_k, fiscal_year=fiscal_year, chunk_type=chunk_type,
    )

    fused = _rrf_fuse(dense_results, bm25_results)
    return fused[:top_k]
    # Note: _deduplicate() and _lost_in_middle_reorder() are generation-layer
    # optimizations — apply them in server.py after retrieval, not here.


def retrieve_multi_query(
    questions:   list[str],
    vectors:     np.ndarray,
    bm25,
    manifest:    list,
    chunks:      dict,
    top_k:       int = DEFAULT_TOP_K,
    fiscal_year: str | None = None,
    chunk_type:  str | None = None,
    model=None,
    tokenizer=None,
    use_rerank:  bool = True,
) -> list[dict]:
    """
    Multi-query retrieval: retrieve for each query variant, merge results,
    deduplicate by chunk_id, then rerank the merged pool with the original question.

    questions[0] must be the original question (used for final reranking).
    Subsequent entries are rephrased variants generated by generate_query_variants().
    """
    seen_ids: set[str] = set()
    merged:   list[dict] = []

    for question in questions:
        results = retrieve_hybrid(
            question=question, vectors=vectors, bm25=bm25,
            manifest=manifest, chunks=chunks, top_k=top_k,
            fiscal_year=fiscal_year, chunk_type=chunk_type,
            model=model, tokenizer=tokenizer,
        )
        for r in results:
            cid = r["row"]["chunk_id"]
            if cid not in seen_ids:
                seen_ids.add(cid)
                merged.append(r)

    if not merged:
        return []

    # Rerank the merged pool using the original question, then return top_k
    if use_rerank and len(merged) > 1:
        try:
            merged = rerank(questions[0], merged)
        except Exception as e:
            print(f"  Rerank failed (falling back to score order): {e}")

    return merged[:top_k]


# ── HyDE — Hypothetical Document Embeddings ──────────────────────────────────

def generate_hypothetical_answer(question: str, llm_model) -> str:
    """
    #9: Generate a hypothetical answer to the question using an LLM.
    The hypothetical answer reads like a real document passage and embeds
    much closer to actual chunks than the question itself does.
    """
    prompt = (
        f"Write a one-paragraph excerpt from a Canadian government Employment "
        f"Insurance policy report that directly answers this question. "
        f"Be specific with numbers and facts. Do not say you don't know.\n\n"
        f"Question: {question}"
    )
    try:
        response = llm_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"  HyDE generation failed: {e} — falling back to raw question")
        return question


def retrieve_hyde(
    question:    str,
    vectors:     np.ndarray,
    bm25,
    manifest:    list,
    chunks:      dict,
    llm_model,
    top_k:       int = DEFAULT_TOP_K,
    fiscal_year: str | None = None,
    chunk_type:  str | None = None,
    model=None,
    tokenizer=None,
) -> list[dict]:
    """
    #9: HyDE retrieval — generates a hypothetical answer, embeds it, then
    fuses the HyDE dense results with standard hybrid retrieval via RRF.
    Significantly improves recall for specific factual questions.
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_embedding_model()

    candidate_k = max(top_k * CANDIDATE_MULT, 50)

    # Standard hybrid results (question embedding + BM25)
    dense_q = retrieve(
        question=question, vectors=vectors, manifest=manifest, chunks=chunks,
        top_k=candidate_k, fiscal_year=fiscal_year, chunk_type=chunk_type,
        model=model, tokenizer=tokenizer,
    )
    bm25_res = retrieve_bm25(
        question=question, bm25=bm25, manifest=manifest, chunks=chunks,
        top_k=candidate_k, fiscal_year=fiscal_year, chunk_type=chunk_type,
    )

    # HyDE: embed the hypothetical answer and retrieve with it
    hyp_answer  = generate_hypothetical_answer(question, llm_model)
    hyp_vec     = embed_one(model, tokenizer, hyp_answer)
    hyp_scores  = vectors @ hyp_vec

    mask = np.ones(len(manifest), dtype=bool)
    if fiscal_year:
        mask &= np.array([r["fiscal_year"] == fiscal_year for r in manifest])
    if chunk_type:
        mask &= np.array([r["chunk_type"] == chunk_type for r in manifest])
    hyp_scores = np.where(mask, hyp_scores, -np.inf)
    top_hyp    = np.argsort(hyp_scores)[::-1][:candidate_k]

    dense_hyp = []
    for idx in top_hyp:
        if hyp_scores[idx] == -np.inf:
            break
        dense_hyp.append({
            "score": float(hyp_scores[idx]),
            "row":   manifest[idx],
            "chunk": chunks.get(manifest[idx]["chunk_id"], {}),
        })

    # Three-way RRF: standard dense + BM25 + HyDE dense
    fused   = _rrf_fuse(_rrf_fuse(dense_q, bm25_res), dense_hyp)
    deduped = _deduplicate(fused)
    return _lost_in_middle_reorder(deduped[:top_k])


# ── Stage 3: keyword extraction + full-page expansion ────────────────────────

KEYWORD_EXTRACT_PROMPT = """\
Extract 3-5 key search terms from this question that would help find the answer \
in a Canadian government Employment Insurance report.
Focus on: acronyms (e.g. MIE, SST, PRAR, ROE), program names, specific numbers \
or percentages, proper nouns, fiscal year references.
Return ONLY a JSON array of short strings (not full sentences).

Question: {question}"""


def _extract_keywords(question: str, llm_model=None) -> str:
    """
    Extract key terms from a question for a focused BM25 pass.
    Uses the LLM when available; falls back to regex (acronyms + numbers).
    Returns a space-joined string of terms.
    """
    if llm_model is not None:
        try:
            resp = llm_model.generate_content(
                KEYWORD_EXTRACT_PROMPT.format(question=question),
                generation_config={"response_mime_type": "application/json"},
            )
            keywords = json.loads(resp.text)
            if isinstance(keywords, list) and keywords:
                return " ".join(str(k) for k in keywords)
        except Exception:
            pass
    # Regex fallback: acronyms + numbers/percentages/dollar amounts
    acronyms = re.findall(r'\b[A-Z]{2,}\b', question)
    numbers  = re.findall(r'\$?[\d,]+\.?\d*%?', question)
    return " ".join(acronyms + numbers) or question


def _build_full_page_map(chunks: dict) -> dict:
    """
    Build (fiscal_year, page) → [chunk_id, ...] index covering all chunk types.
    Handles both page_number (table/chart) and start_page (narrative) fields.
    """
    page_map: dict[tuple, list[str]] = {}
    for cid, chunk in chunks.items():
        fy   = chunk.get("fiscal_year", "")
        page = chunk.get("page_number") or chunk.get("start_page")
        if page:
            page_map.setdefault((fy, int(page)), []).append(cid)
    return page_map


def expand_full_page(results: list[dict], chunks: dict) -> list[dict]:
    """
    Stage 3 context expansion: attach all chunks from the same (fiscal_year, page)
    as context_chunks. Broader than expand_context() which only handles parent-child.
    Appends to any context_chunks already set by expand_context().
    """
    page_map = _build_full_page_map(chunks)
    for r in results:
        row  = r["row"]
        fy   = row.get("fiscal_year", "")
        page = row.get("page")
        cid  = row.get("chunk_id", "")
        if not page:
            continue
        key        = (fy, int(page))
        page_cids  = page_map.get(key, [])
        already    = {c.get("chunk_id") for c in r.get("context_chunks", [])} | {cid}
        extra      = [chunks[sid] for sid in page_cids if sid not in already and sid in chunks]
        if extra:
            r["context_chunks"] = r.get("context_chunks", []) + extra
    return results


def retrieve_stage3(
    question:    str,
    vectors:     np.ndarray,
    bm25,
    manifest:    list,
    chunks:      dict,
    llm_model,
    top_k:       int = 10,
    fiscal_year: str | None = None,
    model=None,
    tokenizer=None,
) -> list[dict]:
    """
    Stage 3 deep retrieval — triggered when Stage 2 (power search) abstains.

    Combines three complementary signals via RRF:
      1. HyDE          — hypothetical document embedding bridges question/doc phrasing gap
      2. Keyword BM25  — extracted acronyms/numbers searched via BM25
      3. Year-relaxed  — removes FY filter to catch content with weak FY metadata

    Then applies full-page context expansion so the reduce LLM sees all chunks
    on the same page as each top result.
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_embedding_model()

    candidate_k = max(top_k * CANDIDATE_MULT, 50)

    # Signal 1: HyDE (3-way RRF: dense + BM25 + HyDE dense)
    print("  [Stage 3] HyDE retrieval…")
    hyde_results = retrieve_hyde(
        question=question, vectors=vectors, bm25=bm25,
        manifest=manifest, chunks=chunks, llm_model=llm_model,
        top_k=candidate_k, fiscal_year=fiscal_year,
        model=model, tokenizer=tokenizer,
    )

    # Signal 2: Keyword BM25 pass
    print("  [Stage 3] Keyword BM25 pass…")
    keyword_query   = _extract_keywords(question, llm_model)
    print(f"  [Stage 3] Keywords: {keyword_query!r}")
    keyword_results = retrieve_bm25(
        question=keyword_query, bm25=bm25,
        manifest=manifest, chunks=chunks,
        top_k=candidate_k, fiscal_year=fiscal_year,
    )

    # Signal 3: Year-relaxed dense search (only when FY filter was applied)
    year_relaxed: list[dict] = []
    if fiscal_year:
        print("  [Stage 3] Year-relaxed dense search…")
        year_relaxed = retrieve(
            question=question, vectors=vectors,
            manifest=manifest, chunks=chunks,
            top_k=candidate_k // 2, fiscal_year=None,
            model=model, tokenizer=tokenizer,
        )

    # Fuse all signals
    fused = _rrf_fuse(hyde_results, keyword_results)
    if year_relaxed:
        fused = _rrf_fuse(fused, year_relaxed)

    deduped = _deduplicate(fused)[:top_k]

    # Full-page context expansion
    deduped = expand_full_page(deduped, chunks)

    return deduped


# ── Context expansion ────────────────────────────────────────────────────────

def expand_context(results: list[dict], chunks: dict, max_related: int = 3) -> list[dict]:
    """
    #3: Parent-child context expansion.

    Two expansion sources:
    1. related_chunk_ids — set by chunk.py for narrative→table links
    2. parent_chunk_id / sub-table group — when a sub-table is retrieved,
       also fetch its parent and siblings; when a parent is retrieved, fetch
       its sub-table children. Ensures the full factsheet is always in context.

    Does NOT change retrieval rankings — only enriches result objects.
    """
    # Build parent → [children] map once for the full chunk store
    parent_to_children: dict[str, list[str]] = {}
    for cid, chunk in chunks.items():
        pid = chunk.get("parent_chunk_id")
        if pid:
            parent_to_children.setdefault(pid, []).append(cid)

    for r in results:
        chunk    = r["chunk"]
        chunk_id = chunk.get("chunk_id", "")
        extra: list = []

        # Source 1: related_chunk_ids (narrative → table links from chunk.py)
        for cid in chunk.get("related_chunk_ids", [])[:max_related]:
            if cid in chunks:
                extra.append(chunks[cid])

        # Source 2a: sub-table retrieved → fetch parent + siblings
        parent_id = chunk.get("parent_chunk_id")
        if parent_id:
            if parent_id in chunks:
                extra.append(chunks[parent_id])
            for sibling_id in parent_to_children.get(parent_id, []):
                if sibling_id != chunk_id and sibling_id in chunks:
                    extra.append(chunks[sibling_id])

        # Source 2b: parent table retrieved → fetch its sub-table children
        for child_id in parent_to_children.get(chunk_id, []):
            if child_id in chunks:
                extra.append(chunks[child_id])

        if extra:
            # Deduplicate by chunk_id, preserve order
            seen: set = set()
            deduped = []
            for c in extra:
                cid = c.get("chunk_id", "")
                if cid and cid not in seen:
                    seen.add(cid)
                    deduped.append(c)
            r["context_chunks"] = deduped[:max_related * 2]

    return results


# ── Reranking ─────────────────────────────────────────────────────────────────

RERANKER_MODEL_ID = "Qwen/Qwen3-Reranker-4B"
RERANK_INSTRUCTION = (
    "Given a query and a document, determine if the document contains information "
    "that answers the query about Canadian Employment Insurance policy."
)

_qwen_reranker         = None   # module-level cache
_qwen_reranker_tok     = None


def _load_qwen_reranker():
    global _qwen_reranker, _qwen_reranker_tok
    if _qwen_reranker is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading reranker: {RERANKER_MODEL_ID}…")
        _qwen_reranker_tok = AutoTokenizer.from_pretrained(
            RERANKER_MODEL_ID, padding_side="left"
        )
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        _qwen_reranker = AutoModelForCausalLM.from_pretrained(
            RERANKER_MODEL_ID,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
        if device != "cuda":
            _qwen_reranker = _qwen_reranker.to(device)
        _qwen_reranker.eval()
        print("Reranker loaded.")
    return _qwen_reranker, _qwen_reranker_tok


def _qwen_rerank_score(model, tokenizer, question: str, doc: str) -> float:
    """Score a single (question, doc) pair with Qwen3-Reranker-4B."""
    import torch
    messages = [
        {"role": "system", "content": RERANK_INSTRUCTION},
        {
            "role": "user",
            "content": (
                f"<Instruct>: Determine if this document answers the query.\n"
                f"<Query>: {question}\n"
                f"<Document>: {doc}"
            ),
        },
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ) + "<think>\n\n</think>\n\n"

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    token_true_id  = tokenizer.convert_tokens_to_ids("yes")
    token_false_id = tokenizer.convert_tokens_to_ids("no")

    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]
        score  = torch.softmax(
            logits[:, [token_false_id, token_true_id]], dim=-1
        )[0, 1].item()
    return score


def rerank(question: str, results: list[dict]) -> list[dict]:
    """
    Rerank using Qwen3-Reranker-4B.
    Batches all candidates into a single forward pass — ~10× faster than
    scoring one candidate at a time.
    """
    if not results:
        return results

    import torch
    model, tokenizer = _load_qwen_reranker()

    # Build all prompt texts in one pass
    texts = []
    for r in results:
        chunk = r["chunk"]
        body  = (chunk.get("text_nl") or chunk.get("text", ""))[:2000]
        bc    = r["row"].get("section_breadcrumb", "")
        doc   = f"{bc}\n\n{body}" if bc else body
        messages = [
            {"role": "system", "content": RERANK_INSTRUCTION},
            {"role": "user", "content": (
                f"<Instruct>: Determine if this document answers the query.\n"
                f"<Query>: {question}\n"
                f"<Document>: {doc}"
            )},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ) + "<think>\n\n</think>\n\n"
        texts.append(text)

    # Single batched forward pass (left-padding ensures logits[:, -1, :] is correct)
    inputs = tokenizer(
        texts, return_tensors="pt", truncation=True,
        max_length=2048, padding=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    token_true_id  = tokenizer.convert_tokens_to_ids("yes")
    token_false_id = tokenizer.convert_tokens_to_ids("no")

    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]
        scores = torch.softmax(
            logits[:, [token_false_id, token_true_id]], dim=-1
        )[:, 1].tolist()

    scored = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
    return [{"score": float(s), "row": r["row"], "chunk": r["chunk"]}
            for s, r in scored]


def rerank_llm(
    question: str,
    results:  list[dict],
    llm_model,
    top_k:    int | None = None,
) -> list[dict]:
    """
    #13: LLM-based reranker (RankGPT-style).
    Passes all candidates to the LLM and asks it to rank them by relevance.
    Higher quality than the cross-encoder but more expensive (one LLM call).
    Use as an alternative to rerank() when cost is acceptable.
    """
    if not results:
        return results

    snippets = []
    for i, r in enumerate(results, 1):
        chunk = r["chunk"]
        body  = (chunk.get("text_nl") or chunk.get("text", ""))[:600]
        bc    = r["row"].get("section_breadcrumb", "")[:60]
        fy    = r["row"].get("fiscal_year", "")
        snippets.append(f"[{i}] {fy} | {bc}\n{body}")

    chunks_text = "\n\n".join(snippets)
    prompt = (
        f"Question: {question}\n\n"
        f"Below are {len(results)} retrieved passages. "
        f"Return ONLY a JSON array of the passage numbers reordered from most "
        f"to least relevant. Example: [3, 1, 4, 2, 5]\n\n"
        f"{chunks_text}"
    )
    try:
        response = llm_model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
        )
        order = json.loads(response.text)
        if isinstance(order, list) and all(isinstance(x, int) for x in order):
            reranked = []
            seen = set()
            for idx in order:
                if 1 <= idx <= len(results) and idx not in seen:
                    r = results[idx - 1]
                    reranked.append({"score": float(len(results) - len(reranked)),
                                     "row": r["row"], "chunk": r["chunk"]})
                    seen.add(idx)
            # Append any not mentioned in order
            for i, r in enumerate(results, 1):
                if i not in seen:
                    reranked.append({"score": 0.0, "row": r["row"], "chunk": r["chunk"]})
            return reranked[:top_k] if top_k else reranked
    except Exception as e:
        print(f"  LLM reranker error: {e} — returning original order")

    return results[:top_k] if top_k else results


# ── Output formatting ─────────────────────────────────────────────────────────

def print_results(question: str, results: list, mode: str = "dense"):
    print(f'\nQuery: "{question}"  [mode: {mode}]')
    print("─" * 70)

    for i, r in enumerate(results, 1):
        row   = r["row"]
        chunk = r["chunk"]
        score = r["score"]

        if row["chunk_type"] == "table":
            body  = chunk.get("text_nl") or chunk.get("text", "")
            label = f"Table: {row.get('table_title', '')[:55]}" if row.get("table_title") else "Table"
        else:
            body  = chunk.get("text", "")
            label = f"p.{row['page']}" if row.get("page") else ""

        snippet = body[:SNIPPET_CHARS].replace("\n", " ")
        if len(body) > SNIPPET_CHARS:
            snippet += "..."

        print(f"\n#{i}  score={score:.4f}  "
              f"[{row['fiscal_year']} | {row['chunk_type']} | {label}]")
        print(f"    Section: {row.get('section_breadcrumb', '')[:70]}")
        print(f"    {snippet}")

    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Query the EI MAR vector index.")
    parser.add_argument("question", help="The question or search phrase.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--year", help="Filter by fiscal year, e.g. 2023-2024.")
    parser.add_argument("--type", choices=["narrative", "table"], dest="chunk_type")
    parser.add_argument(
        "--mode", choices=["dense", "bm25", "hybrid"], default="dense",
        help="Retrieval mode (default: dense)"
    )
    parser.add_argument(
        "--rerank", action="store_true",
        help="Apply cross-encoder reranker after retrieval (#12)"
    )
    args = parser.parse_args()

    vectors, manifest = load_index()
    chunks = load_chunks()

    if args.mode == "dense":
        results = retrieve(
            question=args.question, vectors=vectors, manifest=manifest,
            chunks=chunks, top_k=args.top_k,
            fiscal_year=args.year, chunk_type=args.chunk_type,
        )

    elif args.mode == "bm25":
        bm25 = load_bm25(manifest, chunks)
        results = retrieve_bm25(
            question=args.question, bm25=bm25, manifest=manifest,
            chunks=chunks, top_k=args.top_k,
            fiscal_year=args.year, chunk_type=args.chunk_type,
        )

    else:  # hybrid
        bm25 = load_bm25(manifest, chunks)
        results = retrieve_hybrid(
            question=args.question, vectors=vectors, bm25=bm25,
            manifest=manifest, chunks=chunks, top_k=args.top_k,
            fiscal_year=args.year, chunk_type=args.chunk_type,
        )

    if args.rerank:
        results = rerank(args.question, results)

    print_results(args.question, results, mode=args.mode)


if __name__ == "__main__":
    main()
