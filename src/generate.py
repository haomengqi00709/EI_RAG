#!/usr/bin/env python3
"""
generate.py — Filter → expand → reduce generation pipeline for the EI MAR RAG system.

Patterns implemented:
  Multi-query   — generate N query variants to improve retrieval recall
  Filter phase  — each retrieved chunk classified YES / NEED_ADJACENT / NOT_FOUND (parallel)
  Expand phase  — NEED_ADJACENT chunks fetch ±1 neighbours from the chunk store
  Reduce phase  — single LLM reads full text of all relevant chunks and answers

Shared by server.py, handler.py, and evaluate_e2e.py.
"""

import concurrent.futures
import json

# ── Prompts ───────────────────────────────────────────────────────────────────

MULTI_QUERY_PROMPT = """\
Generate {n} different search query variants for the question below.
Use different phrasing, synonyms, or emphasis to help retrieve relevant documents.
Return ONLY a valid JSON array of strings (no explanation).

Question: {question}"""

FILTER_PROMPT = """\
You are checking whether a document excerpt contains information relevant to answering \
a question.

Question: {question}

Excerpt ({metadata}):
{chunk_text}

Does this excerpt contain information that answers the question?
- YES            — excerpt clearly contains relevant information
- NEED_ADJACENT  — excerpt seems relevant but appears cut off, references a \
continuation, or is part of a larger table/section that likely spans adjacent pages
- NOT_FOUND      — excerpt is not relevant to this question

Respond with ONLY one of: YES, NEED_ADJACENT, NOT_FOUND"""

REDUCE_PROMPT = """\
Question: {question}

Retrieved document excerpts:
{context}

Using ONLY the excerpts above, answer the question concisely.
Rules:
- Read the exact pre-computed value directly from the table or text — do not recalculate \
or derive a value yourself, unless the question explicitly asks you to calculate something
- When a table has multiple rows (e.g. different programs, provinces, benefit types, or \
time periods), read the value from the specific row that matches the exact condition \
stated in the question — do not pick a nearby row or an aggregate
- If multiple excerpts give different values for the same metric (e.g. different \
categories, bodies, or time periods), present ALL of them with enough context for a \
human to distinguish — explain briefly WHY the values differ (e.g. different appeal \
bodies, different time periods, different program categories). Do not silently pick one.
- If partial breakdowns together answer the question, sum them and state the total
- Do not abstain just because you prefer more context — use what is present
- Only respond INSUFFICIENT_CONTEXT if the excerpts are completely unrelated to the question
- Cite the source as [FY YYYY-YYYY | p.N]
- Write a clear, direct answer. Lead with the key fact or value, then briefly describe what \
the data shows (e.g. the breakdown across categories, provinces, or years) if that adds useful context.
- Keep it concise — no preamble, no re-stating the question."""

FAITHFULNESS_PROMPT = """\
Question: {question}
Answer: {answer}
Source excerpts:
{context}

Is every factual claim in the answer directly supported by the source excerpts?
Return ONLY valid JSON: {{"faithful": true or false, "issues": "brief explanation or null"}}"""

CLARIFICATION_PROMPT = """\
Question: {question}
Answer: {answer}

Does this answer present COMPETING values where the user must choose one \
(not complementary values that are all correct, but genuinely different answers)?

Examples that NEED clarification:
- "The value is $23,138 or $30,509 depending on the metric — which do you mean?"
- Multiple dollar figures for the same metric from different categories

Examples that do NOT need clarification:
- A single clear answer
- A breakdown by province (all values are part of the answer)
- The answer already specifies why values differ and they are complementary

Return ONLY valid JSON: \
{{"needs_clarification": true or false, "question": "one short clarifying question or null"}}"""


# ── Multi-query generation ────────────────────────────────────────────────────

def generate_query_variants(question: str, model, n: int = 3) -> list[str]:
    """
    Generate n rephrased variants of the question for multi-query retrieval.
    Always returns [original_question, ...variants].
    Falls back to [original_question] on any error.
    """
    try:
        response = model.generate_content(
            MULTI_QUERY_PROMPT.format(question=question, n=n),
            generation_config={"response_mime_type": "application/json"},
        )
        variants = json.loads(response.text)
        if isinstance(variants, list):
            unique = [
                v for v in variants[:n]
                if isinstance(v, str) and v.strip() and v.strip() != question.strip()
            ]
            return [question] + unique
    except Exception as e:
        print(f"  Multi-query failed: {e}")
    return [question]


# ── Clarification check ───────────────────────────────────────────────────────

def check_clarification(answer: str, question: str, model) -> tuple[bool, str | None]:
    """
    Detect whether the answer presents competing values that require the user
    to pick one. Returns (needs_clarification, clarifying_question).
    Falls back to (False, None) on any error.
    """
    try:
        response = model.generate_content(
            CLARIFICATION_PROMPT.format(question=question, answer=answer),
            generation_config={"response_mime_type": "application/json"},
        )
        data = json.loads(response.text)
        needs = bool(data.get("needs_clarification", False))
        cq    = data.get("question") or None
        return needs, cq
    except Exception as e:
        print(f"  Clarification check failed: {e}")
        return False, None


# ── Adjacent chunk helpers ────────────────────────────────────────────────────

def _build_idx_map(chunks: dict) -> dict:
    """
    Build a (fy_prefix, sequential_idx) → chunk_id lookup for O(1) adjacent access.
    chunk_id format: {FY}_{type}_{page}_{idx}  e.g. 20192020_table_0306_0147
    """
    idx_map = {}
    for cid in chunks:
        fy = cid.split("_")[0]
        try:
            idx = int(cid.rsplit("_", 1)[1])
            idx_map[(fy, idx)] = cid
        except (ValueError, IndexError):
            pass
    return idx_map


def _get_adjacent_chunks(
    chunk_id: str, chunks: dict, idx_map: dict, radius: int = 1
) -> list[dict]:
    """Return adjacent chunks within ±radius by sequential index."""
    fy = chunk_id.split("_")[0]
    try:
        idx = int(chunk_id.rsplit("_", 1)[1])
    except (ValueError, IndexError):
        return []

    adjacent = []
    for delta in range(-radius, radius + 1):
        if delta == 0:
            continue
        neighbour_id = idx_map.get((fy, idx + delta))
        if neighbour_id and neighbour_id in chunks:
            adjacent.append(chunks[neighbour_id])
    return adjacent


def _build_page_table_map(chunks: dict) -> dict:
    """Build page_number → [chunk_id, ...] index for all table chunks."""
    page_map: dict[int, list[str]] = {}
    for cid, chunk in chunks.items():
        if chunk.get("chunk_type") == "table":
            page = chunk.get("page_number", 0)
            page_map.setdefault(page, []).append(cid)
    return page_map


def _get_same_page_siblings(
    chunk_id: str, chunks: dict, page_map: dict, exclude_ids: set
) -> list[dict]:
    """Return other table chunks on the same page, excluding already-relevant ones."""
    chunk = chunks.get(chunk_id)
    if not chunk or chunk.get("chunk_type") != "table":
        return []
    page = chunk.get("page_number", 0)
    return [
        chunks[sid]
        for sid in page_map.get(page, [])
        if sid != chunk_id and sid not in exclude_ids and sid in chunks
    ]


# ── Filter phase — per-chunk relevance classification ────────────────────────

def _filter_one(question: str, result: dict, model, rank: int) -> dict:
    """
    Classify a single chunk: YES / NEED_ADJACENT / NOT_FOUND.
    Returns: {rank, chunk_id, metadata, signal, result}
    """
    row   = result["row"]
    chunk = result["chunk"]
    body  = chunk.get("text_nl") or chunk.get("text", "")
    meta  = (
        f"FY {row.get('fiscal_year', '')} | "
        f"p.{row.get('page', '')} | "
        f"{row.get('section_breadcrumb', '')[:70]}"
    )
    try:
        resp   = model.generate_content(
            FILTER_PROMPT.format(question=question, metadata=meta, chunk_text=body)
        )
        signal = resp.text.strip().upper()
        if signal.startswith("YES"):
            signal = "YES"
        elif "ADJACENT" in signal or signal.startswith("NEED"):
            signal = "NEED_ADJACENT"
        else:
            signal = "NOT_FOUND"
    except Exception:
        signal = "NOT_FOUND"

    return {
        "rank":     rank,
        "chunk_id": row.get("chunk_id", ""),
        "metadata": meta,
        "signal":   signal,
        "result":   result,
    }


# ── Context builder for reduce ────────────────────────────────────────────────

def _format_chunk_text(chunk: dict, row: dict = None, label: str = "") -> str:
    body = chunk.get("text_nl") or chunk.get("text", "")
    if row:
        fy   = row.get("fiscal_year", "")
        page = row.get("page", "")
        bc   = row.get("section_breadcrumb", "")[:60]
    else:
        # Adjacent chunks: pull metadata directly from the chunk dict
        fy   = chunk.get("fiscal_year", "")
        page = chunk.get("page_number") or chunk.get("start_page", "")
        bc   = chunk.get("section_breadcrumb", "")[:60]
    header = f"[{label}] FY {fy} | p.{page} | {bc}"
    return f"{header}\n{body}"


def _build_reduce_context(relevant: list, adjacents: dict) -> str:
    """
    Build full-text context for the reduce LLM.
    relevant: list of filter results with signal YES or NEED_ADJACENT
    adjacents: {chunk_id: [adjacent_chunk_dict, ...]}
    """
    parts = []
    for item in relevant:
        result = item["result"]
        row    = result["row"]
        chunk  = result["chunk"]
        label  = str(item["rank"])
        parts.append(_format_chunk_text(chunk, row, label))
        for adj in adjacents.get(item["chunk_id"], []):
            parts.append(_format_chunk_text(adj, label=f"{label}+adj"))
    return "\n\n---\n\n".join(parts)


def _build_faithfulness_context(results: list) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        row     = r["row"]
        body    = r["chunk"].get("text_nl") or r["chunk"].get("text", "")
        snippet = body if len(body) <= 2000 else body[:1500]
        parts.append(
            f"[{i}] FY {row.get('fiscal_year', '')} | "
            f"p.{row.get('page', '')} | "
            f"{row.get('section_breadcrumb', '')[:60]}\n{snippet}"
        )
    return "\n\n".join(parts)


# ── Full pipeline ─────────────────────────────────────────────────────────────

def generate_answer_map_reduce(
    question:          str,
    results:           list,
    model,
    faithfulness_model=None,
    chunks:            dict = None,
    neighbor_radius:   int  = 1,
) -> dict:
    """
    Filter → expand → reduce pipeline:
      1. Filter  — each chunk classified YES / NEED_ADJACENT / NOT_FOUND (parallel)
      2. Expand  — NEED_ADJACENT chunks get ±radius neighbours attached;
                   in power search (radius > 1) ALL relevant chunks get neighbours
      3. Reduce  — single LLM reads full text of all relevant chunks and answers
      4. Faithfulness check
      5. Clarification check

    chunks: the full chunk store (dict keyed by chunk_id) — required for expansion.
    neighbor_radius: 1 = standard (±1 for NEED_ADJACENT only);
                     3 = power search (±3 for all relevant chunks)
    Returns: {answer, faithful, faithfulness_issues, abstained, mapped,
              clarification_needed, clarification_question}
    """
    if not results:
        return {
            "answer": "No results retrieved.",
            "faithful": None, "abstained": True, "mapped": [],
            "clarification_needed": False, "clarification_question": None,
        }

    # ── 1. Filter (parallel) ──────────────────────────────────────────────────
    filtered = [None] * len(results)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(len(results), 5)
    ) as ex:
        future_to_idx = {
            ex.submit(_filter_one, question, results[i], model, i + 1): i
            for i in range(len(results))
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                filtered[idx] = future.result()
            except Exception:
                filtered[idx] = {
                    "rank": idx + 1, "chunk_id": "", "metadata": "",
                    "signal": "NOT_FOUND", "result": results[idx],
                }

    relevant = [f for f in filtered if f is not None and f["signal"] in ("YES", "NEED_ADJACENT")]

    if not relevant:
        return {
            "answer":   "The retrieved documents do not contain enough information to answer this question.",
            "faithful": None, "abstained": True, "mapped": filtered,
            "clarification_needed": False, "clarification_question": None,
        }

    # ── 2. Expand: adjacent chunks (NEED_ADJACENT) + same-page siblings (tables) ──
    adjacents: dict[str, list] = {}
    if chunks:
        idx_map  = _build_idx_map(chunks)
        page_map = _build_page_table_map(chunks)
        already_relevant = {item["chunk_id"] for item in relevant}
        for item in relevant:
            cid   = item["chunk_id"]
            extra = []
            # Power search (radius > 1): expand all relevant chunks
            # Standard: expand only NEED_ADJACENT chunks
            if item["signal"] == "NEED_ADJACENT" or neighbor_radius > 1:
                extra.extend(
                    _get_adjacent_chunks(cid, chunks, idx_map, radius=neighbor_radius)
                )
            # For table chunks: also include other tables on the same page
            if item["result"]["chunk"].get("chunk_type") == "table":
                extra.extend(_get_same_page_siblings(cid, chunks, page_map, already_relevant))
            if extra:
                adjacents[cid] = extra

    # ── 3. Reduce — single call with full context ─────────────────────────────
    context = _build_reduce_context(relevant, adjacents)
    try:
        resp         = model.generate_content(
            REDUCE_PROMPT.format(question=question, context=context)
        )
        final_answer = resp.text.strip()
    except Exception as e:
        return {
            "answer": f"Generation error: {e}",
            "faithful": None, "abstained": False, "mapped": filtered,
            "clarification_needed": False, "clarification_question": None,
        }

    if "INSUFFICIENT_CONTEXT" in final_answer:
        return {
            "answer":   "The retrieved documents do not contain enough information to answer this question.",
            "faithful": None, "abstained": True, "mapped": filtered,
            "clarification_needed": False, "clarification_question": None,
        }

    # ── 4. Faithfulness ───────────────────────────────────────────────────────
    faithful            = None
    faithfulness_issues = None
    fm = faithfulness_model or model
    try:
        f_resp = fm.generate_content(
            FAITHFULNESS_PROMPT.format(
                question=question,
                answer=final_answer,
                context=_build_faithfulness_context(results)[:4000],
            ),
            generation_config={"response_mime_type": "application/json"},
        )
        f_data              = json.loads(f_resp.text)
        faithful            = f_data.get("faithful")
        faithfulness_issues = f_data.get("issues")
    except Exception:
        pass

    # ── 5. Clarification check ────────────────────────────────────────────────
    clarification_needed   = False
    clarification_question = None
    needs_clar, clar_q = check_clarification(final_answer, question, model)
    clarification_needed   = needs_clar
    clarification_question = clar_q

    return {
        "answer":                  final_answer,
        "faithful":                faithful,
        "faithfulness_issues":     faithfulness_issues,
        "abstained":               False,
        "mapped":                  filtered,
        "clarification_needed":    clarification_needed,
        "clarification_question":  clarification_question,
    }
