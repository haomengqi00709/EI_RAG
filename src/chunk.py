#!/usr/bin/env python3
"""
chunk.py — Converts parsed EI MAR JSON files into embedding-ready chunks.

Input:  data/extracted/*.json
Output: data/chunked/*.jsonl  (one JSON object per line, one line per chunk)

Four chunk types:
  narrative — consecutive paragraphs grouped within the same section (~1800 chars)
  table     — one chunk per table (title + markdown)
  chart     — one chunk per chart (title + source note)
  footnote  — numbered citation blocks (excluded from vector index, kept for provenance)

Run:
    python src/chunk.py                    # process all files
    python src/chunk.py data/extracted/2023-2024-EI-MAR-EN.json  # single file
"""

import json
import re
import sys
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

# Pages of these types carry no useful content for RAG
SKIP_PAGE_TYPES = {"cover", "copyright", "toc", "abbreviations", "blank"}

# Paragraphs shorter than this are noise: chart captions, footnote markers,
# source-note fragments, stray section-title echoes
MIN_PARA_CHARS = 60

# Target narrative chunk size in characters (~450 tokens for English prose).
# A paragraph that exceeds this on its own is still emitted as a single chunk.
NARRATIVE_TARGET_CHARS = 1800

# Tables smaller than this are empty or trivial — skip them.
# Mirrors the MIN_TABLE_CHARS threshold used in enrich_tables.py.
MIN_TABLE_CHARS = 50

# Footnote paragraphs start with a 1-2 digit number followed by a capital letter.
# e.g. "1 These circumstances include..." or "10 Statistics Canada, Table 14-10..."
FOOTNOTE_PATTERN = re.compile(r'^\d{1,2}\s+[A-Z]')

# Annex: extract from section_breadcrumb ("ANNEX 4 > ...") or table_title ("Annex 4.2.3 – ...")
RE_ANNEX_BREADCRUMB = re.compile(r"\bANNEX\s+(\d+(?:\.\d+)?)", re.IGNORECASE)
RE_ANNEX_TITLE = re.compile(r"^Annex\s+(\d+(?:\.\d+)?)\b", re.IGNORECASE)
# In-narrative references we want to resolve to table/chart chunks
RE_TABLE_REF = re.compile(r"\bTable\s+(\d+)\b", re.IGNORECASE)
RE_ANNEX_REF = re.compile(r"\bAnnex\s+(\d+(?:\.\d+)?)\b", re.IGNORECASE)
RE_CHART_REF = re.compile(r"\bChart\s+(\d+)\b", re.IGNORECASE)

# ── Semantic chunking helpers ──────────────────────────────────────────────────
#
# #4: Detect sub-section headers within a section — short paragraphs that look
# like headings (no terminal punctuation, Title Case or ALL CAPS, ≤ 120 chars).
# When found, flush the current narrative and start a new chunk so headings
# always begin a new chunk rather than being buried mid-chunk.

SUBHEADER_MAX_CHARS = 120
_SUBHEADER_RE = re.compile(
    r'^[A-Z][A-Za-z0-9 ,:\-/()\u2019\u2013\u2014]+$'  # no terminal .!?
)


def is_subheader(para: str) -> bool:
    """
    True if the paragraph looks like a sub-section heading.
    Heuristic: short, starts with capital, no sentence-ending punctuation.
    """
    stripped = para.strip()
    if len(stripped) > SUBHEADER_MAX_CHARS:
        return False
    if stripped[-1] in '.!?:':
        return False
    # Must look like a heading phrase — no lowercase sentence structure
    word_count = len(stripped.split())
    if word_count > 12:
        return False
    return bool(_SUBHEADER_RE.match(stripped))

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_chunk_id(fiscal_year: str, chunk_type: str, page: int, index: int) -> str:
    fy = fiscal_year.replace("-", "")   # "2023-2024" → "20232024"
    return f"{fy}_{chunk_type}_{page:04d}_{index:04d}"


def is_footnote_para(para: str) -> bool:
    """True if paragraph is a footnote block (superscript-numbered citation)."""
    return bool(FOOTNOTE_PATTERN.match(para))


def annex_ref_from_context(section_breadcrumb: str, table_title: str = "") -> str | None:
    """Return annex reference if this table/chart is in an annex, e.g. 'Annex 4' or 'Annex 4.2.3'."""
    # Prefer explicit "Annex N.M" from title (more specific)
    if table_title:
        m = RE_ANNEX_TITLE.search(table_title.strip())
        if m:
            return f"Annex {m.group(1)}"
    bc = section_breadcrumb or ""
    m = RE_ANNEX_BREADCRUMB.search(bc)
    if m:
        return f"Annex {m.group(1)}"
    return None


def is_table_title_echo(para: str, table_titles: set) -> bool:
    """True if this paragraph is just a table title already captured in a table chunk."""
    stripped = para.strip()
    for title in table_titles:
        # Exact match, or paragraph starts with a long-enough prefix of the title
        # (handles cases where the title was truncated in the table object)
        if stripped == title:
            return True
        if len(title) >= 40 and stripped.startswith(title[:40]):
            return True
    return False


# ── Core chunking logic ───────────────────────────────────────────────────────

def chunk_document(doc: dict) -> list:
    chunks = []
    fiscal_year = doc["fiscal_year"]
    source_file = doc["source_file"]

    # Per-type counters for unique chunk IDs
    counters = {"narrative": 0, "table": 0, "chart": 0, "footnote": 0}

    # Narrative accumulator — groups paragraphs across pages within one section
    acc = {
        "paras": [],
        "chars": 0,
        "start_page": 0,
        "end_page": 0,
        "breadcrumb": "",
        "hierarchy": {},
        "carry": "",    # last sentence borrowed from the previous chunk for overlap
    }

    _narrative_carry = ""   # staging area: set by flush, consumed on next acc init

    # Footnote accumulator — same structure, separate bucket
    acc_fn = {
        "paras": [],
        "chars": 0,
        "start_page": 0,
        "end_page": 0,
        "breadcrumb": "",
        "hierarchy": {},
    }

    def flush_narrative(propagate_carry=True):
        nonlocal _narrative_carry
        if not acc["paras"]:
            _narrative_carry = ""
            return

        # Prepend any carry borrowed from the previous chunk
        body = "\n\n".join(acc["paras"]).strip()
        carry_in = acc["carry"]
        if carry_in:
            text = carry_in + " " + body
            overlap_chars = len(carry_in) + 1
        else:
            text = body
            overlap_chars = 0

        # Extract carry for the next chunk — only when flushing mid-section
        if propagate_carry:
            last_period = text.rfind('. ')
            _narrative_carry = text[last_period + 2:].strip() if last_period != -1 else ""
        else:
            _narrative_carry = ""

        chunk = {
            "chunk_id": make_chunk_id(fiscal_year, "narrative", acc["start_page"], counters["narrative"]),
            "chunk_type": "narrative",
            "text": text,
            "char_count": len(text),
            "fiscal_year": fiscal_year,
            "source_file": source_file,
            "start_page": acc["start_page"],
            "end_page": acc["end_page"],
            "section_breadcrumb": acc["breadcrumb"],
            "hierarchy": acc["hierarchy"],
        }
        if overlap_chars:
            chunk["overlap_chars"] = overlap_chars

        chunks.append(chunk)
        counters["narrative"] += 1
        acc["paras"] = []
        acc["chars"] = 0
        acc["carry"] = ""
        acc["breadcrumb"] = ""   # reset so next para initialises the accumulator cleanly

    def flush_footnote():
        if not acc_fn["paras"]:
            return
        text = "\n\n".join(acc_fn["paras"]).strip()
        chunk = {
            "chunk_id": make_chunk_id(fiscal_year, "footnote", acc_fn["start_page"], counters["footnote"]),
            "chunk_type": "footnote",
            "text": text,
            "char_count": len(text),
            "fiscal_year": fiscal_year,
            "source_file": source_file,
            "start_page": acc_fn["start_page"],
            "end_page": acc_fn["end_page"],
            "section_breadcrumb": acc_fn["breadcrumb"],
            "hierarchy": acc_fn["hierarchy"],
        }
        chunks.append(chunk)
        counters["footnote"] += 1
        acc_fn["paras"] = []
        acc_fn["chars"] = 0
        acc_fn["breadcrumb"] = ""

    for page in doc["pages"]:
        ptype = page["content_type"]
        pnum = page["page_number"]
        breadcrumb = page["section_breadcrumb"]
        hierarchy = page.get("hierarchy", {})

        # Non-content pages: flush any pending narrative, then skip
        if ptype in SKIP_PAGE_TYPES:
            flush_narrative(propagate_carry=False)
            continue

        # ── Table chunks ──────────────────────────────────────────────────────
        table_titles: set = set()
        for t in page.get("tables", []):
            title = (t.get("title") or "").strip()
            markdown = (t.get("markdown") or "").strip()

            if title:
                table_titles.add(title)

            text_parts = [p for p in [title, markdown] if p]
            text = "\n\n".join(text_parts)
            if len(text) < MIN_TABLE_CHARS:
                continue

            meta = t.get("metadata", {})
            table_page = meta.get("page_number", pnum)

            section_bc = meta.get("section_breadcrumb", breadcrumb)
            chunk = {
                "chunk_id": make_chunk_id(fiscal_year, "table", table_page, counters["table"]),
                "chunk_type": "table",
                "text": text,
                "char_count": len(text),
                "fiscal_year": fiscal_year,
                "source_file": source_file,
                "page_number": table_page,
                "section_breadcrumb": section_bc,
                "hierarchy": meta.get("hierarchy", hierarchy),
                "table_title": title,
                "row_count": t.get("row_count", 0),
                "col_count": t.get("col_count", 0),
            }
            annex_ref = annex_ref_from_context(section_bc, title)
            if annex_ref:
                chunk["annex_ref"] = annex_ref
            if "_stitched_pages" in t:
                chunk["stitched_pages"] = t["_stitched_pages"]

            chunks.append(chunk)
            counters["table"] += 1

        # ── Chart chunks ──────────────────────────────────────────────────────
        for c in page.get("charts", []):
            title = (c.get("title") or "").strip()
            source = (c.get("source") or "").strip()
            text_parts = [p for p in [title, source] if p]
            text = "\n".join(text_parts)
            if not text:
                continue

            meta = c.get("metadata", {})
            section_bc = meta.get("section_breadcrumb", breadcrumb)
            chunk = {
                "chunk_id": make_chunk_id(fiscal_year, "chart", pnum, counters["chart"]),
                "chunk_type": "chart",
                "text": text,
                "char_count": len(text),
                "fiscal_year": fiscal_year,
                "source_file": source_file,
                "page_number": pnum,
                "section_breadcrumb": section_bc,
                "hierarchy": meta.get("hierarchy", hierarchy),
                "chart_number": c.get("chart_number", ""),
            }
            annex_ref = annex_ref_from_context(section_bc, title)
            if annex_ref:
                chunk["annex_ref"] = annex_ref
            chunks.append(chunk)
            counters["chart"] += 1

        # ── Narrative paragraphs ──────────────────────────────────────────────
        for para in page.get("paragraphs", []):
            para = para.strip()

            # Drop noise: very short strings
            if len(para) < MIN_PARA_CHARS:
                continue

            # Drop table title echoes (already captured in the table chunk)
            if is_table_title_echo(para, table_titles):
                continue

            if is_footnote_para(para):
                # Route to footnote accumulator; flush any pending narrative first
                flush_narrative()
                if acc_fn["breadcrumb"] and breadcrumb and breadcrumb != acc_fn["breadcrumb"]:
                    flush_footnote()
                if not acc_fn["paras"]:
                    acc_fn["start_page"] = pnum
                    acc_fn["breadcrumb"] = breadcrumb
                    acc_fn["hierarchy"] = hierarchy
                acc_fn["paras"].append(para)
                acc_fn["chars"] += len(para)
                acc_fn["end_page"] = pnum
                if acc_fn["chars"] >= NARRATIVE_TARGET_CHARS:
                    flush_footnote()
            else:
                # Route to narrative accumulator; flush any pending footnotes first
                flush_footnote()

                # Section boundary: flush and start fresh, no carry across sections
                if acc["breadcrumb"] and breadcrumb and breadcrumb != acc["breadcrumb"]:
                    flush_narrative(propagate_carry=False)


                # Initialise accumulator on first paragraph of a new section
                if not acc["paras"]:
                    acc["start_page"] = pnum
                    acc["breadcrumb"] = breadcrumb
                    acc["hierarchy"] = hierarchy
                    acc["carry"] = _narrative_carry   # inject carry from previous chunk
                    _narrative_carry = ""             # consume it

                acc["paras"].append(para)
                acc["chars"] += len(para)
                acc["end_page"] = pnum

                # Hit size target → flush and let next paragraph start fresh
                if acc["chars"] >= NARRATIVE_TARGET_CHARS:
                    flush_narrative()

    # Flush any remaining paragraphs at end of document
    flush_narrative(propagate_carry=False)
    flush_footnote()

    # Link narrative to adjacent tables/charts (same page ±1) and to explicitly referenced ones (Table N, Annex N.M, Chart N)
    chunks = _link_related_chunks(chunks)
    chunks = _link_referenced_chunks(chunks)
    return chunks


# ── Parent-child linking ──────────────────────────────────────────────────────

def _link_related_chunks(chunks: list) -> list:
    """
    #3: For each narrative chunk, add a related_chunk_ids field pointing to
    table and chart chunks that appear on the same page range (±1 page buffer).

    This allows the generation layer to automatically pull in adjacent tables
    when a narrative chunk is retrieved — solving the problem where the exact
    number lives in a table that the narrative references but does not repeat.
    """
    # Index table/chart chunks by page number
    page_to_ids: dict[int, list[str]] = {}
    for c in chunks:
        if c["chunk_type"] in ("table", "chart"):
            page = c.get("page_number", 0)
            page_to_ids.setdefault(page, []).append(c["chunk_id"])

    for c in chunks:
        if c["chunk_type"] != "narrative":
            continue
        start = c.get("start_page", 0)
        end   = c.get("end_page", start)
        related = []
        for page in range(start - 1, end + 2):   # ±1 page buffer
            related.extend(page_to_ids.get(page, []))
        if related:
            c["related_chunk_ids"] = related

    return chunks


def _ref_keys_from_title(chunk_type: str, title: str, chart_number: str = "") -> list[str]:
    """Extract reference keys for linking: 'table_16', 'annex_2.1', 'chart_4'."""
    keys = []
    t = (title or "").strip()
    if chunk_type == "table":
        for m in RE_TABLE_REF.finditer(t):
            keys.append(f"table_{m.group(1)}")
        for m in RE_ANNEX_REF.finditer(t):
            keys.append(f"annex_{m.group(1)}")
    elif chunk_type == "chart":
        if chart_number and chart_number.isdigit():
            keys.append(f"chart_{chart_number}")
        for m in RE_CHART_REF.finditer(t):
            keys.append(f"chart_{m.group(1)}")
    return keys


def _link_referenced_chunks(chunks: list) -> list:
    """
    For each narrative chunk, find explicit references to "Table N", "Annex N.M", "Chart N"
    and add referenced_chunk_ids pointing to the matching table/chart chunks in the same doc.
    Use when explaining an answer: narrative gives context, referenced chunks give the data.
    """
    # Group chunks by (fiscal_year, source_file) — one doc at a time
    doc_chunks: dict[tuple[str, str], list[dict]] = {}
    for c in chunks:
        key = (c.get("fiscal_year", ""), c.get("source_file", ""))
        doc_chunks.setdefault(key, []).append(c)

    for (fy, src), doc_list in doc_chunks.items():
        # Build ref key -> chunk_ids for tables and charts in this doc
        ref_to_ids: dict[str, list[str]] = {}
        for c in doc_list:
            if c["chunk_type"] == "table":
                for k in _ref_keys_from_title("table", c.get("table_title", "")):
                    ref_to_ids.setdefault(k, []).append(c["chunk_id"])
            elif c["chunk_type"] == "chart":
                for k in _ref_keys_from_title("chart", c.get("text", ""), c.get("chart_number", "")):
                    ref_to_ids.setdefault(k, []).append(c["chunk_id"])

        # For each narrative in this doc, find refs in text and link
        for c in doc_list:
            if c["chunk_type"] != "narrative":
                continue
            text = c.get("text", "")
            ref_ids = []
            for m in RE_TABLE_REF.finditer(text):
                ref_ids.extend(ref_to_ids.get(f"table_{m.group(1)}", []))
            for m in RE_ANNEX_REF.finditer(text):
                ref_ids.extend(ref_to_ids.get(f"annex_{m.group(1)}", []))
            for m in RE_CHART_REF.finditer(text):
                ref_ids.extend(ref_to_ids.get(f"chart_{m.group(1)}", []))
            if ref_ids:
                c["referenced_chunk_ids"] = list(dict.fromkeys(ref_ids))  # preserve order, dedupe

    return chunks


# ── Stats helpers ─────────────────────────────────────────────────────────────

def chunk_stats(chunks: list) -> dict:
    by_type = {}
    char_counts = []
    for c in chunks:
        t = c["chunk_type"]
        by_type[t] = by_type.get(t, 0) + 1
        char_counts.append(c["char_count"])

    if not char_counts:
        return by_type

    char_counts.sort()
    n = len(char_counts)
    return {
        "by_type": by_type,
        "total": n,
        "char_min": char_counts[0],
        "char_median": char_counts[n // 2],
        "char_p90": char_counts[int(0.9 * n)],
        "char_max": char_counts[-1],
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        input_paths = [Path(p) for p in sys.argv[1:]]
    else:
        input_paths = sorted(Path("data/extracted").glob("*.json"))

    if not input_paths:
        print("No JSON files found.")
        return

    output_dir = Path("data/chunked")
    output_dir.mkdir(parents=True, exist_ok=True)

    grand_total = 0
    for json_path in input_paths:
        with open(json_path) as f:
            doc = json.load(f)

        chunks = chunk_document(doc)
        stats = chunk_stats(chunks)

        output_path = output_dir / (json_path.stem + ".jsonl")
        with open(output_path, "w") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        bt = stats.get("by_type", {})
        print(
            f"{doc['fiscal_year']}  →  {stats['total']} chunks  "
            f"(narrative={bt.get('narrative', 0)}, "
            f"table={bt.get('table', 0)}, "
            f"chart={bt.get('chart', 0)}, "
            f"footnote={bt.get('footnote', 0)})  "
            f"| chars: min={stats['char_min']}, "
            f"median={stats['char_median']}, "
            f"p90={stats['char_p90']}, "
            f"max={stats['char_max']}"
        )
        grand_total += stats["total"]

    print(f"\nTotal: {grand_total} chunks across {len(input_paths)} documents")
    print(f"Output: {output_dir}/")


if __name__ == "__main__":
    main()
