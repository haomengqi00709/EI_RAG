"""
PDF Parsing Pipeline for EI MAR Reports (v2)
=============================================
Enhanced with:
  - Hierarchical structure detection (chapter/section/subsection via font analysis)
  - Content type classification (cover, toc, narrative, table, annex, etc.)
  - Temporal markers (fiscal year from filename + in-text year references)

Outputs structured JSON for inspection and downstream chunking.

Usage:
    python src/parse_pdf.py                                    # Parse all PDFs
    python src/parse_pdf.py Documents/2023-2024-EI-MAR-EN.pdf  # Parse one PDF
"""

import json
import os
import re
import sys
import pdfplumber


# ---------------------------------------------------------------------------
# Font-based header detection thresholds
# ---------------------------------------------------------------------------
# Learned from inspecting the EI MAR PDFs:
#   34pt Perpetua-Bold  → Major heading (CHAPTER I, HIGHLIGHTS)
#   20pt Perpetua-Bold  → Chapter subtitle (Labour market context)
#   14pt Perpetua       → Section header (1.1 Economic overview)
#   13pt Cambria-Bold   → Subsection header
#   11pt Calibri        → Body text

FONT_TIER_MAJOR   = 28   # >= this size → major heading
FONT_TIER_CHAPTER = 18   # >= this size → chapter subtitle
FONT_TIER_SECTION = 12.5 # >= this size → section header (13pt Cambria-Bold)
FONT_TIER_SUBSECT = 11.5 # >= this size → subsection header


# ---------------------------------------------------------------------------
# Temporal marker detection
# ---------------------------------------------------------------------------

# Matches fiscal year patterns like "2023-24", "2022-2023", "2020-21"
RE_FISCAL_YEAR = re.compile(r"\b(20\d{2})[–\-](20)?\d{2}\b")

# Matches standalone years like "2023", "2020"
RE_YEAR = re.compile(r"\b(20[12]\d)\b")


def extract_temporal_markers(text: str) -> list[str]:
    """Find all year/fiscal-year references in a block of text."""
    markers = set()
    for m in RE_FISCAL_YEAR.finditer(text):
        markers.add(m.group(0))
    for m in RE_YEAR.finditer(text):
        markers.add(m.group(1))
    return sorted(markers)


def split_temporal_markers(
    markers: list[str], doc_fiscal_year: str
) -> tuple[list[str], list[str]]:
    """
    Split temporal markers into primary vs referenced.

    Primary: years that fall within the document's own fiscal year range.
        e.g. for doc_fiscal_year="2023-2024" → "2023", "2024", "2023-24"
    Referenced: all other years mentioned (historical comparisons, trends).
        e.g. "2019-20", "2021", "2022-23"
    """
    parts = doc_fiscal_year.split("-")
    if len(parts) != 2:
        return markers, []

    start = parts[0]            # "2023"
    end = parts[1]              # "2024"
    short_end = end[-2:]        # "24"

    # All patterns that count as "this document's year"
    primary_set = {
        start, end,
        f"{start}-{short_end}", f"{start}-{end}",
        f"{start}\u2013{short_end}", f"{start}\u2013{end}",  # en-dash variants
    }

    primary = [m for m in markers if m in primary_set]
    referenced = [m for m in markers if m not in primary_set]
    return primary, referenced


# ---------------------------------------------------------------------------
# Hierarchical structure detection
# ---------------------------------------------------------------------------

# Regex patterns for numbered section headers in the text itself
RE_CHAPTER_NUM = re.compile(
    r"^(CHAPTER\s+[IVX]+)", re.IGNORECASE
)
RE_SECTION_NUM = re.compile(
    r"^(\d+\.\d+(?:\.\d+)?)\s"
)
RE_ANNEX_NUM = re.compile(
    r"^(Annex\s+\d+)", re.IGNORECASE
)


def detect_headers_from_fonts(page) -> list[dict]:
    """
    Analyze character-level font data to detect headers on a page.
    Returns a list of detected headers with their hierarchy level.
    """
    if not page.chars:
        return []

    # Group consecutive characters by (fontname, size, y-position)
    # to reconstruct text runs
    runs = []
    current_run = {"text": "", "fontname": "", "size": 0, "top": 0}

    for char in page.chars:
        c_text = char.get("text", "")
        c_font = char.get("fontname", "")
        c_size = round(char.get("size", 0), 1)
        c_top = round(char.get("top", 0), 0)

        # Same run if same font, similar size, similar vertical position
        if (c_font == current_run["fontname"]
                and abs(c_size - current_run["size"]) < 0.5
                and abs(c_top - current_run["top"]) < 3):
            current_run["text"] += c_text
        else:
            if current_run["text"].strip():
                runs.append(current_run.copy())
            current_run = {
                "text": c_text,
                "fontname": c_font,
                "size": c_size,
                "top": c_top,
            }

    if current_run["text"].strip():
        runs.append(current_run)

    # Identify headers based on font size thresholds
    headers = []
    for run in runs:
        text = run["text"].strip()
        size = run["size"]

        if not text or len(text) < 2:
            continue

        if size >= FONT_TIER_MAJOR:
            headers.append({
                "level": "major",
                "text": text,
                "font_size": size,
            })
        elif size >= FONT_TIER_CHAPTER:
            headers.append({
                "level": "chapter_subtitle",
                "text": text,
                "font_size": size,
            })
        elif size >= FONT_TIER_SECTION:
            # Only count as section header if it looks like one
            # (bold font or starts with a number pattern)
            is_bold = "Bold" in run["fontname"] or "bold" in run["fontname"]
            has_number = RE_SECTION_NUM.match(text) or RE_ANNEX_NUM.match(text)
            if is_bold or has_number:
                headers.append({
                    "level": "section",
                    "text": text,
                    "font_size": size,
                })

    # Merge consecutive headers at the same level (fixes line-broken titles)
    merged = []
    for h in headers:
        if merged and merged[-1]["level"] == h["level"]:
            # Same level — likely a continuation across a line break
            prev_text = merged[-1]["text"].rstrip()
            curr_text = h["text"].lstrip()
            merged[-1]["text"] = prev_text + " " + curr_text
        else:
            merged.append(h)

    return merged


def build_section_breadcrumb(headers: list[dict]) -> str:
    """
    Build a breadcrumb string from detected headers.
    e.g. "Chapter II > 2.2 EI regular benefits > 2.2.6 Seasonal claimants"
    """
    if not headers:
        return ""
    parts = [h["text"].strip().replace("\n", " ")[:120] for h in headers]
    return " > ".join(parts)


# ---------------------------------------------------------------------------
# Content type classification
# ---------------------------------------------------------------------------

def classify_page(page_num: int, text: str, tables: list,
                  headers: list[dict], total_pages: int) -> str:
    """
    Classify what type of content this page contains.
    Returns one of: cover, copyright, toc, abbreviations, narrative,
                    table_heavy, annex, blank
    """
    text_lower = text.lower().strip()
    char_count = len(text.strip())

    # Blank or near-blank pages (must also have no real tables — after step 1
    # table-only pages have near-empty text but are not blank)
    if char_count < 30 and not tables:
        return "blank"

    # First few pages heuristics
    if page_num <= 2:
        if "monitoring and" in text_lower or "assessment" in text_lower:
            return "cover"
        if "copyright" in text_lower or "isbn" in text_lower or "issn" in text_lower:
            return "copyright"

    # Copyright page (can appear on page 2-4)
    if page_num <= 5 and ("issn" in text_lower or "cat. no" in text_lower
                          or "reproduction rights" in text_lower):
        return "copyright"

    # Table of contents
    if "table of" in text_lower and "contents" in text_lower:
        return "toc"
    if page_num <= 15 and text.count("...") > 5:
        return "toc"

    # Abbreviations page
    if "list of" in text_lower and "abbreviation" in text_lower:
        return "abbreviations"
    # Abbreviation pages are typically early and have many short rows
    if page_num <= 15 and len(tables) > 0:
        first_table = tables[0]
        if first_table.get("col_count", 0) == 2 and first_table.get("row_count", 0) > 10:
            return "abbreviations"

    # Annex pages
    if any("annex" in h["text"].lower() for h in headers):
        return "annex"
    if RE_ANNEX_NUM.match(text.strip()):
        return "annex"

    # Table-heavy: table content dominates over narrative text.
    # After step 1, text field excludes table regions, so char_count is
    # narrative-only. A page is table_heavy if tables have more content than
    # the remaining narrative text, or if there's no narrative text at all.
    if tables:
        table_chars = sum(len(t["content"]) for t in tables)
        if char_count == 0 or table_chars / max(char_count, 1) > 0.5:
            return "table_heavy"

    return "narrative"


# ---------------------------------------------------------------------------
# Multi-column layout detection and handling
# ---------------------------------------------------------------------------

# Minimum gap width (in pixels) between columns to count as multi-column
MIN_COLUMN_GAP_PX = 20  # ~2 bins of 10px each

# Paragraph detection thresholds
_LINE_TOLERANCE = 3    # points — chars within this y-range are on the same line
_PARA_GAP_RATIO = 1.6  # gap > this × median line height signals a new paragraph


def detect_columns(page) -> dict:
    """
    Analyze character x-positions to detect multi-column layouts.
    Returns:
        {"num_columns": 1} for single-column pages
        {"num_columns": 2, "split_x": <float>} for two-column pages
    """
    chars = [c for c in page.chars if c.get("text", "").strip()]
    if len(chars) < 50:
        return {"num_columns": 1}

    width = page.width

    # Build histogram of character x-positions (10px bins)
    bin_width = 10
    num_bins = int(width // bin_width + 1)
    bins = [0] * num_bins
    for c in chars:
        idx = min(int(c["x0"] // bin_width), num_bins - 1)
        bins[idx] += 1

    # Search for a gap in the middle 40% of the page
    search_start = int(num_bins * 0.3)
    search_end = int(num_bins * 0.7)

    # Find the longest consecutive run of near-empty bins
    best_gap_start = -1
    best_gap_len = 0
    current_start = -1
    current_len = 0

    for i in range(search_start, search_end):
        if bins[i] < 3:  # near-empty bin
            if current_start == -1:
                current_start = i
            current_len += 1
            if current_len > best_gap_len:
                best_gap_len = current_len
                best_gap_start = current_start
        else:
            current_start = -1
            current_len = 0

    min_gap_bins = MIN_COLUMN_GAP_PX // bin_width
    if best_gap_len >= min_gap_bins:
        # Split point is the middle of the gap
        split_x = (best_gap_start + best_gap_len / 2) * bin_width
        return {"num_columns": 2, "split_x": split_x}

    return {"num_columns": 1}


def extract_text_multicolumn(page) -> tuple[str, dict]:
    """
    Extract text from a page, handling multi-column layouts.
    Returns (text, column_info) where column_info describes what was detected.
    """
    col_info = detect_columns(page)

    if col_info["num_columns"] == 1:
        return page.extract_text() or "", col_info

    split_x = col_info["split_x"]

    # Check for full-width content at the top of the page (e.g. page title).
    # Characters whose x-range spans across the split point are "full-width".
    # We find the y-coordinate where full-width content ends.
    full_width_bottom = 0
    for c in page.chars:
        if not c.get("text", "").strip():
            continue
        # If a character sits right of the split, check if there are also
        # characters on the same line to the left (and vice versa).
        # Simpler heuristic: look for large-font text (likely a header) at top
        if c.get("size", 0) >= FONT_TIER_SECTION and c.get("top", 0) < page.height * 0.2:
            full_width_bottom = max(full_width_bottom, c.get("bottom", 0))

    # Crop regions
    top_margin = 0
    bottom_margin = page.height

    # Extract full-width header area (if any)
    header_text = ""
    if full_width_bottom > 0:
        header_area = page.crop((0, top_margin, page.width, full_width_bottom + 5))
        header_text = header_area.extract_text() or ""
        top_margin = full_width_bottom + 5

    # Extract left column
    left_crop = page.crop((0, top_margin, split_x, bottom_margin))
    left_text = left_crop.extract_text() or ""

    # Extract right column
    right_crop = page.crop((split_x, top_margin, page.width, bottom_margin))
    right_text = right_crop.extract_text() or ""

    # Combine: header, then left column, then right column
    parts = []
    if header_text.strip():
        parts.append(header_text.strip())
    if left_text.strip():
        parts.append(left_text.strip())
    if right_text.strip():
        parts.append(right_text.strip())

    return "\n\n".join(parts), col_info


# ---------------------------------------------------------------------------
# Paragraph boundary detection
# ---------------------------------------------------------------------------

def _words_to_paragraphs(words: list[dict]) -> list[str]:
    """
    Convert a list of pdfplumber word dicts (from a single column region)
    into a list of paragraph strings.

    Words are grouped into lines by y-position.  A gap between consecutive
    lines larger than _PARA_GAP_RATIO × the median line height is treated as
    a paragraph boundary.  Lines within a paragraph are joined with a space
    (they are just word-wrap artifacts from the PDF layout).
    """
    if not words:
        return []

    words = sorted(words, key=lambda w: (w["top"], w["x0"]))

    # Group words into lines
    lines: list[list[dict]] = []
    cur: list[dict] = []
    cur_top: float | None = None
    for word in words:
        if cur_top is None or abs(word["top"] - cur_top) <= _LINE_TOLERANCE:
            cur.append(word)
            if cur_top is None:
                cur_top = word["top"]
        else:
            lines.append(cur)
            cur = [word]
            cur_top = word["top"]
    if cur:
        lines.append(cur)

    if not lines:
        return []

    # Compute median line height
    heights = [
        max(w.get("bottom", w["top"]) for w in line) - min(w["top"] for w in line)
        for line in lines
    ]
    median_h = sorted(heights)[len(heights) // 2] if heights else 12

    # Accumulate lines into paragraphs
    paragraphs: list[str] = []
    cur_lines: list[str] = []

    for i, line in enumerate(lines):
        line_text = " ".join(w["text"] for w in sorted(line, key=lambda w: w["x0"]))

        if i > 0 and cur_lines:
            prev_bot = max(w.get("bottom", w["top"]) for w in lines[i - 1])
            this_top = min(w["top"] for w in line)
            gap = this_top - prev_bot
            if gap > median_h * _PARA_GAP_RATIO:
                combined = " ".join(cur_lines)
                combined = combined.strip()
                # Skip lone page numbers that strip_page_footer removed from
                # text but extract_words() still sees at the page bottom.
                if combined and not combined.isdigit():
                    paragraphs.append(combined)
                cur_lines = []

        cur_lines.append(line_text)

    if cur_lines:
        combined = " ".join(cur_lines).strip()
        if combined and not combined.isdigit():
            paragraphs.append(combined)

    return paragraphs


def extract_paragraphs(text_page, col_info: dict) -> list[str]:
    """
    Return a list of paragraph strings for the page by detecting paragraph
    boundaries from vertical line-spacing in the (table-excluded) text_page.

    For multi-column pages the two columns are processed independently so
    words from different columns never interleave.
    """
    words = text_page.extract_words() or []
    if not words:
        return []

    if col_info["num_columns"] == 1:
        return _words_to_paragraphs(words)

    split_x = col_info["split_x"]
    left_words  = [w for w in words if w["x0"] <  split_x]
    right_words = [w for w in words if w["x0"] >= split_x]
    return _words_to_paragraphs(left_words) + _words_to_paragraphs(right_words)


# ---------------------------------------------------------------------------
# Header/footer removal and text normalization
# ---------------------------------------------------------------------------

def strip_page_footer(text: str) -> str:
    """
    Remove page number footers.
    These EI MAR PDFs end nearly every page with a standalone number (the
    page number).  We strip it so it doesn't pollute chunks.
    """
    if not text:
        return text
    lines = text.rstrip().split("\n")
    # If last line is just a number (possibly with whitespace), drop it
    if lines and lines[-1].strip().isdigit():
        lines = lines[:-1]
    return "\n".join(lines)


def normalize_text(text: str) -> str:
    """
    Light normalization pass applied after footer stripping.
    Fixes hyphenated line breaks that pdfplumber introduces when a hyphenated
    compound word is split across lines (e.g. 'socio-\\ndemographic').
    The hyphen is real and kept; only the newline is removed.
    """
    # Covers letter-digit and digit-digit cases too, e.g.:
    #   'COVID-\n19' → 'COVID-19'
    #   '14-10-\n0287-01' → '14-10-0287-01'  (Statistics Canada table refs)
    #   '2018-\n19' → '2018-19'              (fiscal years)
    return re.sub(r"([a-zA-Z0-9])-\n([a-zA-Z0-9])", r"\1-\2", text)


# ---------------------------------------------------------------------------
# Table extraction and formatting
# ---------------------------------------------------------------------------

def clean_cell(cell) -> str:
    """Clean a single table cell: collapse newlines, strip whitespace."""
    if cell is None:
        return ""
    text = str(cell).strip()
    # Collapse internal newlines into spaces (fixes multi-line headers)
    text = re.sub(r"\s*\n\s*", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"  +", " ", text)
    return text


def table_to_text(table: list[list]) -> str:
    """Convert a pdfplumber table (list of rows) into readable pipe-delimited text."""
    if not table:
        return ""
    lines = []
    for row in table:
        cleaned = [clean_cell(cell) for cell in row]
        lines.append(" | ".join(cleaned))
    return "\n".join(lines)


def table_to_markdown(table: list[list]) -> str:
    """
    Convert a pdfplumber table (list of rows) into a Markdown table.
    First row is treated as the header row.
    """
    if not table or len(table) < 2:
        return table_to_text(table)  # fallback for single-row tables

    # Clean all cells
    cleaned = []
    for row in table:
        cleaned.append([clean_cell(cell) for cell in row])

    # Determine column count (use max across all rows for safety)
    num_cols = max(len(row) for row in cleaned)

    # Pad rows that are short
    for row in cleaned:
        while len(row) < num_cols:
            row.append("")

    # Build header
    header_row = cleaned[0]
    header_line = "| " + " | ".join(header_row) + " |"
    separator = "| " + " | ".join(["---"] * num_cols) + " |"

    # Build data rows
    data_lines = []
    for row in cleaned[1:]:
        data_lines.append("| " + " | ".join(row) + " |")

    return "\n".join([header_line, separator] + data_lines)


def is_false_positive_table(tbl: list[list]) -> bool:
    """
    Detect 1-column pdfplumber false positives: bordered text boxes,
    bulleted/numbered lists, and diagram labels that get mis-detected as
    single-column tables.

    Real 1-column tables have short cell content (codes, names, numbers).
    False positives have prose sentences or long descriptive text.

    Note: content is NOT lost — it stays in the page `text` field from
    extract_text(). We just stop treating it as a structured table.
    """
    if not tbl:
        return False

    col_count = max(len(row) for row in tbl)
    if col_count != 1:
        return False

    cells = [clean_cell(row[0]) for row in tbl if row and row[0] is not None]
    if not cells:
        return False

    max_len = max(len(c) for c in cells)
    max_words = max(len(c.split()) for c in cells)

    # Any cell is a long sentence, or contains many words → prose text box
    return max_len > 80 or max_words > 12


def make_table_exclusion_filter(bboxes: list[tuple]) -> callable:
    """
    Return a pdfplumber char-filter function that excludes characters
    falling inside any of the given bounding boxes (x0, top, x1, bottom).
    Used to strip table regions from extract_text() so the text field
    contains only narrative content.
    """
    def not_in_tables(obj):
        ox0 = obj.get("x0", 0)
        ox1 = obj.get("x1", 0)
        otop = obj.get("top", 0)
        obot = obj.get("bottom", 0)
        for (bx0, btop, bx1, bbot) in bboxes:
            if ox0 < bx1 and ox1 > bx0 and otop < bbot and obot > btop:
                return False
        return True
    return not_in_tables


def detect_table_title_above(page, table_bbox: tuple, strip_height: int = 80) -> str:
    """
    Look at the region directly above a table's bounding box and extract
    any table title found there (e.g. 'Table 3 – ...' or 'Annex 2.5.1 – ...').

    Searches a strip of `strip_height` points above the table's top edge,
    full page width, so the title is attributed to this specific table rather
    than to the first title found anywhere on the page.
    """
    _, table_top, _, _ = table_bbox
    strip_top = max(0, table_top - strip_height)
    strip = page.crop((0, strip_top, page.width, table_top))
    strip_text = strip.extract_text() or ""
    return detect_table_title(strip_text)


def detect_table_title(text: str) -> str:
    """
    Try to find the table title from the page text.
    Looks for patterns like 'Table 3 – ...' or 'Annex 2.5.1 – ...'
    These are document table titles (Table 1, Table 2), not StatsCan
    reference numbers (Table 14-10-0068-01).
    """
    patterns = [
        # "Table 3 – Employment Insurance claims..." (doc tables use low numbers)
        r"(Table\s+\d{1,3}\s*[–\-—]\s*[^\n]+)",
        # "Annex 2.5.1 – Regular benefits: ..."
        r"(Annex\s+[\d.]+\s*[–\-—]\s*[^\n]+)",
    ]
    titles = []
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            candidate = m.group(1).strip()
            # Skip StatsCan table references (e.g. "Table 14-10-0068-01")
            if re.match(r"Table\s+\d{2,}-", candidate):
                continue
            titles.append(candidate)

    return titles[0] if titles else ""


# ---------------------------------------------------------------------------
# Chart detection
# ---------------------------------------------------------------------------

# Matches: "Chart 3 – Year-over-year change in the CPI, Canada, Jan 2011 to Mar 2024"
RE_CHART_TITLE = re.compile(
    r"(Chart\s+[\d.]+\s*[–\-—]\s*[^\n]+(?:\n[^\n]{1,80})?)"
)

# Matches: "Source: Statistics Canada, Table 18-10-0004-01." or "Sources: ..."
RE_CHART_SOURCE = re.compile(
    r"(Sources?:\s*[^\n]+(?:\n[^\n]+)*?)(?=\n[A-Z]|\n\d|\Z)"
)


def detect_charts(text: str) -> list[dict]:
    """
    Detect chart references in page text.
    Extracts chart number, title, and source note.
    Charts are images we can't read, but the surrounding text describes them.
    """
    charts = []

    for m in RE_CHART_TITLE.finditer(text):
        title_full = m.group(1).strip()
        # Clean up: collapse newlines in title (titles sometimes wrap)
        title_full = re.sub(r"\s*\n\s*", " ", title_full)
        # Trim if Source: bled into the title
        if "Source" in title_full:
            title_full = title_full[:title_full.index("Source")].rstrip()
        if "Sources" in title_full:
            title_full = title_full[:title_full.index("Sources")].rstrip()

        # Extract chart number
        num_match = re.match(r"Chart\s+([\d.]+)", title_full)
        chart_num = num_match.group(1) if num_match else ""

        # Find the source note that follows this chart title
        # Look in the text after the chart title position
        after_title = text[m.end():]
        source = ""
        source_match = RE_CHART_SOURCE.match(after_title.lstrip())
        if source_match:
            source = re.sub(r"\s*\n\s*", " ", source_match.group(1).strip())

        charts.append({
            "chart_number": chart_num,
            "title": title_full,
            "source": source,
            "is_image": True,  # reminder: actual chart data is not extractable
        })

    return charts


# ---------------------------------------------------------------------------
# Page extraction (enhanced)
# ---------------------------------------------------------------------------

def extract_page(page, page_num: int, total_pages: int,
                 doc_fiscal_year: str = "") -> dict:
    """Extract text, tables, headers, content type, and temporal markers."""
    # --- Tables (step 1+4) ---
    # Use find_tables() to get bounding boxes alongside data, so we can:
    #   (1) exclude real table regions from text extraction (no duplication)
    #   (4) attribute titles per-table by looking above each bbox
    tables_found = page.find_tables() or []
    tables = []
    real_table_bboxes = []
    skipped_false_positives = 0

    for tbl_obj in tables_found:
        tbl_data = tbl_obj.extract()
        pipe_text = table_to_text(tbl_data)
        if not pipe_text.strip():
            continue                          # empty — skip silently
        if is_false_positive_table(tbl_data):
            skipped_false_positives += 1
            continue
        real_table_bboxes.append(tbl_obj.bbox)
        md_text = table_to_markdown(tbl_data)
        # (4) Title comes from the strip just above this table's bbox
        title = detect_table_title_above(page, tbl_obj.bbox)
        tables.append({
            "table_index": len(tables),
            "title": title,
            "content": pipe_text,
            "markdown": md_text,
            "row_count": len(tbl_data),
            "col_count": len(tbl_data[0]) if tbl_data else 0,
            "_bbox": list(tbl_obj.bbox),   # internal: used for stitching
        })

    # --- Text extraction (step 1) ---
    # Filter out characters that fall inside real table bounding boxes so the
    # text field contains only narrative content, not garbled table text.
    if real_table_bboxes:
        char_filter = make_table_exclusion_filter(real_table_bboxes)
        text_page = page.filter(char_filter)
    else:
        text_page = page

    raw_text, column_info = extract_text_multicolumn(text_page)
    text = normalize_text(strip_page_footer(raw_text))
    paragraphs = extract_paragraphs(text_page, column_info)

    # Hierarchical: detect headers from font analysis (use original page)
    headers = detect_headers_from_fonts(page)

    # Charts — detect titles and source notes
    charts = detect_charts(text)

    # Content type classification
    content_type = classify_page(page_num, text, tables, headers, total_pages)

    # Temporal markers — split into primary (this doc's year) vs referenced
    all_markers = extract_temporal_markers(text)
    primary_years, referenced_years = split_temporal_markers(
        all_markers, doc_fiscal_year
    )

    return {
        "page_number": page_num,
        "text": text,
        "tables": tables,
        "has_tables": len(tables) > 0,
        "false_positive_tables_skipped": skipped_false_positives,
        "charts": charts,
        "has_charts": len(charts) > 0,
        "char_count": len(text),
        "content_type": content_type,
        "headers": headers,
        "section_breadcrumb": build_section_breadcrumb(headers),
        "temporal_markers": all_markers,
        "primary_years": primary_years,
        "referenced_years": referenced_years,
        "paragraphs": paragraphs,
        "num_columns": column_info["num_columns"],
        "_page_height": page.height,          # internal: used for stitching
        "_doc_fiscal_year": doc_fiscal_year,  # internal: used by propagate_hierarchy
    }


# ---------------------------------------------------------------------------
# Hierarchical state tracker
# ---------------------------------------------------------------------------

# Hierarchy levels in order of precedence (top → bottom)
HIERARCHY_LEVELS = ["major", "chapter_subtitle", "section"]


def propagate_hierarchy(pages: list[dict]) -> None:
    """
    Walk through pages sequentially and propagate the last-seen header
    at each hierarchy level forward.  Mutates pages in place, updating
    'section_breadcrumb' and adding 'hierarchy' dict.

    Rules:
      - When a new header at level N is seen, it replaces level N
        and clears all levels below N.
      - Pages without headers inherit the current state.
      - Non-content pages (cover, toc, blank, copyright, abbreviations)
        reset the hierarchy state.
    """
    # Current state: one slot per hierarchy level
    state = {level: "" for level in HIERARCHY_LEVELS}

    non_content_types = {"cover", "copyright", "toc", "abbreviations", "blank"}

    for page in pages:
        doc_fiscal_year = page.get("_doc_fiscal_year", "")

        # Reset hierarchy on non-content pages
        if page["content_type"] in non_content_types:
            state = {level: "" for level in HIERARCHY_LEVELS}
            page["hierarchy"] = dict(state)
            page["section_breadcrumb"] = ""
            continue

        # Update state with any headers detected on this page
        if page["headers"]:
            for header in page["headers"]:
                hlevel = header["level"]
                if hlevel not in state:
                    continue
                # Set this level
                state[hlevel] = header["text"].replace("\n", " ").strip()[:150]
                # Clear all levels below
                clear = False
                for level in HIERARCHY_LEVELS:
                    if clear:
                        state[level] = ""
                    if level == hlevel:
                        clear = True

        # Store current hierarchy state and build breadcrumb
        page["hierarchy"] = dict(state)
        parts = [state[lvl] for lvl in HIERARCHY_LEVELS if state[lvl]]
        page["section_breadcrumb"] = " > ".join(parts)

        # Enrich tables and charts with full context metadata
        context_meta = {
            "doc_fiscal_year": doc_fiscal_year,
            "page_number": page["page_number"],
            "content_type": page["content_type"],
            "section_breadcrumb": page["section_breadcrumb"],
            "hierarchy": dict(state),
            "primary_years": page.get("primary_years", []),
            "referenced_years": page.get("referenced_years", []),
        }
        for table in page.get("tables", []):
            table["metadata"] = context_meta
        for chart in page.get("charts", []):
            chart["metadata"] = context_meta


# ---------------------------------------------------------------------------
# Multi-page table stitching
# ---------------------------------------------------------------------------

# Distance (in points) a table edge must be from the page boundary
# to qualify as "cut off" and potentially continuing on the next page.
STITCH_BOTTOM_GAP = 80   # table bottom must be within 80pts of page bottom
STITCH_TOP_GAP    = 95   # continuation top must be within 95pts of page top


def _rebuild_markdown_from_content(content: str, col_count: int) -> str:
    """Rebuild a Markdown table from pipe-delimited content rows."""
    rows = [r for r in content.split("\n") if r.strip()]
    if not rows:
        return ""
    header = "| " + rows[0] + " |"
    separator = "| " + " | ".join(["---"] * col_count) + " |"
    data = ["| " + r + " |" for r in rows[1:]]
    return "\n".join([header, separator] + data)


def _try_stitch_adjacent(pages: list[dict], i: int) -> bool:
    """
    Attempt to merge page[i]'s last table with page[i+1]'s first table.

    Conditions for a stitch:
      - Both pages have tables.
      - Table must have >= 2 columns (1-col tables are often text boxes).
      - Last table's bottom is within STITCH_BOTTOM_GAP of the page bottom.
      - Next page's first table starts within STITCH_TOP_GAP of the page top.
      - Column counts match.

    Merging:
      - Rows from the continuation are appended to the source table.
      - If the continuation's first row is an identical repeat of the source
        header, that row is skipped (many reports repeat headers per page).
      - Markdown is rebuilt from the merged pipe-delimited content.
      - The continuation table is removed from the next page's table list.

    Returns True if a stitch was performed.
    """
    page      = pages[i]
    next_page = pages[i + 1]

    if not page["tables"] or not next_page["tables"]:
        return False

    last_table  = page["tables"][-1]
    first_table = next_page["tables"][0]

    # Must have real multi-column tables
    if last_table["col_count"] < 2 or first_table["col_count"] < 2:
        return False

    # Column counts must match
    if last_table["col_count"] != first_table["col_count"]:
        return False

    # Need bbox info stored during extraction
    last_bbox  = last_table.get("_bbox")
    first_bbox = first_table.get("_bbox")
    if not last_bbox or not first_bbox:
        return False

    page_height = page.get("_page_height", 792)
    _, _, _, last_bottom = last_bbox
    _, first_top, _, _   = first_bbox

    if page_height - last_bottom > STITCH_BOTTOM_GAP:
        return False
    if first_top > STITCH_TOP_GAP:
        return False

    # --- Merge ---
    last_rows = [r for r in last_table["content"].split("\n") if r.strip()]
    cont_rows = [r for r in first_table["content"].split("\n") if r.strip()]

    # Skip a repeated header row in the continuation
    header_repeated = (
        bool(cont_rows) and bool(last_rows)
        and cont_rows[0].strip() == last_rows[0].strip()
    )
    if header_repeated:
        cont_rows = cont_rows[1:]

    merged_rows = last_rows + cont_rows
    merged_content  = "\n".join(merged_rows)
    merged_markdown = _rebuild_markdown_from_content(
        merged_content, last_table["col_count"]
    )

    last_table["content"]   = merged_content
    last_table["markdown"]  = merged_markdown
    last_table["row_count"] = len(merged_rows)

    # Track which pages contributed to this table
    if "_stitched_pages" not in last_table:
        last_table["_stitched_pages"] = [page["page_number"]]
    last_table["_stitched_pages"].append(next_page["page_number"])

    # Remove the now-merged continuation from the next page
    next_page["tables"].pop(0)
    return True


def stitch_tables(pages: list[dict]) -> int:
    """
    Second pass: detect and merge multi-page tables in place.

    Iterates repeatedly until no more adjacent stitches are possible,
    which handles tables spanning 3 or more pages.

    Returns the total number of stitch operations performed.
    """
    total = 0
    changed = True
    while changed:
        changed = False
        for i in range(len(pages) - 1):
            if _try_stitch_adjacent(pages, i):
                changed = True
                total += 1
    return total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_fiscal_year(filename: str) -> str:
    """Pull the fiscal year from the filename, e.g. '2023-2024'."""
    match = re.search(r"(\d{4})[_-](\d{2,4})", filename)
    if match:
        start = match.group(1)
        end = match.group(2)
        if len(end) == 2:
            end = start[:2] + end
        return f"{start}-{end}"
    return "unknown"


# ---------------------------------------------------------------------------
# Main parsing function
# ---------------------------------------------------------------------------

def parse_pdf(pdf_path: str) -> dict:
    """Parse a single PDF and return structured data."""
    filename = os.path.basename(pdf_path)
    fiscal_year = extract_fiscal_year(filename)

    print(f"  Parsing: {filename}")
    print(f"  Fiscal year: {fiscal_year}")

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"  Total pages: {total_pages}")

        for i, page in enumerate(pdf.pages):
            page_data = extract_page(page, page_num=i + 1,
                                     total_pages=total_pages,
                                     doc_fiscal_year=fiscal_year)
            pages.append(page_data)

            if (i + 1) % 50 == 0:
                print(f"    ... processed {i + 1}/{total_pages} pages")

    # Second pass: propagate hierarchy across pages
    propagate_hierarchy(pages)

    # Third pass: stitch multi-page tables
    tables_stitched = stitch_tables(pages)
    print(f"  Multi-page tables stitched: {tables_stitched}")

    # Fix #1: update has_tables and re-classify pages whose only table was
    # stitched away (they are now table-free and should not be table_heavy)
    for page in pages:
        page["has_tables"] = len(page["tables"]) > 0
        if not page["tables"] and page["content_type"] == "table_heavy":
            page["content_type"] = classify_page(
                page["page_number"], page["text"],
                page["tables"], page["headers"], total_pages,
            )

    # Summary stats
    total_chars = sum(p["char_count"] for p in pages)
    pages_with_tables = sum(1 for p in pages if p["has_tables"])
    total_tables = sum(len(p["tables"]) for p in pages)
    total_false_positive_tables = sum(p["false_positive_tables_skipped"] for p in pages)

    # Content type distribution
    type_counts = {}
    for p in pages:
        ct = p["content_type"]
        type_counts[ct] = type_counts.get(ct, 0) + 1

    # Multi-column page count
    multi_col_pages = sum(1 for p in pages if p["num_columns"] > 1)

    # Chart stats
    pages_with_charts = sum(1 for p in pages if p["has_charts"])
    total_charts = sum(len(p["charts"]) for p in pages)

    # Collect all unique section headers found
    all_headers = []
    for p in pages:
        for h in p["headers"]:
            all_headers.append({
                "page": p["page_number"],
                "level": h["level"],
                "text": h["text"].replace("\n", " ")[:120],
            })

    # Collect all charts for summary
    all_charts = []
    for p in pages:
        for c in p["charts"]:
            all_charts.append({
                "chart_number": c["chart_number"],
                "title": c["title"],
                "page": p["page_number"],
                "section_breadcrumb": p["section_breadcrumb"][:100],
            })

    result = {
        "source_file": filename,
        "fiscal_year": fiscal_year,
        "total_pages": total_pages,
        "total_characters": total_chars,
        "pages_with_tables": pages_with_tables,
        "total_tables": total_tables,
        "false_positive_tables_filtered": total_false_positive_tables,
        "tables_stitched": tables_stitched,
        "total_paragraphs": sum(len(p["paragraphs"]) for p in pages),
        "pages_with_charts": pages_with_charts,
        "total_charts": total_charts,
        "content_type_distribution": type_counts,
        "multi_column_pages": multi_col_pages,
        "detected_headers": all_headers,
        "detected_charts": all_charts,
        "pages": pages,
    }

    print(f"  Characters extracted: {total_chars:,}")
    print(f"  Pages with tables: {pages_with_tables}")
    print(f"  Total tables found: {total_tables} (filtered {total_false_positive_tables} 1-col false positives)")
    print(f"  Content type distribution: {type_counts}")
    print(f"  Multi-column pages: {multi_col_pages}")
    print(f"  Charts detected: {total_charts} (on {pages_with_charts} pages)")
    print(f"  Section headers detected: {len(all_headers)}")
    print()

    return result


def save_result(result: dict, output_dir: str):
    """Save parsed result to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(result["source_file"])[0]
    output_path = os.path.join(output_dir, f"{name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved to: {output_path} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_dir = os.path.join(project_root, "Documents")
    output_dir = os.path.join(project_root, "data", "extracted")

    if len(sys.argv) > 1:
        pdf_files = [sys.argv[1]]
    else:
        pdf_files = [
            os.path.join(docs_dir, f)
            for f in sorted(os.listdir(docs_dir))
            if f.endswith(".pdf")
        ]

    print(f"Found {len(pdf_files)} PDF(s) to parse\n")

    for pdf_path in pdf_files:
        result = parse_pdf(pdf_path)
        save_result(result, output_dir)

    print("\nDone. Check data/extracted/ for output.")
