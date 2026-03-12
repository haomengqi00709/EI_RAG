# PDF Parsing Pipeline

Technical reference for `src/parse_pdf.py`.
Covers architecture, output schema, configuration, and known limitations.

---

## Overview

Converts Employment Insurance Monitoring and Assessment Report (EI MAR) PDFs into
structured JSON ready for downstream chunking and embedding.

**Input**: PDF files in `Documents/`
**Output**: One JSON file per PDF in `data/extracted/`
**Run**:

```bash
# Parse all PDFs
python src/parse_pdf.py

# Parse one PDF
python src/parse_pdf.py Documents/2023-2024-EI-MAR-EN.pdf
```

---

## Pipeline Stages

Parsing runs in three sequential passes over each document.

### Pass 1 — Page extraction

For every page, in order:

1. **Table detection** (`page.find_tables()`)
   Finds tables with bounding boxes. Filters out 1-column false positives (bordered
   text boxes and bullet lists mis-detected as tables). Real table bounding boxes are
   retained to exclude those regions from text extraction.

2. **Text extraction** (narrative only)
   Characters falling inside real table bounding boxes are filtered out before calling
   `extract_text()`. This means the `text` field contains only narrative content — no
   duplicate or garbled table text.
   Multi-column pages (detected via x-position histogram gap analysis) have each column
   extracted independently and concatenated.

3. **Text normalization**
   Hyphenated line breaks are joined (`'COVID-\n19'` → `'COVID-19'`). Page number
   footers are stripped.

4. **Paragraph detection**
   Words are grouped into lines using y-position tolerance. A vertical gap between
   consecutive lines exceeding 1.6× the median line height signals a paragraph
   boundary. Multi-column pages process each column independently. Page numbers
   (which `extract_words()` sees but footer stripping removes from `text`) are
   filtered from the paragraph list.

5. **Header detection**
   Character-level font size and name data identifies heading hierarchy:
   - `major` — ≥ 28pt (chapter titles: CHAPTER I, HIGHLIGHTS, INTRODUCTION)
   - `chapter_subtitle` — ≥ 18pt (subtitle lines under chapter headings)
   - `section` — ≥ 12.5pt + bold or numbered (1.1, 2.3, Annex 2.4.1)

6. **Content type classification**
   Each page is classified into one of: `cover`, `copyright`, `toc`, `abbreviations`,
   `narrative`, `table_heavy`, `annex`, `blank`.
   `table_heavy` is assigned when table content exceeds narrative content in length
   (or when the page has tables but no narrative text at all).

7. **Temporal marker extraction**
   All fiscal year patterns (`2023-24`, `2022-2023`) and calendar years (`2020–2024`)
   are extracted. Markers are split into `primary_years` (this document's own fiscal
   year) and `referenced_years` (historical comparisons).

8. **Chart detection**
   Regex identifies chart titles (`Chart 3 – ...`) and source notes in the narrative
   text. Chart visual data is not extractable; only metadata is captured.

---

### Pass 2 — Hierarchy propagation

Walks all pages sequentially and carries the most-recently-seen header at each level
forward. Pages without their own headers inherit the running state. Non-content pages
(cover, toc, blank, copyright, abbreviations) reset the state. Each page receives:

- `hierarchy` dict — current major / chapter_subtitle / section values
- `section_breadcrumb` — human-readable path, e.g. `"CHAPTER II > 2.2 EI regular benefits"`

Tables and charts on each page also receive a `metadata` dict containing the full
hierarchy context at the time of that page.

---

### Pass 3 — Multi-page table stitching

Detects tables that were cut off at a page boundary and merges them.

**Detection criteria (all must be true):**
- Table has ≥ 2 columns (1-column tables excluded)
- Last table on page N: bottom edge within 80 pts of page bottom
- First table on page N+1: top edge within 95 pts of page top
- Column counts match

**Merging:**
- Rows from the continuation are appended to the source table
- If the continuation's first row is an identical repeat of the source header
  (common in government reports), that row is skipped
- Markdown is rebuilt from the merged pipe-delimited content
- The continuation table is removed from page N+1's table list
- `_stitched_pages` records which pages contributed

The pass repeats until no more stitches are possible, handling tables spanning 3+
pages.

**Post-stitch cleanup:**
`has_tables` is recomputed for all pages. Pages that lost their only table to
stitching and were previously classified as `table_heavy` are reclassified using
the updated (now table-free) state.

---

## Output JSON Schema

### Document level

```
{
  "source_file":                    str     — original filename
  "fiscal_year":                    str     — e.g. "2022-2023" (from filename)
  "total_pages":                    int
  "total_characters":               int     — narrative chars only (tables excluded)
  "pages_with_tables":              int
  "total_tables":                   int     — after false-positive filtering + stitching
  "false_positive_tables_filtered": int     — 1-col prose boxes removed
  "tables_stitched":                int     — multi-page merges performed
  "total_paragraphs":               int     — across all pages
  "pages_with_charts":              int
  "total_charts":                   int
  "content_type_distribution":      dict    — counts per type
  "multi_column_pages":             int
  "detected_headers":               list[dict]  — all headers found, with page + level
  "detected_charts":                list[dict]  — all charts found, with page + breadcrumb
  "pages":                          list[dict]  — one entry per page (see below)
}
```

### Page level

```
{
  "page_number":                    int
  "content_type":                   str     — cover | copyright | toc | abbreviations |
                                              narrative | table_heavy | annex | blank
  "text":                           str     — narrative text only (table regions excluded)
  "char_count":                     int     — len(text)
  "paragraphs":                     list[str]  — narrative split at paragraph boundaries
  "num_columns":                    int     — 1 or 2
  "has_tables":                     bool
  "tables":                         list[dict] — see Table schema below
  "false_positive_tables_skipped":  int     — 1-col prose boxes removed on this page
  "has_charts":                     bool
  "charts":                         list[dict] — see Chart schema below
  "headers":                        list[dict] — headers detected on THIS page only
  "hierarchy":                      dict    — propagated {major, chapter_subtitle, section}
  "section_breadcrumb":             str     — e.g. "CHAPTER II > 2.2 EI regular benefits"
  "temporal_markers":               list[str]  — all year refs found on this page
  "primary_years":                  list[str]  — years belonging to this document's FY
  "referenced_years":               list[str]  — historical/comparison years
  "_page_height":                   float   — page height in points (internal, for stitching)
  "_doc_fiscal_year":               str     — document FY (internal, for propagation)
}
```

### Table (element of `page["tables"]`)

```
{
  "table_index":      int
  "title":            str     — from text directly above the table bbox (e.g. "Table 3 – ...")
                                empty string if no title pattern found
  "content":          str     — pipe-delimited rows, one per line: "col1 | col2 | col3"
  "markdown":         str     — GitHub-flavoured Markdown table
  "row_count":        int     — total rows including header
  "col_count":        int     — column count
  "_bbox":            list    — [x0, top, x1, bottom] in points (internal)
  "_stitched_pages":  list[int]  — present only on stitched tables; pages that contributed
                                   e.g. [339, 340]
  "metadata": {
    "doc_fiscal_year":    str
    "page_number":        int     — page where the table STARTS
    "content_type":       str
    "section_breadcrumb": str
    "hierarchy":          dict
    "primary_years":      list[str]
    "referenced_years":   list[str]
  }
}
```

### Chart (element of `page["charts"]`)

```
{
  "chart_number":  str    — e.g. "3"
  "title":         str    — full title line, e.g. "Chart 3 – Year-over-year change..."
  "source":        str    — source note if found, e.g. "Source: Statistics Canada, ..."
  "is_image":      bool   — always True; actual chart data is not extractable
  "metadata":      dict   — same keys as table metadata above
}
```

---

## Configuration & Thresholds

All constants are defined at the top of `src/parse_pdf.py`.

| Constant | Value | What it controls |
|---|---|---|
| `FONT_TIER_MAJOR` | 28 pt | Min font size for major headings |
| `FONT_TIER_CHAPTER` | 18 pt | Min font size for chapter subtitles |
| `FONT_TIER_SECTION` | 12.5 pt | Min font size for section headers (must also be bold or numbered) |
| `FONT_TIER_SUBSECT` | 11.5 pt | Defined but not currently used |
| `MIN_COLUMN_GAP_PX` | 20 px | Minimum gap width to detect a two-column layout |
| `STITCH_BOTTOM_GAP` | 80 pt | Table bottom must be within this of page bottom to be a stitch candidate |
| `STITCH_TOP_GAP` | 95 pt | Continuation table top must be within this of page top |
| `_LINE_TOLERANCE` | 3 pt | Y-position tolerance for grouping chars into the same line |
| `_PARA_GAP_RATIO` | 1.6 | Gap-to-line-height ratio that triggers a paragraph break |

---

## What Is and Isn't Generalized

The **algorithms** are general and would work on any PDF:
column detection, table stitching, paragraph detection, text normalization,
table extraction, temporal marker extraction.

The **tuning** is calibrated to EI MAR reports and would need adjustment for
other document types:

| Hardcoded element | Where | Notes |
|---|---|---|
| Font size thresholds | `FONT_TIER_*` constants | Learned from EI MAR's Perpetua/Cambria fonts. Other documents use different fonts. |
| Cover/copyright keywords | `classify_page()` | `"monitoring and"`, `"issn"`, `"cat. no"` are EI MAR-specific strings |
| TOC detection | `classify_page()` | `text.count("...") > 5` — works for this report style |
| Chart title pattern | `RE_CHART_TITLE` | Matches `"Chart N –"`. Other reports may use `"Figure"`, `"Exhibit"`, etc. |
| Table/annex title patterns | `detect_table_title()` | Matches `"Table N –"` and `"Annex N.N –"` naming conventions |
| Fiscal year from filename | `extract_fiscal_year()` | Expects filenames like `2023-2024-EI-MAR-EN.pdf` |

To support a new document type, at minimum update: font thresholds (inspect fonts
on a sample page), `classify_page()` keywords, and chart/table title regex patterns.

---

## Validation

After parsing, run the validation script from `VALIDATION_CHECKLIST.md`:

```python
import json, os

for f in sorted(os.listdir("data/extracted")):
    if not f.endswith('.json'):
        continue
    with open(os.path.join("data/extracted", f)) as fh:
        d = json.load(fh)
    empty_bc = sum(1 for p in d['pages'] if not p['section_breadcrumb'])
    bc_pct = (1 - empty_bc / d['total_pages']) * 100
    print(f"{d['fiscal_year']} | pages={d['total_pages']} | "
          f"chars={d['total_characters']:,} | tables={d['total_tables']} | "
          f"stitched={d['tables_stitched']} | charts={d['total_charts']} | "
          f"paragraphs={d['total_paragraphs']:,} | breadcrumb={bc_pct:.0f}%")
    print(f"  types: {d['content_type_distribution']}")
```

**Expected ranges for EI MAR reports:**

| Metric | Expected range | Red flag |
|---|---|---|
| Pages | 513–650 | < 400 or > 800 |
| Characters | 816k–1.04M | Much lower = extraction failing |
| Tables | 422–569 | < 300 or > 650 |
| False positives filtered | 35–233 | — |
| Tables stitched | 8–19 | 0 = stitching not working |
| Charts | 67–99 | 0 = chart pattern not matching |
| Paragraphs | 1,200–1,974 | < 800 = paragraph detection failing |
| Breadcrumb coverage | 96–97% | < 90% = hierarchy propagation broken |

**Actual results across all 5 reports:**

| Year | Pages | Chars | Tables | Stitched | Paragraphs | Breadcrumb |
|---|---|---|---|---|---|---|
| 2019-2020 | 595 | 1,010,173 | 422 | 13 | 1,360 | 97% |
| 2020-2021 | 513 | 816,345 | 485 | 8 | 1,250 | 97% |
| 2021-2022 | 538 | 834,278 | 496 | 10 | 1,203 | 96% |
| 2022-2023 | 567 | 822,273 | 569 | 19 | 1,200 | 97% |
| 2023-2024 | 650 | 1,038,597 | 455 | 18 | 1,333 | 97% |

---

## Known Limitations

See `KNOWN_ISSUES.md` for full details. Summary of items not yet addressed:

| Issue | Impact | Priority |
|---|---|---|
| Multi-page tables spanning 3+ pages | Rare; handled by repeat-pass stitching, but a table that skips a page (blank continuation page) would not stitch | Low |
| Footnotes inline in body text | ~31% of chars on narrative pages are small-font footnotes; they appear in `text` and `paragraphs` as part of the narrative | Low |
| Chart visual data not extracted | Only title/source metadata captured; actual data values in charts are not available | Medium (future) |
| Font thresholds are EI MAR-specific | Adding new document types requires re-tuning | High (when expanding) |
| Tables without ruled lines | pdfplumber cannot detect whitespace-aligned tables; extracted as garbled text | Medium |
| Sub-subsection headers (2.2.1) | Only three hierarchy levels detected; fourth level not captured | Low |

*Last updated: 2026-02-19*
