# Post-Parsing Validation Checklist

Run this checklist after parsing all 5 EI MAR PDFs. Compare stats side by side to spot outliers and quality issues.

---

## 1. Summary Stats Comparison

Check that all reports produce roughly similar numbers. Big deviations signal extraction problems.

| Check | What to compare | Red flag |
|---|---|---|
| Total pages | Should be ~550-700 per report | A report with <400 or >800 pages |
| Total characters | Should be ~1.0-1.5M per report | Much lower = extraction failing on many pages |
| Tables found | Should be ~400-600 per report | Much lower = pdfplumber not detecting tables (font/layout change?) |
| Charts found | Should be ~60-100 per report | 0 charts = chart title pattern changed |
| Headers detected | Should be ~50-100 per report | <20 = font thresholds don't match this report's fonts |
| Multi-column pages | Should be ~20-30 | >100 = false positive problem with column detection |

---

## 2. Content Type Distribution

Each report should have a similar breakdown. Compare percentages, not raw counts.

| Content type | Expected % | Red flag |
|---|---|---|
| narrative | ~50-55% | <40% = too much classified as other types |
| table_heavy | ~20-25% | Much higher/lower = classification threshold issue |
| annex | ~20-25% | 0% = annex detection failing |
| blank | ~1-2% | >5% = extraction failing on pages (getting empty text) |
| toc | ~0.5-1% | 0 = TOC not detected (minor issue) |
| cover | ~0.3% | 0 = cover not detected (minor issue) |
| abbreviations | ~0.3% | 0 = not detected (minor issue) |

---

## 3. Hierarchy Detection Quality

### Check: Do all 4 chapters get detected in each report?

Each report should have:
- CHAPTER I (Labour market context)
- CHAPTER II (EI benefits)
- CHAPTER III (Employment benefits and support measures)
- CHAPTER IV (Program administration)
- HIGHLIGHTS, INTRODUCTION
- Multiple ANNEX sections

**Red flag**: Any report missing chapters = font size thresholds don't match.

### Check: Section-level headers detected?

Each report should have section headers like 1.1, 1.2, 2.1, 2.2, etc.

**Red flag**: A report with only major/chapter headers but 0 section headers = 13pt Cambria-Bold threshold doesn't match this report.

### Check: Breadcrumb coverage

After hierarchy propagation, >95% of pages should have non-empty breadcrumbs.

**Red flag**: >10% empty breadcrumbs = hierarchy propagation not working for this report.

---

## 4. Font Consistency Across Years

If header detection fails for a specific report, check the fonts:

```python
# Quick font check for a specific report
import pdfplumber
pdf = pdfplumber.open("Documents/XXXX-XXXX-EI-MAR-EN.pdf")
page = pdf.pages[20]  # Chapter 1 start page
fonts = set()
for c in page.chars:
    fonts.add((c.get('fontname','?'), round(c.get('size',0), 1)))
print(sorted(fonts, key=lambda x: -x[1])[:10])
```

Compare font names and sizes to the 2023-24 report thresholds:
- 34pt Perpetua-Bold → major heading
- 20pt Perpetua-Bold → chapter subtitle
- 13pt Cambria-Bold → section header
- 11pt Calibri → body text

---

## 5. COVID-era Reports (2020-21, 2021-22)

These reports cover pandemic years with temporary EI measures. Watch for:

| Check | Why |
|---|---|
| Additional sections | COVID temporary measures may have added sections not in other reports |
| Different section numbering | Sections may have been restructured for pandemic content |
| Larger page count | COVID reports may be longer due to temporary measure analysis |
| Different terminology | "CERB", "EI-ERB", "temporary measures" — domain-specific terms that appear only in these years |

---

## 6. French Content

Check how many pages contain French text across reports.

```python
# Quick French detection heuristic
french_indicators = ['les', 'des', 'est', 'sont', 'pour', 'dans', 'une', 'avec']
for page in data['pages']:
    words = page['text'].lower().split()
    french_count = sum(1 for w in words if w in french_indicators)
    if french_count > 10:
        print(f"Page {page['page_number']}: likely French ({french_count} indicators)")
```

**If >5% of pages are French**: Consider adding a `language` field per page.

---

## 7. Special Characters

Spot-check a few pages from each report for:
- Fiscal year patterns: do both `2020-21` and `2020–21` (en-dash) appear? Our regex handles both.
- Dollar amounts: `$1,234.5` vs `$1 234,5` (French number formatting)
- Percentage: `5.6%` vs `5,6 %` (French formatting)

---

## 8. Table Extraction Quality

For each report, spot-check one table from each tier:

| Tier | How to find | What to check |
|---|---|---|
| Clean (simple table) | Find a table with <6 columns | Headers readable? Data correct? |
| Medium (multi-line headers) | Find a table with 6-8 columns | Headers cleaned up properly after newline collapse? |
| Complex (annex table) | Find a table with >10 columns | Is it usable at all? |

---

## 9. Temporal Marker Quality

For each report, verify:
- `doc_fiscal_year` matches the filename
- `primary_years` on narrative pages include the report's own year
- `referenced_years` include comparison years (typically previous 1-2 fiscal years)

**Red flag**: A 2020-21 report where no page has `"2020-21"` in primary_years = fiscal year regex not matching.

---

## Quick Validation Script

After running all 5 PDFs, run this to generate the comparison table:

```python
import json, os

output_dir = "data/extracted"
for f in sorted(os.listdir(output_dir)):
    if not f.endswith('.json'):
        continue
    with open(os.path.join(output_dir, f)) as fh:
        d = json.load(fh)

    empty_bc = sum(1 for p in d['pages'] if not p['section_breadcrumb'])
    bc_pct = (1 - empty_bc / d['total_pages']) * 100

    print(f"{d['fiscal_year']} | "
          f"pages={d['total_pages']} | "
          f"chars={d['total_characters']:,} | "
          f"tables={d['total_tables']} | "
          f"charts={d.get('total_charts', '?')} | "
          f"headers={len(d['detected_headers'])} | "
          f"multi_col={d.get('multi_column_pages', '?')} | "
          f"breadcrumb={bc_pct:.0f}%")
    print(f"  types: {d['content_type_distribution']}")
    print()
```

---

*Created: 2025-02-15*
