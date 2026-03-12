#!/usr/bin/env python3
"""
enrich_tables.py — Adds a `text_nl` field to every table chunk.

Reads data/chunked/*.jsonl, calls the Gemini API for each table chunk that
doesn't already have a `text_nl` (v2 cache), and writes the result back in-place.
Non-table chunks are passed through unchanged.

Caching: every generated `text_nl` is stored in data/cache/text_nl_cache.json,
keyed by md5(CACHE_VERSION + table_text). Bumping CACHE_VERSION forces regeneration.

Resume-safe: re-running skips cache hits instantly.

Run:
    python src/enrich_tables.py                    # all files
    python src/enrich_tables.py data/chunked/2023-2024-EI-MAR-EN.jsonl  # one file

Requires: GOOGLE_API_KEY in environment or .env
"""

import hashlib
import json
import os
import re
import time
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

GEMINI_MODEL  = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
CACHE_VERSION = "v4"          # bump to invalidate old cache (v4: include annex # in first sentence when applicable)
SAVE_EVERY    = 25
CACHE_PATH    = Path("data/cache/text_nl_cache.json")
MIN_TABLE_CHARS = 50
RATE_LIMIT_DELAY = 0.5        # seconds between API calls (avoid 429s)

SYSTEM_PROMPT = (
    "You are a government data analyst. Your job is to convert a markdown table "
    "into exhaustive, searchable natural language. "
    "Your FIRST sentence must state the table title/caption exactly as given (e.g. 'Table 1 – Job vacancies...' or 'Annex 4.2.3 – Call Centres...'). "
    "If the table is in an annex (section or title mentions Annex N or ANNEX N), include the annex number in that first sentence so readers can find it. "
    "If a parent table is provided, this is a companion sub-table — your first sentence must name the parent table and briefly describe what metric this sub-table covers based on its column headers "
    "(e.g. 'Sub-table of Table 1 – LMDA Key Facts, Newfoundland and Labrador, FY2021: program expenditures.'). "
    "Then for EVERY data row, write one sentence that states the row label and ALL its "
    "column values explicitly — include every number, percentage, and dollar amount. "
    "Do NOT summarize, skip rows, or describe trends. "
    "Every value in the table must appear verbatim in your output."
)

USER_TEMPLATE = """\
Convert this table into natural language.
1) First sentence: state the table title/caption exactly as given below. If the section or title indicates an annex (e.g. ANNEX 4 or Annex 2.1), include the annex number in that first sentence. If a parent table is provided, your first sentence must state that this is a sub-table of the parent and briefly describe the metric it covers.
2) Then for every row: state the row label and all its column values explicitly. Do not skip any rows or summarize.

Report: Canadian Employment Insurance Monitoring and Assessment Report, fiscal year {fiscal_year}
Section: {section_breadcrumb}
Table title (include this verbatim in your first sentence): {table_title}
{annex_line}
{parent_line}
{text}"""

# ── Gemini setup ──────────────────────────────────────────────────────────────

def setup_gemini():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY not found in environment or .env")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )

# ── Cache ─────────────────────────────────────────────────────────────────────

def cache_key(table_text: str, parent_title: str | None = None) -> str:
    key_str = f"{CACHE_VERSION}:{table_text}"
    if parent_title:
        key_str += f":parent={parent_title}"
    return hashlib.md5(key_str.encode()).hexdigest()


def load_cache() -> dict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ── Generation ────────────────────────────────────────────────────────────────

def _annex_ref_from_chunk(chunk: dict) -> str | None:
    """Derive annex ref from chunk for prompt and for writing back if missing."""
    if chunk.get("annex_ref"):
        return chunk["annex_ref"]
    bc = chunk.get("section_breadcrumb") or ""
    title = chunk.get("table_title") or ""
    if re.search(r"\bANNEX\s+(\d+(?:\.\d+)?)", bc, re.IGNORECASE):
        m = re.search(r"\bANNEX\s+(\d+(?:\.\d+)?)", bc, re.IGNORECASE)
        if m:
            return f"Annex {m.group(1)}"
    if title and re.match(r"^Annex\s+(\d+(?:\.\d+)?)\b", title.strip(), re.IGNORECASE):
        m = re.match(r"^Annex\s+(\d+(?:\.\d+)?)\b", title.strip(), re.IGNORECASE)
        if m:
            return f"Annex {m.group(1)}"
    return None


def convert_table(model, chunk: dict, parent_title: str | None = None) -> str:
    annex_ref = _annex_ref_from_chunk(chunk)
    annex_line = f"Annex (include in first sentence if present): {annex_ref}" if annex_ref else ""
    parent_line = f"Parent table (reference in your first sentence): {parent_title}" if parent_title else ""
    prompt = USER_TEMPLATE.format(
        fiscal_year=chunk.get("fiscal_year") or "Unknown year",
        section_breadcrumb=chunk.get("section_breadcrumb") or "Unknown section",
        table_title=chunk.get("table_title") or "(no title)",
        annex_line=annex_line,
        parent_line=parent_line,
        text=chunk["text"],
    )
    for attempt in range(3):
        try:
            resp = model.generate_content(prompt)
            return resp.text.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower():
                wait = 30 * (attempt + 1)
                print(f"    Rate limit — waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed after 3 attempts: {chunk['chunk_id']}")


# ── File helpers ──────────────────────────────────────────────────────────────

def _write_jsonl(path: Path, chunks: list):
    with open(path, "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


# ── Core logic ────────────────────────────────────────────────────────────────

def _get_sample_chunks(paths: list, n: int, min_chars: int) -> list:
    """Collect up to n valid table chunks across paths (for --sample mode)."""
    samples = []
    for path in paths:
        chunks = [json.loads(line) for line in open(path)]
        for c in chunks:
            if c["chunk_type"] != "table":
                continue
            if c.get("char_count", 0) < max(MIN_TABLE_CHARS, min_chars):
                continue
            samples.append(c)
            if len(samples) >= n:
                return samples
    return samples


def run_sample_mode(model, paths: list, n: int, min_chars: int, out_path: Path | None = None):
    """Enrich n table chunks and print/save prompt + text_nl for review. Does not write to jsonl or cache."""
    chunks = _get_sample_chunks(paths, n, min_chars)
    if not chunks:
        print("No table chunks found meeting min-chars.")
        return
    lines = []
    for i, chunk in enumerate(chunks, 1):
        annex_ref = _annex_ref_from_chunk(chunk)
        annex_line = f"Annex (include in first sentence if present): {annex_ref}" if annex_ref else ""
        prompt = USER_TEMPLATE.format(
            fiscal_year=chunk.get("fiscal_year") or "Unknown year",
            section_breadcrumb=chunk.get("section_breadcrumb") or "Unknown section",
            table_title=chunk.get("table_title") or "(no title)",
            annex_line=annex_line,
            text=chunk["text"],
        )
        try:
            text_nl = convert_table(model, chunk)
            if i < len(chunks):
                time.sleep(RATE_LIMIT_DELAY)
        except Exception as e:
            text_nl = f"[ERROR: {e}]"
        title = chunk.get("table_title") or "(no title)"
        block = (
            f"{'='*80}\n"
            f"SAMPLE {i}/{len(chunks)}  chunk_id={chunk['chunk_id']}\n"
            f"table_title: {title[:120]}{'...' if len(title) > 120 else ''}\n"
            f"{'-'*80}\n"
            f"PROMPT (first 800 chars):\n{prompt[:800]}...\n\n"
            f"TEXT_NL (output):\n{text_nl}\n"
        )
        print(block)
        lines.append(block)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("".join(lines))
        print(f"\nFull output saved to {out_path}")


def _looks_like_table(chunk: dict) -> bool:
    """Narrative chunks that contain embedded table data (misclassified by parser)."""
    text = chunk.get("text", "")
    # Markdown table pipes OR dense numeric patterns (garbled multi-column text)
    has_pipes    = text.count("|") >= 4
    has_numbers  = sum(1 for w in text.split() if any(c.isdigit() for c in w)) > 10
    has_percents = text.count("%") >= 3
    return has_pipes or (has_numbers and has_percents)


def build_parent_child_map(chunks: list) -> dict:
    """
    Returns {child_chunk_id: parent_chunk_id} for untitled table chunks that
    immediately follow a titled table on the same page — the factsheet sub-table pattern.
    """
    from collections import defaultdict
    tables = [c for c in chunks if c["chunk_type"] == "table"]
    by_page: dict = defaultdict(list)
    for t in tables:
        pg = t.get("page_number")
        if pg is not None:
            by_page[pg].append(t)

    child_to_parent: dict = {}
    for ts in by_page.values():
        current_parent = None
        for t in ts:
            if t.get("table_title"):
                current_parent = t
            elif current_parent is not None:
                child_to_parent[t["chunk_id"]] = current_parent["chunk_id"]
    return child_to_parent


def enrich_file(model, jsonl_path: Path, cache: dict, min_chars: int = 0) -> dict:
    chunks = [json.loads(line) for line in open(jsonl_path)]
    chunk_index = {c["chunk_id"]: c for c in chunks}

    # Build parent-child map for sub-table detection
    child_to_parent = build_parent_child_map(chunks)

    def _parent_title(c: dict) -> str | None:
        pid = child_to_parent.get(c["chunk_id"])
        if pid and pid in chunk_index:
            return chunk_index[pid].get("table_title") or None
        return None

    # Include table chunks + narrative chunks that contain embedded table data
    table_chunks = [
        c for c in chunks
        if c["chunk_type"] == "table"
        or (c["chunk_type"] == "narrative" and _looks_like_table(c))
    ]
    tiny  = [c for c in table_chunks if c.get("char_count", 0) < MIN_TABLE_CHARS]
    valid = [c for c in table_chunks if c.get("char_count", 0) >= MIN_TABLE_CHARS]

    todo = [
        c for c in valid
        if cache_key(c["text"], _parent_title(c)) not in cache
        and c.get("char_count", 0) >= min_chars
    ]
    already_done = len(valid) - len(todo)

    stats = {
        "model_calls": 0,
        "cache_hits":  0,
        "already_done": already_done,
        "skipped_tiny": len(tiny),
        "errors": 0,
    }

    # Always write parent_chunk_id for all detected sub-tables (idempotent)
    new_parent_ids = 0
    for cid, pid in child_to_parent.items():
        if cid in chunk_index and chunk_index[cid].get("parent_chunk_id") != pid:
            chunk_index[cid]["parent_chunk_id"] = pid
            new_parent_ids += 1

    if not todo:
        print(f"  {jsonl_path.name}: all {len(valid)} valid table chunks already enriched "
              f"({stats['skipped_tiny']} tiny skipped).")
        if new_parent_ids:
            _write_jsonl(jsonl_path, list(chunk_index.values()))
            print(f"  {jsonl_path.name}: wrote parent_chunk_id for {new_parent_ids} sub-tables.")
        return stats

    print(f"  {jsonl_path.name}: {len(todo)} to process, "
          f"{already_done} already done, "
          f"{stats['skipped_tiny']} tiny skipped, "
          f"{len(child_to_parent)} sub-tables detected.")

    t_file_start = time.time()

    for i, chunk in enumerate(todo, 1):
        pt  = _parent_title(chunk)
        key = cache_key(chunk["text"], pt)
        try:
            if key in cache:
                text_nl = cache[key]
                stats["cache_hits"] += 1
            else:
                text_nl = convert_table(model, chunk, parent_title=pt)
                cache[key] = text_nl
                stats["model_calls"] += 1
                time.sleep(RATE_LIMIT_DELAY)

            chunk_index[chunk["chunk_id"]]["text_nl"] = text_nl
            # Ensure annex_ref is set for downstream (e.g. answers that cite annex)
            annex_ref = _annex_ref_from_chunk(chunk)
            if annex_ref and not chunk_index[chunk["chunk_id"]].get("annex_ref"):
                chunk_index[chunk["chunk_id"]]["annex_ref"] = annex_ref

        except Exception as e:
            print(f"    ERROR on {chunk['chunk_id']}: {e}")
            stats["errors"] += 1

        if i % SAVE_EVERY == 0 or i == len(todo):
            _write_jsonl(jsonl_path, list(chunk_index.values()))
            save_cache(cache)

            elapsed = time.time() - t_file_start
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (len(todo) - i) / rate if rate > 0 else 0
            eta = time.strftime("%H:%M:%S", time.gmtime(remaining))
            pct = 100 * i / len(todo)
            ts = time.strftime("%H:%M:%S")
            print(f"    [{i}/{len(todo)}  {pct:.0f}%] "
                  f"model={stats['model_calls']} cache={stats['cache_hits']} "
                  f"| ETA {eta}  {ts}", flush=True)

    return stats


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="Specific .jsonl files to process")
    parser.add_argument(
        "--min-chars", type=int, default=0,
        help="Only enrich tables with char_count >= this value"
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Test mode: enrich only N table chunks and print prompt + text_nl (no file/cache writes)"
    )
    parser.add_argument(
        "--sample-out", type=str, default="data/eval/enrich_sample_output.txt",
        help="Where to save full sample output when using --sample (default: data/eval/enrich_sample_output.txt)"
    )
    args = parser.parse_args()

    if args.files:
        input_paths = [Path(p) for p in args.files]
    else:
        input_paths = sorted(Path("data/chunked").glob("*.jsonl"))

    if not input_paths:
        print("No .jsonl files found.")
        return

    model = setup_gemini()

    if args.sample > 0:
        print(f"Sample mode: enriching {args.sample} table chunk(s). No .jsonl or cache writes.\n")
        run_sample_mode(
            model, input_paths, args.sample, args.min_chars,
            out_path=Path(args.sample_out),
        )
        return

    if args.min_chars:
        print(f"Selective mode: only enriching tables with char_count >= {args.min_chars}")

    cache = load_cache()
    print(f"Cache: {len(cache)} entries loaded from {CACHE_PATH}")
    print(f"Cache version: {CACHE_VERSION} — old v1 entries will be regenerated.")
    print(f"Model: {GEMINI_MODEL}\n")

    total_model = total_cache = total_tiny = total_errors = 0
    t0 = time.time()

    for path in input_paths:
        print(f"\n{path.name}")
        stats = enrich_file(model, path, cache, min_chars=args.min_chars)
        total_model  += stats["model_calls"]
        total_cache  += stats["cache_hits"]
        total_tiny   += stats["skipped_tiny"]
        total_errors += stats["errors"]
        print(f"  → model_calls={stats['model_calls']}, "
              f"cache_hits={stats['cache_hits']}, "
              f"already_done={stats['already_done']}, "
              f"skipped_tiny={stats['skipped_tiny']}, "
              f"errors={stats['errors']}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min.")
    print(f"API calls: {total_model} | Cache hits: {total_cache} | "
          f"Tiny skipped: {total_tiny} | Errors: {total_errors}")
    print(f"Cache now has {len(cache)} entries → {CACHE_PATH}")


if __name__ == "__main__":
    main()
