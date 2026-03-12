#!/usr/bin/env python3
"""
test_embed_fix.py — Validate the Option 2 embed fix on 10 known-miss table chunks.

Phase 1 (always runs, no model needed):
    Prints before/after embed text for each target chunk to confirm the title
    duplication is removed. Safe to run anytime.

Phase 2 (--embed flag, requires embedding model):
    Re-embeds just the 10 target chunks using the Option 2 logic, stores them
    in a temporary copy of vectors.npy, then runs dense retrieval for the 10
    corresponding questions. Reports mini-recall@10.
    NOTE: dense-only (no BM25, no reranker) — sufficient to test embedding quality.

Usage:
    python src/test_embed_fix.py           # Phase 1 only (fast, ~1 sec)
    python src/test_embed_fix.py --embed   # Phase 1 + Phase 2 (loads embedding model)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ── Target chunks + questions ─────────────────────────────────────────────────
# 6 table regressions (new misses in Mar 3 run) + 4 persistent table misses.

TARGETS = [
    # table regressions
    ("20192020_table_0142_0044",
     "What was the average weekly Work-Sharing benefit rate for women in fiscal year 2018-2019?",
     "2019-2020"),
    ("20222023_table_0253_0179",
     "By what percentage did expenditures for Research & Innovation increase between fiscal year 2021-2022 and 2022-2023?",
     "2022-2023"),
    ("20222023_table_0432_0350",
     "What was the total amount, in thousands of dollars, paid out for compassionate care benefits in Manitoba during the 2022-2023 fiscal year?",
     "2022-2023"),
    ("20222023_table_0518_0478",
     "What were the total premium revenues for the Employment Insurance operating account in fiscal year 2021-2022?",
     "2022-2023"),
    ("20232024_table_0163_0063",
     "What was the average duration of Employment Insurance compassionate care claims in the 2020-21 fiscal year?",
     "2023-2024"),
    ("20232024_table_0575_0295",
     "What was the total number of Targeted Wage Subsidies in Newfoundland for the fiscal year 2023-2024?",
     "2023-2024"),
    # persistent table misses
    ("20202021_table_0245_0186",
     "What is the incidence of employment for the Skills Development Active program?",
     "2020-2021"),
    ("20212022_table_0346_0231",
     "How many new EI claims were established in Newfoundland and Labrador during the first half of fiscal year 2021-2022?",
     "2021-2022"),
    ("20212022_table_0388_0274",
     "What was the total number of new claims established for parental extended benefits in fiscal year 2017-2018?",
     "2021-2022"),
    ("20222023_table_0258_0186",
     "In fiscal year 2022-2023, what was the year-over-year change in expenditures for Targeted Wage Subsidies?",
     "2022-2023"),
]

QUERY_INSTRUCTION = (
    "Given a Canadian government Employment Insurance (EI) policy document, "
    "retrieve the passage that answers the following question: "
)

TOP_K = 10


# ── Embed text builders ───────────────────────────────────────────────────────

def build_embed_text_old(chunk: dict) -> str:
    """Current logic — title appears twice for enriched tables (the bug)."""
    if chunk["chunk_type"] == "table":
        body = chunk.get("text_nl") or chunk["text"]
    else:
        body = chunk["text"]
    header_parts = []
    bc = (chunk.get("section_breadcrumb") or "").strip()
    tt = chunk.get("table_title", "").strip() if chunk["chunk_type"] == "table" else ""
    if bc:
        header_parts.append(f"Section: {bc}")
    if tt:
        header_parts.append(f"Table: {tt}")
    return ("\n".join(header_parts) + "\n\n" + body) if header_parts else body


def build_embed_text_new(chunk: dict) -> str:
    """Option 2 fix — Table: header suppressed when text_nl is available.
    text_nl (v4) already starts with the title verbatim, so the header is
    redundant. Only use the header as a fallback for raw-markdown tables."""
    if chunk["chunk_type"] == "table":
        body = chunk.get("text_nl") or chunk["text"]
    else:
        body = chunk["text"]
    header_parts = []
    bc = (chunk.get("section_breadcrumb") or "").strip()
    tt = chunk.get("table_title", "").strip() if chunk["chunk_type"] == "table" else ""
    if bc:
        header_parts.append(f"Section: {bc}")
    # Only add Table: header when text_nl is NOT available (raw markdown fallback).
    # When text_nl exists, its first sentence already contains the title (v4 format).
    if tt and not chunk.get("text_nl"):
        header_parts.append(f"Table: {tt}")
    return ("\n".join(header_parts) + "\n\n" + body) if header_parts else body


# ── Chunk loader ──────────────────────────────────────────────────────────────

def load_target_chunks(target_ids: set) -> dict:
    chunks = {}
    for f in sorted(Path("data/chunked").glob("*.jsonl")):
        for line in open(f):
            c = json.loads(line)
            if c["chunk_id"] in target_ids:
                chunks[c["chunk_id"]] = c
    return chunks


# ── Phase 1: Text inspection ──────────────────────────────────────────────────

def phase1(chunks: dict):
    print("=" * 80)
    print("PHASE 1: Before / After embed text (no model needed)")
    print("=" * 80)

    all_changed = True
    for chunk_id, question, fy in TARGETS:
        chunk = chunks.get(chunk_id)
        if not chunk:
            print(f"\n[!] {chunk_id} — NOT FOUND in data/chunked/")
            all_changed = False
            continue

        old = build_embed_text_old(chunk)
        new = build_embed_text_new(chunk)
        tt  = chunk.get("table_title", "")
        has_nl = bool(chunk.get("text_nl"))

        print(f"\n{'─'*80}")
        print(f"Chunk : {chunk_id}")
        print(f"Title : {tt[:90]!r}")
        print(f"text_nl present: {has_nl}")
        print()
        print(f"OLD ({len(old)} chars) — first 400:")
        print("  " + old[:400].replace("\n", "\n  "))
        print()
        print(f"NEW ({len(new)} chars) — first 400:")
        print("  " + new[:400].replace("\n", "\n  "))
        changed = old != new
        print(f"\n  → Changed: {'YES ✓' if changed else 'NO  ← unexpected, check logic'}")
        if not changed:
            all_changed = False

    print("\n" + "=" * 80)
    if all_changed:
        print("Phase 1 PASSED — all 10 chunks produce different (de-duplicated) embed text.")
    else:
        print("Phase 1 WARNING — some chunks were unchanged. Review output above.")
    print("=" * 80)


# ── Phase 2: Mini re-embed + retrieval test ───────────────────────────────────

def phase2(chunks: dict):
    print("\n" + "=" * 80)
    print("PHASE 2: Re-embed 10 chunks + dense retrieval test")
    print("NOTE: dense-only (no BM25, no reranker) — used to isolate embedding quality")
    print("=" * 80)

    vectors  = np.load("data/embeddings/vectors.npy")
    manifest = [json.loads(l) for l in open("data/embeddings/manifest.jsonl")]
    chunk_id_to_index = {r["chunk_id"]: r["index"] for r in manifest}
    # group manifest rows by fiscal year for fast FY filtering
    fy_indices: dict[str, list[int]] = {}
    for r in manifest:
        fy_indices.setdefault(r.get("fiscal_year", ""), []).append(r["index"])

    # Load embedding model
    sys.path.insert(0, str(Path(__file__).parent))
    from embed import load_embedding_model, embed_one

    model, tokenizer = load_embedding_model()

    # Re-embed target chunks with Option 2 logic into a temporary copy
    print("\nRe-embedding 10 target chunks with Option 2 fix...")
    tmp_vectors = vectors.copy()
    embedded_ids = []
    for chunk_id, _, _ in TARGETS:
        chunk = chunks.get(chunk_id)
        if not chunk:
            print(f"  [!] {chunk_id} — not in chunked data")
            continue
        idx = chunk_id_to_index.get(chunk_id)
        if idx is None:
            print(f"  [!] {chunk_id} — not in manifest (was it embedded?)")
            continue
        embed_text = build_embed_text_new(chunk)
        vec = embed_one(model, tokenizer, embed_text)
        tmp_vectors[idx] = vec
        embedded_ids.append(chunk_id)
        print(f"  ✓ {chunk_id}  idx={idx}  chars={len(embed_text)}")

    print(f"\nRunning dense retrieval (top-{TOP_K}, FY-filtered) for 10 questions...")
    found = 0
    results = []
    for chunk_id, question, fy in TARGETS:
        query_text = QUERY_INSTRUCTION + question
        q_vec = embed_one(model, tokenizer, query_text)

        indices = fy_indices.get(fy, [])
        if not indices:
            print(f"  [!] No vectors for FY {fy}")
            results.append((chunk_id, None, []))
            continue

        fy_vecs = tmp_vectors[indices]
        scores  = fy_vecs @ q_vec
        top_local  = np.argsort(-scores)[:TOP_K]
        top_global = [indices[i] for i in top_local]
        top_ids    = [manifest[i]["chunk_id"] for i in top_global]

        if chunk_id in top_ids:
            rank = top_ids.index(chunk_id) + 1
            print(f"  ✓ FOUND rank={rank:2d}  {chunk_id}")
            found += 1
            results.append((chunk_id, rank, top_ids))
        else:
            print(f"  ✗ MISS           {chunk_id}")
            print(f"       top-3: {top_ids[:3]}")
            results.append((chunk_id, None, top_ids))

    print(f"\nMini Recall@{TOP_K}: {found}/{len(TARGETS)} = {found / len(TARGETS) * 100:.1f}%")
    print()
    if found == len(TARGETS):
        print("All 10 target chunks found — Option 2 fix validated. Safe to apply to embed.py.")
    elif found > len(TARGETS) // 2:
        print("Majority found — significant improvement. Review remaining misses before full apply.")
    else:
        print("Fewer than half found — investigate before applying the full fix.")
    print("=" * 80)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate Option 2 embed fix on 10 known-miss table chunks."
    )
    parser.add_argument(
        "--embed", action="store_true",
        help="Run Phase 2: re-embed 10 chunks and test dense retrieval."
    )
    args = parser.parse_args()

    target_ids = {t[0] for t in TARGETS}
    chunks = load_target_chunks(target_ids)
    print(f"Loaded {len(chunks)}/{len(target_ids)} target chunks from data/chunked/\n")

    phase1(chunks)

    if args.embed:
        phase2(chunks)
    else:
        print("\n[Phase 2 skipped — run with --embed to also test retrieval]\n")


if __name__ == "__main__":
    main()
