#!/usr/bin/env python3
"""Reconstruct manifest.jsonl from chunked JSONL files."""
import json
from pathlib import Path

EMBED_TYPES = {"narrative", "table"}


def build_embed_text(chunk):
    if chunk["chunk_type"] == "table":
        body = chunk.get("text_nl") or chunk["text"]
    else:
        body = chunk["text"]
    header_parts = []
    bc = (chunk.get("section_breadcrumb") or "").strip()
    tt = chunk.get("table_title", "").strip() if chunk["chunk_type"] == "table" else ""
    if bc:
        header_parts.append(f"Section: {bc}")
    if tt and not chunk.get("text_nl"):
        header_parts.append(f"Table: {tt}")
    return ("\n".join(header_parts) + "\n\n" + body) if header_parts else body


chunks = []
for f in sorted(Path("data/chunked").glob("*.jsonl")):
    for line in open(f):
        c = json.loads(line)
        if c["chunk_type"] in EMBED_TYPES:
            chunks.append(c)

print(f"Collected {len(chunks)} chunks")

with open("data/embeddings/manifest.jsonl", "w") as f:
    for i, chunk in enumerate(chunks):
        row = {
            "index":              i,
            "chunk_id":           chunk["chunk_id"],
            "chunk_type":         chunk["chunk_type"],
            "fiscal_year":        chunk.get("fiscal_year", ""),
            "source_file":        chunk.get("source_file", ""),
            "section_breadcrumb": chunk.get("section_breadcrumb", ""),
            "page":               chunk.get("page_number") or chunk.get("start_page"),
            "embed_chars":        len(build_embed_text(chunk)),
        }
        if chunk["chunk_type"] == "table":
            row["table_title"] = chunk.get("table_title", "")
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Written {len(chunks)} rows to data/embeddings/manifest.jsonl")
print(f"Size: {Path('data/embeddings/manifest.jsonl').stat().st_size / 1024:.0f} KB")
