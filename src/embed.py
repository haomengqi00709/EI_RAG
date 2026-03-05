#!/usr/bin/env python3
"""
embed.py — Generates vector embeddings for all indexable chunks.

Input:   data/chunked/*.jsonl
Output:  data/embeddings/vectors.npy      — float32 array, shape (N, 2560)
         data/embeddings/manifest.jsonl   — one JSON line per vector

Chunk types embedded:   narrative, table (uses text_nl)
Chunk types excluded:   footnote, chart

Breadcrumb prepend: every embed text is prefixed with
    "Section: {breadcrumb}\\n\\n"
when the chunk has a non-empty section_breadcrumb.

For table chunks: text_nl (prose NL) is embedded, not the raw markdown.

Resume-safe: re-running skips chunk_ids already in the manifest.
Use --reindex to force a full re-embed from scratch.
Use --reindex-tables-only to re-embed only table chunks (e.g. after text_nl update), keep narrative vectors.

Run:
    python src/embed.py                    # embed all, resume if interrupted
    python src/embed.py --reindex          # re-embed everything from scratch
    python src/embed.py --reindex-tables-only   # re-embed only table chunks, keep narratives

Model: mlx-community/Qwen3-Embedding-4B-4bit-DWQ
       (~2.3 GB, downloads automatically to HuggingFace cache on first run)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_ID_MLX   = "mlx-community/Qwen3-Embedding-4B-4bit-DWQ"  # Apple Silicon
MODEL_ID_TORCH = "Qwen/Qwen3-Embedding-4B"                    # CUDA / CPU
EMBED_DIM   = 2560       # Qwen3-Embedding-4B native output dimension
MAX_SEQ_LEN = 8192       # model context window (tokens)
SAVE_EVERY  = 50         # checkpoint frequency (chunks)

# Only these chunk types go into the vector index
EMBED_TYPES = {"narrative", "table"}

OUT_DIR       = Path("data/embeddings")
VECTORS_PATH  = OUT_DIR / "vectors.npy"
MANIFEST_PATH = OUT_DIR / "manifest.jsonl"

# #6: Per-type index paths — built alongside the combined index
TYPE_VECTORS_PATH  = {t: OUT_DIR / f"vectors_{t}.npy"   for t in ("narrative", "table")}
TYPE_MANIFEST_PATH = {t: OUT_DIR / f"manifest_{t}.jsonl" for t in ("narrative", "table")}


# ── Embed text construction ───────────────────────────────────────────────────

def build_embed_text(chunk: dict) -> str:
    """
    Construct the string that will actually be embedded.

    #1 — Contextual chunk headers: prepend section breadcrumb and (for tables
    without text_nl) the table title so the embedding captures *where* the
    chunk lives in the document, not just *what* it says.

    - Tables: embed text_nl (prose NL) rather than raw markdown.
    - Table title header: only added when text_nl is NOT present (raw markdown
      fallback). When text_nl exists, its first sentence already contains the
      title verbatim (v4 format), so adding the header would duplicate it and
      over-weight the title in the embedding vector.
    - All types: Section breadcrumb prepended when present.
    """
    if chunk["chunk_type"] == "table":
        body = chunk.get("text_nl") or chunk["text"]
    else:
        body = chunk["text"]

    header_parts = []
    bc = (chunk.get("section_breadcrumb") or "").strip()
    tt = chunk.get("table_title", "").strip() if chunk["chunk_type"] == "table" else ""

    if bc:
        header_parts.append(f"Section: {bc}")
    # Only add Table: header when text_nl is absent (raw markdown fallback).
    # text_nl (v4) starts with the title verbatim, so the header would duplicate it.
    if tt and not chunk.get("text_nl"):
        header_parts.append(f"Table: {tt}")

    if header_parts:
        return "\n".join(header_parts) + "\n\n" + body
    return body


# ── Backend detection ─────────────────────────────────────────────────────────

def _use_mlx() -> bool:
    """True when running on Apple Silicon with MLX available."""
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False


# ── Embedding model ───────────────────────────────────────────────────────────

def load_embedding_model():
    """
    Load the Qwen3-Embedding model and tokenizer.
    - Apple Silicon (MLX available): loads 4-bit MLX-quantised model (fast, low RAM)
    - CUDA / CPU (RunPod / Linux):   loads full float16 model via transformers
    """
    if _use_mlx():
        from mlx_lm import load
        print(f"Loading {MODEL_ID_MLX} (MLX) ...")
        model, tokenizer = load(MODEL_ID_MLX)
        print("Embedding model loaded (MLX).\n")
        return model, tokenizer
    else:
        import torch
        from transformers import AutoModel, AutoTokenizer
        print(f"Loading {MODEL_ID_TORCH} (transformers) ...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID_TORCH)
        device    = "cuda" if torch.cuda.is_available() else "cpu"
        dtype     = torch.float16 if device == "cuda" else torch.float32
        model     = AutoModel.from_pretrained(
            MODEL_ID_TORCH,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        ).eval()
        if device != "cuda":
            model = model.to(device)
        print(f"Embedding model loaded (torch/{device}).\n")
        return model, tokenizer


def embed_one(model, tokenizer, text: str) -> np.ndarray:
    """
    Embed a single text string.
    Returns a L2-normalised float32 numpy array of shape (EMBED_DIM,).
    Uses last-token pooling as recommended by the Qwen3-Embedding model card.
    Dispatches to MLX or torch backend based on model type.
    """
    if _use_mlx():
        return _embed_one_mlx(model, tokenizer, text)
    else:
        return _embed_one_torch(model, tokenizer, text)


def _embed_one_mlx(model, tokenizer, text: str) -> np.ndarray:
    """MLX backend (Apple Silicon)."""
    import mlx.core as mx

    tokens = [int(t) for t in tokenizer.encode(text)]
    if len(tokens) > MAX_SEQ_LEN:
        tokens = tokens[:MAX_SEQ_LEN]

    input_ids = mx.array([tokens], dtype=mx.int32)

    hidden = model.model.embed_tokens(input_ids)
    for layer in model.model.layers:
        hidden = layer(hidden)
    hidden = model.model.norm(hidden)

    emb  = hidden[0, -1, :]
    norm = mx.linalg.norm(emb)
    emb  = emb / mx.maximum(norm, 1e-9)
    emb  = emb.astype(mx.float32)
    mx.eval(emb)
    return np.array(emb)


def _embed_one_torch(model, tokenizer, text: str) -> np.ndarray:
    """Transformers backend (CUDA / CPU)."""
    import torch

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=MAX_SEQ_LEN, padding=False,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Last-token pooling — handle both left and right padding
    hidden = outputs.last_hidden_state          # (1, seq_len, dim)
    mask   = inputs.get("attention_mask")
    if mask is not None:
        seq_lens = mask.sum(dim=1) - 1          # index of last real token
        emb = hidden[0, seq_lens[0], :]
    else:
        emb = hidden[0, -1, :]

    norm = emb.norm().clamp(min=1e-9)
    emb  = (emb / norm).float().cpu()
    return emb.numpy()


# ── Persistence helpers ───────────────────────────────────────────────────────

def load_existing() -> tuple[list, list, set]:
    """
    Load existing vectors and manifest for resume.
    Returns (vectors_list, manifest_rows, done_chunk_ids).
    """
    if not MANIFEST_PATH.exists():
        return [], [], set()
    manifest_rows = [json.loads(l) for l in open(MANIFEST_PATH)]
    done_ids = {r["chunk_id"] for r in manifest_rows}
    vectors = list(np.load(VECTORS_PATH)) if VECTORS_PATH.exists() else []
    return vectors, manifest_rows, done_ids


def save_outputs(vectors: list, manifest_rows: list):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Combined index (all types)
    np.save(VECTORS_PATH, np.array(vectors, dtype=np.float32))
    with open(MANIFEST_PATH, "w") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # #6: Per-type indexes — split combined into narrative / table sub-indexes
    for ctype in ("narrative", "table"):
        type_rows = [r for r in manifest_rows if r["chunk_type"] == ctype]
        if not type_rows:
            continue
        type_indices = [r["index"] for r in type_rows]
        type_vecs    = np.array(vectors, dtype=np.float32)[type_indices]
        # Re-index within the per-type manifest so indices stay consistent
        for local_i, row in enumerate(type_rows):
            row = dict(row)
            row["index"] = local_i
        np.save(TYPE_VECTORS_PATH[ctype], type_vecs)
        with open(TYPE_MANIFEST_PATH[ctype], "w") as f:
            for i, row in enumerate(type_rows):
                row = dict(row)
                row["index"] = i
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── Core logic ────────────────────────────────────────────────────────────────

def collect_chunks() -> list:
    """Load all embeddable chunks from data/chunked/*.jsonl."""
    chunks = []
    for f in sorted(Path("data/chunked").glob("*.jsonl")):
        for line in open(f):
            c = json.loads(line)
            if c["chunk_type"] in EMBED_TYPES:
                chunks.append(c)
    return chunks


def run(reindex: bool = False, reindex_tables_only: bool = False):
    all_chunks = collect_chunks()
    n_narrative = sum(1 for c in all_chunks if c["chunk_type"] == "narrative")
    n_table     = sum(1 for c in all_chunks if c["chunk_type"] == "table")
    print(f"Embeddable chunks: {len(all_chunks)}  "
          f"(narrative={n_narrative}, table={n_table})")

    todo_with_index = None  # (index, chunk) pairs when reindex_tables_only
    # Load existing progress unless full reindex
    if reindex and not reindex_tables_only:
        vectors, manifest_rows, done_ids = [], [], set()
        print("--reindex: starting from scratch.")
        todo = all_chunks
    elif reindex_tables_only:
        vectors, manifest_rows, done_ids = load_existing()
        if not manifest_rows:
            print("No existing index. Run without --reindex-tables-only first, or use --reindex.")
            return
        # Re-embed only table chunks; keep narrative vectors (replace in place by index)
        chunk_id_to_chunk = {c["chunk_id"]: c for c in all_chunks}
        todo_with_index = [(r["index"], chunk_id_to_chunk[r["chunk_id"]]) for r in manifest_rows if r["chunk_type"] == "table"]
        todo = [c for _, c in todo_with_index]
        print(f"--reindex-tables-only: re-embedding {len(todo)} table chunks (keeping {len(vectors) - len(todo)} narrative vectors).")
    else:
        vectors, manifest_rows, done_ids = load_existing()
        if done_ids:
            print(f"Resuming: {len(done_ids)} chunks already embedded.")
        todo = [c for c in all_chunks if c["chunk_id"] not in done_ids]
        print(f"To embed: {len(todo)}")

    if not todo:
        print("Nothing to do. Use --reindex to force a full re-embed.")
        return

    model, tokenizer = load_embedding_model()

    t0 = time.time()
    errors = 0

    for i, chunk in enumerate(todo, 1):
        try:
            embed_text = build_embed_text(chunk)
            vec = embed_one(model, tokenizer, embed_text)

            if reindex_tables_only and todo_with_index:
                idx, _ = todo_with_index[i - 1]
                vectors[idx] = vec
                manifest_rows[idx]["embed_chars"] = len(embed_text)
            else:
                vectors.append(vec)
                row = {
                    "index":              len(vectors) - 1,
                    "chunk_id":           chunk["chunk_id"],
                    "chunk_type":         chunk["chunk_type"],
                    "fiscal_year":        chunk.get("fiscal_year", ""),
                    "source_file":        chunk.get("source_file", ""),
                    "section_breadcrumb": chunk.get("section_breadcrumb", ""),
                    "page":               chunk.get("page_number") or chunk.get("start_page"),
                    "embed_chars":        len(embed_text),
                }
                if chunk["chunk_type"] == "table":
                    row["table_title"] = chunk.get("table_title", "")
                manifest_rows.append(row)

        except Exception as e:
            print(f"  ERROR on {chunk['chunk_id']}: {e}")
            errors += 1

        if i % SAVE_EVERY == 0 or i == len(todo):
            save_outputs(vectors, manifest_rows)
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta_s = (len(todo) - i) / rate if rate > 0 else 0
            print(f"  [{i}/{len(todo)}  {100*i/len(todo):.0f}%] "
                  f"{rate:.1f} chunks/s | ETA {eta_s/60:.1f} min | errors={errors}",
                  flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min.")
    print(f"Vectors:  {VECTORS_PATH}  shape=({len(vectors)}, {EMBED_DIM})")
    print(f"Manifest: {MANIFEST_PATH}  ({len(manifest_rows)} entries)")
    if errors:
        print(f"Errors: {errors} chunks skipped — re-run to retry failed chunks.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Embed EI MAR chunks with Qwen3-Embedding-4B (MLX)."
    )
    parser.add_argument(
        "--reindex", action="store_true",
        help="Re-embed everything from scratch, ignoring existing progress."
    )
    parser.add_argument(
        "--reindex-tables-only", action="store_true",
        help="Re-embed only table chunks (e.g. after text_nl update); keep narrative vectors."
    )
    args = parser.parse_args()
    run(reindex=args.reindex, reindex_tables_only=args.reindex_tables_only)


if __name__ == "__main__":
    main()
