#!/usr/bin/env python3
"""
tmp_viz_vectors.py — 2D visualisation of the vector index.

Shows all embedded chunks projected to 2D via UMAP (falls back to t-SNE).
Colour = fiscal year, shape = chunk type (narrative ● / table ▲)

Run:
    python tmp_viz_vectors.py                  # all chunks
    python tmp_viz_vectors.py --type narrative # narrative only
    python tmp_viz_vectors.py --type table     # table only
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Load data ──────────────────────────────────────────────────────────────────

VECTORS_PATH  = Path("data/embeddings/vectors.npy")
MANIFEST_PATH = Path("data/embeddings/manifest.jsonl")

print("Loading vectors and manifest…")
vectors  = np.load(VECTORS_PATH)
manifest = [json.loads(l) for l in open(MANIFEST_PATH)]

assert len(vectors) == len(manifest), "Vector/manifest length mismatch"
print(f"Loaded {len(vectors)} vectors  shape={vectors.shape}")

# ── Filter ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--type", choices=["narrative", "table", "all"], default="all")
args = parser.parse_args()

if args.type != "all":
    idx      = [i for i, m in enumerate(manifest) if m["chunk_type"] == args.type]
    vectors  = vectors[idx]
    manifest = [manifest[i] for i in idx]
    print(f"Filtered to {args.type}: {len(vectors)} vectors")

# ── Dimensionality reduction ───────────────────────────────────────────────────

try:
    import umap
    print("Reducing with UMAP…")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords  = reducer.fit_transform(vectors)
    method  = "UMAP"
except ImportError:
    from sklearn.manifold import TSNE
    print("UMAP not found — falling back to t-SNE (slower)…")
    sample = min(len(vectors), 3000)
    if sample < len(vectors):
        print(f"  Sampling {sample}/{len(vectors)} for t-SNE speed")
        idx      = np.random.choice(len(vectors), sample, replace=False)
        vectors  = vectors[idx]
        manifest = [manifest[i] for i in idx]
    coords = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(vectors)
    method = "t-SNE"

print(f"{method} done. Plotting…")

# ── Colour / shape maps ────────────────────────────────────────────────────────

FY_COLOURS = {
    "2019-2020": "#ef4444",
    "2020-2021": "#f97316",
    "2021-2022": "#eab308",
    "2022-2023": "#22c55e",
    "2023-2024": "#3b82f6",
}
DEFAULT_COLOUR = "#a855f7"

MARKERS = {"narrative": "o", "table": "^"}

# ── Plot ───────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_title(f"EI MAR Vector Index — {method} projection\n"
             f"{len(vectors)} chunks  ({args.type})",
             fontsize=14, fontweight="bold")

for ctype, marker in MARKERS.items():
    sub_idx = [i for i, m in enumerate(manifest) if m["chunk_type"] == ctype]
    if not sub_idx:
        continue
    for i in sub_idx:
        fy     = manifest[i].get("fiscal_year", "")
        colour = FY_COLOURS.get(fy, DEFAULT_COLOUR)
        ax.scatter(coords[i, 0], coords[i, 1],
                   c=colour, marker=marker, s=18, alpha=0.65, linewidths=0)

# ── Legend ─────────────────────────────────────────────────────────────────────

fy_patches = [mpatches.Patch(color=c, label=fy) for fy, c in FY_COLOURS.items()]
type_handles = [
    plt.Line2D([0], [0], marker="o", color="grey", linestyle="", markersize=8, label="narrative"),
    plt.Line2D([0], [0], marker="^", color="grey", linestyle="", markersize=8, label="table"),
]
ax.legend(handles=fy_patches + type_handles,
          title="Fiscal year / chunk type",
          loc="upper left", fontsize=9, framealpha=0.8)

ax.set_xlabel(f"{method} dim 1"), ax.set_ylabel(f"{method} dim 2")
ax.set_xticks([]), ax.set_yticks([])

plt.tight_layout()
out = Path("tmp_vector_plot.png")
plt.savefig(out, dpi=150)
print(f"Saved → {out}")
plt.show()
