import json, numpy as np
from pathlib import Path

vectors  = np.load("data/embeddings/vectors.npy")
manifest = [json.loads(l) for l in open("data/embeddings/manifest.jsonl")]

print(f"Total vectors: {vectors.shape}\n")
for m, v in zip(manifest[:10], vectors[:10]):
    print(f"[{m['chunk_id']}] {m['fiscal_year']} | {m['chunk_type']} | p.{m['page']} | {m['section_breadcrumb'][:60]}")
    print(f"  vector: [{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}, ... {v[-1]:.4f}]\n")
