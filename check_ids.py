import json
from pathlib import Path

eval_set = json.load(open('data/eval/eval_set.json'))
manifest = json.load(open('data/embeddings/manifest.json'))

manifest_ids = {r['chunk_id'] for r in manifest}
eval_ids     = [q.get('chunk_id') for q in eval_set]

matches = sum(1 for cid in eval_ids if cid in manifest_ids)

print(f"Eval questions:      {len(eval_set)}")
print(f"Manifest chunks:     {len(manifest_ids)}")
print(f"Matching chunk IDs:  {matches}/{len(eval_set)}")
print()
print(f"Sample eval chunk_id:     {eval_ids[0]}")
print(f"Sample manifest chunk_id: {list(manifest_ids)[0]}")
