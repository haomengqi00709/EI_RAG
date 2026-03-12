# RETRIEVAL_EXPERIMENTS.md

A change log and testing guide for every modification made during the retrieval
improvement sprint. Test changes **one at a time** and compare against the
confirmed baseline (83.3% @5).

---

## Metric: What Is Answered@k?

All experiments use `evaluate_llm_judge.py` which runs an **LLM-as-judge** evaluation:

1. For each of 60 questions, retrieve the top-k chunks
2. Ask Gemini: "Do any of these chunks contain enough information to answer the question?"
3. **Answered@k** = % of questions where the answer was found in the top-k chunks

This is similar to Recall@k but more lenient — it measures whether the *information* is
present in retrieved chunks, not whether the exact source chunk_id was returned.

> Note: this is not the same as F1 score. There is no false-positive penalty here —
> Answered@k only measures retrieval *recall* (did we find the right information?).
> A true F1 would also penalise retrieving irrelevant chunks, which we are not yet measuring.

---

## Confirmed Baseline

| Metric      | Score     | Config |
|-------------|-----------|--------|
| Answered@1  | 10.0%     | hybrid, top_k=5 |
| Answered@3  | 60.0%     | CANDIDATE_MULT=4, min 20 |
| **Answered@5**  | **83.3%** | BM25 body-only, original chunks, original text_nl |

Run label: `true_baseline` — confirmed reproducible on 2026-02-22.

**What "baseline" means in code:**
- `chunk.py`: no `is_subheader()` flush (original chunk boundaries)
- `enrich_tables.py`: original summary-style system prompt, cache key = `md5(table_text)`
- `embed.py`: breadcrumb + table title prepended to embed text (unchanged throughout)
- `retrieve.py`: `CANDIDATE_MULT=4`, min 20 candidates, BM25 = body text only

---

## Results Summary

| # | Label | @1 | @3 | @5 | Δ@5 | Verdict |
|---|-------|----|----|----|-----|---------|
| — | **true_baseline** | 10.0% | 60.0% | **83.3%** | — | ✅ Reference |
| 1 | after_reindex (all 22 changes at once) | 11.7% | 26.7% | 55.0% | -28.3% | ❌ Major regression |
| 2 | after_reindex_fixed (dedup/reorder moved to server.py) | 15.0% | 40.0% | 56.7% | -26.6% | ❌ Still regressed |
| 3 | bm25_fix (removed double-breadcrumb) | 11.7% | 36.7% | 51.7% | -31.6% | ❌ Worse |
| 4 | **D: BM25 body-only** (on new chunks) | 10.0% | 31.7% | 50.0% | -33.3% | ❌ Worse — confirms BM25 is not the issue |
| 5 | **D: candidate_pool_10x** | 16.7% | 45.0% | 66.7% | -16.6% | ❌ Worse — noise in RRF pool |
| 6 | **D: candidate_pool_4x** | 10.0% | 60.0% | **83.3%** | 0% | ✅ Matches baseline — keep at 4x |
| 7 | B: text_nl_v2 | — | — | — | — | ⏳ Not yet tested |
| 8 | A: semantic_chunking | — | — | — | — | ⏳ Not yet tested |
| 9 | A: parent_child_linking | — | — | — | — | ⏳ Not yet tested (bug to fix first) |

---

## Key Findings So Far

### ❌ Larger candidate pool (10×) hurts badly (-16.6% @5)
**Why:** RRF fusion with 50 candidates introduces borderline chunks that accumulate
enough score (from appearing mid-list in both dense and BM25) to outrank the
truly relevant chunk. The @1 metric actually improved slightly (10% → 16.7%),
meaning the very top result is marginally better — but ranks 2-5 fill with noise.
**Conclusion: 4× (20 candidates) is the sweet spot for this dataset. Do not increase.**

### ❌ BM25 metadata injection (tested on new chunks) — not the regression cause
Tested body-only BM25 vs injection — both gave ~50% with the semantically-chunked
data. Neither version recovered performance. This confirmed the regression was caused
by the **chunk boundary changes** (semantic chunking), not BM25 configuration.
**Conclusion: BM25 body-only is the baseline. Injection variants untested on clean baseline chunks.**

### ✅ Baseline is reproducible at 83.3%
After full revert (chunk.py, enrich_tables.py, retrieve.py), we confirmed the
baseline is stable and reproducible. All future experiments start from here.

---

## Pending Experiments

### Next: B — Text_nl v2 Prompt (Re-Enrich + Re-Embed, ~3-4 hrs)

**What it changes:** Re-generate all 2,354 table `text_nl` fields using a
row-enumeration prompt that explicitly lists every value (e.g. "Women: $99,
Men: $142, National average: $133") instead of a summary-style description.

**Why it might help:** 3 of the 10 baseline failures are demographic queries
(e.g. "what % of EI sickness claimants were women?"). The exact values only
appear in tables. Row-enumeration makes those values directly searchable by BM25.

**How to run:**
```bash
# 1. Update enrich_tables.py — add PROMPT_VERSION and new SYSTEM_PROMPT (see Change Inventory below)
# 2. Clear cache so all tables are re-generated
rm data/cache/text_nl_cache.json
python src/enrich_tables.py        # ~2-3 hrs (model inference for all tables)
python src/embed.py --reindex      # ~1-2 hrs
rm data/embeddings/bm25_corpus.json
python src/evaluate_llm_judge.py --label text_nl_v2
```

**Risk:** Text_nl v2 is longer → more terms in BM25 → possible precision loss.
But should significantly improve demographic/geographic query recall.

---

### After That: A — Semantic Chunking (Full Re-Run, ~4-6 hrs)

**What it changes:** Flushes narrative chunks at sub-section headers so headings
always start a new chunk rather than being buried mid-chunk.

**Why it might help:** Section-level queries find the right section more precisely.

**Why it might hurt:** Creates more, shorter chunks — relevant information that
previously lived in one chunk may now be split across two.

**⚠️ Important:** `evaluate_retrieval.py` (Recall@k) breaks when semantic chunking
is active because chunk IDs shift. Always use `evaluate_llm_judge.py` for these runs.

**How to run:**
```bash
# In chunk.py, restore the is_subheader flush:
#   elif acc["paras"] and is_subheader(para):
#       flush_narrative(propagate_carry=False)
python src/chunk.py
python src/enrich_tables.py        # cache hits — fast
python src/embed.py --reindex
rm data/embeddings/bm25_corpus.json
python src/evaluate_llm_judge.py --label semantic_chunking
```

---

### Future: A — Parent-Child Linking (Bug Fix Required First)

`_link_related_chunks()` in `chunk.py` is written but **never called** —
there is a recursive call inside the function body and no call site in
`chunk_document()`. Fix the bug before testing.

---

## Change Inventory (Full Details)

### retrieve.py changes

#### Candidate Pool Size
```python
# Baseline (keep this)
CANDIDATE_MULT = 4
candidate_k = max(top_k * CANDIDATE_MULT, 20)   # → 20 candidates at top_k=5

# Tested — worse
CANDIDATE_MULT = 10
candidate_k = max(top_k * CANDIDATE_MULT, 50)   # → 50 candidates at top_k=5
```
Result: 10× → 66.7% @5 (❌ -16.6%). **Keep at 4×.**

#### BM25 Text Function
```python
# Baseline (current)
def _chunk_to_bm25_text(row, chunk):
    if row["chunk_type"] == "table":
        return chunk.get("text_nl") or chunk.get("text", "")
    return chunk.get("text", "")

# Tested variant (body + metadata suffix) — tested only on new chunks, inconclusive
def _chunk_to_bm25_text(row, chunk):
    body = ...
    meta = f"fiscal year {fy} {bc}".strip()
    return f"{body}\n\n{meta}" if meta else body
```

### enrich_tables.py changes

#### Text_nl Prompt (v1 → v2)
```python
# Baseline (current)
SYSTEM_PROMPT = (
    "You are a government data analyst. Convert the following table into clear, "
    "searchable natural language prose. Describe the key figures, trends, and "
    "comparisons in a way that would help someone find this information by keyword search. "
    "Write in plain prose — no bullet points, no headers, no markdown."
)
cache_key = md5(table_text)

# v2 (to test)
SYSTEM_PROMPT = (
    "You are a precise data analyst converting government statistical tables into "
    "searchable natural language. Your output must enumerate EVERY row and column "
    "value explicitly so the text can be searched by keyword. "
    "Rules: (1) State each row's label and its value(s) in full — e.g. "
    "'Women: $99, Men: $142, National average: $133.' "
    "(2) Include all percentages, dollar amounts, counts, and year references exactly. "
    "(3) Do not summarise, interpret, or omit any row. "
    "(4) Write in plain prose — no bullet points, no headers, no markdown."
)
cache_key = md5(f"v2:{table_text}")   # version prefix invalidates old cache
```

### chunk.py changes

#### Semantic Chunking (is_subheader flush)
```python
# Baseline (current) — no flush on subheader
# (is_subheader function exists but is not called in the narrative loop)

# To test: add this block inside the narrative paragraph loop,
# after the section-boundary flush check:
elif acc["paras"] and is_subheader(para):
    flush_narrative(propagate_carry=False)
```

---

## Quick Reference Commands

```bash
source venv/bin/activate

# Eval only (LLM judge — use for all experiments)
python src/evaluate_llm_judge.py --label LABEL

# Delete BM25 corpus (auto-rebuilds on next eval run)
rm data/embeddings/bm25_corpus.json

# Re-enrich tables (slow if cache miss, fast if cache hit)
python src/enrich_tables.py

# Re-embed (slow, ~1-2 hrs)
python src/embed.py --reindex

# Full pipeline
python src/chunk.py && python src/enrich_tables.py && python src/embed.py --reindex

# View latest run scores
python -c "
import json
data = json.load(open('data/eval/llm_judge_results.json'))
a = data['answered']
print(f\"@1={a['@1']:.1%}  @3={a['@3']:.1%}  @5={a['@5']:.1%}  label={data.get('run_label','')}\")
"

# Compare all runs
python -c "
import json, glob
runs = sorted(glob.glob('data/eval/runs/judge_*.json'))
for r in runs:
    d = json.load(open(r))
    a = d['answered']
    print(f\"{d.get('run_label',''):<35} @1={a['@1']:.1%}  @3={a['@3']:.1%}  @5={a['@5']:.1%}\")
"
```
