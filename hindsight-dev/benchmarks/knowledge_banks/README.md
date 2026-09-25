# Knowledge banks — retrieval benchmark

Measures a knowledge bank's search against a vector-store baseline on a standard
multi-hop retrieval set, scored by exact recall@k with no LLM in the loop.

**Dataset.** AMB's `multihop` (2WikiMultihopQA). Splits: `2wiki-50` (50 questions, ~446
docs), `2wiki-100` (~835 docs), `2wiki` (1000 questions, ~6.3k docs). HotpotQA is not in
AMB; 2Wiki is the multi-hop set it ships.

**Baseline.** AMB's `qdrant` provider — embedded Qdrant, dense Qwen3-Embedding-0.6B
(1024d) + sparse BM42, RRF fusion, 512-token chunks. The closest thing to "a hybrid
search stack a customer would build".

## Running it

The provider lives in AMB (`src/memory_bench/memory/hindsight_kb.py`);
`amb_provider_hindsight_kb.py` here is the copy to upstream. Until it is merged, drop it
into the AMB checkout (`~/.cache/hindsight/amb` by default) and register it in
`memory/__init__.py` as `hindsight-kb`.

```bash
# 1. a server with a local embedding model and a real cross-encoder reranker
HINDSIGHT_API_DATABASE_URL=pg0://kbbench:5801 \
HINDSIGHT_API_EMBEDDINGS_PROVIDER=local \
HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL=BAAI/bge-base-en-v1.5 \
HINDSIGHT_API_RERANKER_PROVIDER=local \
  uv run --project hindsight-api-slim hindsight-api --port 8901

# 2. us
cd ~/.cache/hindsight/amb
HINDSIGHT_HTTP_URL=http://localhost:8901 HINDSIGHT_KB_BANK=amb-2wiki \
  uv run amb run --dataset multihop --split 2wiki --mode retrieval --memory hindsight-kb

# 3. the baseline
uv run amb run --dataset multihop --split 2wiki --mode retrieval --memory qdrant
```

The provider creates the bank, writes the corpus in batches of 100 through the async
write operations, waits for each operation to finish, then queries with
`collapse_documents` so top-k means k distinct documents.

## Results (2026-09-25)

Corpus: **2wiki, 1000 questions over ~6.3k documents**. Knowledge bank: bge-base-en-v1.5
(110M, 768d), 512-token chunks with 64 overlap, `collapse_documents`. Baseline: embedded
Qdrant with Qwen3-Embedding-0.6B (1024d) + BM42 sparse, RRF, no reranker.

| System | Reranker | Candidates/arm | R@2 | R@5 | R@10 | R@20 |
|---|---|---|---|---|---|---|
| **Knowledge bank** | **jev** (TypeSafe) | **200** | 66.25 | **83.03** | **86.02** | **86.50** |
| Knowledge bank | jev | 100 | **67.45** | 81.27 | 83.60 | 84.15 |
| Knowledge bank | jev | 50 | 66.57 | 78.50 | 80.80 | 81.20 |
| Knowledge bank | MiniLM-L6 cross-encoder | 50 | 62.30 | 71.43 | 76.20 | 78.90 |
| Knowledge bank | none (fusion only) | 50 | 43.15 | 58.55 | 72.20 | 76.88 |
| qdrant hybrid | none | 50 (prefetch 100) | 62.05 | 70.15 | 74.42 | 76.70 |

**The best configuration beats the baseline by +4.2 R@2, +12.9 R@5, +11.6 R@10, +9.8 R@20.**

On the smaller `2wiki-100` split (MiniLM, 50 candidates) the knowledge bank scored
65.75 / 72.75 / 79.25 / 83.75 against the baseline's 62.00 / 73.50 / 77.50 / 80.75.

What each change was worth:

| Change | Effect |
|---|---|
| jev over the MiniLM cross-encoder (50 candidates) | +4.3 R@2, +7.1 R@5, +4.6 R@10, +2.3 R@20 |
| Reranking at all, MiniLM vs none | +19.2 R@2, +12.9 R@5, +4.0 R@10, +2.0 R@20 |
| Candidates 50 → 100 → 200, **with jev** | +1.9 then +2.4 at R@20; R@2 peaks at 100 |
| Candidates 50 → 150, with MiniLM | nothing. A deeper pool only pays with a ranker good enough to sort it |
| bge-base over bge-small (2wiki-100) | +1.5 R@20 |
| `collapse_documents` | +0.25 R@20, and no query returns fewer than k documents |
| Indexing the document title with each chunk | nothing here (2Wiki paragraphs name their subject anyway); kept because it is free and matters on corpora that do not |

**The honest reading.** Our fusion alone is *behind* the baseline's — 43 / 59 / 72 / 77 with
reranking off, because their dense model is five times the size of ours. The win comes from
reranking, and jev is what turns a narrow win into a wide one. The deeper pool matters for
the same reason: jev ranks 200 candidates listwise in one call, so widening the pool feeds
it more to sort instead of just costing time.

