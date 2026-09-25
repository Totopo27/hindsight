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

Knowledge bank: bge-base-en-v1.5 (110M, 768d), 512-token chunks with 64 overlap, 50
candidates per arm, cross-encoder rerank (ms-marco-MiniLM-L-6-v2), `collapse_documents`.
Baseline: Qwen3-Embedding-0.6B (1024d) + BM42, no reranker.

**2wiki — 1000 questions, ~6.3k documents**

| System | R@2 | R@5 | R@10 | R@20 |
|---|---|---|---|---|
| Hindsight knowledge bank | **62.30** | **71.43** | **76.20** | **78.90** |
| qdrant hybrid | 62.05 | 70.15 | 74.42 | 76.70 |

**2wiki-100 — 100 questions, ~835 documents**

| System | R@2 | R@5 | R@10 | R@20 |
|---|---|---|---|---|
| Hindsight knowledge bank | **65.75** | 72.75 | **79.25** | **83.75** |
| qdrant hybrid | 62.00 | **73.50** | 77.50 | 80.75 |

What each change was worth, measured:

| Change | Effect |
|---|---|
| Cross-encoder rerank on vs off (full split) | **+19.2 R@2, +12.9 R@5, +4.0 R@10, +2.0 R@20** |
| bge-base over bge-small (2wiki-100) | +1.5 R@20 |
| `collapse_documents` | +0.25 R@20; no query short of k documents |
| Candidates per arm 50 → 150 | nothing (−0.5 R@2) |
| Indexing the document title with each chunk | nothing here (2Wiki paragraphs usually name their subject); kept because it costs nothing and matters on corpora that do not |

**The honest reading.** Our fusion alone is *behind* the baseline's: with reranking off we
score 43.15 / 58.55 / 72.20 / 76.88, because their dense model is five times larger than
ours. The cross-encoder more than makes up for it. Two obvious ways to widen the gap: a
better embedding model on the dense arm, and a stronger reranker than MiniLM-L6.
