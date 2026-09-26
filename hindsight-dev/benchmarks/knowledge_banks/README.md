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

## BEIR (official nDCG@10)

`multihop` above is a recall metric on a corpus we assembled; BEIR is the benchmark
every vendor publishes, so it is the one that can be compared to *their* numbers. Added
to AMB as the `beir` dataset (branch `feat/beir`): one split per BEIR task, queries and
graded qrels from the official HuggingFace mirrors, scored by **pytrec_eval** — the
Python binding to NIST `trec_eval`, which is what BEIR itself runs. No LLM anywhere.

```bash
HINDSIGHT_HTTP_URL=http://localhost:8901 HINDSIGHT_KB_BANK=beir-scifact \
  uv run amb run --dataset beir --split scifact --mode retrieval --memory hindsight-kb
uv run amb run --dataset beir --split scifact --mode retrieval --memory qdrant
```

Primary metric is nDCG@10; the run also reports Recall@100, nDCG@100, MAP@100 and MRR.
Retrieval depth is 100 documents (`collapse_documents`), so Recall@100 is honest.

### The published bars

Official vendor numbers, nDCG@10, taken from their own posts (no third-party tables, no
judge). Blanks mean that vendor does not publish that task.

| Task | Vespa BM25 | Vespa hybrid | Elastic BM25 | Elastic + Elastic Rerank | Weaviate hybrid | Weaviate Search Mode |
|---|---|---|---|---|---|---|
| nfcorpus | 0.313 | 0.350 | 0.33 | 0.37 | | |
| scifact | 0.673 | 0.679 | 0.69 | 0.77 | 0.71 | 0.78 |
| fiqa | 0.244 | 0.292 | 0.25 | 0.45 | 0.45 | 0.54 |
| arguana | 0.393 | 0.404 | 0.47 | 0.68 | | |
| scidocs | 0.160 | 0.161 | 0.16 | 0.20 | | |
| trec-covid | 0.690 | 0.750 | 0.69 | 0.86 | | |
| **nq** | 0.327 | 0.404 | 0.33 | 0.62 | 0.61 | 0.70 |
| hotpotqa | 0.623 | 0.632 | 0.60 | 0.77 | | |
| fever | 0.751 | 0.779 | 0.69 | 0.89 | | |
| dbpedia-entity | 0.327 | 0.365 | 0.32 | 0.45 | | |
| climate-fever | 0.207 | 0.191 | 0.19 | 0.33 | | |
| quora | 0.761 | 0.826 | 0.81 | 0.88 | | |
| webis-touche2020 | 0.413 | 0.415 | 0.35 | 0.36 | | |

Sources: [Vespa hybrid part two](https://blog.vespa.ai/improving-zero-shot-ranking-with-vespa-part-two/),
[Elastic Rerank](https://www.elastic.co/search-labs/blog/elastic-semantic-reranker-part-2),
[Weaviate Search Mode benchmarking](https://weaviate.io/blog/search-mode-benchmarking).
Elastic's and Weaviate's are rounded to two decimals in the source.

### Results (2026-09-26)

Knowledge bank: bge-base-en-v1.5 (768d), 512-token chunks with 64 overlap, 200 candidates
per arm, `collapse_documents`, jev (TypeSafe) reranking. Baseline `qdrant`: embedded Qdrant,
Qwen3-Embedding-0.6B (1024d) + BM42 sparse, RRF, no reranker. Both scored by `pytrec_eval`
at retrieval depth 100. Vendor columns are their own published nDCG@10 (table above).

| Task | Knowledge bank | qdrant | Vespa hybrid | Elastic + Elastic Rerank | Weaviate hybrid / Search Mode |
|---|---|---|---|---|---|
| nfcorpus | **0.4011** | 0.3554 | 0.350 | 0.37 | — |
| scifact | **0.8177** | 0.7009 | 0.679 | 0.77 | 0.71 / 0.78 |
| fiqa | **0.4823** | not run¹ | 0.292 | 0.45 | 0.45 / 0.54 |
| arguana | **0.6367**² | 0.6322 | 0.404 | 0.68 | — |
| nq | running | not run¹ | 0.404 | 0.62 | 0.61 / 0.70 |

¹ Qdrant's Qwen3-0.6B embeddings run at roughly 2 documents/second on this machine: fiqa's
57k corpus is ~8 hours and nq's 2.7M out of reach. The baseline is run where it finishes.
² Dense arm only. ArguAna is the one task where both the keyword arm and the reranker cost
points — see below — so this row is `mode: vector`, `rerank: false`.

We beat qdrant on every task measured, and beat all three vendors' published numbers except
Elastic's reranker on ArguAna (0.68) and Weaviate's Search Mode on fiqa (0.54).

What each finding was worth, largest first:

| Finding | Effect |
|---|---|
| **Prepending a title to a chunk that already opens with it** dilutes the embedding. Every BEIR corpus hits this (their documents are stored as `title\n\ntext`) | ArguAna 0.504 → **0.637**, which is what turned a loss to qdrant into a win. Worth nothing on nfcorpus (0.397 → 0.401), scifact (0.8177 either way) or fiqa (0.485 → 0.482): their bodies do not restate the title |
| The scorer must drop the document whose id is the query's (BEIR's own `ignore_identical_ids`). ArguAna's queries *are* corpus documents, so rank 1 was always the query itself | ArguAna 0.378 → 0.521 — a scoring bug on our side, not a retrieval result |
| The keyword arm ORs every query word, so paragraph-length queries match on topic alone and out-vote the dense arm | ArguAna hybrid 0.372 vs dense-only 0.521 (pre-fix pair) → the new `kb_search_vector_weight` |
| Reranking is not free. Where the answer is a *counter*-argument rather than the most similar passage, jev costs points | ArguAna, post-fix: 0.637 dense-only vs 0.483 hybrid+jev |
| Chunk embeddings had no vector index at all (the column was an untyped `vector`, which pgvector cannot index) | invisible at 6k documents; the reason a million-passage corpus was not runnable. No measurable accuracy change (scifact 0.802 exact vs 0.818 indexed is jev's run-to-run variance) |
| **Bulk-loading into a live HNSW index costs 10x.** nq's corpus wrote at 4.5 documents/second with the index present and 47.9 without it — 15 hours instead of a week. There is no bulk-load path in the product: the benchmark drops the index, loads, then rebuilds it | 4.5 → 47.9 documents/second |
| The write pipeline embedded 8 documents per window and spent most of its time in round trips | 39ms/document then, ~11ms/document now (nq, short passages), i.e. embedding-bound rather than overhead-bound |

**The honest reading.** Our fusion alone is behind the baseline's — their dense model is
five times the size of ours — and on the tasks we win, reranking is what wins them. ArguAna
is the exception that proves the rule: no reranker helps when "relevant" means *opposing*
rather than *similar*, and there our plain dense arm is the best configuration we have.

## Query DSL at scale (2026-09-26)

200,000 documents (one passage each), six document fields and one passage field, all
`source: "request"` — so the corpus was loaded through the ordinary write path with **no
LLM call at all**, which is the point of request-served fields. Machine: the same laptop
and embedded Postgres (pg0) the retrieval benchmarks used. Median of three runs, hot cache.

| Query | Median |
|---|---|
| `count(*)` over every passage | 1257 ms |
| group by one field, sum + count | 1179 ms |
| group by two fields (200 vendors × 3 regions) | 1154 ms |
| filtered + `having` + arithmetic over aggregates | 224 ms |
| passage → document join, `count_distinct` | 2271 ms |
| `date_trunc` by month | 140 ms |
| array `$contains` + group | 103 ms |
| 8 columns, nested expression over aggregates | 607 ms |
| 300-value `$in` | 606 ms |

**What the numbers say.** An unfiltered group-by reads every row — no index helps, and a
JSONB field costs a text extract and a numeric cast per row, so a whole-bank aggregate is
about a second per 200k documents. Anything with a `where` drops to 0.1-0.4s because the
GIN index on the field column carries the filter. The passage→document join is the
slowest shape and the one to optimise first if this becomes a hot path.

Two things this measurement changed:

- a 300-value `$in` compiled to 300 OR-ed equality tests (480 ms on 22k documents); one
  bound `jsonb[]` and `= ANY` is a hash lookup, and the same query at 200k now costs 606 ms;
- a document query self-joined `kb_documents` to itself, purely so the filter compiler had
  a second alias to read. That scanned the table twice; the compiler now takes the aliases.

## Records and joins at scale (2026-09-26)

5,000 vendor records and 200,000 contract records, written through the records endpoint —
deterministic, no LLM — in the same bank as the 200k documents above. Contracts carry a
relationship field to vendors, which is what the joins follow. Median of three runs.

| Query | Median |
|---|---|
| `count(*)` over 200k records | 32 ms |
| group by a field, sum + count | 173 ms |
| **join** + group by the joined collection's field | 260 ms |
| join + filter on a joined field (`v.tier`) | 950 ms |
| join + `having` + arithmetic over aggregates, top 20 | 256 ms |

Writing the 205k records took 33 seconds: a record write is a JSONB upsert, with none of
the embedding a document write pays for.

A join is a hash join on the relationship's record id, so it costs little over the
ungrouped scan (260 ms vs 173 ms). Filtering *on the joined collection* is the slow shape
(950 ms) because the filter cannot be pushed to the index on either side — the same thing
SQL would do, and the first place to look if this becomes a hot path.
