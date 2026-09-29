"""LegalBench-RAG: legal retrieval scored against character-level gold spans. No LLM.

Four source sets — PrivacyQA (privacy policies), ContractNLI (NDAs), MAUD (M&A agreements)
and CUAD (commercial contracts) — each query carrying gold snippets as (file, start, end)
character ranges in a raw-text corpus. The split ``mini`` is the paper's LegalBench-RAG-mini:
194 queries per set, chosen with the exact procedure in the official ``benchmark.py``
(sorted by a random key seeded with the snippet's file path, so the 194 cover the fewest
documents), and the corpus is the documents those queries use.

Two scores are reported, because the published numbers use two different ones:

- ``char`` — the paper's metric, verbatim from ``run_benchmark.py``: precision is the share
  of retrieved *characters* inside a gold span, recall the share of gold characters
  retrieved. Its tables (RCTS, no rerank) are the baseline.
- ``hit`` — a retrieved passage counts when it overlaps a gold span at all; recall is the
  share of gold snippets that some retrieved passage overlaps. Ragie's published table is
  measured "slightly differently due to Ragie's chunking approach"; this is that reading,
  the one a production chunker can be held to.

A passage's character range is recovered by locating its text in the source file: the
knowledge bank's passages are exact slices of the document, so this is exact, not fuzzy.

Data: the official download (``data/corpus``, ``data/benchmarks``), mirrored at
``awinml/legalbench-rag``; set ``LEGALBENCH_RAG_DATA`` to its local path.
"""

from __future__ import annotations

import functools
import json
import os
import random
from pathlib import Path
from typing import Literal

from .base import Dataset
from ..models import Document, Query

K_VALUES = (1, 2, 4, 8, 16, 32, 64)
SETS = ("privacy_qa", "contractnli", "maud", "cuad")
MINI_PER_SET = 194

_DATA = Path(os.environ.get("LEGALBENCH_RAG_DATA", Path.home() / ".cache/hindsight/kbbench-data/lbr-data"))


def _mini(tests: list[dict]) -> list[dict]:
    """The official sampler: SORT_BY_DOCUMENT=True, MAX_TESTS_PER_BENCHMARK=194."""
    if len(tests) <= MINI_PER_SET:
        return tests

    def key(test: dict) -> float:
        random.seed(test["snippets"][0]["file_path"])
        return random.random()

    return sorted(tests, key=key)[:MINI_PER_SET]


@functools.lru_cache(maxsize=None)
def _load(split: str) -> tuple[dict, ...]:
    if split not in ("mini", "full"):
        raise ValueError(f"Unknown split '{split}'. Available: ['mini', 'full']")
    records = []
    for name in SETS:
        tests = json.loads((_DATA / "benchmarks" / f"{name}.json").read_text())["tests"]
        if split == "mini":
            tests = _mini(tests)
        for i, test in enumerate(tests):
            records.append(
                {
                    "id": f"{name}-{i}",
                    "set": name,
                    "query": test["query"],
                    "snippets": [
                        {"file": s["file_path"], "span": tuple(s["span"]), "answer": s.get("answer", "")}
                        for s in test["snippets"]
                    ],
                }
            )
    return tuple(records)


@functools.lru_cache(maxsize=None)
def _corpus_text(file_path: str) -> str:
    return (_DATA / "corpus" / file_path).read_text()


def _union(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def _locate(file_path: str, passage: str) -> tuple[int, int] | None:
    text = _corpus_text(file_path)
    start = text.find(passage)
    if start < 0:
        # Chunkers normalise edges; the stripped body is still an exact slice.
        body = passage.strip()
        start = text.find(body)
        if start < 0:
            return None
        return start, start + len(body)
    return start, start + len(passage)


class LegalBenchRAGDataset(Dataset):
    name = "legalbench-rag"
    description = (
        "LegalBench-RAG (PrivacyQA, ContractNLI, MAUD, CUAD): retrieval scored by character-level "
        "precision/recall@k against gold spans, plus passage-hit precision/recall. No judge."
    )
    splits = ["mini", "full"]
    task_type: Literal["retrieval"] = "retrieval"
    isolation_unit = None
    published = False
    links = [
        {"label": "Paper", "url": "https://arxiv.org/abs/2408.10343"},
        {"label": "Code", "url": "https://github.com/zeroentropy-ai/legalbenchrag"},
    ]

    def categories(self, split: str) -> list[str] | None:
        return list(SETS)

    def category_type(self, split: str, category: str) -> Literal["doc", "query"]:
        return "doc"

    def load_queries(self, split: str, category: str | None = None, limit: int | None = None) -> list[Query]:
        records = [r for r in _load(split) if category in (None, r["set"])][: limit or None]
        return [
            Query(
                id=r["id"],
                query=r["query"],
                gold_ids=sorted({s["file"] for s in r["snippets"]}),
                gold_answers=[s["answer"] for s in r["snippets"]],
                meta={
                    "category": r["set"],
                    "retrieval_limit": max(K_VALUES),
                    "snippets": [{"file": s["file"], "span": list(s["span"])} for s in r["snippets"]],
                },
            )
            for r in records
        ]

    def load_documents(
        self,
        split: str,
        category: str | None = None,
        limit: int | None = None,
        ids: set[str] | None = None,
        user_ids: set[str] | None = None,
    ) -> list[Document]:
        files = sorted({s["file"] for r in _load(split) if category in (None, r["set"]) for s in r["snippets"]})
        if ids is not None:
            files = [f for f in files if f in ids]
        # The file name carries the parties and the agreement's name, which is what the
        # queries name — the title a production ingest would have.
        docs = [Document(id=f, content=_corpus_text(f), context=Path(f).stem) for f in files]
        return docs[: limit or None]

    def score_retrieval(self, query: Query, retrieved: list[Document]) -> tuple[bool, str]:
        gold = [(s["file"], tuple(s["span"])) for s in query.meta["snippets"]]
        spans: list[tuple[str, tuple[int, int]]] = []
        unlocated = 0
        for doc in retrieved:
            file_path = next((s for s in (doc.source_ids or []) if s), None) or doc.id
            span = _locate(file_path, doc.content) if file_path else None
            if span is None:
                unlocated += 1
                continue
            spans.append((file_path, span))

        metrics: dict[str, float] = {}
        gold_len = sum(e - s for _, (s, e) in gold)
        for k in K_VALUES:
            top = spans[:k]
            retrieved_len = sum(e - s for _, (s, e) in top)
            hit_chars = sum(_overlap(span, g) for f, span in top for gf, g in gold if f == gf)
            metrics[f"char_precision_at_{k}"] = hit_chars / retrieved_len if retrieved_len else 0.0
            # Recall over the *union* of what was retrieved: overlapping passages (a chunker's
            # overlap) would otherwise count the same gold characters twice and pass 100%.
            # The paper's retrievers do not overlap, so for them the two are the same.
            covered = sum(_overlap(u, g) for gf, g in gold for u in _union([sp for f, sp in top if f == gf]))
            metrics[f"char_recall_at_{k}"] = covered / gold_len if gold_len else 0.0
            hits = [any(f == gf and _overlap(span, g) > 0 for gf, g in gold) for f, span in top]
            metrics[f"hit_precision_at_{k}"] = sum(hits) / len(top) if top else 0.0
            found = [any(f == gf and _overlap(span, g) > 0 for f, span in top) for gf, g in gold]
            metrics[f"hit_recall_at_{k}"] = sum(found) / len(gold) if gold else 0.0
        query.meta.update({key: round(value, 4) for key, value in metrics.items()})
        query.meta["retrieved_count"] = len(retrieved)
        query.meta["unlocated"] = unlocated
        top = max(K_VALUES)
        return metrics[f"hit_recall_at_{top}"] == 1.0, (
            f"hitR@{top}={metrics[f'hit_recall_at_{top}']:.2f} hitP@1={metrics['hit_precision_at_1']:.2f} "
            f"charR@{top}={metrics[f'char_recall_at_{top}']:.2f} charP@1={metrics['char_precision_at_1']:.3f}"
            + (f" unlocated={unlocated}" if unlocated else "")
        )

    def summary_metrics(self, results: list) -> dict:
        if not results:
            return {}
        out: dict = {}
        # The paper weights each set equally (0.25 each) — the same as a mean of set means.
        groups = {name: [r for r in results if r.meta.get("category") == name] for name in SETS}
        groups = {name: rs for name, rs in groups.items() if rs}
        for metric in ("char_precision", "char_recall", "hit_precision", "hit_recall"):
            for k in K_VALUES:
                key = f"{metric}_at_{k}"
                per_set = {name: sum(r.meta.get(key, 0.0) for r in rs) / len(rs) for name, rs in groups.items()}
                for name, value in per_set.items():
                    out[f"{name}|{key}"] = round(value, 4)
                out[key] = round(sum(per_set.values()) / len(per_set), 4)
        out["unlocated_passages"] = sum(r.meta.get("unlocated", 0) for r in results)
        return out

    def summarize_run(self, results: list, console) -> None:
        metrics = self.summary_metrics(results)
        if not metrics:
            return
        from rich.table import Table

        for metric, label in (("char", "character-level (paper)"), ("hit", "passage-hit (Ragie-style)")):
            table = Table(title=f"LegalBench-RAG — {label}")
            table.add_column("k")
            for name in (*SETS, "ALL"):
                table.add_column(f"{name} P", justify="right")
                table.add_column(f"{name} R", justify="right")
            for k in K_VALUES:
                row = [str(k)]
                for name in (*SETS, "ALL"):
                    prefix = "" if name == "ALL" else f"{name}|"
                    p = metrics.get(f"{prefix}{metric}_precision_at_{k}")
                    r = metrics.get(f"{prefix}{metric}_recall_at_{k}")
                    row += [f"{p * 100:.2f}" if p is not None else "-", f"{r * 100:.2f}" if r is not None else "-"]
                table.add_row(*row)
            console.print(table)
        if metrics["unlocated_passages"]:
            console.print(f"[yellow]{metrics['unlocated_passages']} passages could not be located in their source[/yellow]")
