"""PageIndex-OSS-Benchmark: 62 document questions over 34 PDFs, one PDF per question.

VectifyAI's own benchmark for open-source PageIndex (github.com/VectifyAI/PageIndex-OSS-
Benchmark): questions drawn from MMLongBench-Doc-V2 where the answer is a fact in running
text. PageIndex answers each question against the one document it names
(``client.responses(question, doc_id=...)``), so here every question is scoped to its PDF
(isolation unit = the document), and a provider is told which one.

Graded by MMLongBench-Doc-V2's judge — its prompt, loaded from the repo so it is the exact
text (``MMLB_REPO``), on its model (``gpt-5.6-luna``). Their published table is per chat
model; the comparable rows are the ones on the same model tier as ours.

PDFs are converted once to markdown with page markers (``PAGEINDEX_OSS_MD``), the same
converter as FinanceBench.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import os
from pathlib import Path
from typing import Literal

from .base import Dataset
from ..models import Document, Query

_ROOT = Path.home() / ".cache/hindsight/kbbench-data"
_QUESTIONS = Path(os.environ.get("PAGEINDEX_OSS_QUESTIONS", _ROOT / "pioss/questions.json"))
_MD = Path(os.environ.get("PAGEINDEX_OSS_MD", _ROOT / "pioss-md"))
_MMLB = Path(os.environ.get("MMLB_REPO", _ROOT / "mmlb"))


@functools.lru_cache(maxsize=1)
def _judge_prompt() -> str:
    spec = importlib.util.spec_from_file_location("mmlb_judge", _MMLB / "eval" / "judge.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.PROMPT


@functools.lru_cache(maxsize=1)
def _rows() -> tuple[dict, ...]:
    return tuple(json.loads(_QUESTIONS.read_text()))


def _doc_key(doc_id: str) -> str:
    return Path(doc_id).stem


class PageIndexOSSDataset(Dataset):
    name = "pageindex-oss"
    description = (
        "PageIndex-OSS-Benchmark: 62 questions over 34 PDFs (MMLongBench-Doc-V2 lookups), each "
        "answered against its own document, graded by MMLongBench-Doc-V2's judge."
    )
    splits = ["test"]
    task_type: Literal["open"] = "open"
    isolation_unit = None  # scoping is per question (user_id), the corpus is one bank
    published = False
    links = [
        {"label": "Benchmark", "url": "https://github.com/VectifyAI/PageIndex-OSS-Benchmark"},
        {"label": "Judge", "url": "https://github.com/VectifyAI/MMLongBench-Doc-V2"},
    ]

    def load_queries(self, split: str, category: str | None = None, limit: int | None = None) -> list[Query]:
        rows = [r for r in _rows() if category in (None, r.get("task_type"))][: limit or None]
        return [
            Query(
                id=f"q{i:02d}-{_doc_key(r['doc_id'])[:12]}",
                query=r["question"],
                gold_ids=[_doc_key(r["doc_id"])],
                gold_answers=[r["answer"]],
                user_id=_doc_key(r["doc_id"]),
                meta={
                    "category": r.get("task_type"),
                    "answer_format": r.get("answer_format", "Str"),
                    "evidence_pages": r.get("evidence_pages"),
                    "doc_type": r.get("doc_type"),
                },
            )
            for i, r in enumerate(rows)
        ]

    def load_documents(
        self,
        split: str,
        category: str | None = None,
        limit: int | None = None,
        ids: set[str] | None = None,
        user_ids: set[str] | None = None,
    ) -> list[Document]:
        doc_types = {_doc_key(r["doc_id"]): r.get("doc_type") for r in _rows()}
        keys = sorted(doc_types)
        if ids is not None:
            keys = [k for k in keys if k in ids]
        if user_ids is not None:
            keys = [k for k in keys if k in user_ids]
        docs = []
        for key in keys[: limit or None]:
            path = _MD / f"{key}.md"
            if not path.exists():
                raise FileNotFoundError(f"{path} — convert the PDFs first")
            docs.append(Document(id=key, content=path.read_text(), context=key, user_id=key))
        return docs

    def build_judge_prompt(self, query: str, gold_answers: list[str], answer: str) -> str:
        # AMB does not pass answer_format through the judge hook; MMLongBench's rows carry
        # it, and the prompt only uses it as a hint, so the common format stands in.
        return _judge_prompt().format(
            question=" ".join(query.split()),
            answer=gold_answers[0] if gold_answers else "",
            answer_format="Str",
            response=answer[:12000],
        ) + '\n\nReply as JSON with "correct" (the value you would give "equivalent") and "reason".'

    def default_judge_llm(self):
        from ..llm.openai import OpenAILLM

        return OpenAILLM(os.environ.get("PAGEINDEX_OSS_JUDGE_MODEL", "gpt-5.6-luna"))
