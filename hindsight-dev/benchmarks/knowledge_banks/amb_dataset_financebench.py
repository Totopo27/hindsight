"""FinanceBench: 150 open questions over SEC filings, graded against analyst gold answers.

The public set (``PatronusAI/financebench``): 150 questions over 84 filings (10-K, 10-Q,
8-K, earnings releases) — 50 metrics-generated, 50 domain-relevant, 50 novel-generated.
The published numbers use three corpus setups, and each is a split here:

- ``open-single`` — Ragie's *single store*: a question searches only its own filing
  (isolation unit = the filing). Ragie: 51% (top_k=32, no rerank, GPT-4o/4-Turbo).
- ``open`` — one shared store holding the 84 filings the questions use.
- ``open-all`` — one shared store holding all 368 filings of the repository. Ragie's
  *shared store* (27% at top_k=8; the FinanceBench paper's shared vector store: 19%) and
  PageIndex's "all documents stored in a single database" (Mafin 2.5: 98.7%).

Grading is PageIndex's own judge, verbatim from VectifyAI/Mafin2.5-FinanceBench
``eval.py`` (numbers within rounding, fraction/percentage equivalence, supersets correct),
on gpt-4o-2024-11-20 as they ran it. They additionally took the OR of three judges and
human re-annotation of ambiguous items; ``scripts/financebench_rejudge.py`` reproduces the
three-judge OR on a finished run. Human re-annotation is not reproduced.

PDFs are converted once to markdown (tables kept) with a ``<!-- page N -->`` marker per
page, cached in ``FINANCEBENCH_MD`` (see ``scripts/financebench_convert.py``).
"""

from __future__ import annotations

import functools
import os
import re
from pathlib import Path
from typing import Literal

from .base import Dataset
from ..models import Document, Query

_MD = Path(os.environ.get("FINANCEBENCH_MD", Path.home() / ".cache/hindsight/kbbench-data/fb-md"))

_JUDGE = """
    You are an expert evaluator for AI-generated responses to queries. Your task is to determine whether the AI-generated answer correctly answers the query based on the golden answer provided by a human expert.

    Numerical Accuracy:
    - Rounding differences should be **ignored** if they do not meaningfully change the conclusion.
    - You can allow some flexibility in accuracy. For example, 1.2 is considered similar to 1.23. Two numbers are considered similar if one can be rounded to the other.
    - Fractions, percentage, and numerics could be considered similar, for example: "11 of 14" is considered equivalent to "79%" and "0.79".

    Evaluation Criteria:
    - If the golden answer or any of its equivalence can be inferred or generated from the AI-generated answer, then the AI-generated answer is considered correct.
    - If any number, percentage, fraction, or figure in the golden answer is not present in the AI-generated answer, but can be inferred or generated from the AI-generated answer or implicitly exist in the AI-generated answer, then the AI-generated answer is considered correct.
    - The AI-generated answer is considered correct if it conveys the same or similar meaning, conclusion, or rationale as the golden answer.
    - If the AI-generated answer is a superset of the golden answer, it is also considered correct.
    - If the AI-generated answer provides a valid answer or reasonable interpretation compared to the golden answer, it is considered correct.
    - If the AI-generated answer contains subjective judgments or opinions, it is considered correct as long as they are reasonable and justifiable compared to the golden answer.

    - Otherwise, the AI-generated answer is incorrect.

    Inputs:
    - Query: {query}
    - AI-Generated Answer: {answer}
    - Golden Answer: {gold}

    Set "correct" to true or false by the criteria above, and give a one-line reason.
    """

_RAG_PROMPT = """\
You are a financial analyst answering a question about company filings, using ONLY the excerpts below.

Excerpts:
{context}

Question: {query}

Work through the relevant figures step by step (show any calculation), then give the final answer \
clearly. If the excerpts do not contain what is needed, say so.\
"""

_DOC_TYPES = {"10k": "10-K", "10q": "10-Q", "8k": "8-K", "earnings": "Earnings release"}


@functools.lru_cache(maxsize=1)
def _questions() -> tuple[dict, ...]:
    from datasets import load_dataset

    return tuple(load_dataset("PatronusAI/financebench", split="train"))


@functools.lru_cache(maxsize=1)
def _company_names() -> dict[str, str]:
    """The dataset's own company names ("Activision Blizzard"), for the filings it asks about."""
    return {r["doc_name"]: r["company"] for r in _questions()}


def _doc_meta(doc_name: str) -> dict[str, str]:
    """``3M_2018_10K`` -> company, period, doc type. Filenames are COMPANY_PERIOD_TYPE;
    the company comes from the dataset where it names one, since filenames drop spaces."""
    meta = _doc_meta_from_name(doc_name)
    meta["company"] = _company_names().get(doc_name, meta["company"])
    return meta


def _doc_meta_from_name(doc_name: str) -> dict[str, str]:
    m = re.match(r"^(?P<company>.+?)_(?P<period>\d{4}(?:Q\d)?)_(?P<type>[A-Za-z0-9]+?)(?:_dated.*)?$", doc_name)
    if not m:
        return {"company": doc_name, "period": "", "doc_type": ""}
    return {
        "company": m["company"].replace("_", " "),
        "period": m["period"],
        "doc_type": _DOC_TYPES.get(m["type"].lower(), m["type"]),
    }


def _title(doc_name: str) -> str:
    meta = _doc_meta(doc_name)
    return f"{meta['company']} {meta['doc_type']} {meta['period']}".strip()


@functools.lru_cache(maxsize=None)
def _text(doc_name: str) -> str:
    path = _MD / f"{doc_name}.md"
    if not path.exists():
        raise FileNotFoundError(f"{path} — run scripts/financebench_convert.py first")
    return path.read_text()


class FinanceBenchDataset(Dataset):
    name = "financebench"
    description = (
        "FinanceBench open set: 150 analyst questions over SEC filings, graded against gold "
        "answers with PageIndex's published judge prompt."
    )
    splits = ["open", "open-single", "open-all"]
    task_type: Literal["open"] = "open"
    isolation_unit = None  # scoping is per question (user_id), the corpus is one bank
    published = False
    links = [
        {"label": "Paper", "url": "https://arxiv.org/abs/2311.11944"},
        {"label": "Ragie", "url": "https://www.ragie.ai/blog/ragie-outperformed-financebench"},
        {"label": "PageIndex eval", "url": "https://github.com/VectifyAI/Mafin2.5-FinanceBench"},
    ]

    def categories(self, split: str) -> list[str] | None:
        return ["metrics-generated", "domain-relevant", "novel-generated"]

    def category_type(self, split: str, category: str) -> Literal["doc", "query"]:
        return "query"

    def get_isolation_id(self, doc: Document) -> str | None:
        return doc.user_id

    def load_queries(self, split: str, category: str | None = None, limit: int | None = None) -> list[Query]:
        rows = [r for r in _questions() if category in (None, r["question_type"])][: limit or None]
        return [
            Query(
                id=r["financebench_id"],
                query=r["question"],
                gold_ids=[r["doc_name"]],
                gold_answers=[r["answer"]],
                # Only the single-store split scopes a question to its filing.
                user_id=r["doc_name"] if split == "open-single" else None,
                meta={
                    "category": r["question_type"],
                    "reasoning": r["question_reasoning"],
                    "doc_name": r["doc_name"],
                    "justification": r["justification"],
                },
            )
            for r in rows
        ]

    def load_documents(
        self,
        split: str,
        category: str | None = None,
        limit: int | None = None,
        ids: set[str] | None = None,
        user_ids: set[str] | None = None,
    ) -> list[Document]:
        if split == "open-all":
            names = sorted(p.stem for p in _MD.glob("*.md"))
        else:
            names = sorted({r["doc_name"] for r in _questions()})
        if ids is not None:
            names = [n for n in names if n in ids]
        if user_ids is not None:
            names = [n for n in names if n in user_ids]
        docs = []
        for name in names[: limit or None]:
            meta = _doc_meta(name)
            docs.append(
                Document(
                    id=name,
                    content=_text(name),
                    context=_title(name),
                    user_id=name if split == "open-single" else None,
                    # Tags a provider may turn into filterable fields: the filing's company,
                    # period and type are facts of the file, not something to extract.
                    tags=[f"company:{meta['company']}", f"period:{meta['period']}", f"doc_type:{meta['doc_type']}"],
                )
            )
        return docs

    def build_rag_prompt(self, query, context, task_type, split, category=None, meta=None) -> str:
        return _RAG_PROMPT.format(context=context, query=query)

    def build_judge_prompt(self, query: str, gold_answers: list[str], answer: str) -> str:
        return _JUDGE.format(query=query, answer=answer, gold=gold_answers[0] if gold_answers else "")

    def default_judge_llm(self):
        from ..llm.openai import OpenAILLM

        return OpenAILLM(os.environ.get("FINANCEBENCH_JUDGE_MODEL", "gpt-4o-2024-11-20"))
