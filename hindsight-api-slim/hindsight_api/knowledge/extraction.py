"""Filling a bank's metadata schema with the LLM.

Two calls per document at most: one that reads the document and fills the document-level
properties, and one per passage for the passage-level ones. The LLM is the bank's own — the
same resolved config (provider, model, key) retain uses — so a knowledge bank inherits
whatever the deployment or the bank configured, and a bank can run extraction on a cheaper
model than the rest of the server.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from .fields import extraction_model, jsonable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PassageValues:
    """One passage's extracted values, carrying the index they belong to."""

    index: int
    values: dict[str, Any]


_SYSTEM = (
    "You extract structured metadata from documents. Fill only the properties the text "
    "actually supports: leave a property out rather than guessing, and never invent a value "
    "to fill the shape. Where a property lists allowed values, use one of them exactly."
)


def _prompt(*, title: str | None, text: str, scope: str, limit: int) -> list[dict[str, str]]:
    body = text[:limit]
    heading = f"Title: {title}\n" if title else ""
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"Extract the metadata of this {scope}.\n\n{heading}{scope.capitalize()}:\n{body}"},
    ]


async def _call(llm: Any, model: type[BaseModel], messages: list[dict[str, str]]) -> dict[str, Any]:
    result = await llm.call(messages=messages, response_format=model, scope="knowledge_metadata")
    content = result.content
    if isinstance(content, BaseModel):
        return jsonable(content.model_dump())
    if isinstance(content, dict):
        # A provider that returned raw JSON: validate it through the model so a bad value
        # is dropped here rather than stored and filtered on later.
        return jsonable(model.model_validate(content).model_dump())
    return {}


async def extract_document(
    llm: Any,
    schema: dict[str, dict[str, Any]],
    *,
    doc_id: str,
    title: str | None,
    text: str,
    char_limit: int,
) -> dict[str, Any]:
    """Document-level properties for one document. Returns {} when nothing applies."""
    model = extraction_model(schema, name="DocumentMetadata")
    if model is None or not text.strip():
        return {}
    try:
        return await _call(llm, model, _prompt(title=title, text=text, scope="document", limit=char_limit))
    except Exception as e:
        # A document whose metadata could not be extracted is still a document worth
        # storing and searching; the write must not fail with it.
        logger.warning("knowledge metadata extraction failed for %s: %s", doc_id, e)
        return {}


async def extract_passages(
    llm: Any,
    schema: dict[str, dict[str, Any]],
    *,
    doc_id: str,
    title: str | None,
    passages: list[tuple[int, str]],
    char_limit: int,
    concurrency: int,
) -> dict[int, dict[str, Any]]:
    """Passage-level fields, one LLM call per passage, at most ``concurrency`` in flight."""
    model = extraction_model(schema, name="ChunkMetadata")
    if model is None or not passages:
        return {}
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def one(index: int, text: str) -> PassageValues:
        async with semaphore:
            try:
                values = await _call(llm, model, _prompt(title=title, text=text, scope="passage", limit=char_limit))
            except Exception as e:
                logger.warning("knowledge metadata extraction failed for %s#%s: %s", doc_id, index, e)
                values = {}
        return PassageValues(index=index, values=values)

    results = await asyncio.gather(*(one(index, text) for index, text in passages))
    return {result.index: result.values for result in results if result.values}
