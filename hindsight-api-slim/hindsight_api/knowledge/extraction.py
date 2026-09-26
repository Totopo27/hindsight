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
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

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


_CLASSIFY_SYSTEM = (
    "You decide which schema a document belongs to. Answer with one of the offered schema "
    "ids, or 'none' when the document is not any of them. Never invent an id."
)


async def classify_schema(
    llm: Any,
    schemas: list[dict[str, Any]],
    *,
    doc_id: str,
    title: str | None,
    text: str,
    char_limit: int,
) -> str | None:
    """Which of these schemas describes this document, if any.

    The model is shown each schema's id, name, description and field names — the same
    things a person would read to decide — and answers with an id from a fixed set, so a
    hallucinated name cannot get through. 'none' is a real answer: a bank that holds
    contracts and invoices should not force a memo into one of them.
    """
    if not schemas:
        return None
    choices = [schema["schema_id"] for schema in schemas]
    catalogue = "\n".join(
        f"- {schema['schema_id']}: {schema.get('name') or schema['schema_id']}"
        + (f" — {schema['description']}" if schema.get("description") else "")
        + (
            f" (fields: {', '.join(sorted(schema.get('document_fields') or {}))})"
            if schema.get("document_fields")
            else ""
        )
        for schema in schemas
    )
    model = create_model(
        "SchemaChoice",
        schema_id=(Literal[tuple([*choices, "none"])], Field(description="The schema this document is")),  # type: ignore[valid-type]
    )
    messages = [
        {"role": "system", "content": _CLASSIFY_SYSTEM},
        {
            "role": "user",
            "content": f"Schemas:\n{catalogue}\n\nDocument{f' titled {title}' if title else ''}:\n{text[:char_limit]}",
        },
    ]
    try:
        result = await llm.call(messages=messages, response_format=model, scope="knowledge_classify")
        content = result.content
        chosen = content.schema_id if isinstance(content, BaseModel) else (content or {}).get("schema_id")
    except Exception as e:
        # A document that could not be classified is still worth storing; it simply has
        # no fields until someone says which schema it is.
        logger.warning("knowledge schema classification failed for %s: %s", doc_id, e)
        return None
    return None if chosen in (None, "none") else str(chosen)


_RECORDS_SYSTEM = (
    "You pull structured records out of a document. One record is one real-world thing — "
    "a vendor, a contract, a person — not one per document: a document may describe "
    "several, or add detail to one you have seen before. Fill only the fields the text "
    "supports, quote the sentence each value came from, and return nothing at all rather "
    "than inventing a record the document does not describe."
)


async def derive_records(
    llm: Any,
    fields: dict[str, dict[str, Any]],
    *,
    collection_name: str,
    doc_id: str,
    title: str | None,
    text: str,
    char_limit: int,
) -> list[dict[str, Any]]:
    """The records this document contributes to one collection, with their evidence.

    Each returned item is ``{"values": {...}, "evidence": {field: quote}}``. The quote is
    what makes a derived number auditable: a total nobody can trace is a claim, not data.
    """
    if not fields:
        return []
    value_model = extraction_model(fields, name="RecordValues")
    if value_model is None:
        return []
    record_model = create_model(
        "Record",
        values=(value_model, Field(description="The record's fields")),
        evidence=(dict[str, str], Field(default_factory=dict, description="field name -> the sentence it came from")),
    )
    batch_model = create_model(
        "Records",
        records=(
            list[record_model],
            Field(default_factory=list, description=f"Every {collection_name} this text describes"),
        ),  # type: ignore[valid-type]
    )
    messages = [
        {"role": "system", "content": _RECORDS_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Collection: {collection_name}\n\nDocument{f' titled {title}' if title else ''}:\n{text[:char_limit]}"
            ),
        },
    ]
    try:
        result = await llm.call(messages=messages, response_format=batch_model, scope="knowledge_records")
        content = result.content
        records = content.records if isinstance(content, BaseModel) else (content or {}).get("records") or []
    except Exception as e:
        logger.warning("knowledge record derivation failed for %s: %s", doc_id, e)
        return []

    out: list[dict[str, Any]] = []
    for record in records:
        if isinstance(record, BaseModel):
            values, evidence = jsonable(record.values.model_dump()), dict(record.evidence or {})
        elif isinstance(record, dict):
            values, evidence = jsonable(record.get("values") or {}), dict(record.get("evidence") or {})
        else:
            continue
        if values:
            out.append({"values": values, "evidence": {k: v for k, v in evidence.items() if k in values}})
    return out
