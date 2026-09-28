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


class _FieldEvidence(BaseModel):
    """One field of a record, and the sentence the model read it from."""

    field: str
    quote: str


def _evidence_map(evidence: Any) -> dict[str, str]:
    """The evidence list as {field: quote}, tolerating a model that returned a dict."""
    if isinstance(evidence, dict):
        return {str(k): str(v) for k, v in evidence.items()}
    out: dict[str, str] = {}
    for item in evidence or []:
        if isinstance(item, BaseModel):
            out[item.field] = item.quote
        elif isinstance(item, dict) and item.get("field"):
            out[str(item["field"])] = str(item.get("quote") or "")
    return out


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
        # A list of pairs rather than a dict: a free-form dict becomes
        # `additionalProperties` in the JSON schema, and the Gemini Developer API refuses
        # any schema containing it ("additionalProperties is only supported in Gemini
        # Enterprise Agent Platform mode"), which failed every derivation on that provider.
        # A closed shape every provider accepts costs one comprehension to fold back.
        evidence=(
            list[_FieldEvidence],
            Field(default_factory=list, description="For each field filled, the sentence it came from"),
        ),
    )
    # list[record_model] is a type built at runtime, which the checker cannot follow: the
    # element type only exists once the collection's fields are known.
    record_list: Any = list[record_model]  # type: ignore[valid-type]
    batch_model = create_model(
        "Records",
        records=(record_list, Field(default_factory=list, description=f"Every {collection_name} this text describes")),
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
            values, evidence = jsonable(record.values.model_dump()), _evidence_map(record.evidence)
        elif isinstance(record, dict):
            values, evidence = jsonable(record.get("values") or {}), _evidence_map(record.get("evidence"))
        else:
            continue
        if values:
            out.append({"values": values, "evidence": {k: v for k, v in evidence.items() if k in values}})
    return out


# ---- proposing collections


class ProposedField(BaseModel):
    """One field of a proposed collection.

    Flat and closed on purpose: a free-form dict becomes ``additionalProperties`` in the
    JSON schema, which the Gemini Developer API refuses outright. The API's nested shape
    is built from this after the call, not asked of the model.
    """

    name: str = Field(description="snake_case field name")
    type: Literal["string", "integer", "number", "boolean", "date", "datetime", "array"] = "string"
    description: str = Field(default="", description="What this field is, in the documents' words")
    values: list[str] = Field(default_factory=list, description="The allowed values, if it is a classification")
    collection: str = Field(
        default="",
        description="Another collection's id when this field points at one of its records; otherwise empty",
    )


class ProposedCollection(BaseModel):
    """One change a proposal suggests. Nothing here is applied by the model."""

    action: Literal["create", "update", "delete"] = "create"
    collection_id: str = Field(description="snake_case id, plural, e.g. vendors")
    name: str = ""
    description: str = Field(default="", description="What one record of this collection is")
    identity: str = Field(
        default="",
        description="The field that says WHICH thing a record is, so the same thing found twice is one record",
    )
    fields: list[ProposedField] = Field(default_factory=list)
    reason: str = Field(default="", description="Why this collection, in one sentence, from the documents")


class CollectionProposal(BaseModel):
    collections: list[ProposedCollection] = Field(default_factory=list)


_PROPOSE_SYSTEM = (
    "You design collections for a knowledge bank. A collection is a kind of thing the documents "
    "talk about — a vendor, a contract, an incident — and each record folds together what every "
    "document said about one of them.\n"
    "Rules:\n"
    "- Propose only what the documents shown actually support. A field no document answers is a "
    "field that will sit empty forever.\n"
    "- Every collection needs an identity field: the one that says WHICH thing a record is, so the "
    "same thing named in two documents becomes one record.\n"
    "- Use a relationship (the `collection` key on a field) when a field's value IS another "
    "collection's record, rather than repeating that thing's attributes.\n"
    "- Keep what already exists unless it is wrong: 'update' an existing collection only to add or "
    "fix fields, and give the full field list you want it to end with.\n"
    "- Prefer few collections that earn their place over a schema of everything."
)


async def propose_collections(
    llm: Any,
    *,
    instruction: str | None,
    existing: list[dict[str, Any]],
    documents: list[dict[str, str]],
) -> list[ProposedCollection]:
    """Ask the LLM what collections this corpus deserves. It proposes; it never writes.

    The existing collections are sent too, so the answer is a change to what is there
    rather than a design that ignores it.
    """
    lines = ["Collections that already exist:"]
    if existing:
        for collection in existing:
            fields = ", ".join(
                f"{name} ({spec.get('collection') or spec.get('type', 'string')})"
                for name, spec in (collection.get("fields") or {}).items()
            )
            lines.append(
                f"- {collection['collection_id']}: identity={collection.get('identity') or 'none'}; fields: {fields or 'none'}"
            )
    else:
        lines.append("- none yet")

    lines.append("\nDocuments from this bank:")
    for document in documents:
        heading = document.get("title") or document["doc_id"]
        lines.append(f"\n### {heading}\n{document['text']}")

    if instruction:
        lines.append(f"\nWhat the user asked for:\n{instruction}")

    messages = [
        {"role": "system", "content": _PROPOSE_SYSTEM},
        {"role": "user", "content": "\n".join(lines)},
    ]
    try:
        result = await llm.call(
            messages=messages, response_format=CollectionProposal, scope="knowledge_collection_proposal"
        )
    except Exception as e:
        logger.warning("knowledge collection proposal failed: %s", e)
        raise

    content = result.content
    if isinstance(content, CollectionProposal):
        return list(content.collections)
    if isinstance(content, dict):
        return list(CollectionProposal.model_validate(content).collections)
    return []
