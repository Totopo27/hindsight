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
import re
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
    "supports, and return nothing at all rather than inventing a record the document does "
    "not describe.\n\n"
    "Two ways that goes wrong, both of which produce rows nobody wants:\n"
    "- A phrase that mentions the thing is not the thing. 'the Acme logistics agreement' "
    "names the agreement, and the party is 'Acme' — do not turn the phrase into a record "
    "of its own. Use the thing's own name, as the document writes it when naming it.\n"
    "- Only things the collection is about. A document usually names others — the reader, "
    "the author, the other party to an agreement — and they belong in this collection only "
    "if the collection's definition says they do.\n\n"
    "A field that points at another collection's record takes that record's name, exactly "
    "as the list of known records writes it, when the thing is one of them. Say it in the "
    "document's own words only when it is not.\n\n"
    "Every field you fill needs an evidence entry: the field's name and the sentence you "
    "read it from, quoted from the document. A value nobody can trace is a claim, not data."
)


class _FieldEvidence(BaseModel):
    """One field of a record, and the sentence the model read it from.

    A list of these rather than a ``{field: quote}`` dict: a free-form dict becomes
    ``additionalProperties`` in the JSON schema, and the Gemini Developer API refuses any
    schema containing it ("additionalProperties is only supported in Gemini Enterprise
    Agent Platform mode"), which failed every derivation on that provider. A closed shape
    every provider accepts costs one comprehension to fold back.
    """

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


@dataclass(frozen=True)
class CollectionSpec:
    """One collection as a slice of a document is asked about it."""

    collection_id: str
    name: str
    description: str | None
    fields: dict[str, dict[str, Any]]
    #: Records that already exist in the collections this one's relationships point at,
    #: as ``field: record`` lines. Offered so the model names an existing record rather
    #: than a spelling that resolves to nothing.
    candidates: list[str]


async def derive_records(
    llm: Any,
    specs: list[CollectionSpec],
    *,
    doc_id: str,
    title: str | None,
    text: str,
) -> dict[str, list[dict[str, Any]]]:
    """Every record one slice of a document describes, for several collections at once.

    Returns ``{collection_id: [{"values": {...}, "evidence": {field: quote}}]}``. The
    quote is what makes a derived number auditable: a total nobody can trace is a claim.

    One call covers up to the whole set rather than one call per collection. A slice
    already has to be read to answer "which vendors are here"; asking "and which
    contracts" in the same breath costs a few hundred tokens of schema instead of a
    second pass over the same text. On two collections that halves the calls; on ten it
    is a tenth of them.

    The response is one field per collection rather than a list of tagged records,
    because a tagged list needs a union of per-collection value models and a union makes
    a JSON schema that providers reject or mangle. One closed field each is flat.
    """
    per_collection: dict[str, Any] = {}
    for spec in specs:
        value_model = extraction_model(spec.fields, name=f"{_model_name(spec.collection_id)}Values")
        if value_model is None:
            continue
        record_model = create_model(
            f"{_model_name(spec.collection_id)}Record",
            # Values first, evidence second, and evidence required. Declared optional it
            # was omitted on every record by gemini-2.5-flash and nothing derived could
            # be traced. Declared *before* the values it was worse than either: the model
            # wrote commentary into the fields it was quoting for ("Ireland (Ireland is a
            # country - so just Ireland here.)") and left most values null, because it
            # was asked to quote for fields it had not decided on yet. Extract, then cite.
            values=(value_model, Field(description="The record's fields")),
            evidence=(
                list[_FieldEvidence],
                Field(description="For each field you filled in values, the sentence you read it from. Required."),
            ),
        )
        per_collection[spec.collection_id] = (
            list[record_model],  # type: ignore[valid-type]
            Field(default_factory=list, description=f"Records of {spec.name} this text describes"),
        )
    if not per_collection:
        return {}
    batch_model = create_model("Extraction", **per_collection)

    described = []
    for spec in specs:
        line = f"- {spec.collection_id} ({spec.name})"
        if spec.description:
            line += f": {spec.description}"
        described.append(line)
        described.extend(f"    may point at: {candidate}" for candidate in spec.candidates)

    messages = [
        {"role": "system", "content": _RECORDS_SYSTEM},
        {
            "role": "user",
            "content": (
                "Collections to fill:\n"
                + "\n".join(described)
                + f"\n\nDocument{f' titled {title}' if title else ''}:\n{text}"
            ),
        },
    ]
    # Deliberately not caught here. A slice that fails is a slice of the document that
    # was never read, and swallowing it returns {} — which the caller cannot tell from
    # "this slice describes nothing". That silence lost a whole derivation once: two runs
    # collided, every call failed, and the operation reported success with 0 records
    # written over a table that had just been emptied. The caller counts the failures and
    # decides; see run_derive_records.
    result = await llm.call(messages=messages, response_format=batch_model, scope="knowledge_records")
    content = result.content

    out: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        raw = (
            getattr(content, spec.collection_id, None)
            if isinstance(content, BaseModel)
            else (content or {}).get(spec.collection_id)
        )
        records = []
        for record in raw or []:
            if isinstance(record, BaseModel):
                values, evidence = jsonable(record.values.model_dump()), _evidence_map(record.evidence)
            elif isinstance(record, dict):
                values, evidence = jsonable(record.get("values") or {}), _evidence_map(record.get("evidence"))
            else:
                continue
            if values:
                records.append({"values": values, "evidence": {k: v for k, v in evidence.items() if k in values}})
        if records:
            out[spec.collection_id] = records
    return out


def _model_name(collection_id: str) -> str:
    """A collection id as a python identifier, for the model class it builds."""
    cleaned = "".join(part.capitalize() for part in re.split(r"[^0-9a-zA-Z]+", collection_id) if part)
    return cleaned or "Collection"


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


_CHAT_SYSTEM = (
    "You help someone design the collections of a knowledge bank, by talking with them.\n"
    "A collection is a kind of thing the documents talk about — a vendor, a contract, an "
    "incident — and each record folds together what every document said about one of them. "
    "A record's fields are read out of the documents by a model, one call per document.\n"
    "\n"
    "You have one tool, propose_collection_changes. You cannot change anything yourself: the "
    "tool records a proposal that the person reviews and applies. So:\n"
    "- Answer in plain text when they ask a question, or when you need to know more.\n"
    "- Call the tool when you have a concrete change to suggest, and say in your reply what it "
    "does and why.\n"
    "- Never claim to have changed something. You proposed it; they decide.\n"
    "\n"
    "Design rules:\n"
    "- Propose only what the documents support. A field no document answers sits empty forever.\n"
    "- Every collection needs an identity field: the one that says WHICH thing a record is, so "
    "the same thing named in two documents becomes one record.\n"
    "- Use a relationship (a field's `collection`) when the value IS another collection's "
    "record, rather than repeating that thing's attributes.\n"
    "- An update replaces the collection's field list, so give every field it should end with, "
    "not only the new ones.\n"
    "- Prefer few collections that earn their place over a schema of everything.\n"
    "- Every change carries a reason: one sentence, grounded in the documents, saying what it "
    "buys. A change you cannot explain is one you should not propose."
)

#: The one tool the chat has. Its arguments are the change, flat and closed: a free-form
#: map becomes ``additionalProperties``, which the Gemini Developer API refuses outright.
COLLECTION_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "propose_collection_changes",
        "description": (
            "Propose changes to this bank's collections for the person to review. This does not "
            "apply anything — they approve it in the UI."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "changes": {
                    "type": "array",
                    "description": "One entry per collection to create, update or delete.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": (
                                    "Why this change, in one sentence, from the documents. Required: a "
                                    "change nobody can explain is a change nobody should approve."
                                ),
                            },
                            "action": {"type": "string", "enum": ["create", "update", "delete"]},
                            "collection_id": {"type": "string", "description": "snake_case, plural, e.g. vendors"},
                            "name": {"type": "string"},
                            "description": {"type": "string", "description": "What one record of it is"},
                            "identity": {
                                "type": "string",
                                "description": "The field that says which thing a record is",
                            },
                            "fields": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "type": {
                                            "type": "string",
                                            "enum": [
                                                "string",
                                                "integer",
                                                "number",
                                                "boolean",
                                                "date",
                                                "datetime",
                                                "array",
                                            ],
                                        },
                                        "description": {"type": "string"},
                                        "values": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "The allowed values, if it is a classification",
                                        },
                                        "collection": {
                                            "type": "string",
                                            "description": "Another collection's id when this field points at one",
                                        },
                                    },
                                    "required": ["name"],
                                },
                            },
                        },
                        # reason is listed first here and in "properties" on purpose: a model
                        # that fills fields in schema order writes the justification before the
                        # change, and gemini-2.5-flash-lite omitted a trailing "reason" every
                        # time even with it marked required.
                        "required": ["reason", "action", "collection_id"],
                    },
                }
            },
            "required": ["changes"],
        },
    },
}


@dataclass(frozen=True)
class CollectionChat:
    """What one turn came to: what to say, and what to propose."""

    reply: str
    proposals: list[ProposedCollection]


def collections_context(existing: list[dict[str, Any]], documents: list[dict[str, str]]) -> str:
    """The bank as the agent sees it: what it defines now, and what its documents say."""
    lines = ["Collections this bank has now:"]
    if existing:
        for collection in existing:
            fields = ", ".join(
                f"{name} ({spec.get('collection') or spec.get('type', 'string')})"
                for name, spec in (collection.get("fields") or {}).items()
            )
            lines.append(
                f"- {collection['collection_id']}: identity={collection.get('identity') or 'none'}; "
                f"fields: {fields or 'none'}"
            )
    else:
        lines.append("- none yet")
    if documents:
        lines.append("\nA sample of its documents:")
        for document in documents:
            heading = document.get("title") or document["doc_id"]
            lines.append(f"\n### {heading}\n{document['text']}")
    return "\n".join(lines)


async def chat_about_collections(
    llm: Any,
    *,
    messages: list[dict[str, str]],
    context: str,
) -> CollectionChat:
    """One turn of the collection chat: an answer, a proposal, or both.

    The agent is given the bank as it is now on every turn, so after a proposal is
    applied the next turn sees the result rather than its own memory of it.
    """
    conversation = [
        {"role": "system", "content": _CHAT_SYSTEM},
        {"role": "system", "content": context},
        *messages,
    ]
    result = await llm.call_with_tools(
        messages=conversation, tools=[COLLECTION_TOOL], scope="knowledge_collection_chat"
    )
    proposals = _parse_proposals(result)
    # Small models drop "reason" perhaps half the time even with it required and listed
    # first (gemini-2.5-flash-lite does). A reasonless change is discarded downstream, so
    # asking once more is cheaper than handing the person a dropped proposal.
    if any(not proposal.reason.strip() for proposal in proposals):
        retry = await llm.call_with_tools(
            messages=[
                *conversation,
                {
                    "role": "user",
                    "content": (
                        "Call propose_collection_changes again with the same changes, and give "
                        "every change a 'reason': one sentence on why, from the documents."
                    ),
                },
            ],
            tools=[COLLECTION_TOOL],
            scope="knowledge_collection_chat",
        )
        # Take only the reasons from the second answer: the first one is the change the
        # person asked about, and a retry is free to reword the rest of it.
        reasons = {p.collection_id: p.reason.strip() for p in _parse_proposals(retry) if p.reason.strip()}
        proposals = [
            p if p.reason.strip() else p.model_copy(update={"reason": reasons.get(p.collection_id, "")})
            for p in proposals
        ]
    return CollectionChat(reply=(result.content or "").strip(), proposals=proposals)


def _parse_proposals(result: Any) -> list[ProposedCollection]:
    """The tool calls of one turn, skipping any change too malformed to be a proposal."""
    logger.debug(
        "knowledge collection chat: finish=%s tools=%s",
        result.finish_reason,
        [call.name for call in result.tool_calls],
    )
    proposals: list[ProposedCollection] = []
    for call in result.tool_calls:
        if call.name != "propose_collection_changes":
            continue
        for change in call.arguments.get("changes") or []:
            try:
                proposals.append(ProposedCollection.model_validate(change))
            except Exception as e:  # noqa: BLE001 - one bad change must not lose the others
                logger.warning("knowledge collection chat: unusable change %s: %s", change, e)
    return proposals


_LINK_SYSTEM = (
    "You match records to the records they point at. You are given a list of records that "
    "have no value for one field, the closed list of records that field may point at, and "
    "the text the records were read from. Answer for every record you can, and leave out "
    "the ones the text does not place — a wrong link is worse than an empty one: an empty "
    "one is a gap somebody can see, a wrong one is a join that returns a confident wrong "
    "row. Read the whole list before answering: the text usually separates them as a set "
    "(these are one household's servants, those are the prince's kinsmen) rather than "
    "stating each one on its own."
)


class _Link(BaseModel):
    """One record, the record it points at, and the sentence that says so."""

    record: str = Field(description="The record being linked, exactly as it was listed")
    points_at: str = Field(description="One of the candidates, exactly as it was listed")
    quote: str = Field(default="", description="The sentence that says so, quoted from the text")


async def resolve_links(
    llm: Any,
    *,
    collection_name: str,
    field_name: str,
    field_description: str | None,
    target_name: str,
    records: list[str],
    candidates: list[str],
    evidence: str,
) -> list[_Link]:
    """Which record each unlinked record points at, decided for the whole set at once.

    Derivation reads one slice at a time, so a relationship stated elsewhere — a cast
    list, a contract's preamble, a header — is not in front of the model when the record
    is written: on thirty pages of a play, five characters the cast list places in a
    house came back with none, because that list and their scenes are different slices.

    One call for the whole set rather than one per gap. It is cheaper, and it is also
    better: the model sees the records together, and a text that never says "Tybalt is a
    Capulet" in so many words still says it by listing him among them.
    """
    if not records or not candidates:
        return []
    batch_model = create_model(
        "Links",
        links=(list[_Link], Field(default_factory=list, description="One entry per record you can place")),
    )
    messages = [
        {"role": "system", "content": _LINK_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Collection: {collection_name}\n"
                f"Field: {field_name}"
                + (f" — {field_description}" if field_description else "")
                + f" → a record of {target_name!r}\n\n"
                f"Records with no {field_name}:\n" + "\n".join(f"  {record}" for record in records) + "\n\n"
                "It may point at exactly one of:\n" + "\n".join(f"  {c}" for c in candidates) + "\n\n"
                f"What the documents say:\n{evidence}"
            ),
        },
    ]
    try:
        result = await llm.call(messages=messages, response_format=batch_model, scope="knowledge_links")
        content = result.content
    except Exception as e:
        logger.warning("knowledge link pass failed for %s.%s: %s", collection_name, field_name, e)
        return []
    raw = content.links if isinstance(content, BaseModel) else (content or {}).get("links") or []
    out: list[_Link] = []
    for item in raw:
        if isinstance(item, _Link):
            out.append(item)
        elif isinstance(item, dict) and item.get("record") and item.get("points_at"):
            out.append(_Link.model_validate(item))
    return out
