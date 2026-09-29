"""Knowledge banks: write documents (always async, in batches) and search them.

Writes never happen in the request. A write submits one ``knowledge_write_batch``
operation — the same ``async_operations`` row, worker claim, retry and status API the
memory banks use — and returns its id. The worker passages, embeds and stores.

Search is two arms fused with RRF: vector over the passage embeddings and keyword over the
generated tsvector, then the configured cross-encoder reranks the fused candidates.
Everything (passage size, overlap, candidate count, rerank on/off) is per-bank config,
resolved the same way memory-bank settings are.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any

from ..config import get_config
from ..engine.db_utils import acquire_with_retry
from ..engine.llm_wrapper import sanitize_llm_output
from ..engine.retain.bank_utils import DEFAULT_DISPOSITION
from ..engine.schema import fq_table
from ..engine.storage import bank_storage_prefix
from . import collections as collections_store
from . import store
from .extraction import (
    chat_about_collections,
    classify_schema,
    collections_context,
    derive_records,
    extract_document,
    extract_passages,
)
from .fields import SchemaError, extract_only, filterable_names, validate_field_schema, validate_values
from .filters import FilterError
from .identity import normalise
from .passages import split_into_passages
from .query import CompiledQuery, Limits, compile_query, json_safe
from .records_query import compile_record_query
from .transfer import KnowledgeTransferScope

logger = logging.getLogger(__name__)

#: How many documents of one batch are embedded at a time. Each window costs two DB
#: round trips plus one embed call, so a small window spends most of the write in
#: overhead: at 8 the pipeline ran at a third of the embedding model's own throughput.
_EMBED_BATCH_DOCUMENTS = 64
MAX_BATCH_DOCUMENTS = 500


class KnowledgeBankError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class DocumentInput:
    doc_id: str
    text: str
    title: str | None = None
    metadata: dict[str, Any] | None = None
    #: Values for the schema's document fields, supplied instead of extracted. They land in
    #: the same object the LLM would have filled, and a field supplied here is not read by it.
    fields: dict[str, Any] | None = None
    #: The same for the schema's passage fields, by passage index.
    passage_fields: dict[int, dict[str, Any]] | None = None
    #: Which schema fills this document's fields. Omitted, a bank with exactly one schema
    #: uses it; a bank with several extracts nothing rather than guessing.
    schema_id: str | None = None


@dataclass(frozen=True)
class UploadedFile:
    """One file on its way into a knowledge bank, read into memory by the route."""

    doc_id: str
    filename: str
    content: bytes
    content_type: str
    parser: list[str]
    title: str | None = None
    metadata: dict[str, Any] | None = None
    schema_id: str | None = None


@dataclass(frozen=True)
class ExtractedFields:
    """What one document's fields came to: its own values, and its passages' by index."""

    document: dict[str, Any]
    passages: dict[int, dict[str, Any]]


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    passage_index: int
    text: str
    score: float
    ranks: dict[str, int]


def query_terms(query: str) -> list[str]:
    """Words worth searching for, from a natural-language question."""
    return [t for t in re.findall(r"\w+", query.lower()) if len(t) > 1][:32]


def passage_heading(title: str | None, section: str | None) -> str | None:
    """What a passage is indexed under: its document's title, then the section it sits in.

    Both arms read it — it is embedded with the passage and it is the ``heading`` column
    the keyword index is generated from.
    """
    return " — ".join(part for part in (title, section) if part) or None


def embedding_text(title: str | None, passage_text: str, indexed: dict[str, Any] | None = None) -> str:
    """What gets embedded for a passage: indexed field values, the title, then the passage.

    A field marked ``indexed`` is prepended as ``name: value`` so the passage can be found
    by something its own words never say — a region, a document type, a counterparty. The
    title is skipped when the passage already opens with it, because repeating it only
    dilutes the embedding.
    """
    parts = []
    if indexed:
        parts.append(", ".join(f"{name}: {_as_text(value)}" for name, value in sorted(indexed.items())))
    if title and not passage_text.lstrip().lower().startswith(title.strip().lower()):
        parts.append(title)
    parts.append(passage_text)
    return "\n\n".join(parts)


def _as_text(value: Any) -> str:
    return ", ".join(str(item) for item in value) if isinstance(value, list) else str(value)


def indexed_values(
    schema: dict[str, Any] | None, document_values: dict[str, Any], passage_values: dict[str, Any]
) -> dict[str, Any]:
    """The field values this schema marks ``indexed``, document-level and passage-level."""
    if not schema:
        return {}
    indexed: dict[str, Any] = {}
    for level, values in (("document_fields", document_values), ("passage_fields", passage_values)):
        for name, spec in (schema.get(level) or {}).items():
            if spec.get("indexed") and values.get(name) not in (None, [], {}):
                indexed[name] = values[name]
    return indexed


class KnowledgeService:
    """Knowledge-bank operations. Holds the engine for its pool, embeddings and reranker."""

    def __init__(self, memory: Any) -> None:
        self.memory = memory
        self._bm25: dict[str, bool] = {}

    async def _pool(self) -> Any:
        return await self.memory._get_pool()

    async def _has_bm25(self, conn: Any) -> bool:
        # ponytail: checked once per schema per process; an index created later needs a restart.
        key = fq_table("kb_passages")
        if key not in self._bm25:
            self._bm25[key] = await store.has_bm25_index(conn)
        return self._bm25[key]

    async def _config(self, bank_id: str, request_context: Any) -> Any:
        return await self.memory._config_resolver.resolve_full_config(bank_id, request_context)

    async def authorize(self, bank_id: str, *, write: bool, request_context: Any) -> None:
        """Let a tenant extension refuse this call before it touches the bank.

        A knowledge bank is a bank, so it goes through the same validator the memory
        banks use. One operation name covers the whole knowledge read surface and one
        the whole write surface, and the gate lives at the two entry points (the HTTP
        router's dependency and the MCP tools) rather than in each method, so a route
        added later is gated by construction rather than by remembering.

        One name per direction rather than one per endpoint: split the enum when an
        extension actually needs to allow a search while refusing a schema change.
        """
        validator = getattr(self.memory, "_operation_validator", None)
        if validator is None:
            return
        from ..extensions import BankReadContext, BankReadOperation, BankWriteContext, BankWriteOperation

        if write:
            ctx: Any = BankWriteContext(
                bank_id=bank_id,
                operation=BankWriteOperation.KNOWLEDGE_BANK_WRITE,
                request_context=request_context,
            )
            await self.memory._validate_operation(validator.validate_bank_write(ctx))
        else:
            ctx = BankReadContext(
                bank_id=bank_id,
                operation=BankReadOperation.KNOWLEDGE_BANK_READ,
                request_context=request_context,
            )
            await self.memory._validate_operation(validator.validate_bank_read(ctx))

    # ---- banks

    async def create_bank(self, bank_id: str, name: str | None) -> dict[str, Any]:
        async with acquire_with_retry(await self._pool()) as conn:
            kind = await store.bank_kind(conn, bank_id)
            if kind == store.KNOWLEDGE_KIND:
                raise KnowledgeBankError(409, f"knowledge bank {bank_id!r} already exists")
            if kind is not None:
                raise KnowledgeBankError(409, f"bank {bank_id!r} already exists as a {kind} bank")
            await store.create_bank(conn, bank_id, name or bank_id, json.dumps(DEFAULT_DISPOSITION), uuid.uuid4())
        return await self.get_bank(bank_id)

    async def list_banks(
        self, limit: int, offset: int, query: str | None, request_context: Any = None
    ) -> dict[str, Any]:
        """The knowledge banks this caller may see.

        A deployment's operation validator decides which banks a caller is allowed to
        know about, and a knowledge bank is a bank: the same hook that filters the memory
        bank list filters this one, or a tenant extension that hides a bank would hide it
        everywhere except here.
        """
        validator = getattr(self.memory, "_operation_validator", None)
        if validator is None:
            async with acquire_with_retry(await self._pool()) as conn:
                return await store.list_banks(conn, limit, offset, query)

        # The validator may drop any bank, so the page is cut after it runs — the same
        # trade the memory bank list makes, and paid only where a validator is installed.
        from ..extensions import BankListContext

        async with acquire_with_retry(await self._pool()) as conn:
            everything = await store.list_banks(conn, limit=10_000, offset=0, query=query)
        result = await validator.filter_bank_list(
            BankListContext(banks=everything["items"], request_context=request_context)
        )
        banks = result.banks
        return {
            "items": banks[offset : offset + limit],
            "total": len(banks),
            "limit": limit,
            "offset": offset,
        }

    async def get_bank(self, bank_id: str) -> dict[str, Any]:
        async with acquire_with_retry(await self._pool()) as conn:
            stats = await store.bank_stats(conn, bank_id)
        if stats is None:
            raise KnowledgeBankError(404, f"knowledge bank {bank_id!r} not found")
        return stats

    async def delete_bank(self, bank_id: str) -> dict[str, Any]:
        await self._require_bank(bank_id)
        async with acquire_with_retry(await self._pool()) as conn:
            # The bank row owns everything by FK cascade: documents, passages, operations.
            await conn.execute(f"DELETE FROM {store.fq_table('banks')} WHERE bank_id = $1", bank_id)
        return {"bank_id": bank_id, "deleted": True}

    async def _require_bank(self, bank_id: str) -> None:
        async with acquire_with_retry(await self._pool()) as conn:
            kind = await store.bank_kind(conn, bank_id)
        if kind is None:
            raise KnowledgeBankError(404, f"knowledge bank {bank_id!r} not found")
        if kind != store.KNOWLEDGE_KIND:
            raise KnowledgeBankError(409, f"bank {bank_id!r} is a {kind} bank")

    async def _check_filterable(self, bank_id: str, fields: dict[str, Any]) -> None:
        """A search may filter on a schema field only when the schema says it may.

        Names the schema does not define are left alone: they are the caller's own
        document metadata, which was filterable before any schema existed.
        """
        async with acquire_with_retry(await self._pool()) as conn:
            schemas = await store.list_schemas(conn, bank_id)
        if not schemas:
            return
        defined: set[str] = set()
        allowed: set[str] = set()
        for schema in schemas:
            for level in ("document_fields", "passage_fields"):
                defined |= set(schema.get(level) or {})
            allowed |= filterable_names(schema) or set()
        refused = sorted(name for name in fields if name in defined and name not in allowed)
        if refused:
            raise KnowledgeBankError(400, f"fields not filterable in this bank's schemas: {refused}")

    # ---- schemas

    DEFAULT_SCHEMA_ID = "default"

    async def list_schemas(self, bank_id: str) -> dict[str, Any]:
        await self._require_bank(bank_id)
        async with acquire_with_retry(await self._pool()) as conn:
            schemas = await store.list_schemas(conn, bank_id)
            for schema in schemas:
                schema.update(await store.schema_usage(conn, bank_id, schema["schema_id"]))
        return {"items": schemas, "total": len(schemas)}

    async def get_schema(self, bank_id: str, schema_id: str) -> dict[str, Any]:
        await self._require_bank(bank_id)
        async with acquire_with_retry(await self._pool()) as conn:
            schema = await store.get_schema(conn, bank_id, schema_id)
            if schema is None:
                raise KnowledgeBankError(404, f"schema {schema_id!r} not found")
            schema.update(await store.schema_usage(conn, bank_id, schema_id))
        return schema

    async def put_schema(
        self,
        bank_id: str,
        schema_id: str,
        *,
        name: str | None,
        description: str | None,
        document_fields: Any,
        passage_fields: Any,
    ) -> dict[str, Any]:
        """Define (or redefine) one schema: the fields for a document and for its passages."""
        await self._require_bank(bank_id)
        if not schema_id or not schema_id.replace("_", "").replace("-", "").isalnum():
            raise KnowledgeBankError(400, "schema id must be alphanumeric with _ or -")
        try:
            document_definition = validate_field_schema(document_fields, level="document")
            passage_definition = validate_field_schema(passage_fields, level="passage")
        except SchemaError as e:
            raise KnowledgeBankError(400, str(e)) from e
        async with acquire_with_retry(await self._pool()) as conn:
            await store.put_schema(
                conn,
                bank_id,
                schema_id,
                name=name,
                description=description,
                document_fields=document_definition,
                passage_fields=passage_definition,
            )
        return await self.get_schema(bank_id, schema_id)

    async def delete_schema(self, bank_id: str, schema_id: str) -> dict[str, Any]:
        await self._require_bank(bank_id)
        async with acquire_with_retry(await self._pool()) as conn:
            if not await store.delete_schema(conn, bank_id, schema_id):
                raise KnowledgeBankError(404, f"schema {schema_id!r} not found")
        # The fields already extracted stay: they are the documents' data, not the schema's.
        return {"schema_id": schema_id, "deleted": True}

    async def _schema_for_write(
        self,
        bank_id: str,
        schema_id: str | None,
        *,
        document: DocumentInput | None = None,
        config: Any = None,
    ) -> dict[str, Any] | None:
        """Which schema fills a document's fields.

        Named explicitly, or — when the bank has exactly one — that one, because a bank
        with a single schema should not make every write repeat its name. With several and
        no name, the LLM classifies the document; with classification off it gets no
        fields, which is better than the wrong ones.
        """
        async with acquire_with_retry(await self._pool()) as conn:
            if schema_id is not None:
                schema = await store.get_schema(conn, bank_id, schema_id)
                if schema is None:
                    raise KnowledgeBankError(400, f"schema {schema_id!r} not found in this bank")
                return schema
            schemas = await store.list_schemas(conn, bank_id)
        if len(schemas) == 1:
            return schemas[0]
        if not schemas or document is None or config is None or not config.kb_schema_classification:
            return None
        chosen = await classify_schema(
            self._field_extraction_llm(bank_id, config),
            schemas,
            doc_id=document.doc_id,
            title=document.title,
            text=document.text,
            char_limit=int(config.kb_field_extraction_max_chars),
        )
        return next((schema for schema in schemas if schema["schema_id"] == chosen), None)

    async def field_values(self, bank_id: str, field_name: str, level: str) -> dict[str, Any]:
        """How the corpus splits per value of one field."""
        await self._require_bank(bank_id)
        async with acquire_with_retry(await self._pool()) as conn:
            values = await store.value_counts(conn, bank_id, field_name, level=level)
        return {"field": field_name, "level": level, "values": values}

    async def submit_extract(
        self, bank_id: str, doc_ids: list[str] | None, only_missing: bool, schema_id: str | None
    ) -> dict[str, Any]:
        """Queue field extraction for documents already in the bank (after a schema change)."""
        await self._require_bank(bank_id)
        schema = await self._schema_for_write(bank_id, schema_id)
        if schema is None:
            raise KnowledgeBankError(400, "name the schema to extract with: this bank has none, or several")
        if not schema["document_fields"] and not schema["passage_fields"]:
            raise KnowledgeBankError(400, f"schema {schema['schema_id']!r} defines no fields")
        result = await self.memory._submit_async_operation(
            bank_id=bank_id,
            operation_type="knowledge_extract_fields",
            task_type="knowledge_extract_fields",
            task_payload={
                "doc_ids": doc_ids or [],
                "only_missing": only_missing,
                "schema_id": schema["schema_id"],
            },
        )
        return {"operation_id": result["operation_id"], "schema_id": schema["schema_id"], "status": "pending"}

    def _field_extraction_llm(self, bank_id: str, config: Any) -> Any:
        """The bank's LLM for extraction — the same resolved config retain would use."""
        return self.memory._llm_config.with_config(config, bank_id=bank_id, operation="knowledge_fields")

    async def _extract_for_document(
        self,
        *,
        bank_id: str,
        config: Any,
        schema: dict[str, Any],
        doc_id: str,
        title: str | None,
        text: str,
        passage_texts: list[tuple[int, str]],
    ) -> ExtractedFields:
        llm = self._field_extraction_llm(bank_id, config)
        char_limit = int(config.kb_field_extraction_max_chars)
        document_values = await extract_document(
            llm, schema["document_fields"], doc_id=doc_id, title=title, text=text, char_limit=char_limit
        )
        passage_values = await extract_passages(
            llm,
            schema["passage_fields"],
            doc_id=doc_id,
            title=title,
            passages=passage_texts,
            char_limit=char_limit,
            concurrency=int(config.kb_field_extraction_concurrency),
        )
        return ExtractedFields(document=document_values, passages=passage_values)

    async def _fields_for_document(
        self,
        *,
        bank_id: str,
        config: Any,
        schema: dict[str, Any] | None,
        document: DocumentInput,
        passage_texts: list[tuple[int, str]],
        extract: bool = True,
    ) -> ExtractedFields:
        """One document's metadata: what the write supplied, plus what is left to extract.

        Both halves land in the same object. Supplying every property of the schema means
        no LLM call for that document, which is how a corpus can be loaded with full
        metadata and no model spend at all.
        """
        document_definition = (schema or {}).get("document_fields", {})
        passage_definition = (schema or {}).get("passage_fields", {})
        supplied = self._validated_fields(schema, document)
        supplied_document, supplied_passages = supplied.document, supplied.passages

        # An import supplies every field the source had and must not spend a token
        # filling one it did not, so it turns extraction off rather than relying on the
        # supplied values happening to cover the schema.
        if schema is None or not extract:
            return ExtractedFields(document=supplied_document, passages=supplied_passages)

        pending_document = extract_only(document_definition, document.fields)
        pending_passages = extract_only(passage_definition, None)
        passages_to_read = [
            (index, text)
            for index, text in passage_texts
            if not set(passage_definition) <= set(supplied_passages.get(index, {}))
        ]
        if not pending_document and not pending_passages:
            return ExtractedFields(document=supplied_document, passages=supplied_passages)

        llm = self._field_extraction_llm(bank_id, config)
        char_limit = int(config.kb_field_extraction_max_chars)
        document_values = await extract_document(
            llm,
            pending_document,
            doc_id=document.doc_id,
            title=document.title,
            text=document.text,
            char_limit=char_limit,
        )
        passage_values = await extract_passages(
            llm,
            pending_passages,
            doc_id=document.doc_id,
            title=document.title,
            passages=passages_to_read,
            char_limit=char_limit,
            concurrency=int(config.kb_field_extraction_concurrency),
        )
        # The supplied value wins: the caller knew it, the model only guessed at it.
        merged_passages = {
            index: {**passage_values.get(index, {}), **supplied_passages.get(index, {})}
            for index in set(passage_values) | set(supplied_passages)
        }
        return ExtractedFields(
            document={**document_values, **supplied_document},
            passages={index: values for index, values in merged_passages.items() if values},
        )

    async def run_extract_fields(self, task: dict[str, Any]) -> dict[str, Any]:
        """Worker side of a re-extraction: read stored documents, fill the schema again."""
        bank_id = task["bank_id"]
        config = await self._config(bank_id, None)
        schema = await self._schema_for_write(bank_id, task.get("schema_id"))
        pool = await self._pool()
        doc_ids: list[str] = task.get("doc_ids") or []
        only_missing = bool(task.get("only_missing"))

        where = "bank_id = $1"
        params: list[Any] = [bank_id]
        if doc_ids:
            params.append(doc_ids)
            where += f" AND doc_id = ANY(${len(params)}::text[])"
        elif schema is not None:
            # Only this schema's documents, plus the ones no schema has claimed — a bank
            # with several schemas must not have its memos re-read as invoices, and a
            # document written before any schema existed is exactly what a first
            # extraction is for.
            params.append(schema["schema_id"])
            where += f" AND (schema_id = ${len(params)} OR schema_id IS NULL)"
        if only_missing:
            where += " AND fields = '{}'::jsonb"
        async with acquire_with_retry(pool) as conn:
            rows = await conn.fetch(
                f"SELECT doc_id, title, text FROM {store.fq_table('kb_documents')} WHERE {where} ORDER BY doc_id",
                *params,
            )

        documents = updated_passages = 0
        for row in rows:
            async with acquire_with_retry(pool) as conn:
                passage_rows = await conn.fetch(
                    f"SELECT passage_index, text FROM {store.fq_table('kb_passages')} "
                    "WHERE bank_id = $1 AND doc_id = $2 ORDER BY passage_index",
                    bank_id,
                    row["doc_id"],
                )
            extracted = await self._extract_for_document(
                bank_id=bank_id,
                config=config,
                schema=schema,
                doc_id=row["doc_id"],
                title=row["title"],
                text=row["text"],
                passage_texts=[(c["passage_index"], c["text"]) for c in passage_rows],
            )
            async with acquire_with_retry(pool) as conn:
                await store.set_document_fields(conn, bank_id, row["doc_id"], extracted.document)
                await store.set_passage_fields(conn, bank_id, row["doc_id"], extracted.passages)
                if schema is not None:
                    await conn.execute(
                        f"UPDATE {store.fq_table('kb_documents')} SET schema_id = $3 "
                        "WHERE bank_id = $1 AND doc_id = $2",
                        bank_id,
                        row["doc_id"],
                        schema["schema_id"],
                    )
            documents += 1
            updated_passages += len(extracted.passages)

        counts = {"documents_extracted": documents, "passages_extracted": updated_passages}
        operation_id = task.get("operation_id")
        if operation_id:
            async with acquire_with_retry(pool) as conn:
                await conn.execute(
                    f"UPDATE {store.fq_table('async_operations')} "
                    "SET result_metadata = COALESCE(result_metadata, '{}'::jsonb) || $1::jsonb "
                    "WHERE operation_id = $2",
                    json.dumps(counts),
                    uuid.UUID(str(operation_id)),
                )
        logger.info("knowledge field extraction bank=%s %s", bank_id, counts)
        return counts

    # ---- collections and records

    async def list_collections(self, bank_id: str) -> dict[str, Any]:
        await self._require_bank(bank_id)
        async with acquire_with_retry(await self._pool()) as conn:
            items = await collections_store.list_collections(conn, bank_id)
        return {"items": items, "total": len(items)}

    async def get_collection(self, bank_id: str, collection_id: str) -> dict[str, Any]:
        await self._require_bank(bank_id)
        async with acquire_with_retry(await self._pool()) as conn:
            collection = await collections_store.get_collection(conn, bank_id, collection_id)
            if collection is None:
                raise KnowledgeBankError(404, f"collection {collection_id!r} not found")
            collection["records"] = await conn.fetchval(
                f"SELECT count(*) FROM {store.fq_table('kb_records')} WHERE bank_id = $1 AND collection_id = $2",
                bank_id,
                collection_id,
            )
        return collection

    async def chat_about_collections(
        self,
        bank_id: str,
        messages: list[dict[str, str]],
        sample: int,
        request_context: Any = None,
    ) -> dict[str, Any]:
        """One turn of the collection chat: what to say back, and what it proposes.

        The agent answers or calls its one tool; the tool records a proposal. Nothing
        here writes. That separation is the point: applying a schema change re-reads the
        corpus at one LLM call per document, so it is a decision a person makes.
        """
        await self._require_bank(bank_id)
        if not messages:
            raise KnowledgeBankError(400, "messages must not be empty")
        config = await self._config(bank_id, request_context)
        async with acquire_with_retry(await self._pool()) as conn:
            existing = await collections_store.list_collections(conn, bank_id)
            rows = await conn.fetch(
                f"""
                SELECT doc_id, title, left(text, $2) AS text
                FROM {store.fq_table("kb_documents")}
                WHERE bank_id = $1 AND text <> ''
                ORDER BY updated_at DESC
                LIMIT $3
                """,
                bank_id,
                int(config.kb_field_extraction_max_chars),
                sample,
            )
        documents = [dict(row) for row in rows]

        llm = self._field_extraction_llm(bank_id, config)
        try:
            turn = await chat_about_collections(
                llm,
                messages=messages,
                context=collections_context(existing, documents),
            )
        except Exception as e:
            raise KnowledgeBankError(502, f"the model could not answer: {e}") from e

        proposals, warnings = self._validated_proposals(turn.proposals, existing)
        return {
            "reply": turn.reply,
            "proposals": proposals,
            "warnings": warnings,
            "documents_read": len(documents),
        }

    def _validated_proposals(
        self, proposed: list[Any], existing: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Every proposal, checked by the rules a PUT applies.

        What comes back has to be applyable as-is: a definition the model got wrong is
        reported here rather than handed on to fail when someone approves it.
        """
        known = {c["collection_id"] for c in existing} | {p.collection_id for p in proposed if p.action != "delete"}
        existing_ids = {c["collection_id"] for c in existing}
        out: list[dict[str, Any]] = []
        warnings: list[str] = []
        for proposal in proposed:
            if not proposal.collection_id:
                continue
            # A change nobody can explain is a change nobody should approve, so one that
            # arrives without a reason is reported rather than shown as a bare diff.
            if not proposal.reason.strip():
                warnings.append(f"{proposal.collection_id}: proposed with no reason, so it was dropped")
                continue
            if proposal.action == "delete":
                if proposal.collection_id in existing_ids:
                    out.append(
                        {
                            "action": "delete",
                            "collection_id": proposal.collection_id,
                            "reason": proposal.reason,
                            "definition": None,
                        }
                    )
                continue
            fields = {
                field.name: (
                    {
                        "collection": field.collection,
                        **({"description": field.description} if field.description else {}),
                    }
                    if field.collection
                    else {
                        "type": field.type,
                        **({"description": field.description} if field.description else {}),
                        **({"values": field.values} if field.values else {}),
                    }
                )
                for field in proposal.fields
                if field.name
            }
            try:
                validated = collections_store.validate_collection_fields(fields, known_collections=known)
            except SchemaError as e:
                warnings.append(f"{proposal.collection_id}: {e}")
                continue
            identity = proposal.identity if proposal.identity in validated else None
            if proposal.identity and identity is None:
                warnings.append(f"{proposal.collection_id}: identity {proposal.identity!r} is not one of its fields")
            # An update the model left blank keeps what the collection has: a PUT replaces
            # the definition, and "add a field" must not erase a name or an identity.
            current = next((c for c in existing if c["collection_id"] == proposal.collection_id), None)
            if identity is None and current and current.get("identity") in validated:
                identity = current["identity"]
            out.append(
                {
                    # An id the bank already has is an update whatever the model called it.
                    "action": "update" if proposal.collection_id in existing_ids else "create",
                    "collection_id": proposal.collection_id,
                    "reason": proposal.reason,
                    "definition": {
                        "name": proposal.name or (current or {}).get("name") or None,
                        "description": proposal.description or (current or {}).get("description") or None,
                        "identity": identity,
                        "fields": validated,
                    },
                }
            )
        return out, warnings

    async def put_collection(
        self,
        bank_id: str,
        collection_id: str,
        *,
        name: str | None,
        description: str | None,
        fields: Any,
        identity: str | None,
        derive_on_write: bool = False,
        reprocess: bool = False,
    ) -> dict[str, Any]:
        """Define or redefine a collection, and say what the change did to its records.

        Changing the definition is a migration, not an edit: a dropped field has to leave
        the rows, and a new one is empty everywhere until something fills it. Both are
        reported, and ``reprocess`` queues the re-derivation that fills the new one from
        the documents already in the bank.
        """
        await self._require_bank(bank_id)
        if not collection_id or not collection_id.replace("_", "").replace("-", "").isalnum():
            raise KnowledgeBankError(400, "collection id must be alphanumeric with _ or -")
        async with acquire_with_retry(await self._pool()) as conn:
            existing = {c["collection_id"] for c in await collections_store.list_collections(conn, bank_id)}
            before = await collections_store.get_collection(conn, bank_id, collection_id)
        try:
            # A relationship may point at this collection itself (a contract that
            # supersedes another), so its own id counts as known.
            definition = collections_store.validate_collection_fields(
                fields, known_collections=existing | {collection_id}
            )
        except SchemaError as e:
            raise KnowledgeBankError(400, str(e)) from e
        if identity is not None and identity not in definition:
            raise KnowledgeBankError(400, f"identity {identity!r} is not one of this collection's fields")

        previous = (before or {}).get("fields") or {}
        changes = {
            "added": sorted(set(definition) - set(previous)),
            "removed": sorted(set(previous) - set(definition)),
            "retyped": sorted(
                name
                for name in set(previous) & set(definition)
                if previous[name].get("type") != definition[name].get("type")
                or previous[name].get("collection") != definition[name].get("collection")
            ),
        }
        if before and identity != before.get("identity") and before.get("identity") is not None:
            # The identity decides a record's id, so changing it re-groups every row:
            # nothing short of re-deriving can do that, and silently keeping the old ids
            # would leave rows whose id no longer matches their own identity value.
            changes["identity_changed"] = True

        async with acquire_with_retry(await self._pool()) as conn:
            await collections_store.put_collection(
                conn,
                bank_id,
                collection_id,
                name=name,
                description=description,
                fields=definition,
                identity=identity,
                derive_on_write=derive_on_write,
            )
            pruned = 0
            if changes["removed"] or changes["retyped"]:
                # A retyped field's old values are the old type; they go with the dropped
                # ones rather than being coerced into something the caller did not write.
                keep = set(definition) - set(changes["retyped"])
                pruned = await collections_store.prune_fields(conn, bank_id, collection_id, keep)

        result = await self.get_collection(bank_id, collection_id)
        result["changes"] = {**changes, "records_pruned": pruned}
        if reprocess:
            queued = await self.submit_derive_records(bank_id, collection_id, None, replace=True)
            result["reprocess"] = queued
        return result

    async def delete_collection(self, bank_id: str, collection_id: str) -> dict[str, Any]:
        await self._require_bank(bank_id)
        async with acquire_with_retry(await self._pool()) as conn:
            if not await collections_store.delete_collection(conn, bank_id, collection_id):
                raise KnowledgeBankError(404, f"collection {collection_id!r} not found")
        return {"collection_id": collection_id, "deleted": True}

    async def put_records(self, bank_id: str, collection_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Write records directly — the deterministic path, no LLM involved.

        A write is a contribution like any other, attributed to the documents it names
        (or to nothing), so writing the same record twice updates it instead of stacking.
        """
        collection = await self.get_collection(bank_id, collection_id)
        collection = {**collection, "bank_id": bank_id}
        config = await self._config(bank_id, None)
        written = 0
        async with acquire_with_retry(await self._pool()) as conn:
            for record in records:
                values = await self._resolved_relationships(
                    conn, bank_id, collection, record.get("values") or {}, config=config
                )
                record_id = await self._record_id(conn, collection, record.get("record_id"), values, config=config)
                doc_ids = record.get("doc_ids") or [""]
                for doc_id in doc_ids:
                    await collections_store.contribute(
                        conn,
                        bank_id,
                        collection_id,
                        record_id,
                        doc_id=doc_id,
                        values=values,
                        evidence=record.get("evidence") or {},
                    )
                await collections_store.materialize(conn, bank_id, collection_id, record_id)
                written += 1
        return {"collection_id": collection_id, "records_written": written}

    async def delete_record(self, bank_id: str, collection_id: str, record_id: str) -> dict[str, Any]:
        """Delete a record and everything that fed it."""
        await self.get_collection(bank_id, collection_id)
        async with acquire_with_retry(await self._pool()) as conn:
            resolved = await self._existing_record_id(conn, bank_id, collection_id, record_id)
            if resolved is None or not await collections_store.delete_record(conn, bank_id, collection_id, resolved):
                raise KnowledgeBankError(404, f"record {record_id!r} not found")
        return {"record_id": resolved, "deleted": True}

    async def _existing_record_id(self, conn: Any, bank_id: str, collection_id: str, given: str) -> str | None:
        """The record a caller means by this id.

        A record's id is its normalised identity, so "c-1" is stored as "c 1" and
        "Apple Inc." as "apple". A caller should not have to know that: they name the
        record the way their data names it, and it resolves the same way a write does —
        exactly, then through a merge, then through normalisation.
        """
        for candidate in (given, await collections_store.follow_aliases(conn, bank_id, collection_id, given)):
            if await collections_store.get_record(conn, bank_id, collection_id, candidate) is not None:
                return candidate
        key = normalise(given)
        if key and key != given:
            resolved = await collections_store.follow_aliases(conn, bank_id, collection_id, key)
            if await collections_store.get_record(conn, bank_id, collection_id, resolved) is not None:
                return resolved
        return None

    async def collection_stats(self, bank_id: str, collection_id: str) -> dict[str, Any]:
        """What a collection holds, as opposed to what it defines."""
        collection = await self.get_collection(bank_id, collection_id)
        async with acquire_with_retry(await self._pool()) as conn:
            return await collections_store.collection_stats(
                conn, bank_id, collection_id, sorted(collection.get("fields") or {})
            )

    async def list_records(
        self, bank_id: str, collection_id: str, limit: int, offset: int, query: str | None
    ) -> dict[str, Any]:
        """One page of a collection's records."""
        await self.get_collection(bank_id, collection_id)
        async with acquire_with_retry(await self._pool()) as conn:
            return await collections_store.list_records(conn, bank_id, collection_id, limit, offset, query)

    async def get_record(self, bank_id: str, collection_id: str, record_id: str) -> dict[str, Any]:
        await self._require_bank(bank_id)
        async with acquire_with_retry(await self._pool()) as conn:
            resolved = await self._existing_record_id(conn, bank_id, collection_id, record_id)
            record = await collections_store.get_record(conn, bank_id, collection_id, resolved) if resolved else None
        if record is None:
            raise KnowledgeBankError(404, f"record {record_id!r} not found")
        return record

    async def pin_record_values(
        self, bank_id: str, collection_id: str, record_id: str, pinned: dict[str, Any]
    ) -> dict[str, Any]:
        await self._require_bank(bank_id)
        async with acquire_with_retry(await self._pool()) as conn:
            resolved = await self._existing_record_id(conn, bank_id, collection_id, record_id)
            if resolved is None or not await collections_store.pin_values(
                conn, bank_id, collection_id, resolved, pinned
            ):
                raise KnowledgeBankError(404, f"record {record_id!r} not found")
            record_id = resolved
            # Rebuild from the parts so the pin wins by the same rule every other
            # precedence question is answered by, not because it was written last.
            await collections_store.materialize(conn, bank_id, collection_id, record_id)
        return await self.get_record(bank_id, collection_id, record_id)

    async def _derivation_windows(self, pool: Any, bank_id: str, row: Any, char_limit: int) -> list[str]:
        """One document as the windows a derivation reads, largest that still fit.

        Passages are what the bank already cut the document into, so they are the unit:
        consecutive ones are packed up to ``char_limit`` and each pack is one prompt. A
        document with no passages yet (or one short enough to fit) is a single window of
        its own text, which is what the whole corpus used to be.
        """
        if len(row["text"] or "") <= char_limit:
            return [row["text"] or ""]
        async with acquire_with_retry(pool) as conn:
            passages = await conn.fetch(
                f"SELECT text FROM {store.fq_table('kb_passages')} "
                "WHERE bank_id = $1 AND doc_id = $2 ORDER BY passage_index",
                bank_id,
                row["doc_id"],
            )
        if not passages:
            return [row["text"][:char_limit]]
        windows: list[str] = []
        current = ""
        for passage in passages:
            text = passage["text"] or ""
            if current and len(current) + len(text) + 2 > char_limit:
                windows.append(current)
                current = text
            else:
                current = f"{current}\n\n{text}" if current else text
        if current:
            windows.append(current)
        return windows

    async def _relationship_candidates(
        self, conn: Any, bank_id: str, collection: dict[str, Any], limit: int = 200
    ) -> list[str]:
        """The records this collection's relationships may point at, as ``field: name``.

        Capped: a prompt cannot carry a million records, and past a point the list stops
        helping a model choose. Beyond the cap the value falls back to what the document
        says, which resolution still tries to match.
        """
        lines: list[str] = []
        for name, target in collections_store.relationships(collection.get("fields") or {}).items():
            rows = await conn.fetch(
                f"SELECT record_id FROM {store.fq_table('kb_records')} "
                "WHERE bank_id = $1 AND collection_id = $2 ORDER BY record_id LIMIT $3",
                bank_id,
                target,
                limit,
            )
            lines.extend(f"{name}: {row['record_id']}" for row in rows)
        return lines

    async def _unresolved_relationships(self, conn: Any, bank_id: str, collection: dict[str, Any]) -> int:
        """How many relationship values name no record in the collection they point at."""
        unresolved = 0
        for name, target in collections_store.relationships(collection.get("fields") or {}).items():
            unresolved += (
                await conn.fetchval(
                    f"""
                    SELECT count(*) FROM {store.fq_table("kb_records")} r
                    WHERE r.bank_id = $1 AND r.collection_id = $2
                      AND r.values ? $3
                      AND NOT EXISTS (
                        SELECT 1 FROM {store.fq_table("kb_records")} t
                        WHERE t.bank_id = r.bank_id AND t.collection_id = $4
                          AND t.record_id = (r.values -> $3 #>> '{{}}')
                      )
                    """,
                    bank_id,
                    collection["collection_id"],
                    name,
                    target,
                )
                or 0
            )
        return unresolved

    async def _resolved_relationships(
        self, conn: Any, bank_id: str, collection: dict[str, Any], values: dict[str, Any], *, config: Any
    ) -> dict[str, Any]:
        """Point every relationship at a record of the collection it names.

        A relationship field is a foreign key: the query compiler joins it straight
        against the other collection's ``record_id``. So it has to be written in the
        same alphabet record ids are written in — ``normalise`` plus the resolver, the
        same pair :meth:`_record_id` uses — and not the lowercased name the model or the
        caller happened to say. Two normalisations for one concept is how "Acme Ltd"
        came to point at nothing while the record sat there as "acme" (the join returned
        null for every contract in the collections eval).

        Resolution runs against the target collection even when it is empty: the
        resolver then hands back the normalised key, so the link starts dangling and
        lands the moment that record is derived, rather than being wrong forever.
        """
        relationships = collections_store.relationships(collection.get("fields") or {})
        if not relationships:
            return values
        similarity = float(getattr(config, "kb_record_identity_similarity", 0.82)) if config else 0.82
        resolved = dict(values)
        for name, target in relationships.items():
            raw = resolved.get(name)
            if raw in (None, ""):
                continue
            key = normalise(raw)
            if not key:
                # A value that normalises away names nothing; a blank is honest, a
                # record id of "" would collide with every other nameless one.
                resolved[name] = None
                continue
            resolved[name] = await collections_store.resolve_record_id(
                conn, bank_id, target, key, similarity=similarity
            )
        return resolved

    async def _record_id(
        self, conn: Any, collection: dict[str, Any], given: str | None, values: dict[str, Any], *, config: Any = None
    ) -> str:
        """Which record this is.

        The identity field is what makes two documents about the same vendor one record
        rather than two, so it decides the id — after resolution, because documents do
        not agree on how to write a name. Without an identity field a record is whatever
        the caller called it, and failing that a hash of its values.
        """
        if given:
            return str(given)
        identity = collection.get("identity")
        if identity and values.get(identity) not in (None, ""):
            key = normalise(values[identity])
            if not key:
                # A name that normalises to nothing (punctuation, an empty string) must
                # not become the record every nameless thing falls into.
                return hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()[:24]
            similarity = float(getattr(config, "kb_record_identity_similarity", 0.82)) if config else 0.82
            return await collections_store.resolve_record_id(
                conn, collection["bank_id"], collection["collection_id"], key, similarity=similarity
            )
        return hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()[:24]

    async def merge_records(self, bank_id: str, collection_id: str, record_id: str, into: str) -> dict[str, Any]:
        """Declare that two records are the same thing, and keep the decision.

        The losing id becomes an alias of the winner, so a document that names it again
        lands on the merged record instead of re-creating what someone just cleaned up.
        """
        await self.get_collection(bank_id, collection_id)
        async with acquire_with_retry(await self._pool()) as conn:
            # Either side may name a record that was itself merged away; following the
            # alias means a caller working from a stale list still merges into the record
            # that exists rather than being told the id is gone.
            source = await self._existing_record_id(conn, bank_id, collection_id, record_id) or record_id
            target = await self._existing_record_id(conn, bank_id, collection_id, into) or into
            if source == target:
                raise KnowledgeBankError(400, "a record cannot be merged into itself")
            for candidate in (source, target):
                if await collections_store.get_record(conn, bank_id, collection_id, candidate) is None:
                    raise KnowledgeBankError(404, f"record {candidate!r} not found")
            await collections_store.merge_records(conn, bank_id, collection_id, source, target)
            # The id the caller used is now a way of naming the survivor.
            await collections_store.add_alias(conn, bank_id, collection_id, record_id, target, source="merge")
        return await self.get_record(bank_id, collection_id, target)

    async def submit_derive_records(
        self, bank_id: str, collection_id: str, doc_ids: list[str] | None, *, replace: bool = False
    ) -> dict[str, Any]:
        """Queue LLM derivation of this collection's records from the bank's documents.

        ``replace`` drops what those documents contributed before re-reading them, which
        is what a definition change needs: without it the old values would fold back in
        beside the new ones.
        """
        await self.get_collection(bank_id, collection_id)
        result = await self.memory._submit_async_operation(
            bank_id=bank_id,
            operation_type="knowledge_derive_records",
            task_type="knowledge_derive_records",
            task_payload={"collection_id": collection_id, "doc_ids": doc_ids or [], "replace": replace},
        )
        return {"operation_id": result["operation_id"], "collection_id": collection_id, "status": "pending"}

    async def run_derive_records(self, task: dict[str, Any]) -> dict[str, Any]:
        """Worker side: read the documents, ask for records, fold them in by identity."""
        bank_id = task["bank_id"]
        collection_id = task["collection_id"]
        collection = await self.get_collection(bank_id, collection_id)
        collection = {**collection, "bank_id": bank_id}
        config = await self._config(bank_id, None)
        pool = await self._pool()
        llm = self._field_extraction_llm(bank_id, config)
        replace = bool(task.get("replace"))

        where = "bank_id = $1"
        params: list[Any] = [bank_id]
        if task.get("doc_ids"):
            params.append(task["doc_ids"])
            where += f" AND doc_id = ANY(${len(params)}::text[])"
        async with acquire_with_retry(pool) as conn:
            documents = await conn.fetch(
                f"SELECT doc_id, title, text FROM {store.fq_table('kb_documents')} WHERE {where} ORDER BY doc_id",
                *params,
            )
            # The records a relationship may point at, offered to the model so it names
            # one instead of inventing a spelling that resolves to nothing.
            known_records = await self._relationship_candidates(conn, bank_id, collection)

        derivable = {name: spec for name, spec in (collection["fields"] or {}).items() if not spec.get("collection")}
        relationship_fields = collections_store.relationships(collection["fields"] or {})
        fields_for_model = {**derivable, **{name: {"type": "string"} for name in relationship_fields}}
        char_limit = int(config.kb_field_extraction_max_chars)
        records_written = 0
        records_skipped = 0
        identity_field = collection.get("identity")

        async def read(row: Any) -> tuple[Any, list[dict[str, Any]]]:
            """Every record one document describes, read a window at a time.

            A document is not one prompt. It used to be sent whole and truncated at
            ``kb_field_extraction_max_chars``, so a contract longer than that lost
            everything past the cut with nothing said about it — invisible on a corpus of
            paragraphs, total on a real one. Its passages are already stored, so they are
            read in windows of the same budget and what each window found is merged.
            """
            windows = await self._derivation_windows(pool, bank_id, row, char_limit)
            merged: dict[str, dict[str, Any]] = {}
            for window in windows:
                async with semaphore:
                    derived = await derive_records(
                        llm,
                        fields_for_model,
                        collection_name=collection.get("name") or collection_id,
                        collection_description=collection.get("description"),
                        known_records=known_records,
                        doc_id=row["doc_id"],
                        title=row["title"],
                        text=window,
                        char_limit=char_limit,
                    )
                for record in derived:
                    # Windows of one document are one contribution per record, not one
                    # each: a contribution is keyed by (record, document), so writing
                    # them separately would leave only the last window's values.
                    key = normalise(record["values"].get(identity_field)) if identity_field else str(len(merged))
                    into = merged.setdefault(key, {"values": {}, "evidence": {}})
                    into["values"].update({k: v for k, v in record["values"].items() if v is not None})
                    into["evidence"].update(record["evidence"] or {})
            return row, list(merged.values())

        # The model calls run together; the writes do not. Two documents folding the same
        # record would otherwise race each other through materialize.
        semaphore = asyncio.Semaphore(max(1, int(config.kb_field_extraction_concurrency)))
        for row, derived in await asyncio.gather(*(read(row) for row in documents)):
            async with acquire_with_retry(pool) as conn:
                stale = (
                    {
                        r
                        for c, r in await collections_store.records_touched_by(conn, bank_id, row["doc_id"])
                        if c == collection_id
                    }  # fmt: skip
                    if replace
                    else set()
                )
                if replace:
                    await collections_store.drop_contributions_of(conn, bank_id, row["doc_id"], collection_id)
                for record in derived:
                    # A collection with an identity field says what one of its records
                    # IS. A derived record with no value for it is not a thing, it is
                    # half a sentence — and hashing its values into an id files it as a
                    # row of its own: one run produced 14 contract rows for 3 contracts,
                    # rows like {"value": 120000} with no reference, which then carried
                    # their numbers into every aggregate. Dropping it loses nothing: a
                    # document that really describes the record says which one.
                    if identity_field and record["values"].get(identity_field) in (None, ""):
                        records_skipped += 1
                        continue
                    values = await self._resolved_relationships(
                        conn, bank_id, collection, dict(record["values"]), config=config
                    )
                    record_id = await self._record_id(conn, collection, None, values, config=config)
                    await collections_store.contribute(
                        conn,
                        bank_id,
                        collection_id,
                        record_id,
                        doc_id=row["doc_id"],
                        values=values,
                        evidence=record["evidence"] or {},
                    )
                    await collections_store.materialize(conn, bank_id, collection_id, record_id)
                    stale.discard(record_id)
                    records_written += 1
                # A record this document used to feed and no longer mentions: rebuild it,
                # which removes it entirely when nothing else was ever behind it.
                for record_id in stale:
                    await collections_store.materialize(conn, bank_id, collection_id, record_id)

        async with acquire_with_retry(pool) as conn:
            dangling = await self._unresolved_relationships(conn, bank_id, collection)

        counts = {
            "documents_read": len(documents),
            "records_written": records_written,
            # Reported, not swallowed: a derivation that drops most of what the model
            # returned is a prompt or schema problem the operator should see.
            "records_skipped": records_skipped,
            # A relationship whose value names no record in the collection it points at.
            # The link is kept — the record may arrive later — but a join over it returns
            # nothing, and silence there is the failure nobody notices.
            "unresolved_links": dangling,
        }
        operation_id = task.get("operation_id")
        if operation_id:
            async with acquire_with_retry(pool) as conn:
                await conn.execute(
                    f"UPDATE {store.fq_table('async_operations')} "
                    "SET result_metadata = COALESCE(result_metadata, '{}'::jsonb) || $1::jsonb "
                    "WHERE operation_id = $2",
                    json.dumps(counts),
                    uuid.UUID(str(operation_id)),
                )
        logger.info("knowledge records derived bank=%s collection=%s %s", bank_id, collection_id, counts)
        return counts

    async def query_records(
        self, bank_id: str, collection_id: str, body: dict[str, Any], request_context: Any = None
    ) -> dict[str, Any]:
        """Run one record query, joins included."""
        collection = await self.get_collection(bank_id, collection_id)
        async with acquire_with_retry(await self._pool()) as conn:
            joined_fields = {
                other["collection_id"]: frozenset(other.get("fields") or {})
                for other in await collections_store.list_collections(conn, bank_id)
            }
        try:
            compiled = compile_record_query(body, bank_id, collection, limits=Limits(), joined_fields=joined_fields)
        except FilterError as e:
            raise KnowledgeBankError(400, str(e)) from e
        async with acquire_with_retry(await self._pool()) as conn:
            try:
                rows = await conn.fetch(compiled.sql, *compiled.params)
            except Exception as e:
                logger.info("knowledge record query failed bank=%s: %s", bank_id, e)
                raise KnowledgeBankError(400, f"query could not run: {e}") from e
        return {
            "columns": compiled.columns,
            "rows": [[json_safe(value) for value in row] for row in rows],
            "row_count": len(rows),
            "grouped": compiled.grouped,
        }

    # ---- query

    async def query(self, bank_id: str, body: dict[str, Any], request_context: Any = None) -> dict[str, Any]:
        """Run one DSL query. The bank is bound as $1, so it cannot read another bank."""
        await self._require_bank(bank_id)
        try:
            compiled: CompiledQuery = compile_query(body, bank_id, limits=Limits())
        except FilterError as e:
            raise KnowledgeBankError(400, str(e)) from e
        pool = await self._pool()
        async with acquire_with_retry(pool) as conn:
            try:
                rows = await conn.fetch(compiled.sql, *compiled.params)
            except Exception as e:
                # A query that Postgres rejects is the caller's structure, not a server
                # fault: a bad cast or an unorderable type gets the message, not a 500.
                logger.info("knowledge query failed bank=%s: %s", bank_id, e)
                raise KnowledgeBankError(400, f"query could not run: {e}") from e
        return {
            "columns": compiled.columns,
            "rows": [[json_safe(value) for value in row] for row in rows],
            "row_count": len(rows),
            "grouped": compiled.grouped,
        }

    # ---- writing (always async, always a batch)

    async def submit_write(self, bank_id: str, documents: list[DocumentInput]) -> dict[str, Any]:
        """Queue one ``knowledge_write_batch`` operation for these documents."""
        await self._require_bank(bank_id)
        # Supplied property values are validated here, not in the worker: the write is
        # async, so a value that does not fit the schema has to fail the request the caller
        # is holding rather than an operation they would have to go and read.
        for document in documents:
            schema = await self._schema_for_write(bank_id, document.schema_id)
            self._validated_fields(schema, document)
        if not documents:
            raise KnowledgeBankError(400, "documents must not be empty")
        if len(documents) > MAX_BATCH_DOCUMENTS:
            raise KnowledgeBankError(400, f"at most {MAX_BATCH_DOCUMENTS} documents per batch")
        seen: set[str] = set()
        payload_docs = []
        for document in documents:
            if not document.doc_id:
                raise KnowledgeBankError(400, "every document needs an id")
            if document.doc_id in seen:
                raise KnowledgeBankError(400, f"duplicate document id in batch: {document.doc_id}")
            seen.add(document.doc_id)
            payload_docs.append(
                {
                    "doc_id": document.doc_id,
                    # Postgres stores no NUL character in text or JSON, and one in a
                    # document fails the whole batch; it carries nothing, so it goes.
                    "text": document.text.replace("\x00", ""),
                    "title": document.title.replace("\x00", "") if document.title else document.title,
                    "metadata": document.metadata or {},
                    "fields": document.fields or {},
                    "schema_id": document.schema_id,
                    "passage_fields": {str(k): v for k, v in (document.passage_fields or {}).items()},
                }
            )
        result = await self.memory._submit_async_operation(
            bank_id=bank_id,
            operation_type="knowledge_write_batch",
            task_type="knowledge_write_batch",
            task_payload={"documents": payload_docs},
            result_metadata={"documents": len(payload_docs)},
        )
        return {"operation_id": result["operation_id"], "documents": len(payload_docs), "status": "pending"}

    # ---- writing a file

    async def submit_file_write(self, bank_id: str, files: list[UploadedFile], request_context: Any) -> dict[str, Any]:
        """Store these files and queue one conversion operation each.

        Conversion is where a PDF costs seconds and a scanned one costs an OCR pass, so it
        does not happen in the request — each file becomes a ``knowledge_file_convert``
        operation that converts it and then queues the ordinary write. Two operations per
        file, the same shape the memory banks' file retain uses.
        """
        await self._require_bank(bank_id)
        config = get_config()
        if not config.enable_file_upload_api:
            raise KnowledgeBankError(
                404, "File upload API is disabled. Set HINDSIGHT_API_ENABLE_FILE_UPLOAD_API=true to enable."
            )
        if len(files) > config.file_conversion_max_batch_size:
            raise KnowledgeBankError(400, f"at most {config.file_conversion_max_batch_size} files per request")
        total = sum(len(f.content) for f in files)
        if total > config.file_conversion_max_batch_size_bytes:
            raise KnowledgeBankError(
                400,
                f"total batch size ({total / (1024 * 1024):.1f}MB) exceeds "
                f"{config.file_conversion_max_batch_size_mb}MB",
            )

        operation_ids = []
        for file in files:
            # The key carries a uuid rather than the filename or the document id: two files
            # in one batch may share both, and the storage backend would overwrite one with
            # the other (the same defect #3226 fixed on the memory side).
            storage_key = f"{bank_storage_prefix(bank_id)}files/{uuid.uuid4()}/{file.filename}"
            await self.memory._file_storage.store(
                file_data=file.content,
                key=storage_key,
                metadata={
                    "content_type": file.content_type,
                    "original_filename": file.filename,
                    "bank_id": bank_id,
                    "document_id": file.doc_id,
                },
            )
            payload: dict[str, Any] = {
                "doc_id": file.doc_id,
                "storage_key": storage_key,
                "original_filename": file.filename,
                "content_type": file.content_type,
                "parser": file.parser,
                "title": file.title,
                "metadata": file.metadata,
                "schema_id": file.schema_id,
            }
            if request_context is not None and getattr(request_context, "tenant_id", None):
                payload["_tenant_id"] = request_context.tenant_id
            if request_context is not None and getattr(request_context, "api_key_id", None):
                payload["_api_key_id"] = request_context.api_key_id
            result = await self.memory._submit_async_operation(
                bank_id=bank_id,
                operation_type="knowledge_file_convert",
                task_type="knowledge_file_convert",
                task_payload=payload,
                result_metadata={"original_filename": file.filename, "document_id": file.doc_id},
                dedupe_by_bank=False,
            )
            operation_ids.append(result["operation_id"])
        return {"operation_ids": operation_ids, "files": len(files)}

    async def run_file_convert(self, task: dict[str, Any]) -> dict[str, Any]:
        """Worker side of a file write: convert to markdown, then queue the write."""
        bank_id = task["bank_id"]
        filename = task.get("original_filename") or "unknown"
        file_data = await self.memory._file_storage.retrieve(task["storage_key"])
        parsers = task.get("parser") or get_config().file_parser
        try:
            converted = await self.memory._parser_registry.convert_with_fallback(
                parsers=parsers,
                file_data=file_data,
                filename=filename,
                content_type=task.get("content_type"),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to parse file {filename!r}: {e}") from e
        text = sanitize_llm_output(converted.content) or ""

        # The same hook the memory side fires, and for the same reason: a deployment bills
        # conversions, and a knowledge bank's conversions are conversions.
        validator = getattr(self.memory, "_operation_validator", None)
        if validator is not None:
            try:
                from ..extensions.operation_validator import FileConvertResult
                from ..models import RequestContext

                await validator.on_file_convert_complete(
                    FileConvertResult(
                        bank_id=bank_id,
                        parser_name=converted.parser_name,
                        filename=filename,
                        output_chars=len(text),
                        output_text=text,
                        request_context=RequestContext(
                            internal=True,
                            user_initiated=True,
                            tenant_id=task.get("_tenant_id"),
                            api_key_id=task.get("_api_key_id"),
                        ),
                    )
                )
            except Exception as e:
                logger.warning("knowledge file convert: on_file_convert_complete failed: %s", e)

        metadata = dict(task.get("metadata") or {})
        metadata.setdefault("file_original_name", filename)
        metadata.setdefault("file_content_type", task.get("content_type"))
        metadata.setdefault("file_storage_key", task["storage_key"])
        write = await self.submit_write(
            bank_id,
            [
                DocumentInput(
                    doc_id=task["doc_id"],
                    text=text,
                    title=task.get("title") or filename,
                    metadata=metadata,
                    schema_id=task.get("schema_id"),
                )
            ],
        )
        return {"chars": len(text), "parser": converted.parser_name, "write_operation_id": write["operation_id"]}

    def _validated_fields(self, schema: dict[str, Any] | None, document: DocumentInput) -> ExtractedFields:
        """The field values this write supplied, coerced against the schema."""
        definition = schema or {"document_fields": {}, "passage_fields": {}}
        try:
            return ExtractedFields(
                document=validate_values(definition["document_fields"], document.fields or {}, level="document"),
                passages={
                    index: validate_values(definition["passage_fields"], values, level=f"passage {index}")
                    for index, values in (document.passage_fields or {}).items()
                },
            )
        except SchemaError as e:
            raise KnowledgeBankError(400, str(e)) from e

    async def run_write_batch(self, task: dict[str, Any]) -> dict[str, Any]:
        """Worker side of a write batch: passage, embed and store each document."""
        bank_id = task["bank_id"]
        documents = [
            DocumentInput(
                **{k: d.get(k) for k in ("doc_id", "text", "title", "metadata", "fields", "schema_id")},
                passage_fields={int(k): v for k, v in (d.get("passage_fields") or {}).items()},
            )
            for d in task["documents"]
        ]
        extract = bool(task.get("extract", True))
        config = await self._config(bank_id, None)
        passage_size = int(config.kb_passage_size)
        overlap = min(int(config.kb_passage_overlap), max(passage_size - 1, 0))
        pool = await self._pool()
        # The schema is read once for the batch: extraction is per document, but what to
        # extract is a property of the bank.
        # Each document may name its own schema, so the resolution is per document; the
        # lookups are cached for the batch because a batch is usually one kind of document.
        schemas: dict[str | None, dict[str, Any] | None] = {}

        async def schema_for(document: DocumentInput) -> dict[str, Any] | None:
            if not config.kb_field_extraction:
                return None
            if document.schema_id is not None:
                if document.schema_id not in schemas:
                    schemas[document.schema_id] = await self._schema_for_write(bank_id, document.schema_id)
                return schemas[document.schema_id]
            # Unnamed: classification reads the document, so the answer is per document
            # rather than per id and cannot come from the cache.
            return await self._schema_for_write(bank_id, None, document=document, config=config)

        written = skipped = passage_total = 0
        pending_documents: list[tuple[DocumentInput, str, list[Any]]] = []
        for start in range(0, len(documents), _EMBED_BATCH_DOCUMENTS):
            window = documents[start : start + _EMBED_BATCH_DOCUMENTS]
            async with acquire_with_retry(pool) as conn:
                hashes = await store.existing_hashes(conn, bank_id, [d.doc_id for d in window])
            pending = []
            for document in window:
                content_hash = hashlib.sha256(f"{document.text}\x00{passage_size}\x00{overlap}".encode()).hexdigest()
                if hashes.get(document.doc_id) == content_hash:
                    skipped += 1
                    continue
                pending.append(
                    (
                        document,
                        content_hash,
                        split_into_passages(document.text, passage_size=passage_size, passage_overlap=overlap),
                    )
                )
            if not pending:
                continue
            pending_documents.extend(pending)
            # Fields are resolved *before* the embedding, not after: a field marked
            # ``indexed`` becomes part of the text that gets embedded, so the passage is
            # findable by a value that its own words never say ("region: emea").
            extracted: dict[str, ExtractedFields] = {}
            document_schemas: dict[str, dict[str, Any] | None] = {}
            for document, _, passages in pending:
                schema = await schema_for(document)
                document_schemas[document.doc_id] = schema
                extracted[document.doc_id] = await self._fields_for_document(
                    bank_id=bank_id,
                    config=config,
                    schema=schema,
                    document=document,
                    passage_texts=[(passage.index, passage.text) for passage in passages],
                    extract=extract,
                )

            # Embed the title with the passage. A paragraph usually names its subject once,
            # in the title, and a passage without it is unfindable by that name — unless the
            # passage already opens with it, in which case repeating it only dilutes the
            # embedding (worth 12 nDCG@10 points on BEIR ArguAna, whose bodies restate
            # their title).
            texts = [
                embedding_text(
                    passage_heading(document.title, passage.section),
                    passage.text,
                    indexed_values(
                        document_schemas.get(document.doc_id),
                        extracted[document.doc_id].document,
                        extracted[document.doc_id].passages.get(passage.index, {}),
                    ),
                )
                for document, _, passages in pending
                for passage in passages
            ]
            vectors = await self.memory.embeddings.encode_documents(texts) if texts else []
            offset = 0
            async with acquire_with_retry(pool) as conn:
                async with conn.transaction():
                    for document, content_hash, passages in pending:
                        rows = [
                            (
                                bank_id,
                                document.doc_id,
                                passage.index,
                                passage.text,
                                passage_heading(document.title, passage.section),
                                passage.token_count,
                                store.vector_literal(vectors[offset + i]),
                                None,
                            )
                            for i, passage in enumerate(passages)
                        ]
                        offset += len(passages)
                        await store.upsert_document(
                            conn,
                            bank_id,
                            document.doc_id,
                            text=document.text,
                            title=document.title,
                            metadata=document.metadata or {},
                            content_hash=content_hash,
                            passage_count=len(passages),
                            fields=(extracted[document.doc_id].document if document.doc_id in extracted else {}),
                            schema_id=document.schema_id
                            or (document_schemas.get(document.doc_id) or {}).get("schema_id"),
                        )
                        await store.replace_passages(conn, bank_id, document.doc_id, rows)
                        await store.set_passage_fields(
                            conn,
                            bank_id,
                            document.doc_id,
                            extracted[document.doc_id].passages if document.doc_id in extracted else {},
                        )
                        written += 1
                        passage_total += len(passages)
        logger.info(
            "knowledge write batch bank=%s written=%d unchanged=%d passages=%d",
            bank_id,
            written,
            skipped,
            passage_total,
        )
        # Collections that derive on write see the documents this batch just stored, so a
        # record is up to date with the corpus without anyone running a job. `replace`,
        # because a rewritten document must not contribute twice.
        if written:
            async with acquire_with_retry(pool) as conn:
                auto = [
                    collection
                    for collection in await collections_store.list_collections(conn, bank_id)
                    if collection.get("derive_on_write")
                ]
            for collection in auto:
                await self.run_derive_records(
                    {
                        "bank_id": bank_id,
                        "collection_id": collection["collection_id"],
                        "doc_ids": [d.doc_id for d, _, _ in pending_documents],
                        "replace": True,
                    }
                )

        counts = {"documents_written": written, "documents_unchanged": skipped, "passages": passage_total}
        operation_id = task.get("operation_id")
        if operation_id:
            # Merge the counts into the operation row, as the retain handlers do, so the
            # caller can see what a finished write actually did.
            async with acquire_with_retry(pool) as conn:
                await conn.execute(
                    f"UPDATE {store.fq_table('async_operations')} "
                    "SET result_metadata = COALESCE(result_metadata, '{}'::jsonb) || $1::jsonb "
                    "WHERE operation_id = $2",
                    json.dumps(counts),
                    uuid.UUID(operation_id),
                )
        return counts

    # ---- reading

    async def list_documents(
        self,
        bank_id: str,
        limit: int,
        offset: int,
        query: str | None,
        schema_id: str | None = None,
        fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        await self._require_bank(bank_id)
        async with acquire_with_retry(await self._pool()) as conn:
            return await store.list_documents(conn, bank_id, limit, offset, query, schema_id, fields)

    async def passage_map(self, bank_id: str, limit: int) -> dict[str, Any]:
        """The bank's passages as points, for the map view."""
        await self._require_bank(bank_id)
        async with acquire_with_retry(await self._pool()) as conn:
            items = await store.passage_map(conn, bank_id, limit)
            total = await conn.fetchval(
                f"SELECT count(*) FROM {store.fq_table('kb_passages')} WHERE bank_id = $1", bank_id
            )
        return {"items": items, "total": total, "limit": limit}

    async def passage_counts(self, bank_id: str, doc_ids: list[str]) -> dict[str, int]:
        """How many passages each document has — so a reader of one knows there is more."""
        async with acquire_with_retry(await self._pool()) as conn:
            return await store.passage_counts(conn, bank_id, doc_ids)

    async def get_document(self, bank_id: str, doc_id: str) -> dict[str, Any]:
        await self._require_bank(bank_id)
        async with acquire_with_retry(await self._pool()) as conn:
            document = await store.get_document(conn, bank_id, doc_id)
        if document is None:
            raise KnowledgeBankError(404, f"document {doc_id!r} not found")
        return document

    async def delete_document(self, bank_id: str, doc_id: str) -> dict[str, Any]:
        await self._require_bank(bank_id)
        async with acquire_with_retry(await self._pool()) as conn:
            if not await store.delete_document(conn, bank_id, doc_id):
                raise KnowledgeBankError(404, f"document {doc_id!r} not found")
            # Records are derived from documents: dropping this document's contributions
            # and rebuilding what they fed removes a record nothing else was behind, and
            # leaves one with other sources standing, minus what this document said.
            touched = await collections_store.records_touched_by(conn, bank_id, doc_id)
            await collections_store.drop_contributions_of(conn, bank_id, doc_id)
            for collection_id, record_id in touched:
                await collections_store.materialize(conn, bank_id, collection_id, record_id)
        return {"doc_id": doc_id, "deleted": True}

    async def search(
        self,
        bank_id: str,
        query: str,
        *,
        top_k: int,
        mode: str = "hybrid",
        fields: dict[str, Any] | None = None,
        schema_id: str | None = None,
        rerank: bool | None = None,
        collapse_documents: bool = False,
        request_context: Any = None,
    ) -> list[SearchHit]:
        await self._require_bank(bank_id)
        config = await self._config(bank_id, request_context)
        candidates = max(int(config.kb_search_candidates), top_k)
        use_rerank = config.kb_search_rerank if rerank is None else rerank

        if fields:
            await self._check_filterable(bank_id, fields)
        pool = await self._pool()
        try:
            async with acquire_with_retry(pool) as conn:
                arms: dict[str, list[store.PassageHit]] = {}
                if mode in ("hybrid", "vector"):
                    [vector] = await self.memory.embeddings.encode_query([query])
                    arms["vector"] = await store.search_semantic(conn, bank_id, vector, candidates, fields, schema_id)
                if mode in ("hybrid", "keyword"):
                    arms["keyword"] = await store.search_keyword(
                        conn,
                        bank_id,
                        query_terms(query),
                        candidates,
                        fields,
                        schema_id,
                        bm25=await self._has_bm25(conn),
                    )
                # A query that names a document ("the agreement between X and Y") is also
                # searched inside that document. Across a corpus of near-identical NDAs the
                # global arms rank on the clause and cannot tell the agreements apart, and
                # in a 300-page merger agreement the right clause can sit below the
                # candidate cut; the routed arms put that document's best passages in the
                # pool either way, and the fusion and the reranker decide from there.
                routed = (
                    await store.documents_named_in(
                        conn, bank_id, query, float(config.kb_search_title_routing_similarity)
                    )
                    if config.kb_search_title_routing
                    else []
                )
                if routed:
                    if "vector" in arms:
                        arms["routed_vector"] = await store.search_semantic(
                            conn, bank_id, vector, candidates, fields, schema_id, routed
                        )
                    if "keyword" in arms:
                        arms["routed_keyword"] = await store.search_keyword(
                            conn, bank_id, query_terms(query), candidates, fields, schema_id, routed
                        )
        except FilterError as e:
            # A filter the caller cannot have meant is a 400, not a 500.
            raise KnowledgeBankError(400, str(e)) from e

        # Weighted reciprocal rank fusion, k=60 — the same constant recall uses. The weight
        # is what stops a long query's keyword arm from out-voting the vector arm: every
        # word of the query is OR-ed, so a paragraph-length query matches on topic alone.
        vector_weight = min(max(float(config.kb_search_vector_weight), 0.0), 1.0)
        weights = {
            "vector": vector_weight,
            "keyword": 1.0 - vector_weight,
            "routed_vector": vector_weight,
            "routed_keyword": 1.0 - vector_weight,
        }
        scores: dict[tuple[str, int], float] = {}
        ranks: dict[tuple[str, int], dict[str, int]] = {}
        texts: dict[tuple[str, int], str] = {}
        for arm, hits in arms.items():
            # A single-arm search must not be scaled down by a weight meant for the mix.
            weight = weights[arm] if len(arms) > 1 else 1.0
            for hit in hits:
                key = (hit.doc_id, hit.passage_index)
                scores[key] = scores.get(key, 0.0) + weight / (60 + hit.rank)
                ranks.setdefault(key, {})[arm] = hit.rank
                texts[key] = hit.text
        ordered = sorted(scores, key=lambda key: -scores[key])
        final_scores: dict[tuple[str, int], float] = {key: round(scores[key], 6) for key in ordered}
        pool = ordered[:candidates] if use_rerank else ordered

        if use_rerank and pool:
            # The reranker reads the document's title with each passage, as the embedding
            # does (see embedding_text): a clause is only relevant in the agreement the
            # query names, and the clause alone rarely says which agreement it is in.
            async with acquire_with_retry(await self._pool()) as conn:
                titles = await store.document_titles(conn, bank_id, sorted({key[0] for key in pool}))
            rerank_texts = {
                key: embedding_text(titles.get(key[0]), texts[key]) if titles.get(key[0]) else texts[key]
                for key in pool
            }
            reranked = await self._rerank(query, pool, rerank_texts)
            if reranked is not None:
                pool = [key for key, _ in reranked]
                final_scores = dict(reranked)

        if collapse_documents:
            # One passage per document: the caller asked for k documents, not k passages.
            # Without this, a document with two good passages costs a slot another document
            # could have filled.
            seen: set[str] = set()
            collapsed: list[tuple[str, int]] = []
            for key in pool:
                if key[0] in seen:
                    continue
                seen.add(key[0])
                collapsed.append(key)
            pool = collapsed

        return [SearchHit(key[0], key[1], texts[key], final_scores[key], ranks[key]) for key in pool[:top_k]]

    async def _rerank(
        self, query: str, keys: list[tuple[str, int]], texts: dict[tuple[str, int], str]
    ) -> list[tuple[tuple[str, int], float]] | None:
        """Score the fused candidates with the configured cross-encoder.

        Calls the model directly rather than through ``CrossEncoderReranker``: that wrapper
        also applies recency and temporal scoring, which are memory-fact notions a document
        passage does not have.
        """
        reranker = getattr(self.memory, "_cross_encoder_reranker", None)
        model = getattr(reranker, "cross_encoder", None) if reranker is not None else None
        if model is None:
            return None
        try:
            if hasattr(model, "initialize"):
                await model.initialize()
            scores = await model.predict([(query, texts[key]) for key in keys])
        except Exception as e:  # a reranker outage must not fail the search
            logger.warning("knowledge search: rerank failed, keeping fusion order (%s)", e)
            return None
        if not scores or len(scores) != len(keys):
            return None
        ordered = sorted(zip(keys, (float(s) for s in scores)), key=lambda pair: -pair[1])
        return ordered

    # ---- moving a bank

    async def submit_export(self, bank_id: str, *, scope: KnowledgeTransferScope) -> dict[str, Any]:
        """Queue an export. Building and compressing an archive is not request work."""
        await self._require_bank(bank_id)
        if not (scope.data or scope.config):
            raise KnowledgeBankError(400, "nothing to export: set include_data, include_config, or both")
        result = await self.memory._submit_async_operation(
            bank_id=bank_id,
            operation_type="knowledge_export",
            task_type="knowledge_export",
            task_payload={"include_data": scope.data, "include_config": scope.config},
        )
        return {"operation_id": result["operation_id"], "status": "pending"}

    async def run_export(self, task: dict[str, Any]) -> dict[str, Any]:
        """Worker side: stream the archive into file storage, hand back a download url."""
        from .transfer import stream_export

        bank_id = task["bank_id"]
        scope = KnowledgeTransferScope(
            data=bool(task.get("include_data", True)), config=bool(task.get("include_config", True))
        )
        from ..engine.memory_engine import ByteStreamCounter

        stream = stream_export(await self._pool(), bank_id, scope=scope)
        counter = ByteStreamCounter(stream)
        storage_key = f"banks/{bank_id}/exports/{uuid.uuid4()}/knowledge.zip"
        await self.memory._file_storage.store_stream(
            key=storage_key,
            stream=counter,
            metadata={"content_type": "application/zip", "bank_id": bank_id},
        )
        result = {
            "storage_key": storage_key,
            "download_url": await self.memory._file_storage.get_download_url(storage_key),
            "byte_size": counter.total_bytes,
            "filename": f"{bank_id}-knowledge.zip",
        }
        await self._record_result(task.get("operation_id"), result)
        return result

    async def submit_import(
        self,
        bank_id: str,
        archive_bytes: bytes,
        *,
        target_bank_id: str | None,
        mode: str,
        scope: KnowledgeTransferScope,
    ) -> dict[str, Any]:
        """Queue an import of this archive. The archive is parsed here, not in the worker.

        ``bank_id`` is the bank the operation is recorded against — ``async_operations``
        has a foreign key to ``banks``, and in restore mode the target does not exist yet.
        """
        from .transfer import KnowledgeArchiveError, parse_archive

        await self._require_bank(bank_id)
        if mode not in ("restore", "merge"):
            raise KnowledgeBankError(400, f"invalid mode {mode!r} (expected restore|merge)")
        try:
            archive = parse_archive(archive_bytes)
        except KnowledgeArchiveError as e:
            raise KnowledgeBankError(400, str(e)) from e
        target = target_bank_id or (bank_id if mode == "merge" else archive.source_bank_id)
        async with acquire_with_retry(await self._pool()) as conn:
            kind = await store.bank_kind(conn, target)
        if mode == "restore" and kind is not None:
            raise KnowledgeBankError(
                409,
                f"bank {target!r} already exists; a restore writes into a fresh bank. "
                f"Use mode=merge to fold this archive into it, or choose another target.",
            )
        if mode == "merge" and kind != store.KNOWLEDGE_KIND:
            raise KnowledgeBankError(404, f"knowledge bank {target!r} not found")

        storage_key = f"banks/{bank_id}/imports/{uuid.uuid4()}/knowledge.zip"
        await self.memory._file_storage.store(
            file_data=archive_bytes,
            key=storage_key,
            metadata={"content_type": "application/zip", "bank_id": bank_id},
        )
        result = await self.memory._submit_async_operation(
            bank_id=bank_id,
            operation_type="knowledge_import",
            task_type="knowledge_import",
            task_payload={
                "storage_key": storage_key,
                "target_bank_id": target,
                "mode": mode,
                # An archive cannot be asked for what it does not carry, so the flags
                # narrow what is restored rather than adding to it.
                "include_data": scope.data and archive.has_data,
                "include_config": scope.config and archive.has_config,
            },
            result_metadata={"target_bank_id": target, "documents": archive.document_count},
        )
        return {"operation_id": result["operation_id"], "target_bank_id": target, "status": "pending"}

    async def run_import(self, task: dict[str, Any]) -> dict[str, Any]:
        """Worker side: create the target if needed, restore config, then data.

        Config first, always: a document names the schema that gives its fields meaning,
        and a record belongs to a collection that has to exist before it can be written.
        """
        from .transfer import parse_archive

        target = task["target_bank_id"]
        archive = parse_archive(await self.memory._file_storage.retrieve(task["storage_key"]))
        counts = {"schemas": 0, "collections": 0, "documents": 0, "records": 0}

        async with acquire_with_retry(await self._pool()) as conn:
            if await store.bank_kind(conn, target) is None:
                await store.create_bank(
                    conn, target, archive.name or target, json.dumps(DEFAULT_DISPOSITION), uuid.uuid4()
                )

        if task.get("include_config"):
            for schema in archive.schemas:
                await self.put_schema(
                    target,
                    schema["schema_id"],
                    name=schema.get("name"),
                    description=schema.get("description"),
                    document_fields=schema.get("document_fields") or {},
                    passage_fields=schema.get("passage_fields") or {},
                )
                counts["schemas"] += 1
            # Two passes: a relationship field names another collection, and
            # put_collection refuses one that does not exist yet. The first pass writes
            # every definition without its relationships, the second puts them back.
            plain = [
                {**c, "fields": {k: v for k, v in (c.get("fields") or {}).items() if not v.get("collection")}}
                for c in archive.collections
            ]
            for definition in plain + archive.collections:
                await self.put_collection(
                    target,
                    definition["collection_id"],
                    name=definition.get("name"),
                    description=definition.get("description"),
                    fields=definition.get("fields") or {},
                    identity=definition.get("identity"),
                    derive_on_write=bool(definition.get("derive_on_write")),
                )
            counts["collections"] = len(archive.collections)

        if task.get("include_data"):
            documents = archive.documents()
            for start in range(0, len(documents), MAX_BATCH_DOCUMENTS):
                window = documents[start : start + MAX_BATCH_DOCUMENTS]
                # extract=False: the archive carries the fields the source extracted, so
                # an import costs no tokens and cannot invent a value the source never had.
                written = await self.run_write_batch({"bank_id": target, "documents": window, "extract": False})
                counts["documents"] += written["documents_written"]
            counts["records"] = await self._restore_records(target, archive)

        await self._record_result(task.get("operation_id"), counts)
        try:
            await self.memory._file_storage.delete(task["storage_key"])
        except Exception:
            logger.warning("knowledge import: failed to delete %s", task["storage_key"], exc_info=True)
        logger.info("knowledge import bank=%s %s", target, counts)
        return counts

    async def _restore_records(self, target: str, archive: Any) -> int:
        """Replay each record's contributions and fold them back into a record.

        A record is written as the sum of what each document said about it, so restoring
        the parts and re-materializing rebuilds the same row — including its evidence —
        without trusting a folded copy that could disagree with them.
        """
        touched: set[tuple[str, str]] = set()
        async with acquire_with_retry(await self._pool()) as conn:
            for row in archive.contributions:
                await collections_store.contribute(
                    conn,
                    target,
                    row["collection_id"],
                    row["record_id"],
                    doc_id=row.get("doc_id") or "",
                    values=row.get("values") or {},
                    evidence=row.get("evidence") or {},
                )
                touched.add((row["collection_id"], row["record_id"]))
            for row in archive.pinned:
                await collections_store.pin_values(
                    conn, target, row["collection_id"], row["record_id"], row.get("pinned") or {}
                )
                touched.add((row["collection_id"], row["record_id"]))
            for collection_id, record_id in sorted(touched):
                await collections_store.materialize(conn, target, collection_id, record_id)
            for row in archive.aliases:
                await collections_store.add_alias(
                    conn,
                    target,
                    row["collection_id"],
                    row["alias_key"],
                    row["record_id"],
                    source=row.get("source") or "merge",
                )
        return len(touched)

    async def _record_result(self, operation_id: str | None, result: dict[str, Any]) -> None:
        """Merge a finished transfer's counts into its operation row."""
        if not operation_id:
            return
        async with acquire_with_retry(await self._pool()) as conn:
            await conn.execute(
                f"UPDATE {store.fq_table('async_operations')} "
                "SET result_metadata = COALESCE(result_metadata, '{}'::jsonb) || $1::jsonb "
                "WHERE operation_id = $2",
                json.dumps(result),
                uuid.UUID(operation_id),
            )


async def run_write_batch_task(memory: Any, task: dict[str, Any]) -> dict[str, Any]:
    """Entry point the engine's task dispatch calls for ``knowledge_write_batch``."""
    return await KnowledgeService(memory).run_write_batch(task)


async def run_extract_fields_task(memory: Any, task: dict[str, Any]) -> dict[str, Any]:
    """Entry point the engine's task dispatch calls for ``knowledge_extract_fields``."""
    return await KnowledgeService(memory).run_extract_fields(task)


async def run_file_convert_task(memory: Any, task: dict[str, Any]) -> dict[str, Any]:
    """Entry point the engine's task dispatch calls for ``knowledge_file_convert``."""
    return await KnowledgeService(memory).run_file_convert(task)


async def run_derive_records_task(memory: Any, task: dict[str, Any]) -> dict[str, Any]:
    """Entry point the engine's task dispatch calls for ``knowledge_derive_records``."""
    return await KnowledgeService(memory).run_derive_records(task)


async def run_export_task(memory: Any, task: dict[str, Any]) -> dict[str, Any]:
    """Entry point the engine's task dispatch calls for ``knowledge_export``."""
    return await KnowledgeService(memory).run_export(task)


async def run_import_task(memory: Any, task: dict[str, Any]) -> dict[str, Any]:
    """Entry point the engine's task dispatch calls for ``knowledge_import``."""
    return await KnowledgeService(memory).run_import(task)


__all__ = [
    "DocumentInput",
    "KnowledgeBankError",
    "KnowledgeService",
    "SearchHit",
    "run_derive_records_task",
    "run_export_task",
    "run_extract_fields_task",
    "run_import_task",
    "run_write_batch_task",
]
