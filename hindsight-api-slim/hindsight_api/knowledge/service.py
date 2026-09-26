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

import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any

from ..engine.db_utils import acquire_with_retry
from ..engine.retain.bank_utils import DEFAULT_DISPOSITION
from . import store
from .extraction import extract_document, extract_passages
from .fields import SchemaError, extract_only, validate_field_schema, validate_values
from .filters import FilterError
from .passages import split_into_passages
from .query import CompiledQuery, Limits, compile_query, json_safe

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
    tags: list[str] | None = None
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


def embedding_text(title: str | None, passage_text: str) -> str:
    """What gets embedded for a passage: the title, then the passage — unless it is already there."""
    if not title:
        return passage_text
    if passage_text.lstrip().lower().startswith(title.strip().lower()):
        return passage_text
    return f"{title}\n\n{passage_text}"


class KnowledgeService:
    """Knowledge-bank operations. Holds the engine for its pool, embeddings and reranker."""

    def __init__(self, memory: Any) -> None:
        self.memory = memory

    async def _pool(self) -> Any:
        return await self.memory._get_pool()

    async def _config(self, bank_id: str, request_context: Any) -> Any:
        return await self.memory._config_resolver.resolve_full_config(bank_id, request_context)

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

    async def list_banks(self, limit: int, offset: int, query: str | None) -> dict[str, Any]:
        async with acquire_with_retry(await self._pool()) as conn:
            return await store.list_banks(conn, limit, offset, query)

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

    async def _schema_for_write(self, bank_id: str, schema_id: str | None) -> dict[str, Any] | None:
        """Which schema fills a document's fields.

        Named explicitly, or — when the bank has exactly one — that one, because a bank
        with a single schema should not make every write repeat its name. A bank with
        several and no name on the write extracts nothing rather than guessing.
        """
        async with acquire_with_retry(await self._pool()) as conn:
            if schema_id is not None:
                schema = await store.get_schema(conn, bank_id, schema_id)
                if schema is None:
                    raise KnowledgeBankError(400, f"schema {schema_id!r} not found in this bank")
                return schema
            schemas = await store.list_schemas(conn, bank_id)
        return schemas[0] if len(schemas) == 1 else None

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

        if schema is None:
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
                    "text": document.text,
                    "title": document.title,
                    "tags": document.tags or [],
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
                **{k: d.get(k) for k in ("doc_id", "text", "title", "tags", "metadata", "fields", "schema_id")},
                passage_fields={int(k): v for k, v in (d.get("passage_fields") or {}).items()},
            )
            for d in task["documents"]
        ]
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
            if document.schema_id not in schemas:
                schemas[document.schema_id] = await self._schema_for_write(bank_id, document.schema_id)
            return schemas[document.schema_id]

        written = skipped = passage_total = 0
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
            # Embed the title with the passage. A paragraph usually names its subject once,
            # in the title, and a passage without it is unfindable by that name — unless the
            # passage already opens with it, in which case repeating it only dilutes the
            # embedding (worth 12 nDCG@10 points on BEIR ArguAna, whose bodies restate
            # their title).
            texts = [
                embedding_text(document.title, passage.text)
                for document, _, passages in pending
                for passage in passages
            ]
            vectors = await self.memory.embeddings.encode_documents(texts) if texts else []
            extracted: dict[str, ExtractedFields] = {}
            for document, _, passages in pending:
                extracted[document.doc_id] = await self._fields_for_document(
                    bank_id=bank_id,
                    config=config,
                    schema=await schema_for(document),
                    document=document,
                    passage_texts=[(passage.index, passage.text) for passage in passages],
                )
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
                                document.title,
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
                            tags=document.tags or [],
                            metadata=document.metadata or {},
                            content_hash=content_hash,
                            passage_count=len(passages),
                            fields=(extracted[document.doc_id].document if document.doc_id in extracted else {}),
                            schema_id=document.schema_id or (await schema_for(document) or {}).get("schema_id"),
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

    async def list_documents(self, bank_id: str, limit: int, offset: int, query: str | None) -> dict[str, Any]:
        await self._require_bank(bank_id)
        async with acquire_with_retry(await self._pool()) as conn:
            return await store.list_documents(conn, bank_id, limit, offset, query)

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
        return {"doc_id": doc_id, "deleted": True}

    async def search(
        self,
        bank_id: str,
        query: str,
        *,
        top_k: int,
        mode: str = "hybrid",
        tags: list[str] | None = None,
        fields: dict[str, Any] | None = None,
        rerank: bool | None = None,
        collapse_documents: bool = False,
        request_context: Any = None,
    ) -> list[SearchHit]:
        await self._require_bank(bank_id)
        config = await self._config(bank_id, request_context)
        candidates = max(int(config.kb_search_candidates), top_k)
        use_rerank = config.kb_search_rerank if rerank is None else rerank

        pool = await self._pool()
        try:
            async with acquire_with_retry(pool) as conn:
                arms: dict[str, list[store.PassageHit]] = {}
                if mode in ("hybrid", "vector"):
                    [vector] = await self.memory.embeddings.encode_query([query])
                    arms["vector"] = await store.search_semantic(conn, bank_id, vector, candidates, tags, fields)
                if mode in ("hybrid", "keyword"):
                    arms["keyword"] = await store.search_keyword(
                        conn, bank_id, query_terms(query), candidates, tags, fields
                    )
        except FilterError as e:
            # A filter the caller cannot have meant is a 400, not a 500.
            raise KnowledgeBankError(400, str(e)) from e

        # Weighted reciprocal rank fusion, k=60 — the same constant recall uses. The weight
        # is what stops a long query's keyword arm from out-voting the vector arm: every
        # word of the query is OR-ed, so a paragraph-length query matches on topic alone.
        vector_weight = min(max(float(config.kb_search_vector_weight), 0.0), 1.0)
        weights = {"vector": vector_weight, "keyword": 1.0 - vector_weight}
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
            reranked = await self._rerank(query, pool, texts)
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


async def run_write_batch_task(memory: Any, task: dict[str, Any]) -> dict[str, Any]:
    """Entry point the engine's task dispatch calls for ``knowledge_write_batch``."""
    return await KnowledgeService(memory).run_write_batch(task)


async def run_extract_fields_task(memory: Any, task: dict[str, Any]) -> dict[str, Any]:
    """Entry point the engine's task dispatch calls for ``knowledge_extract_fields``."""
    return await KnowledgeService(memory).run_extract_fields(task)


__all__ = [
    "DocumentInput",
    "KnowledgeBankError",
    "KnowledgeService",
    "SearchHit",
    "run_extract_fields_task",
    "run_write_batch_task",
]
