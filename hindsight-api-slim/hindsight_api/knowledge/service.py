"""Knowledge banks: write documents (always async, in batches) and search them.

Writes never happen in the request. A write submits one ``knowledge_write_batch``
operation — the same ``async_operations`` row, worker claim, retry and status API the
memory banks use — and returns its id. The worker chunks, embeds and stores.

Search is two arms fused with RRF: vector over the chunk embeddings and keyword over the
generated tsvector, then the configured cross-encoder reranks the fused candidates.
Everything (chunk size, overlap, candidate count, rerank on/off) is per-bank config,
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
from .chunking import chunk_document

logger = logging.getLogger(__name__)

#: How many documents of one batch are embedded at a time.
_EMBED_BATCH_DOCUMENTS = 8
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


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    chunk_index: int
    text: str
    score: float
    ranks: dict[str, int]


def query_terms(query: str) -> list[str]:
    """Words worth searching for, from a natural-language question."""
    return [t for t in re.findall(r"\w+", query.lower()) if len(t) > 1][:32]


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
            # The bank row owns everything by FK cascade: documents, chunks, operations.
            await conn.execute(f"DELETE FROM {store.fq_table('banks')} WHERE bank_id = $1", bank_id)
        return {"bank_id": bank_id, "deleted": True}

    async def _require_bank(self, bank_id: str) -> None:
        async with acquire_with_retry(await self._pool()) as conn:
            kind = await store.bank_kind(conn, bank_id)
        if kind is None:
            raise KnowledgeBankError(404, f"knowledge bank {bank_id!r} not found")
        if kind != store.KNOWLEDGE_KIND:
            raise KnowledgeBankError(409, f"bank {bank_id!r} is a {kind} bank")

    # ---- writing (always async, always a batch)

    async def submit_write(self, bank_id: str, documents: list[DocumentInput]) -> dict[str, Any]:
        """Queue one ``knowledge_write_batch`` operation for these documents."""
        await self._require_bank(bank_id)
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

    async def run_write_batch(self, task: dict[str, Any]) -> dict[str, Any]:
        """Worker side of a write batch: chunk, embed and store each document."""
        bank_id = task["bank_id"]
        documents = [DocumentInput(**{k: d.get(k) for k in ("doc_id", "text", "title", "tags", "metadata")})
                     for d in task["documents"]]  # fmt: skip
        config = await self._config(bank_id, None)
        chunk_size = int(config.kb_chunk_size)
        overlap = min(int(config.kb_chunk_overlap), max(chunk_size - 1, 0))
        pool = await self._pool()

        written = skipped = chunk_total = 0
        for start in range(0, len(documents), _EMBED_BATCH_DOCUMENTS):
            window = documents[start : start + _EMBED_BATCH_DOCUMENTS]
            async with acquire_with_retry(pool) as conn:
                hashes = await store.existing_hashes(conn, bank_id, [d.doc_id for d in window])
            pending = []
            for document in window:
                content_hash = hashlib.sha256(f"{document.text}\x00{chunk_size}\x00{overlap}".encode()).hexdigest()
                if hashes.get(document.doc_id) == content_hash:
                    skipped += 1
                    continue
                pending.append(
                    (
                        document,
                        content_hash,
                        chunk_document(document.text, chunk_size=chunk_size, chunk_overlap=overlap),
                    )
                )
            if not pending:
                continue
            texts = [chunk.text for _, _, chunks in pending for chunk in chunks]
            vectors = await self.memory.embeddings.encode_documents(texts) if texts else []
            offset = 0
            async with acquire_with_retry(pool) as conn:
                async with conn.transaction():
                    for document, content_hash, chunks in pending:
                        rows = [
                            (
                                bank_id,
                                document.doc_id,
                                chunk.index,
                                chunk.text,
                                chunk.token_count,
                                store.vector_literal(vectors[offset + i]),
                            )
                            for i, chunk in enumerate(chunks)
                        ]
                        offset += len(chunks)
                        await store.upsert_document(
                            conn,
                            bank_id,
                            document.doc_id,
                            text=document.text,
                            title=document.title,
                            tags=document.tags or [],
                            metadata=document.metadata or {},
                            content_hash=content_hash,
                            chunk_count=len(chunks),
                        )
                        await store.replace_chunks(conn, bank_id, document.doc_id, rows)
                        written += 1
                        chunk_total += len(chunks)
        logger.info(
            "knowledge write batch bank=%s written=%d unchanged=%d chunks=%d", bank_id, written, skipped, chunk_total
        )
        counts = {"documents_written": written, "documents_unchanged": skipped, "chunks": chunk_total}
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
        rerank: bool | None = None,
        request_context: Any = None,
    ) -> list[SearchHit]:
        await self._require_bank(bank_id)
        config = await self._config(bank_id, request_context)
        candidates = max(int(config.kb_search_candidates), top_k)
        use_rerank = config.kb_search_rerank if rerank is None else rerank

        pool = await self._pool()
        async with acquire_with_retry(pool) as conn:
            arms: dict[str, list[store.ChunkHit]] = {}
            if mode in ("hybrid", "vector"):
                [vector] = await self.memory.embeddings.encode_query([query])
                arms["vector"] = await store.search_semantic(conn, bank_id, vector, candidates, tags)
            if mode in ("hybrid", "keyword"):
                arms["keyword"] = await store.search_keyword(conn, bank_id, query_terms(query), candidates, tags)

        # Reciprocal rank fusion, k=60 — the same constant recall uses.
        scores: dict[tuple[str, int], float] = {}
        ranks: dict[tuple[str, int], dict[str, int]] = {}
        texts: dict[tuple[str, int], str] = {}
        for arm, hits in arms.items():
            for hit in hits:
                key = (hit.doc_id, hit.chunk_index)
                scores[key] = scores.get(key, 0.0) + 1.0 / (60 + hit.rank)
                ranks.setdefault(key, {})[arm] = hit.rank
                texts[key] = hit.text
        fused = sorted(scores, key=lambda key: -scores[key])[: candidates if use_rerank else top_k]

        if use_rerank and fused:
            reranked = await self._rerank(query, fused, texts)
            if reranked is not None:
                fused = reranked[:top_k]
                return [
                    SearchHit(doc_id, index, texts[(doc_id, index)], score, ranks[(doc_id, index)])
                    for (doc_id, index), score in fused
                ]
        return [SearchHit(key[0], key[1], texts[key], round(scores[key], 6), ranks[key]) for key in fused[:top_k]]

    async def _rerank(
        self, query: str, keys: list[tuple[str, int]], texts: dict[tuple[str, int], str]
    ) -> list[tuple[tuple[str, int], float]] | None:
        """Score the fused candidates with the configured cross-encoder.

        Calls the model directly rather than through ``CrossEncoderReranker``: that wrapper
        also applies recency and temporal scoring, which are memory-fact notions a document
        chunk does not have.
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


__all__ = [
    "DocumentInput",
    "KnowledgeBankError",
    "KnowledgeService",
    "SearchHit",
    "run_write_batch_task",
]
