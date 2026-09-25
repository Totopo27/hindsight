"""SQL for knowledge banks: banks, documents, chunks and the two search arms.

Every statement is scoped by ``bank_id`` and schema-qualified with ``fq_table`` so it
lands in the caller's tenant schema, exactly like the memory tables.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..engine.schema import fq_table

KNOWLEDGE_KIND = "knowledge"


@dataclass(frozen=True)
class DocumentRow:
    doc_id: str
    title: str | None
    tags: list[str]
    metadata: dict[str, Any]
    chunk_count: int
    chars: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class ChunkHit:
    doc_id: str
    chunk_index: int
    text: str
    rank: int


def vector_literal(vector: list[float]) -> str:
    """pgvector's text input form. asyncpg has no vector codec, so values go as text."""
    return "[" + ",".join(f"{value:.7f}" for value in vector) + "]"


async def create_bank(conn: Any, bank_id: str, name: str, disposition: str, internal_id: Any) -> bool:
    """Insert the bank row. Returns False when a bank with that id already exists."""
    inserted = await conn.fetchval(
        f"""
        INSERT INTO {fq_table("banks")} (bank_id, name, disposition, mission, internal_id, kind)
        VALUES ($1, $2, $3::jsonb, '', $4, '{KNOWLEDGE_KIND}')
        ON CONFLICT (bank_id) DO NOTHING
        RETURNING bank_id
        """,
        bank_id,
        name,
        disposition,
        internal_id,
    )
    return inserted is not None


async def bank_kind(conn: Any, bank_id: str) -> str | None:
    return await conn.fetchval(f"SELECT kind FROM {fq_table('banks')} WHERE bank_id = $1", bank_id)


async def list_banks(conn: Any, limit: int, offset: int, query: str | None) -> dict[str, Any]:
    where = f"b.kind = '{KNOWLEDGE_KIND}'"
    params: list[Any] = []
    if query:
        params.append(f"%{query}%")
        where += f" AND (b.bank_id ILIKE ${len(params)} OR b.name ILIKE ${len(params)})"
    total = await conn.fetchval(f"SELECT count(*) FROM {fq_table('banks')} b WHERE {where}", *params)
    rows = await conn.fetch(
        f"""
        SELECT b.bank_id, b.name, b.created_at, b.updated_at,
               (SELECT count(*) FROM {fq_table("kb_documents")} d WHERE d.bank_id = b.bank_id) AS documents,
               (SELECT count(*) FROM {fq_table("kb_chunks")} c WHERE c.bank_id = b.bank_id) AS chunks
        FROM {fq_table("banks")} b
        WHERE {where}
        ORDER BY b.bank_id
        LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
        """,
        *params,
        limit,
        offset,
    )
    return {"items": [dict(r) for r in rows], "total": total, "limit": limit, "offset": offset}


async def bank_stats(conn: Any, bank_id: str) -> dict[str, Any] | None:
    row = await conn.fetchrow(
        f"""
        SELECT b.bank_id, b.name, b.created_at, b.updated_at,
               (SELECT count(*) FROM {fq_table("kb_documents")} d WHERE d.bank_id = b.bank_id) AS documents,
               (SELECT count(*) FROM {fq_table("kb_chunks")} c WHERE c.bank_id = b.bank_id) AS chunks,
               (SELECT max(d.updated_at) FROM {fq_table("kb_documents")} d WHERE d.bank_id = b.bank_id)
                   AS last_write_at,
               (SELECT count(*) FROM {fq_table("async_operations")} o
                WHERE o.bank_id = b.bank_id AND o.status IN ('pending', 'processing')) AS operations_in_flight
        FROM {fq_table("banks")} b
        WHERE b.bank_id = $1 AND b.kind = '{KNOWLEDGE_KIND}'
        """,
        bank_id,
    )
    return dict(row) if row else None


async def existing_hashes(conn: Any, bank_id: str, doc_ids: list[str]) -> dict[str, str]:
    rows = await conn.fetch(
        f"SELECT doc_id, content_hash FROM {fq_table('kb_documents')} WHERE bank_id = $1 AND doc_id = ANY($2::text[])",
        bank_id,
        doc_ids,
    )
    return {r["doc_id"]: r["content_hash"] for r in rows}


async def upsert_document(
    conn: Any,
    bank_id: str,
    doc_id: str,
    *,
    text: str,
    title: str | None,
    tags: list[str],
    metadata: dict[str, Any],
    content_hash: str,
    chunk_count: int,
) -> None:
    await conn.execute(
        f"""
        INSERT INTO {fq_table("kb_documents")}
            (bank_id, doc_id, text, title, tags, metadata, content_hash, chunk_count)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8)
        ON CONFLICT (bank_id, doc_id) DO UPDATE SET
            text = EXCLUDED.text, title = EXCLUDED.title, tags = EXCLUDED.tags,
            metadata = EXCLUDED.metadata, content_hash = EXCLUDED.content_hash,
            chunk_count = EXCLUDED.chunk_count, updated_at = now()
        """,
        bank_id,
        doc_id,
        text,
        title,
        tags,
        json.dumps(metadata),
        content_hash,
        chunk_count,
    )


async def replace_chunks(conn: Any, bank_id: str, doc_id: str, rows: list[tuple[Any, ...]]) -> None:
    await conn.execute(f"DELETE FROM {fq_table('kb_chunks')} WHERE bank_id = $1 AND doc_id = $2", bank_id, doc_id)
    if rows:
        await conn.executemany(
            f"INSERT INTO {fq_table('kb_chunks')} (bank_id, doc_id, chunk_index, text, token_count, embedding) "
            "VALUES ($1, $2, $3, $4, $5, $6::vector)",
            rows,
        )


async def list_documents(conn: Any, bank_id: str, limit: int, offset: int, query: str | None) -> dict[str, Any]:
    where = "bank_id = $1"
    params: list[Any] = [bank_id]
    if query:
        params.append(f"%{query}%")
        where += f" AND (doc_id ILIKE ${len(params)} OR title ILIKE ${len(params)})"
    total = await conn.fetchval(f"SELECT count(*) FROM {fq_table('kb_documents')} WHERE {where}", *params)
    rows = await conn.fetch(
        f"""
        SELECT doc_id, title, tags, metadata, chunk_count, length(text) AS chars, created_at, updated_at
        FROM {fq_table("kb_documents")}
        WHERE {where}
        ORDER BY updated_at DESC, doc_id
        LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
        """,
        *params,
        limit,
        offset,
    )
    return {"items": [dict(r) for r in rows], "total": total, "limit": limit, "offset": offset}


async def get_document(conn: Any, bank_id: str, doc_id: str) -> dict[str, Any] | None:
    row = await conn.fetchrow(
        f"""
        SELECT doc_id, title, tags, metadata, text, chunk_count, created_at, updated_at
        FROM {fq_table("kb_documents")} WHERE bank_id = $1 AND doc_id = $2
        """,
        bank_id,
        doc_id,
    )
    if row is None:
        return None
    chunks = await conn.fetch(
        f"SELECT chunk_index, text, token_count FROM {fq_table('kb_chunks')} "
        "WHERE bank_id = $1 AND doc_id = $2 ORDER BY chunk_index",
        bank_id,
        doc_id,
    )
    return {**dict(row), "chunks": [dict(c) for c in chunks]}


async def delete_document(conn: Any, bank_id: str, doc_id: str) -> bool:
    deleted = await conn.fetchval(
        f"DELETE FROM {fq_table('kb_documents')} WHERE bank_id = $1 AND doc_id = $2 RETURNING doc_id",
        bank_id,
        doc_id,
    )
    return deleted is not None


def _tag_filter(tags: list[str] | None, params: list[Any]) -> str:
    if not tags:
        return ""
    params.append(tags)
    return (
        f" AND EXISTS (SELECT 1 FROM {fq_table('kb_documents')} d "
        f"WHERE d.bank_id = c.bank_id AND d.doc_id = c.doc_id AND d.tags && ${len(params)}::text[])"
    )


async def search_semantic(
    conn: Any, bank_id: str, query_vector: list[float], limit: int, tags: list[str] | None
) -> list[ChunkHit]:
    params: list[Any] = [bank_id]
    filters = _tag_filter(tags, params)
    params.append(vector_literal(query_vector))
    rows = await conn.fetch(
        f"""
        SELECT c.doc_id, c.chunk_index, c.text
        FROM {fq_table("kb_chunks")} c
        WHERE c.bank_id = $1{filters}
        ORDER BY c.embedding <=> ${len(params)}::vector
        LIMIT {int(limit)}
        """,
        *params,
    )
    return [ChunkHit(r["doc_id"], r["chunk_index"], r["text"], i + 1) for i, r in enumerate(rows)]


async def search_keyword(
    conn: Any, bank_id: str, terms: list[str], limit: int, tags: list[str] | None
) -> list[ChunkHit]:
    """Keyword arm. Terms are OR-ed: a question rarely has every word in one chunk."""
    if not terms:
        return []
    params: list[Any] = [bank_id]
    filters = _tag_filter(tags, params)
    params.append(" | ".join(terms))
    rows = await conn.fetch(
        f"""
        SELECT c.doc_id, c.chunk_index, c.text, ts_rank_cd(c.search_vector, q) AS rank
        FROM {fq_table("kb_chunks")} c, to_tsquery('english', ${len(params)}) q
        WHERE c.bank_id = $1 AND c.search_vector @@ q{filters}
        ORDER BY rank DESC
        LIMIT {int(limit)}
        """,
        *params,
    )
    return [ChunkHit(r["doc_id"], r["chunk_index"], r["text"], i + 1) for i, r in enumerate(rows)]
