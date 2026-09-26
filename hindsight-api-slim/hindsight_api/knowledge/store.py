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
from .filters import compile_filters

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
class SearchScope:
    """The JOIN and the extra WHERE a search arm needs for its tag/metadata filters."""

    join: str
    where: str


@dataclass(frozen=True)
class ChunkHit:
    doc_id: str
    chunk_index: int
    text: str
    rank: int


def vector_literal(vector: list[float]) -> str:
    """pgvector's text input form. asyncpg has no vector codec, so values go as text."""
    return "[" + ",".join(f"{value:.7f}" for value in vector) + "]"


def _loaded(row: Any, *columns: str) -> dict[str, Any]:
    """Row as a dict with its JSONB columns parsed.

    asyncpg has no automatic JSONB codec on this pool, so a jsonb column arrives as text
    and would otherwise reach the caller as a string that looks like JSON.
    """
    out = dict(row)
    for column in columns:
        value = out.get(column)
        if isinstance(value, str):
            out[column] = json.loads(value)
    return out


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


async def get_metadata_schema(conn: Any, bank_id: str) -> dict[str, Any]:
    row = await conn.fetchrow(
        f"SELECT document_schema, chunk_schema, updated_at FROM {fq_table('kb_metadata_schemas')} WHERE bank_id = $1",
        bank_id,
    )
    if row is None:
        return {"document": {}, "chunks": {}, "updated_at": None}
    return {
        "document": json.loads(row["document_schema"])
        if isinstance(row["document_schema"], str)
        else row["document_schema"],  # fmt: skip
        "chunks": json.loads(row["chunk_schema"]) if isinstance(row["chunk_schema"], str) else row["chunk_schema"],
        "updated_at": row["updated_at"],
    }


async def put_metadata_schema(
    conn: Any, bank_id: str, document_schema: dict[str, Any], chunk_schema: dict[str, Any]
) -> None:
    await conn.execute(
        f"""
        INSERT INTO {fq_table("kb_metadata_schemas")} (bank_id, document_schema, chunk_schema)
        VALUES ($1, $2::jsonb, $3::jsonb)
        ON CONFLICT (bank_id) DO UPDATE SET
            document_schema = EXCLUDED.document_schema,
            chunk_schema = EXCLUDED.chunk_schema,
            updated_at = now()
        """,
        bank_id,
        json.dumps(document_schema),
        json.dumps(chunk_schema),
    )


async def value_counts(conn: Any, bank_id: str, property_name: str, *, level: str) -> list[dict[str, Any]]:
    """How the corpus splits per value of one property — the facet behind the schema page.

    An array property counts once per element, so "parties: [Acme, Globex]" lands in both
    buckets; a scalar counts once. The two cases are separate legs because a set-returning
    function cannot live inside a CASE.
    """
    table, column = (
        (fq_table("kb_chunks"), "metadata") if level == "chunks" else (fq_table("kb_documents"), "extracted_metadata")
    )
    rows = await conn.fetch(
        f"""
        SELECT value, count(*) AS count FROM (
            SELECT jsonb_array_elements_text({column} -> $2) AS value
            FROM {table}
            WHERE bank_id = $1 AND jsonb_typeof({column} -> $2) = 'array'
            UNION ALL
            SELECT {column} #>> ARRAY[$2] AS value
            FROM {table}
            WHERE bank_id = $1 AND {column} ? $2 AND jsonb_typeof({column} -> $2) <> 'array'
        ) v
        WHERE value IS NOT NULL
        GROUP BY value ORDER BY count DESC, value
        LIMIT 100
        """,
        bank_id,
        property_name,
    )
    return [dict(r) for r in rows]


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
    extracted_metadata: dict[str, Any] | None = None,
) -> None:
    await conn.execute(
        f"""
        INSERT INTO {fq_table("kb_documents")}
            (bank_id, doc_id, text, title, tags, metadata, content_hash, chunk_count, extracted_metadata)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9::jsonb)
        ON CONFLICT (bank_id, doc_id) DO UPDATE SET
            text = EXCLUDED.text, title = EXCLUDED.title, tags = EXCLUDED.tags,
            metadata = EXCLUDED.metadata, content_hash = EXCLUDED.content_hash,
            chunk_count = EXCLUDED.chunk_count, extracted_metadata = EXCLUDED.extracted_metadata,
            updated_at = now()
        """,
        bank_id,
        doc_id,
        text,
        title,
        tags,
        json.dumps(metadata),
        content_hash,
        chunk_count,
        json.dumps(extracted_metadata or {}),
    )


async def set_document_extracted(conn: Any, bank_id: str, doc_id: str, values: dict[str, Any]) -> None:
    await conn.execute(
        f"UPDATE {fq_table('kb_documents')} SET extracted_metadata = $3::jsonb, updated_at = now() "
        "WHERE bank_id = $1 AND doc_id = $2",
        bank_id,
        doc_id,
        json.dumps(values),
    )


async def set_chunk_metadata(conn: Any, bank_id: str, doc_id: str, values: dict[int, dict[str, Any]]) -> None:
    if not values:
        return
    await conn.executemany(
        f"UPDATE {fq_table('kb_chunks')} SET metadata = $4::jsonb "
        "WHERE bank_id = $1 AND doc_id = $2 AND chunk_index = $3",
        [(bank_id, doc_id, index, json.dumps(v)) for index, v in values.items()],
    )


async def replace_chunks(conn: Any, bank_id: str, doc_id: str, rows: list[tuple[Any, ...]]) -> None:
    await conn.execute(f"DELETE FROM {fq_table('kb_chunks')} WHERE bank_id = $1 AND doc_id = $2", bank_id, doc_id)
    if rows:
        await conn.executemany(
            f"INSERT INTO {fq_table('kb_chunks')} "
            "(bank_id, doc_id, chunk_index, text, heading, token_count, embedding, metadata) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7::vector, COALESCE($8::jsonb, '{}'::jsonb))",
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
        SELECT doc_id, title, tags, metadata, extracted_metadata, chunk_count,
               length(text) AS chars, created_at, updated_at
        FROM {fq_table("kb_documents")}
        WHERE {where}
        ORDER BY updated_at DESC, doc_id
        LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
        """,
        *params,
        limit,
        offset,
    )
    return {
        "items": [_loaded(r, "metadata", "extracted_metadata") for r in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


async def get_document(conn: Any, bank_id: str, doc_id: str) -> dict[str, Any] | None:
    row = await conn.fetchrow(
        f"""
        SELECT doc_id, title, tags, metadata, extracted_metadata, text, chunk_count, created_at, updated_at
        FROM {fq_table("kb_documents")} WHERE bank_id = $1 AND doc_id = $2
        """,
        bank_id,
        doc_id,
    )
    if row is None:
        return None
    chunks = await conn.fetch(
        f"SELECT chunk_index, text, token_count, metadata FROM {fq_table('kb_chunks')} "
        "WHERE bank_id = $1 AND doc_id = $2 ORDER BY chunk_index",
        bank_id,
        doc_id,
    )
    return {
        **_loaded(row, "metadata", "extracted_metadata"),
        "chunks": [_loaded(c, "metadata") for c in chunks],
    }


async def delete_document(conn: Any, bank_id: str, doc_id: str) -> bool:
    deleted = await conn.fetchval(
        f"DELETE FROM {fq_table('kb_documents')} WHERE bank_id = $1 AND doc_id = $2 RETURNING doc_id",
        bank_id,
        doc_id,
    )
    return deleted is not None


def _scope(tags: list[str] | None, metadata: dict[str, Any] | None, params: list[Any]) -> SearchScope:
    """The JOIN and the WHERE additions for tag and metadata filters.

    Metadata filters read the document row, so they need it joined rather than probed with
    EXISTS; tags then come off the same join instead of a second subquery.
    """
    if not tags and not metadata:
        return SearchScope(join="", where="")
    join = f" JOIN {fq_table('kb_documents')} d ON d.bank_id = c.bank_id AND d.doc_id = c.doc_id"
    where = ""
    if tags:
        params.append(tags)
        where += f" AND d.tags && ${len(params)}::text[]"
    where += compile_filters(metadata, params)
    return SearchScope(join=join, where=where)


async def search_semantic(
    conn: Any,
    bank_id: str,
    query_vector: list[float],
    limit: int,
    tags: list[str] | None,
    metadata: dict[str, Any] | None = None,
) -> list[ChunkHit]:
    params: list[Any] = [bank_id]
    scope = _scope(tags, metadata, params)
    params.append(vector_literal(query_vector))
    rows = await conn.fetch(
        f"""
        SELECT c.doc_id, c.chunk_index, c.text
        FROM {fq_table("kb_chunks")} c{scope.join}
        WHERE c.bank_id = $1{scope.where}
        ORDER BY c.embedding <=> ${len(params)}::vector
        LIMIT {int(limit)}
        """,
        *params,
    )
    return [ChunkHit(r["doc_id"], r["chunk_index"], r["text"], i + 1) for i, r in enumerate(rows)]


async def search_keyword(
    conn: Any,
    bank_id: str,
    terms: list[str],
    limit: int,
    tags: list[str] | None,
    metadata: dict[str, Any] | None = None,
) -> list[ChunkHit]:
    """Keyword arm. Terms are OR-ed: a question rarely has every word in one chunk."""
    if not terms:
        return []
    params: list[Any] = [bank_id]
    scope = _scope(tags, metadata, params)
    params.append(" | ".join(terms))
    rows = await conn.fetch(
        f"""
        SELECT c.doc_id, c.chunk_index, c.text, ts_rank_cd(c.search_vector, q) AS rank
        FROM {fq_table("kb_chunks")} c{scope.join}, to_tsquery('english', ${len(params)}) q
        WHERE c.bank_id = $1 AND c.search_vector @@ q{scope.where}
        ORDER BY rank DESC
        LIMIT {int(limit)}
        """,
        *params,
    )
    return [ChunkHit(r["doc_id"], r["chunk_index"], r["text"], i + 1) for i, r in enumerate(rows)]
