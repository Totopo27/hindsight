"""SQL for knowledge banks: banks, documents, passages and the two search arms.

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
    passage_count: int
    chars: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class SearchScope:
    """The JOIN and the extra WHERE a search arm needs for its tag/metadata filters."""

    join: str
    where: str


@dataclass(frozen=True)
class PassageHit:
    doc_id: str
    passage_index: int
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
               (SELECT count(*) FROM {fq_table("kb_passages")} c WHERE c.bank_id = b.bank_id) AS passages
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
               (SELECT count(*) FROM {fq_table("kb_passages")} c WHERE c.bank_id = b.bank_id) AS passages,
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


async def get_schema(conn: Any, bank_id: str, schema_id: str) -> dict[str, Any] | None:
    """One schema of a bank: the fields it defines for documents and for passages."""
    row = await conn.fetchrow(
        f"""
        SELECT schema_id, name, description, document_fields, passage_fields, created_at, updated_at
        FROM {fq_table("kb_schemas")} WHERE bank_id = $1 AND schema_id = $2
        """,
        bank_id,
        schema_id,
    )
    return _loaded(row, "document_fields", "passage_fields") if row else None


async def list_schemas(conn: Any, bank_id: str) -> list[dict[str, Any]]:
    rows = await conn.fetch(
        f"""
        SELECT schema_id, name, description, document_fields, passage_fields, created_at, updated_at
        FROM {fq_table("kb_schemas")} WHERE bank_id = $1 ORDER BY schema_id
        """,
        bank_id,
    )
    return [_loaded(row, "document_fields", "passage_fields") for row in rows]


async def put_schema(
    conn: Any,
    bank_id: str,
    schema_id: str,
    *,
    name: str | None,
    description: str | None,
    document_fields: dict[str, Any],
    passage_fields: dict[str, Any],
) -> None:
    await conn.execute(
        f"""
        INSERT INTO {fq_table("kb_schemas")}
            (bank_id, schema_id, name, description, document_fields, passage_fields)
        VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb)
        ON CONFLICT (bank_id, schema_id) DO UPDATE SET
            name = EXCLUDED.name, description = EXCLUDED.description,
            document_fields = EXCLUDED.document_fields, passage_fields = EXCLUDED.passage_fields,
            updated_at = now()
        """,
        bank_id,
        schema_id,
        name,
        description,
        json.dumps(document_fields),
        json.dumps(passage_fields),
    )


async def delete_schema(conn: Any, bank_id: str, schema_id: str) -> bool:
    deleted = await conn.fetchval(
        f"DELETE FROM {fq_table('kb_schemas')} WHERE bank_id = $1 AND schema_id = $2 RETURNING schema_id",
        bank_id,
        schema_id,
    )
    return deleted is not None


async def schema_usage(conn: Any, bank_id: str, schema_id: str) -> dict[str, Any]:
    """How much of the bank this schema has actually filled."""
    documents = await conn.fetchval(
        f"SELECT count(*) FROM {fq_table('kb_documents')} "
        "WHERE bank_id = $1 AND schema_id = $2 AND fields <> '{}'::jsonb",
        bank_id,
        schema_id,
    )
    passages = await conn.fetchval(
        f"""
        SELECT count(*) FROM {fq_table("kb_passages")} p
        JOIN {fq_table("kb_documents")} d ON d.bank_id = p.bank_id AND d.doc_id = p.doc_id
        WHERE p.bank_id = $1 AND d.schema_id = $2 AND p.fields <> '{{}}'::jsonb
        """,
        bank_id,
        schema_id,
    )
    return {"documents_with_fields": documents, "passages_with_fields": passages}


async def value_counts(conn: Any, bank_id: str, property_name: str, *, level: str) -> list[dict[str, Any]]:
    """How the corpus splits per value of one property — the facet behind the schema page.

    An array property counts once per element, so "parties: [Acme, Globex]" lands in both
    buckets; a scalar counts once. The two cases are separate legs because a set-returning
    function cannot live inside a CASE.
    """
    table, column = (fq_table("kb_passages"), "fields") if level == "passages" else (fq_table("kb_documents"), "fields")
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
    passage_count: int,
    fields: dict[str, Any] | None = None,
    schema_id: str | None = None,
) -> None:
    await conn.execute(
        f"""
        INSERT INTO {fq_table("kb_documents")}
            (bank_id, doc_id, text, title, tags, metadata, content_hash, passage_count, fields, schema_id)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9::jsonb, $10)
        ON CONFLICT (bank_id, doc_id) DO UPDATE SET
            text = EXCLUDED.text, title = EXCLUDED.title, tags = EXCLUDED.tags,
            metadata = EXCLUDED.metadata, content_hash = EXCLUDED.content_hash,
            passage_count = EXCLUDED.passage_count, fields = EXCLUDED.fields,
            schema_id = EXCLUDED.schema_id, updated_at = now()
        """,
        bank_id,
        doc_id,
        text,
        title,
        tags,
        json.dumps(metadata),
        content_hash,
        passage_count,
        json.dumps(fields or {}),
        schema_id,
    )


async def set_document_fields(conn: Any, bank_id: str, doc_id: str, values: dict[str, Any]) -> None:
    await conn.execute(
        f"UPDATE {fq_table('kb_documents')} SET fields = $3::jsonb, updated_at = now() "
        "WHERE bank_id = $1 AND doc_id = $2",
        bank_id,
        doc_id,
        json.dumps(values),
    )


async def set_passage_fields(conn: Any, bank_id: str, doc_id: str, values: dict[int, dict[str, Any]]) -> None:
    if not values:
        return
    await conn.executemany(
        f"UPDATE {fq_table('kb_passages')} SET fields = $4::jsonb "
        "WHERE bank_id = $1 AND doc_id = $2 AND passage_index = $3",
        [(bank_id, doc_id, index, json.dumps(v)) for index, v in values.items()],
    )


async def replace_passages(conn: Any, bank_id: str, doc_id: str, rows: list[tuple[Any, ...]]) -> None:
    await conn.execute(f"DELETE FROM {fq_table('kb_passages')} WHERE bank_id = $1 AND doc_id = $2", bank_id, doc_id)
    if rows:
        await conn.executemany(
            f"INSERT INTO {fq_table('kb_passages')} "
            "(bank_id, doc_id, passage_index, text, heading, token_count, embedding, fields) "
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
        SELECT doc_id, title, tags, metadata, fields, schema_id, passage_count,
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
        "items": [_loaded(r, "metadata", "fields") for r in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


async def get_document(conn: Any, bank_id: str, doc_id: str) -> dict[str, Any] | None:
    row = await conn.fetchrow(
        f"""
        SELECT doc_id, title, tags, metadata, fields, schema_id, text, passage_count, created_at, updated_at
        FROM {fq_table("kb_documents")} WHERE bank_id = $1 AND doc_id = $2
        """,
        bank_id,
        doc_id,
    )
    if row is None:
        return None
    passages = await conn.fetch(
        f"SELECT passage_index, text, token_count, fields FROM {fq_table('kb_passages')} "
        "WHERE bank_id = $1 AND doc_id = $2 ORDER BY passage_index",
        bank_id,
        doc_id,
    )
    return {
        **_loaded(row, "metadata", "fields"),
        "passages": [_loaded(c, "fields") for c in passages],
    }


async def delete_document(conn: Any, bank_id: str, doc_id: str) -> bool:
    deleted = await conn.fetchval(
        f"DELETE FROM {fq_table('kb_documents')} WHERE bank_id = $1 AND doc_id = $2 RETURNING doc_id",
        bank_id,
        doc_id,
    )
    return deleted is not None


def _scope(tags: list[str] | None, fields: dict[str, Any] | None, params: list[Any]) -> SearchScope:
    """The JOIN and the WHERE additions for tag and metadata filters.

    Metadata filters read the document row, so they need it joined rather than probed with
    EXISTS; tags then come off the same join instead of a second subquery.
    """
    if not tags and not fields:
        return SearchScope(join="", where="")
    join = f" JOIN {fq_table('kb_documents')} d ON d.bank_id = c.bank_id AND d.doc_id = c.doc_id"
    where = ""
    if tags:
        params.append(tags)
        where += f" AND d.tags && ${len(params)}::text[]"
    where += compile_filters(fields, params)
    return SearchScope(join=join, where=where)


async def search_semantic(
    conn: Any,
    bank_id: str,
    query_vector: list[float],
    limit: int,
    tags: list[str] | None,
    fields: dict[str, Any] | None = None,
) -> list[PassageHit]:
    params: list[Any] = [bank_id]
    scope = _scope(tags, fields, params)
    params.append(vector_literal(query_vector))
    rows = await conn.fetch(
        f"""
        SELECT c.doc_id, c.passage_index, c.text
        FROM {fq_table("kb_passages")} c{scope.join}
        WHERE c.bank_id = $1{scope.where}
        ORDER BY c.embedding <=> ${len(params)}::vector
        LIMIT {int(limit)}
        """,
        *params,
    )
    return [PassageHit(r["doc_id"], r["passage_index"], r["text"], i + 1) for i, r in enumerate(rows)]


async def search_keyword(
    conn: Any,
    bank_id: str,
    terms: list[str],
    limit: int,
    tags: list[str] | None,
    fields: dict[str, Any] | None = None,
) -> list[PassageHit]:
    """Keyword arm. Terms are OR-ed: a question rarely has every word in one passage."""
    if not terms:
        return []
    params: list[Any] = [bank_id]
    scope = _scope(tags, fields, params)
    params.append(" | ".join(terms))
    rows = await conn.fetch(
        f"""
        SELECT c.doc_id, c.passage_index, c.text, ts_rank_cd(c.search_vector, q) AS rank
        FROM {fq_table("kb_passages")} c{scope.join}, to_tsquery('english', ${len(params)}) q
        WHERE c.bank_id = $1 AND c.search_vector @@ q{scope.where}
        ORDER BY rank DESC
        LIMIT {int(limit)}
        """,
        *params,
    )
    return [PassageHit(r["doc_id"], r["passage_index"], r["text"], i + 1) for i, r in enumerate(rows)]
