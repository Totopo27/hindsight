"""Collections: structured datasets derived from a bank's documents, and their records.

A collection says what one record is — its fields, and which field identifies it. A
record is one real-world thing (a vendor, a contract), gathered across every document
that mentions it rather than one row per file, with the document behind each value kept
as evidence.

A field whose spec carries ``collection`` is a **relationship**: its value is another
collection's record id, and that is what the query endpoint joins on.

Records arrive two ways, and both end in the same table:

- **derived**, by an LLM reading the documents against the collection's fields, merging
  what it finds into the record its identity field names;
- **written**, by a caller who already has the data — which is also how a corpus can be
  set up, and measured, without spending anything on a model.
"""

from __future__ import annotations

import json
from typing import Any

from ..engine.schema import fq_table
from .fields import PROPERTY_TYPES, SchemaError

MAX_FIELDS = 50


def validate_collection_fields(raw: Any, *, known_collections: set[str]) -> dict[str, dict[str, Any]]:
    """Check a collection's field definitions and return them normalised."""
    if not isinstance(raw, dict) or not raw:
        raise SchemaError("a collection needs at least one field")
    if len(raw) > MAX_FIELDS:
        raise SchemaError(f"a collection may have at most {MAX_FIELDS} fields")

    out: dict[str, dict[str, Any]] = {}
    for name, spec in raw.items():
        if not isinstance(name, str) or not name.replace("_", "").replace("-", "").isalnum():
            raise SchemaError(f"field name {name!r} must be alphanumeric with _ or -")
        if not isinstance(spec, dict):
            raise SchemaError(f"field {name!r} must be an object, e.g. {{'type': 'string'}}")

        collection = spec.get("collection")
        if collection is not None:
            # A relationship's type is fixed: it holds the id of a record in that
            # collection, so declaring it "number" would be a lie the joins believe.
            if not isinstance(collection, str) or collection not in known_collections:
                raise SchemaError(f"field {name!r}: collection {collection!r} is not a collection of this bank")
            out[name] = {"type": "string", "collection": collection, "description": spec.get("description")}
            continue

        field_type = spec.get("type", "string")
        if field_type not in PROPERTY_TYPES:
            raise SchemaError(f"field {name!r} has unknown type {field_type!r}; one of {PROPERTY_TYPES}")
        normalised: dict[str, Any] = {"type": field_type}
        if spec.get("description") is not None:
            normalised["description"] = spec["description"]
        if field_type == "array":
            items = spec.get("items", "string")
            if items not in PROPERTY_TYPES:
                raise SchemaError(f"field {name!r}: items must be one of {PROPERTY_TYPES}")
            normalised["items"] = items
        values = spec.get("values")
        if values is not None:
            if not isinstance(values, list) or not values:
                raise SchemaError(f"field {name!r}: values must be a non-empty list")
            normalised["values"] = values
        out[name] = normalised
    return out


def relationships(fields: dict[str, Any]) -> dict[str, str]:
    """Field name -> the collection it points at."""
    return {name: spec["collection"] for name, spec in fields.items() if spec.get("collection")}


# ---- storage


async def put_collection(
    conn: Any,
    bank_id: str,
    collection_id: str,
    *,
    name: str | None,
    description: str | None,
    fields: dict[str, Any],
    identity: str | None,
) -> None:
    await conn.execute(
        f"""
        INSERT INTO {fq_table("kb_collections")} (bank_id, collection_id, name, description, fields, identity)
        VALUES ($1, $2, $3, $4, $5::jsonb, $6)
        ON CONFLICT (bank_id, collection_id) DO UPDATE SET
            name = EXCLUDED.name, description = EXCLUDED.description,
            fields = EXCLUDED.fields, identity = EXCLUDED.identity, updated_at = now()
        """,
        bank_id,
        collection_id,
        name,
        description,
        json.dumps(fields),
        identity,
    )


def _loaded(row: Any, *columns: str) -> dict[str, Any]:
    out = dict(row)
    for column in columns:
        if isinstance(out.get(column), str):
            out[column] = json.loads(out[column])
    return out


async def get_collection(conn: Any, bank_id: str, collection_id: str) -> dict[str, Any] | None:
    row = await conn.fetchrow(
        f"""
        SELECT collection_id, name, description, fields, identity, created_at, updated_at
        FROM {fq_table("kb_collections")} WHERE bank_id = $1 AND collection_id = $2
        """,
        bank_id,
        collection_id,
    )
    return _loaded(row, "fields") if row else None


async def list_collections(conn: Any, bank_id: str) -> list[dict[str, Any]]:
    rows = await conn.fetch(
        f"""
        SELECT c.collection_id, c.name, c.description, c.fields, c.identity, c.created_at, c.updated_at,
               (SELECT count(*) FROM {fq_table("kb_records")} r
                WHERE r.bank_id = c.bank_id AND r.collection_id = c.collection_id) AS records
        FROM {fq_table("kb_collections")} c WHERE c.bank_id = $1 ORDER BY c.collection_id
        """,
        bank_id,
    )
    return [_loaded(row, "fields") for row in rows]


async def delete_collection(conn: Any, bank_id: str, collection_id: str) -> bool:
    deleted = await conn.fetchval(
        f"DELETE FROM {fq_table('kb_collections')} WHERE bank_id = $1 AND collection_id = $2 RETURNING collection_id",
        bank_id,
        collection_id,
    )
    return deleted is not None


async def upsert_record(
    conn: Any,
    bank_id: str,
    collection_id: str,
    record_id: str,
    *,
    values: dict[str, Any],
    evidence: dict[str, Any],
    doc_ids: list[str],
) -> None:
    """Merge one record's values into whatever is already there.

    Two documents about the same vendor each fill part of the row, so a later write adds
    to the record rather than replacing it — except where a value is pinned, which a
    human set deliberately and no document may overwrite.
    """
    await conn.execute(
        f"""
        INSERT INTO {fq_table("kb_records")} (bank_id, collection_id, record_id, values, evidence, doc_ids)
        VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6::text[])
        ON CONFLICT (bank_id, collection_id, record_id) DO UPDATE SET
            values = {fq_table("kb_records")}.values || EXCLUDED.values || {fq_table("kb_records")}.pinned,
            evidence = {fq_table("kb_records")}.evidence || EXCLUDED.evidence,
            doc_ids = ARRAY(
                SELECT DISTINCT unnest({fq_table("kb_records")}.doc_ids || EXCLUDED.doc_ids)
            ),
            updated_at = now()
        """,
        bank_id,
        collection_id,
        record_id,
        json.dumps(values),
        json.dumps(evidence),
        doc_ids,
    )


async def pin_values(conn: Any, bank_id: str, collection_id: str, record_id: str, pinned: dict[str, Any]) -> bool:
    """A human's correction, which outranks the documents from here on."""
    updated = await conn.fetchval(
        f"""
        UPDATE {fq_table("kb_records")}
        SET pinned = pinned || $4::jsonb, values = values || $4::jsonb, updated_at = now()
        WHERE bank_id = $1 AND collection_id = $2 AND record_id = $3
        RETURNING record_id
        """,
        bank_id,
        collection_id,
        record_id,
        json.dumps(pinned),
    )
    return updated is not None


async def get_record(conn: Any, bank_id: str, collection_id: str, record_id: str) -> dict[str, Any] | None:
    row = await conn.fetchrow(
        f"""
        SELECT record_id, values, evidence, pinned, doc_ids, created_at, updated_at
        FROM {fq_table("kb_records")} WHERE bank_id = $1 AND collection_id = $2 AND record_id = $3
        """,
        bank_id,
        collection_id,
        record_id,
    )
    return _loaded(row, "values", "evidence", "pinned") if row else None


async def delete_records_for_document(conn: Any, bank_id: str, doc_id: str) -> None:
    """Forget what one document contributed: drop records it alone is behind."""
    await conn.execute(
        f"""
        DELETE FROM {fq_table("kb_records")}
        WHERE bank_id = $1 AND doc_ids = ARRAY[$2]::text[]
        """,
        bank_id,
        doc_id,
    )
    await conn.execute(
        f"""
        UPDATE {fq_table("kb_records")}
        SET doc_ids = array_remove(doc_ids, $2), updated_at = now()
        WHERE bank_id = $1 AND $2 = ANY(doc_ids)
        """,
        bank_id,
        doc_id,
    )
