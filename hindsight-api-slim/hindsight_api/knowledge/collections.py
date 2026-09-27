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
from .identity import can_compare_fuzzily, is_typo_of

MAX_FIELDS = 50
#: Quotes kept per field on the materialised record. A vendor mentioned in ten thousand
#: documents would otherwise carry ten thousand quotes in every response that returns it.
#: The contributions keep them all — this is the summary, not the archive.
MAX_EVIDENCE_PER_FIELD = 20


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
    derive_on_write: bool = False,
) -> None:
    await conn.execute(
        f"""
        INSERT INTO {fq_table("kb_collections")}
            (bank_id, collection_id, name, description, fields, identity, derive_on_write)
        VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)
        ON CONFLICT (bank_id, collection_id) DO UPDATE SET
            name = EXCLUDED.name, description = EXCLUDED.description,
            fields = EXCLUDED.fields, identity = EXCLUDED.identity,
            derive_on_write = EXCLUDED.derive_on_write, updated_at = now()
        """,
        bank_id,
        collection_id,
        name,
        description,
        json.dumps(fields),
        identity,
        derive_on_write,
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
        SELECT collection_id, name, description, fields, identity, derive_on_write, created_at, updated_at
        FROM {fq_table("kb_collections")} WHERE bank_id = $1 AND collection_id = $2
        """,
        bank_id,
        collection_id,
    )
    return _loaded(row, "fields") if row else None


async def list_collections(conn: Any, bank_id: str) -> list[dict[str, Any]]:
    rows = await conn.fetch(
        f"""
        SELECT c.collection_id, c.name, c.description, c.fields, c.identity, c.derive_on_write,
               c.created_at, c.updated_at,
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


async def contribute(
    conn: Any,
    bank_id: str,
    collection_id: str,
    record_id: str,
    *,
    doc_id: str,
    values: dict[str, Any],
    evidence: dict[str, Any],
) -> None:
    """Record what one document says about one record, replacing what it said before.

    Re-deriving a document must not double its values, and deleting it must take them
    away again, so a document's contribution is a row of its own rather than something
    merged into the record and forgotten.
    """
    await conn.execute(
        f"""
        INSERT INTO {fq_table("kb_record_contributions")}
            (bank_id, collection_id, record_id, doc_id, values, evidence)
        VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb)
        ON CONFLICT (bank_id, collection_id, record_id, doc_id) DO UPDATE SET
            values = EXCLUDED.values, evidence = EXCLUDED.evidence, updated_at = now()
        """,
        bank_id,
        collection_id,
        record_id,
        doc_id,
        json.dumps(values),
        json.dumps(evidence),
    )


async def materialize(conn: Any, bank_id: str, collection_id: str, record_id: str) -> bool:
    """Fold a record's contributions into the record. Returns False if nothing is left.

    Later contributions win a disagreement, and a pinned value wins everything: that is
    the whole precedence rule, in one statement, so a record can always be rebuilt from
    its parts rather than depending on the order writes happened to arrive.
    """
    pinned = await conn.fetchval(
        f"SELECT pinned FROM {fq_table('kb_records')} WHERE bank_id = $1 AND collection_id = $2 AND record_id = $3",
        bank_id,
        collection_id,
        record_id,
    )
    pinned_values = json.loads(pinned) if isinstance(pinned, str) else (pinned or {})
    rows = await conn.fetch(
        f"""
        SELECT doc_id, values, evidence FROM {fq_table("kb_record_contributions")}
        WHERE bank_id = $1 AND collection_id = $2 AND record_id = $3
        ORDER BY updated_at, doc_id
        """,
        bank_id,
        collection_id,
        record_id,
    )
    if not rows and not pinned_values:
        await conn.execute(
            f"DELETE FROM {fq_table('kb_records')} WHERE bank_id = $1 AND collection_id = $2 AND record_id = $3",
            bank_id,
            collection_id,
            record_id,
        )
        return False

    values: dict[str, Any] = {}
    evidence: dict[str, list[dict[str, Any]]] = {}
    doc_ids: list[str] = []
    for row in rows:
        row_values = json.loads(row["values"]) if isinstance(row["values"], str) else row["values"]
        row_evidence = json.loads(row["evidence"]) if isinstance(row["evidence"], str) else row["evidence"]
        values.update({k: v for k, v in (row_values or {}).items() if v is not None})
        for name, quote in (row_evidence or {}).items():
            evidence.setdefault(name, []).append({"doc_id": row["doc_id"], "quote": quote})
        if row["doc_id"]:
            doc_ids.append(row["doc_id"])
    values.update(pinned_values)
    # Keep the most recent quotes per field: contributions are folded oldest first, so
    # the tail is the newest, and the newest is what a reader wants to see first.
    evidence = {name: quotes[-MAX_EVIDENCE_PER_FIELD:] for name, quotes in evidence.items()}

    await conn.execute(
        f"""
        INSERT INTO {fq_table("kb_records")} (bank_id, collection_id, record_id, values, evidence, doc_ids)
        VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6::text[])
        ON CONFLICT (bank_id, collection_id, record_id) DO UPDATE SET
            values = EXCLUDED.values, evidence = EXCLUDED.evidence,
            doc_ids = EXCLUDED.doc_ids, updated_at = now()
        """,
        bank_id,
        collection_id,
        record_id,
        json.dumps(values),
        json.dumps(evidence),
        sorted(set(doc_ids)),
    )
    return True


async def records_touched_by(conn: Any, bank_id: str, doc_id: str) -> list[tuple[str, str]]:
    rows = await conn.fetch(
        f"SELECT collection_id, record_id FROM {fq_table('kb_record_contributions')} "
        "WHERE bank_id = $1 AND doc_id = $2",
        bank_id,
        doc_id,
    )
    return [(row["collection_id"], row["record_id"]) for row in rows]


async def drop_contributions_of(conn: Any, bank_id: str, doc_id: str, collection_id: str | None = None) -> None:
    where = "bank_id = $1 AND doc_id = $2"
    params: list[Any] = [bank_id, doc_id]
    if collection_id is not None:
        params.append(collection_id)
        where += f" AND collection_id = ${len(params)}"
    await conn.execute(f"DELETE FROM {fq_table('kb_record_contributions')} WHERE {where}", *params)


async def prune_fields(conn: Any, bank_id: str, collection_id: str, keep: set[str]) -> int:
    """Forget values of fields the collection no longer defines.

    A field dropped from a collection has to leave the rows too, or a query would keep
    returning a column the schema says does not exist.
    """
    rows = await conn.fetch(
        f"SELECT record_id, doc_id, values, evidence FROM {fq_table('kb_record_contributions')} "
        "WHERE bank_id = $1 AND collection_id = $2",
        bank_id,
        collection_id,
    )
    touched: set[str] = set()
    for row in rows:
        values = json.loads(row["values"]) if isinstance(row["values"], str) else row["values"]
        evidence = json.loads(row["evidence"]) if isinstance(row["evidence"], str) else row["evidence"]
        kept = {k: v for k, v in (values or {}).items() if k in keep}
        kept_evidence = {k: v for k, v in (evidence or {}).items() if k in keep}
        if kept != values or kept_evidence != evidence:
            await conn.execute(
                f"UPDATE {fq_table('kb_record_contributions')} SET values = $4::jsonb, evidence = $5::jsonb "
                "WHERE bank_id = $1 AND collection_id = $2 AND record_id = $3 AND doc_id = $6",
                bank_id,
                collection_id,
                row["record_id"],
                json.dumps(kept),
                json.dumps(kept_evidence),
                row["doc_id"],
            )
            touched.add(row["record_id"])
    await conn.execute(
        f"""
        UPDATE {fq_table("kb_records")}
        SET pinned = (SELECT COALESCE(jsonb_object_agg(key, value), '{{}}'::jsonb)
                      FROM jsonb_each(pinned) WHERE key = ANY($3::text[])),
            updated_at = now()
        WHERE bank_id = $1 AND collection_id = $2
        """,
        bank_id,
        collection_id,
        sorted(keep),
    )
    for record_id in touched:
        await materialize(conn, bank_id, collection_id, record_id)
    return len(touched)


async def delete_record(conn: Any, bank_id: str, collection_id: str, record_id: str) -> bool:
    deleted = await conn.fetchval(
        f"DELETE FROM {fq_table('kb_records')} WHERE bank_id = $1 AND collection_id = $2 AND record_id = $3 "
        "RETURNING record_id",
        bank_id,
        collection_id,
        record_id,
    )
    await conn.execute(
        f"DELETE FROM {fq_table('kb_record_contributions')} "
        "WHERE bank_id = $1 AND collection_id = $2 AND record_id = $3",
        bank_id,
        collection_id,
        record_id,
    )
    return deleted is not None


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


# ---- identity resolution


async def resolve_record_id(
    conn: Any,
    bank_id: str,
    collection_id: str,
    key: str,
    *,
    similarity: float,
) -> str:
    """Which record a name belongs to: an alias, an existing record, or a new one.

    The order is the point. An alias is somebody's decision and wins outright; an exact
    key match is certainty; similarity is a guess, so it comes last, only for names long
    enough to measure, and only above the configured threshold.
    """
    if not key:
        return key
    alias = await conn.fetchval(
        f"SELECT record_id FROM {fq_table('kb_record_aliases')} "
        "WHERE bank_id = $1 AND collection_id = $2 AND alias_key = $3",
        bank_id,
        collection_id,
        key,
    )
    if alias:
        return alias

    exact = await conn.fetchval(
        f"SELECT record_id FROM {fq_table('kb_records')} WHERE bank_id = $1 AND collection_id = $2 AND record_id = $3",
        bank_id,
        collection_id,
        key,
    )
    if exact:
        return exact

    if similarity <= 0 or not can_compare_fuzzily(key):
        return key
    # Stage one: the trigram index narrows thousands of records to a handful. It is a
    # prefilter, not the verdict — a low threshold on purpose, because the verdict is the
    # edit distance below and a prefilter that is too strict hides the typos from it.
    candidates = await conn.fetch(
        f"""
        SELECT record_id FROM {fq_table("kb_records")}
        WHERE bank_id = $1 AND collection_id = $2
          AND length(record_id) BETWEEN $4 AND $5
          AND similarity(record_id, $3) >= $6
        ORDER BY similarity(record_id, $3) DESC
        LIMIT 5
        """,
        bank_id,
        collection_id,
        key,
        len(key) - 2,
        len(key) + 2,
        similarity,
    )
    # Stage two: the decision. Nearly the same length, a couple of characters apart.
    for row in candidates:
        if is_typo_of(key, str(row["record_id"])):
            # Remember it, so the next document spelled that way costs a lookup instead of
            # a scan, and the grouping cannot drift as records come and go.
            await add_alias(conn, bank_id, collection_id, key, str(row["record_id"]), source="variant")
            return str(row["record_id"])
    return key


async def follow_aliases(conn: Any, bank_id: str, collection_id: str, record_id: str) -> str:
    """Where a record id points today, after any merges it was part of."""
    return (
        await conn.fetchval(
            f"SELECT record_id FROM {fq_table('kb_record_aliases')} "
            "WHERE bank_id = $1 AND collection_id = $2 AND alias_key = $3",
            bank_id,
            collection_id,
            record_id,
        )
        or record_id
    )


async def add_alias(
    conn: Any, bank_id: str, collection_id: str, alias_key: str, record_id: str, *, source: str = "merge"
) -> None:
    if not alias_key or alias_key == record_id:
        return
    await conn.execute(
        f"""
        INSERT INTO {fq_table("kb_record_aliases")} (bank_id, collection_id, alias_key, record_id, source)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (bank_id, collection_id, alias_key) DO UPDATE SET record_id = EXCLUDED.record_id,
            source = EXCLUDED.source
        """,
        bank_id,
        collection_id,
        alias_key,
        record_id,
        source,
    )


async def merge_records(conn: Any, bank_id: str, collection_id: str, source_id: str, target_id: str) -> None:
    """Fold one record into another and make the decision stick.

    Everything that fed the loser now feeds the winner, the loser's id becomes an alias
    of the winner so the next document does not undo the merge, and anything that pointed
    at the loser is repointed — a contract whose vendor was merged still has a vendor.
    """
    contributions = await conn.fetch(
        f"SELECT doc_id, values, evidence FROM {fq_table('kb_record_contributions')} "
        "WHERE bank_id = $1 AND collection_id = $2 AND record_id = $3 ORDER BY updated_at",
        bank_id,
        collection_id,
        source_id,
    )
    for row in contributions:
        values = json.loads(row["values"]) if isinstance(row["values"], str) else row["values"]
        evidence = json.loads(row["evidence"]) if isinstance(row["evidence"], str) else row["evidence"]
        existing = await conn.fetchrow(
            f"SELECT values, evidence FROM {fq_table('kb_record_contributions')} "
            "WHERE bank_id = $1 AND collection_id = $2 AND record_id = $3 AND doc_id = $4",
            bank_id,
            collection_id,
            target_id,
            row["doc_id"],
        )
        if existing:
            # The same document fed both sides: its two halves are one contribution now.
            current = json.loads(existing["values"]) if isinstance(existing["values"], str) else existing["values"]
            current_evidence = (
                json.loads(existing["evidence"]) if isinstance(existing["evidence"], str) else existing["evidence"]
            )
            values = {**(values or {}), **(current or {})}
            evidence = {**(evidence or {}), **(current_evidence or {})}
        await contribute(
            conn,
            bank_id,
            collection_id,
            target_id,
            doc_id=row["doc_id"],
            values=values or {},
            evidence=evidence or {},
        )

    pinned = await conn.fetchval(
        f"SELECT pinned FROM {fq_table('kb_records')} WHERE bank_id = $1 AND collection_id = $2 AND record_id = $3",
        bank_id,
        collection_id,
        source_id,
    )
    pinned_values = json.loads(pinned) if isinstance(pinned, str) else (pinned or {})
    if pinned_values:
        # A correction someone made on the losing record was still a correction.
        await conn.execute(
            f"UPDATE {fq_table('kb_records')} SET pinned = pinned || $4::jsonb "
            "WHERE bank_id = $1 AND collection_id = $2 AND record_id = $3",
            bank_id,
            collection_id,
            target_id,
            json.dumps(pinned_values),
        )

    await conn.execute(
        f"DELETE FROM {fq_table('kb_record_contributions')} "
        "WHERE bank_id = $1 AND collection_id = $2 AND record_id = $3",
        bank_id,
        collection_id,
        source_id,
    )
    await conn.execute(
        f"DELETE FROM {fq_table('kb_records')} WHERE bank_id = $1 AND collection_id = $2 AND record_id = $3",
        bank_id,
        collection_id,
        source_id,
    )
    await add_alias(conn, bank_id, collection_id, source_id, target_id, source="merge")
    # Aliases that pointed at the loser now point at the winner, so a chain of merges
    # stays one hop deep.
    await conn.execute(
        f"UPDATE {fq_table('kb_record_aliases')} SET record_id = $4 "
        "WHERE bank_id = $1 AND collection_id = $2 AND record_id = $3",
        bank_id,
        collection_id,
        source_id,
        target_id,
    )
    await repoint_relationships(conn, bank_id, collection_id, source_id, target_id)
    await materialize(conn, bank_id, collection_id, target_id)


async def repoint_relationships(conn: Any, bank_id: str, collection_id: str, source_id: str, target_id: str) -> None:
    """Move every reference to the merged-away record onto the surviving one."""
    collections = await list_collections(conn, bank_id)
    for other in collections:
        pointing = [name for name, target in relationships(other.get("fields") or {}).items() if target == collection_id]  # fmt: skip
        for field_name in pointing:
            for table in ("kb_records", "kb_record_contributions"):
                await conn.execute(
                    f"""
                    UPDATE {fq_table(table)}
                    SET values = jsonb_set(values, ARRAY[$4], to_jsonb($5::text))
                    WHERE bank_id = $1 AND collection_id = $2 AND values ->> $4 = $3
                    """,
                    bank_id,
                    other["collection_id"],
                    source_id,
                    field_name,
                    target_id,
                )
