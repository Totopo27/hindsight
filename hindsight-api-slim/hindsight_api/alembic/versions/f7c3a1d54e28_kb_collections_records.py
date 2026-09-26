"""Knowledge banks: collections and their records

A collection is a structured dataset derived from the documents — vendors, contracts,
invoices — and a record is one row of it: one real-world thing, gathered across every
document that mentions it, not one per file. A record's ``values`` hold its fields, and
``evidence`` holds, per field, the document it came from.

Relationship fields point at another collection's record, which is what the query
endpoint joins on.

Revision ID: f7c3a1d54e28
Revises: e4b2c8d61a37
Create Date: 2026-09-26
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "f7c3a1d54e28"
down_revision: str | Sequence[str] | None = "e4b2c8d61a37"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    s = _pg_schema_prefix()
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS {s}kb_collections (
            bank_id TEXT NOT NULL REFERENCES {s}banks(bank_id) ON DELETE CASCADE,
            collection_id TEXT NOT NULL,
            name TEXT,
            description TEXT,
            -- field name -> {{type, description, values, items, collection}}; `collection`
            -- makes it a relationship to another collection's record.
            fields JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            -- Which field identifies one record, so the same real-world thing found in two
            -- documents lands in one row instead of two.
            identity TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (bank_id, collection_id)
        )
    """)
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS {s}kb_records (
            bank_id TEXT NOT NULL,
            collection_id TEXT NOT NULL,
            record_id TEXT NOT NULL,
            values JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            -- field name -> [{{doc_id, quote}}]: where each value came from, which is what
            -- makes a derived number auditable instead of a claim.
            evidence JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            -- Values a human pinned; they outrank whatever the documents say.
            pinned JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            doc_ids TEXT[] NOT NULL DEFAULT '{{}}',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (bank_id, collection_id, record_id),
            FOREIGN KEY (bank_id, collection_id)
                REFERENCES {s}kb_collections(bank_id, collection_id) ON DELETE CASCADE
        )
    """)
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_kb_records_values ON {s}kb_records USING GIN (values)")
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_kb_records_docs ON {s}kb_records USING GIN (doc_ids)")


def _pg_downgrade() -> None:
    s = _pg_schema_prefix()
    op.execute(f"DROP TABLE IF EXISTS {s}kb_records")
    op.execute(f"DROP TABLE IF EXISTS {s}kb_collections")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
