"""Knowledge banks: what each document contributed to each record

A record is the fold of its contributions, one per document, plus whatever a human
pinned. Merging straight into the record made a write easy and everything else
impossible: a document could never be *un*-contributed, so re-deriving it doubled its
values, deleting it left them behind, and dropping a field from the collection left the
old values in every row.

Revision ID: a9d4f2b71c63
Revises: f7c3a1d54e28
Create Date: 2026-09-26
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "a9d4f2b71c63"
down_revision: str | Sequence[str] | None = "f7c3a1d54e28"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    s = _pg_schema_prefix()
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS {s}kb_record_contributions (
            bank_id TEXT NOT NULL,
            collection_id TEXT NOT NULL,
            record_id TEXT NOT NULL,
            -- The empty string is a write that came from no document: a caller's own
            -- record. It folds like any other contribution.
            doc_id TEXT NOT NULL DEFAULT '',
            values JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            evidence JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (bank_id, collection_id, record_id, doc_id),
            FOREIGN KEY (bank_id, collection_id)
                REFERENCES {s}kb_collections(bank_id, collection_id) ON DELETE CASCADE
        )
    """)
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_kb_contributions_doc ON {s}kb_record_contributions (bank_id, doc_id)")  # fmt: skip
    # Records written before this become one contribution each, attributed to nothing in
    # particular, so an existing bank keeps its rows and can still be re-derived.
    op.execute(f"""
        INSERT INTO {s}kb_record_contributions (bank_id, collection_id, record_id, doc_id, values, evidence)
        SELECT bank_id, collection_id, record_id, '', values, evidence FROM {s}kb_records
        ON CONFLICT DO NOTHING
    """)
    # A collection can now derive on write, without anyone asking for it per batch.
    op.execute(f"ALTER TABLE {s}kb_collections ADD COLUMN IF NOT EXISTS derive_on_write BOOLEAN NOT NULL DEFAULT false")  # fmt: skip


def _pg_downgrade() -> None:
    s = _pg_schema_prefix()
    op.execute(f"ALTER TABLE {s}kb_collections DROP COLUMN IF EXISTS derive_on_write")
    op.execute(f"DROP TABLE IF EXISTS {s}kb_record_contributions")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
