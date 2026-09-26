"""Knowledge-bank metadata: per-bank schemas, extracted values on documents and chunks

Revision ID: d3f1a9c27b45
Revises: b0c1d2e3f4a5
Create Date: 2026-09-26
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "d3f1a9c27b45"
down_revision: str | Sequence[str] | None = "b0c1d2e3f4a5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    s = _pg_schema_prefix()
    # One schema per bank: what the LLM should extract from a document, and from each of
    # its chunks. Kept out of banks.config because it is a nested document, not a setting.
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS {s}kb_metadata_schemas (
            bank_id TEXT PRIMARY KEY REFERENCES {s}banks(bank_id) ON DELETE CASCADE,
            document_schema JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            chunk_schema JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    # Extracted values live beside the caller's own metadata rather than in it, so a
    # re-extraction can never overwrite what the caller wrote.
    op.execute(f"ALTER TABLE {s}kb_documents ADD COLUMN IF NOT EXISTS extracted_metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb")  # fmt: skip
    op.execute(f"ALTER TABLE {s}kb_chunks ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb")
    # Filtering reads these on every search arm.
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_kb_documents_extracted ON {s}kb_documents USING GIN (extracted_metadata)")  # fmt: skip
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_kb_documents_metadata ON {s}kb_documents USING GIN (metadata)")
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_kb_chunks_metadata ON {s}kb_chunks USING GIN (metadata)")


def _pg_downgrade() -> None:
    s = _pg_schema_prefix()
    op.execute(f"DROP INDEX IF EXISTS {s}idx_kb_chunks_metadata")
    op.execute(f"DROP INDEX IF EXISTS {s}idx_kb_documents_metadata")
    op.execute(f"DROP INDEX IF EXISTS {s}idx_kb_documents_extracted")
    op.execute(f"ALTER TABLE {s}kb_chunks DROP COLUMN IF EXISTS metadata")
    op.execute(f"ALTER TABLE {s}kb_documents DROP COLUMN IF EXISTS extracted_metadata")
    op.execute(f"DROP TABLE IF EXISTS {s}kb_metadata_schemas")


def upgrade() -> None:
    # Knowledge banks are PostgreSQL-only (see b0c1d2e3f4a5): the oracle slot is
    # deliberately absent rather than forgotten.
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
