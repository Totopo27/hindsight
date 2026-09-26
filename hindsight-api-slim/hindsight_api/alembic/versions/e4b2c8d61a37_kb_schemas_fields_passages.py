"""Knowledge banks: passages, fields, and schemas as a resource

The vocabulary settles here, before anything depends on the old one: a document is
split into **passages** (was chunks), what an LLM or a write fills in are **fields**
(was metadata), and the definition of those fields is a **schema** — of which a bank
may now have several, one per kind of document, rather than exactly one.

Everything is a rename of existing objects, so no data moves: a bank that already had
a metadata schema keeps it as the schema with id ``default``.

Revision ID: e4b2c8d61a37
Revises: d3f1a9c27b45
Create Date: 2026-09-26
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "e4b2c8d61a37"
down_revision: str | Sequence[str] | None = "d3f1a9c27b45"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    s = _pg_schema_prefix()
    # Passages
    op.execute(f"ALTER TABLE IF EXISTS {s}kb_chunks RENAME TO kb_passages")
    op.execute(f"ALTER TABLE IF EXISTS {s}kb_passages RENAME COLUMN chunk_index TO passage_index")
    op.execute(f"ALTER TABLE IF EXISTS {s}kb_passages RENAME COLUMN metadata TO fields")
    op.execute(f"ALTER TABLE IF EXISTS {s}kb_documents RENAME COLUMN chunk_count TO passage_count")
    # Fields: what the caller wrote stays `metadata` (free-form, theirs); what the schema
    # defines — extracted or supplied — is `fields`.
    op.execute(f"ALTER TABLE IF EXISTS {s}kb_documents RENAME COLUMN extracted_metadata TO fields")
    op.execute(f"ALTER INDEX IF EXISTS {s}idx_kb_chunks_search RENAME TO idx_kb_passages_search")
    op.execute(f"ALTER INDEX IF EXISTS {s}idx_kb_chunks_bank RENAME TO idx_kb_passages_bank")
    op.execute(f"ALTER INDEX IF EXISTS {s}idx_kb_chunks_metadata RENAME TO idx_kb_passages_fields")
    op.execute(f"ALTER INDEX IF EXISTS {s}idx_kb_documents_extracted RENAME TO idx_kb_documents_fields")
    op.execute(f"ALTER INDEX IF EXISTS {s}idx_kb_chunks_embedding_hnsw RENAME TO idx_kb_passages_embedding_hnsw")

    # Schemas: one row per schema instead of one per bank.
    op.execute(f"ALTER TABLE IF EXISTS {s}kb_metadata_schemas RENAME TO kb_schemas")
    op.execute(f"ALTER TABLE IF EXISTS {s}kb_schemas RENAME COLUMN document_schema TO document_fields")
    op.execute(f"ALTER TABLE IF EXISTS {s}kb_schemas RENAME COLUMN chunk_schema TO passage_fields")
    op.execute(f"ALTER TABLE IF EXISTS {s}kb_schemas ADD COLUMN IF NOT EXISTS schema_id TEXT NOT NULL DEFAULT 'default'")  # fmt: skip
    op.execute(f"ALTER TABLE IF EXISTS {s}kb_schemas ADD COLUMN IF NOT EXISTS name TEXT")
    op.execute(f"ALTER TABLE IF EXISTS {s}kb_schemas ADD COLUMN IF NOT EXISTS description TEXT")
    # The bank alone was the key; the pair is now, so a bank can hold a schema per kind of
    # document (contract, invoice, report) and a write says which one it is.
    op.execute(f"ALTER TABLE IF EXISTS {s}kb_schemas DROP CONSTRAINT IF EXISTS kb_metadata_schemas_pkey")
    op.execute(f"ALTER TABLE IF EXISTS {s}kb_schemas ADD PRIMARY KEY (bank_id, schema_id)")

    # A document records which schema filled its fields.
    op.execute(f"ALTER TABLE IF EXISTS {s}kb_documents ADD COLUMN IF NOT EXISTS schema_id TEXT")


def _pg_downgrade() -> None:
    s = _pg_schema_prefix()
    op.execute(f"ALTER TABLE IF EXISTS {s}kb_documents DROP COLUMN IF EXISTS schema_id")
    op.execute(f"ALTER TABLE IF EXISTS {s}kb_schemas DROP CONSTRAINT IF EXISTS kb_schemas_pkey")
    op.execute(f"ALTER TABLE IF EXISTS {s}kb_schemas DROP COLUMN IF EXISTS description")
    op.execute(f"ALTER TABLE IF EXISTS {s}kb_schemas DROP COLUMN IF EXISTS name")
    op.execute(f"ALTER TABLE IF EXISTS {s}kb_schemas DROP COLUMN IF EXISTS schema_id")
    op.execute(f"ALTER TABLE IF EXISTS {s}kb_schemas RENAME COLUMN passage_fields TO chunk_schema")
    op.execute(f"ALTER TABLE IF EXISTS {s}kb_schemas RENAME COLUMN document_fields TO document_schema")
    op.execute(f"ALTER TABLE IF EXISTS {s}kb_schemas RENAME TO kb_metadata_schemas")
    op.execute(f"ALTER TABLE IF EXISTS {s}kb_metadata_schemas ADD PRIMARY KEY (bank_id)")
    op.execute(f"ALTER INDEX IF EXISTS {s}idx_kb_passages_embedding_hnsw RENAME TO idx_kb_chunks_embedding_hnsw")
    op.execute(f"ALTER INDEX IF EXISTS {s}idx_kb_documents_fields RENAME TO idx_kb_documents_extracted")
    op.execute(f"ALTER INDEX IF EXISTS {s}idx_kb_passages_fields RENAME TO idx_kb_chunks_metadata")
    op.execute(f"ALTER INDEX IF EXISTS {s}idx_kb_passages_bank RENAME TO idx_kb_chunks_bank")
    op.execute(f"ALTER INDEX IF EXISTS {s}idx_kb_passages_search RENAME TO idx_kb_chunks_search")
    op.execute(f"ALTER TABLE IF EXISTS {s}kb_documents RENAME COLUMN fields TO extracted_metadata")
    op.execute(f"ALTER TABLE IF EXISTS {s}kb_documents RENAME COLUMN passage_count TO chunk_count")
    op.execute(f"ALTER TABLE IF EXISTS {s}kb_passages RENAME COLUMN fields TO metadata")
    op.execute(f"ALTER TABLE IF EXISTS {s}kb_passages RENAME COLUMN passage_index TO chunk_index")
    op.execute(f"ALTER TABLE IF EXISTS {s}kb_passages RENAME TO kb_chunks")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
