"""Knowledge banks: documents and chunks for retrieval.

A knowledge bank is a ``banks`` row with ``kind = 'knowledge'``. Keeping it in
``banks`` is what lets the whole operations substrate work unchanged: the
``async_operations`` FK, per-bank ``config``, audit, usage and the delete cascade
all key off ``banks.bank_id``. The HTTP surface is separate
(``/v1/default/knowledge-banks``) because a memory bank's retain/recall/reflect
API means nothing for documents.

Two tables:

* ``kb_documents`` — one row per document the user wrote, with its text, tags,
  metadata and a content hash so an unchanged re-write is a no-op.
* ``kb_chunks`` — the retrieval unit: chunk text, its embedding and a generated
  ``tsvector`` for the keyword arm. Deleting a document takes its chunks.

Revision ID: b0c1d2e3f4a5
Revises: e5b1c7d3a902
Create Date: 2026-09-25
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "b0c1d2e3f4a5"
down_revision: str | Sequence[str] | None = "e5b1c7d3a902"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    s = _pg_schema_prefix()
    op.execute(f"ALTER TABLE {s}banks ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'memory'")
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS {s}kb_documents (
            bank_id TEXT NOT NULL REFERENCES {s}banks(bank_id) ON DELETE CASCADE,
            doc_id TEXT NOT NULL,
            text TEXT NOT NULL,
            title TEXT,
            tags TEXT[] NOT NULL DEFAULT '{{}}',
            metadata JSONB NOT NULL DEFAULT '{{}}',
            content_hash TEXT NOT NULL,
            chunk_count INT NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (bank_id, doc_id)
        )
    """)
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_kb_documents_tags ON {s}kb_documents USING GIN (tags)")
    # The embedding column is dimension-less on purpose: the dimension follows the
    # configured provider, exactly as ensure_embedding_dimension handles memory_units.
    # The ANN index is created per bank once it is worth one (see kb vector index
    # maintenance); a small bank is an exact scan, which is faster than a bad index.
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS {s}kb_chunks (
            bank_id TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            chunk_index INT NOT NULL,
            text TEXT NOT NULL,
            heading TEXT,
            token_count INT NOT NULL DEFAULT 0,
            embedding vector NOT NULL,
            -- The document's title is indexed with every chunk of it: a paragraph often
            -- never repeats the subject it is about, and both arms need it to match.
            search_vector tsvector GENERATED ALWAYS AS (
                to_tsvector('english', coalesce(heading, '') || ' ' || text)
            ) STORED,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (bank_id, doc_id, chunk_index),
            FOREIGN KEY (bank_id, doc_id) REFERENCES {s}kb_documents(bank_id, doc_id) ON DELETE CASCADE
        )
    """)
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_kb_chunks_search ON {s}kb_chunks USING GIN (search_vector)")
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_kb_chunks_bank ON {s}kb_chunks (bank_id)")


def _pg_downgrade() -> None:
    s = _pg_schema_prefix()
    op.execute(f"DROP TABLE IF EXISTS {s}kb_chunks")
    op.execute(f"DROP TABLE IF EXISTS {s}kb_documents")
    op.execute(f"ALTER TABLE {s}banks DROP COLUMN IF EXISTS kind")


def upgrade() -> None:
    # PostgreSQL only for now: the Oracle half lands with the phase that ships
    # knowledge banks on Oracle, rather than a copy nothing exercises.
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
