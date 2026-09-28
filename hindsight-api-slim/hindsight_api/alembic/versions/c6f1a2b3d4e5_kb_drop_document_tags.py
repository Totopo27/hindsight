"""Knowledge banks: documents have no tags

Tags are a memory-bank idea — they scope what a recall may see. A knowledge bank
answers with passages, and what a search should scope by is the schema a document was
read with and the fields that schema filled, both of which say something about the
document rather than being pinned to it by hand.

Revision ID: c6f1a2b3d4e5
Revises: b5e8c07d9a14
Create Date: 2026-09-28
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "c6f1a2b3d4e5"
down_revision: str | Sequence[str] | None = "b5e8c07d9a14"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    s = _pg_schema_prefix()
    op.execute(f"DROP INDEX IF EXISTS {s}idx_kb_documents_tags")
    op.execute(f"ALTER TABLE {s}kb_documents DROP COLUMN IF EXISTS tags")


def _pg_downgrade() -> None:
    s = _pg_schema_prefix()
    op.execute(f"ALTER TABLE {s}kb_documents ADD COLUMN IF NOT EXISTS tags TEXT[] NOT NULL DEFAULT '{{}}'")
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_kb_documents_tags ON {s}kb_documents USING GIN (tags)")


def _oracle_upgrade() -> None:
    op.execute("DROP INDEX idx_kb_documents_tags")
    op.execute("ALTER TABLE kb_documents DROP COLUMN tags")


def _oracle_downgrade() -> None:
    op.execute("ALTER TABLE kb_documents ADD tags CLOB CHECK (tags IS JSON)")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
