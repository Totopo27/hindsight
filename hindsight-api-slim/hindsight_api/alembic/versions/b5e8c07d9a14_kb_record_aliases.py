"""Knowledge banks: record aliases, so a merge sticks

Resolution can be wrong, so a person can merge two records. That decision has to outlive
the merge itself: without somewhere to keep it, the next document naming the losing side
re-creates the duplicate that was just cleaned up.

Revision ID: b5e8c07d9a14
Revises: a9d4f2b71c63
Create Date: 2026-09-27
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "b5e8c07d9a14"
down_revision: str | Sequence[str] | None = "a9d4f2b71c63"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    s = _pg_schema_prefix()
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS {s}kb_record_aliases (
            bank_id TEXT NOT NULL,
            collection_id TEXT NOT NULL,
            -- The normalised key that should resolve to this record.
            alias_key TEXT NOT NULL,
            record_id TEXT NOT NULL,
            -- 'merge' when a person merged two records, 'variant' when resolution
            -- decided it: the first is a judgement to keep, the second is a cache.
            source TEXT NOT NULL DEFAULT 'merge',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (bank_id, collection_id, alias_key),
            FOREIGN KEY (bank_id, collection_id)
                REFERENCES {s}kb_collections(bank_id, collection_id) ON DELETE CASCADE
        )
    """)
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_kb_aliases_record ON {s}kb_record_aliases (bank_id, collection_id, record_id)")  # fmt: skip
    # Similarity matching for the typo case reads this; the extension is already used for
    # entity names, but a bank that only ever used knowledge banks may not have it yet.
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_kb_records_id_trgm ON {s}kb_records USING GIN (record_id gin_trgm_ops)")  # fmt: skip


def _pg_downgrade() -> None:
    s = _pg_schema_prefix()
    op.execute(f"DROP INDEX IF EXISTS {s}idx_kb_records_id_trgm")
    op.execute(f"DROP TABLE IF EXISTS {s}kb_record_aliases")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
