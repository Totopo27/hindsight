"""Moving a knowledge bank: what an archive carries, and how it is read back.

Two scopes, the same pair a memory bank's transfer offers:

* ``config`` — the bank row and the shapes it works with: its schemas and its
  collection definitions. Config alone *is* the template format: a bank exported
  with ``include_data=false`` is a bank someone else can start from.
* ``data`` — the documents (text, fields, metadata) and the records derived from
  them, carried as their per-document contributions so the importer rebuilds each
  record from its parts rather than trusting a folded copy.

Passages and embeddings are never carried. The importer re-passages and re-embeds
with the target bank's own settings, so an archive moves between instances
configured with different embedding models, and a document's stored fields travel
with it so the import costs no LLM tokens.
"""

from __future__ import annotations

import json
import logging
import zipfile
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from typing import Any

from ..engine.db_utils import acquire_with_retry
from ..engine.schema import fq_table
from ..engine.transfer.stream_archive import ZipStreamer

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
ARCHIVE_KIND = "knowledge"

#: Documents go one JSON object per line so neither side ever holds the corpus.
DOCUMENTS_ENTRY = "documents.jsonl"
MANIFEST_ENTRY = "manifest.json"

_EXPORT_BATCH = 100


@dataclass(frozen=True)
class KnowledgeTransferScope:
    """What an archive carries. Neither flag implies the other."""

    data: bool = True
    config: bool = True


@dataclass(frozen=True)
class KnowledgeArchive:
    """A parsed archive: everything but the documents, which stay on the stream."""

    source_bank_id: str
    name: str | None
    config: dict[str, Any]
    schemas: list[dict[str, Any]]
    collections: list[dict[str, Any]]
    contributions: list[dict[str, Any]]
    pinned: list[dict[str, Any]]
    aliases: list[dict[str, Any]]
    document_count: int
    has_data: bool
    has_config: bool
    raw: bytes

    def documents(self) -> list[dict[str, Any]]:
        """The archive's documents, read back from the JSONL entry."""
        with zipfile.ZipFile(BytesIO(self.raw)) as zf:
            if DOCUMENTS_ENTRY not in zf.namelist():
                return []
            with zf.open(DOCUMENTS_ENTRY) as handle:
                return [json.loads(line) for line in handle if line.strip()]


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _loaded(row: Any, *columns: str) -> dict[str, Any]:
    """A row as a dict, with its JSONB columns parsed (this pool has no JSON codec)."""
    out = {k: v for k, v in dict(row).items()}
    for column in columns:
        if isinstance(out.get(column), str):
            out[column] = json.loads(out[column])
    return out


async def stream_export(
    pool: Any,
    bank_id: str,
    *,
    scope: KnowledgeTransferScope,
) -> AsyncIterator[bytes]:
    """Stream the bank as the chunks of a ZIP archive.

    Sections other than the documents are small by construction (a bank has tens of
    schemas and collections, not millions), so they are read whole; the documents are
    read a page at a time and written as they go.
    """
    streamer = ZipStreamer()
    document_count = 0

    async def write_json(entry: str, payload: Any) -> AsyncIterator[bytes]:
        async for chunk in streamer.write_file_bytes(entry, json.dumps(payload, default=_json_default).encode()):
            yield chunk

    if scope.config:
        async with acquire_with_retry(pool) as conn:
            bank = await conn.fetchrow(
                f"SELECT bank_id, name, config FROM {fq_table('banks')} WHERE bank_id = $1",
                bank_id,
            )
            schemas = await conn.fetch(
                f"SELECT schema_id, name, description, document_fields, passage_fields "
                f"FROM {fq_table('kb_schemas')} WHERE bank_id = $1 ORDER BY schema_id",
                bank_id,
            )
            collections = await conn.fetch(
                f"SELECT collection_id, name, description, fields, identity, derive_on_write "
                f"FROM {fq_table('kb_collections')} WHERE bank_id = $1 ORDER BY collection_id",
                bank_id,
            )
        bank_row = _loaded(bank, "config") if bank else {"bank_id": bank_id, "name": None, "config": {}}
        async for chunk in write_json("bank.json", bank_row):
            yield chunk
        async for chunk in write_json(
            "schemas.json", [_loaded(row, "document_fields", "passage_fields") for row in schemas]
        ):
            yield chunk
        async for chunk in write_json("collections.json", [_loaded(row, "fields") for row in collections]):
            yield chunk

    if scope.data:

        async def document_lines() -> AsyncIterator[bytes]:
            nonlocal document_count
            offset = 0
            while True:
                async with acquire_with_retry(pool) as conn:
                    rows = await conn.fetch(
                        f"SELECT doc_id, title, text, metadata, fields, schema_id "
                        f"FROM {fq_table('kb_documents')} WHERE bank_id = $1 "
                        f"ORDER BY doc_id LIMIT {_EXPORT_BATCH} OFFSET $2",
                        bank_id,
                        offset,
                    )
                    if not rows:
                        return
                    # The passage fields of a document travel with it: they are keyed by
                    # passage index, and the importer re-passages with the same settings.
                    passage_fields = await conn.fetch(
                        f"SELECT doc_id, passage_index, fields FROM {fq_table('kb_passages')} "
                        f"WHERE bank_id = $1 AND doc_id = ANY($2::text[]) AND fields <> '{{}}'::jsonb",
                        bank_id,
                        [row["doc_id"] for row in rows],
                    )
                by_document: dict[str, dict[str, Any]] = {}
                for row in passage_fields:
                    loaded = _loaded(row, "fields")
                    by_document.setdefault(row["doc_id"], {})[str(row["passage_index"])] = loaded["fields"]
                for row in rows:
                    document = _loaded(row, "metadata", "fields")
                    document["passage_fields"] = by_document.get(row["doc_id"], {})
                    document_count += 1
                    yield (json.dumps(document, default=_json_default) + "\n").encode()
                offset += len(rows)

        async for chunk in streamer.write_file_chunks(DOCUMENTS_ENTRY, document_lines()):
            yield chunk

        async with acquire_with_retry(pool) as conn:
            contributions = await conn.fetch(
                f"SELECT collection_id, record_id, doc_id, values, evidence "
                f"FROM {fq_table('kb_record_contributions')} WHERE bank_id = $1 "
                f"ORDER BY collection_id, record_id, doc_id",
                bank_id,
            )
            # Only the pinned values are taken from the record itself: everything else
            # about it is rebuilt from the contributions above.
            pinned = await conn.fetch(
                f"SELECT collection_id, record_id, pinned FROM {fq_table('kb_records')} "
                f"WHERE bank_id = $1 AND pinned <> '{{}}'::jsonb ORDER BY collection_id, record_id",
                bank_id,
            )
            aliases = await conn.fetch(
                f"SELECT collection_id, alias_key, record_id, source FROM {fq_table('kb_record_aliases')} "
                f"WHERE bank_id = $1 ORDER BY collection_id, alias_key",
                bank_id,
            )
        async for chunk in write_json(
            "records.json",
            {
                "contributions": [_loaded(row, "values", "evidence") for row in contributions],
                "pinned": [_loaded(row, "pinned") for row in pinned],
                "aliases": [dict(row) for row in aliases],
            },
        ):
            yield chunk

    async for chunk in write_json(
        MANIFEST_ENTRY,
        {
            "schema_version": SCHEMA_VERSION,
            "kind": ARCHIVE_KIND,
            "source_bank_id": bank_id,
            "exported_at": datetime.now(UTC).isoformat(),
            "scope": {"data": scope.data, "config": scope.config},
            "document_count": document_count,
        },
    ):
        yield chunk
    yield streamer.finish()

    logger.info(
        "knowledge export bank=%s data=%s config=%s documents=%d",
        bank_id,
        scope.data,
        scope.config,
        document_count,
    )


class KnowledgeArchiveError(ValueError):
    """The bytes handed in are not a knowledge-bank archive this version can read."""


def parse_archive(archive_bytes: bytes) -> KnowledgeArchive:
    """Read an archive's manifest and small sections. Raises on anything unreadable.

    Called on the request path so a malformed upload is a 400 the caller sees, rather
    than a background operation they have to go and read.
    """
    try:
        zf = zipfile.ZipFile(BytesIO(archive_bytes))
    except zipfile.BadZipFile as e:
        raise KnowledgeArchiveError("not a ZIP archive") from e
    with zf:
        names = set(zf.namelist())
        if MANIFEST_ENTRY not in names:
            raise KnowledgeArchiveError("archive has no manifest.json")
        manifest = json.loads(zf.read(MANIFEST_ENTRY))
        if manifest.get("kind") != ARCHIVE_KIND:
            raise KnowledgeArchiveError(f"archive is a {manifest.get('kind') or 'memory'} bank, not a knowledge bank")
        if int(manifest.get("schema_version", 0)) > SCHEMA_VERSION:
            raise KnowledgeArchiveError(
                f"archive was written by a newer version (schema_version {manifest['schema_version']})"
            )

        def section(entry: str, fallback: Any) -> Any:
            return json.loads(zf.read(entry)) if entry in names else fallback

        bank = section("bank.json", {})
        records = section("records.json", {})
        return KnowledgeArchive(
            source_bank_id=manifest["source_bank_id"],
            name=bank.get("name"),
            config=bank.get("config") or {},
            schemas=section("schemas.json", []),
            collections=section("collections.json", []),
            contributions=records.get("contributions") or [],
            pinned=records.get("pinned") or [],
            aliases=records.get("aliases") or [],
            document_count=int(manifest.get("document_count") or 0),
            has_data=bool((manifest.get("scope") or {}).get("data")),
            has_config=bool((manifest.get("scope") or {}).get("config")),
            raw=archive_bytes,
        )
