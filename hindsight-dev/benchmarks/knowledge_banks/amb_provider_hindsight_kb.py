"""Hindsight knowledge banks: plain document retrieval over the HTTP API.

Unlike the `hindsight` providers, this one does no memory extraction. It writes the
documents into a knowledge bank (chunked and embedded server-side) and retrieves chunks
with the bank's hybrid search — the like-for-like comparison against `qdrant`
(dense + sparse + RRF over 512-token chunks).

    HINDSIGHT_HTTP_URL   the API (default http://localhost:8888)
    HINDSIGHT_HTTP_KEY   optional bearer token
    HINDSIGHT_KB_BANK    bank id to use (default amb-kb)
    HINDSIGHT_KB_MODE    hybrid | vector | keyword (default hybrid)
    HINDSIGHT_KB_RERANK  true | false (default: the bank's setting)
    HINDSIGHT_KB_WRITE_BATCH      documents per write operation (default 100, max 500)
    HINDSIGHT_KB_INGEST_TIMEOUT_S how long the whole ingest may take (default 1800)
"""

import os
import time
from pathlib import Path

import httpx

from ..models import Document
from .base import MemoryProvider

# Documents per write operation, and how long the whole corpus may take to land.
# A million-passage corpus (BEIR nq) needs both raised: 500 is the API's per-batch
# ceiling, and its ingest runs for hours, not minutes.
_WRITE_BATCH = int(os.environ.get("HINDSIGHT_KB_WRITE_BATCH", "100"))
_OPERATION_TIMEOUT_S = int(os.environ.get("HINDSIGHT_KB_INGEST_TIMEOUT_S", "1800"))


class HindsightKnowledgeBankProvider(MemoryProvider):
    name = "hindsight-kb"
    description = (
        "Hindsight knowledge bank: documents chunked (512 tokens) and embedded server-side, "
        "retrieved with hybrid vector + keyword search fused by RRF and reranked by the "
        "configured cross-encoder."
    )
    kind = "local"
    provider = "hindsight"
    variant = "knowledge-bank"
    link = "https://hindsight.vectorize.io"
    concurrency = 8

    def __init__(self):
        self._base_url = os.environ.get("HINDSIGHT_HTTP_URL", "http://localhost:8888").rstrip("/")
        self._api_key = os.environ.get("HINDSIGHT_HTTP_KEY", "")
        self._bank = os.environ.get("HINDSIGHT_KB_BANK", "amb-kb")
        self._mode = os.environ.get("HINDSIGHT_KB_MODE", "hybrid")
        rerank = os.environ.get("HINDSIGHT_KB_RERANK")
        self._rerank = None if rerank is None else rerank.strip().lower() in ("1", "true", "yes")
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        self._http = httpx.Client(base_url=self._base_url, headers=headers, timeout=600)

    # ---- lifecycle

    def _kb(self, suffix: str = "") -> str:
        return f"/v1/default/knowledge-banks/{self._bank}{suffix}"

    def prepare(self, store_dir: Path, unit_ids: set[str] | None = None, reset: bool = True) -> None:
        if reset:
            self._http.delete(self._kb())
        created = self._http.post("/v1/default/knowledge-banks", json={"id": self._bank})
        if created.status_code not in (201, 409):
            created.raise_for_status()

    def cleanup(self) -> None:
        self._http.close()

    # ---- ingest

    def ingest(self, documents: list[Document]) -> None:
        operations: list[str] = []
        for start in range(0, len(documents), _WRITE_BATCH):
            payload = [
                {
                    "id": doc.id,
                    "text": doc.content,
                    # Indexed with every chunk. Datasets that carry a real title put
                    # it in `context` (BEIR); 2Wiki's document id IS the page title.
                    "title": doc.context or doc.id,
                    "tags": [f"user:{doc.user_id}"] if doc.user_id else [],
                    "metadata": {"user_id": doc.user_id} if doc.user_id else {},
                }
                for doc in documents[start : start + _WRITE_BATCH]
            ]
            response = self._http.post(self._kb("/documents"), json={"documents": payload})
            response.raise_for_status()
            operations.append(response.json()["operation_id"])
        self._await_operations(operations)

    def _await_operations(self, operation_ids: list[str]) -> None:
        deadline = time.time() + _OPERATION_TIMEOUT_S
        for operation_id in operation_ids:
            while True:
                response = self._http.get(self._kb(f"/operations/{operation_id}"))
                response.raise_for_status()
                status = response.json().get("status")
                if status == "completed":
                    break
                if status in ("failed", "cancelled"):
                    raise RuntimeError(f"knowledge write {operation_id} {status}: {response.json()}")
                if time.time() > deadline:
                    raise TimeoutError(f"knowledge write {operation_id} still {status}")
                time.sleep(1.0)

    # ---- retrieve

    def retrieve(
        self,
        query: str,
        k: int = 10,
        user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[Document], dict | None]:
        body: dict = {"query": query, "top_k": k, "mode": self._mode, "collapse_documents": True}
        if user_id:
            body["tags"] = [f"user:{user_id}"]
        if self._rerank is not None:
            body["rerank"] = self._rerank
        response = self._http.post(self._kb("/search"), json=body)
        response.raise_for_status()
        raw = response.json()
        documents = [
            Document(
                id=f"{hit['document_id']}#{hit['chunk_index']}",
                content=hit["text"],
                user_id=user_id,
                source_ids=[hit["document_id"]],
            )
            for hit in raw.get("results", [])
        ]
        return documents, raw
