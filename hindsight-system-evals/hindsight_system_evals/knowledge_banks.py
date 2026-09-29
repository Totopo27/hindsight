"""Driving a knowledge bank over its public HTTP API.

Every other suite here goes through the published Python client, which is the rule:
an eval that reaches past the API measures something no user can run. This one
cannot, yet — the generated client has no knowledge-banks surface, because the
`/v1/default/knowledge-banks` router post-dates the last `scripts/generate-clients.sh`
run. So this module is a thin wrapper over the same HTTP endpoints the client would
expose, and nothing more: no engine import, no SQL, no server internals. When the
client gains the surface, this file is what gets deleted.

Writes are asynchronous — a document write and a derivation are both operations —
so every call that submits one waits for it here. A suite that polls its own
operations reads as a suite about polling.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)

#: A derivation is one LLM call per document per collection, so a corpus of a
#: dozen documents can sit for a while behind a slow provider.
OPERATION_TIMEOUT_S = 900.0
_POLL_S = 1.0


class KnowledgeBank:
    """One bank, over HTTP. Constructed per test, closed by the fixture."""

    def __init__(self, base_url: str, bank_id: str, api_key: str | None = None) -> None:
        self.bank_id = bank_id
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers, timeout=120.0)

    async def aclose(self) -> None:
        await self._http.aclose()

    def _path(self, suffix: str = "") -> str:
        return f"/v1/default/knowledge-banks/{self.bank_id}{suffix}"

    async def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        response = await self._http.request(method, path, **kwargs)
        if response.status_code >= 400:
            raise RuntimeError(f"{method} {path} -> {response.status_code}: {response.text[:400]}")
        return response.json() if response.content else {}

    # ---- lifecycle

    async def create(self, name: str | None = None) -> None:
        await self._json("POST", "/v1/default/knowledge-banks", json={"id": self.bank_id, "name": name})

    async def delete(self) -> None:
        await self._http.delete(self._path())

    # ---- writing

    async def write(self, documents: list[dict[str, Any]]) -> None:
        """Write documents and wait for the batch to land."""
        submitted = await self._json("POST", self._path("/documents"), json={"documents": documents})
        await self.await_operation(submitted["operation_id"])

    async def delete_document(self, doc_id: str) -> None:
        await self._json("DELETE", self._path(f"/documents/{doc_id}"))

    async def put_collection(self, collection_id: str, definition: dict[str, Any]) -> None:
        await self._json("PUT", self._path(f"/collections/{collection_id}"), json=definition)

    async def derive(self, collection_id: str, doc_ids: list[str] | None = None, *, replace: bool = True) -> None:
        submitted = await self._json(
            "POST",
            self._path(f"/collections/{collection_id}/derive"),
            json={"doc_ids": doc_ids or [], "replace": replace},
        )
        await self.await_operation(submitted["operation_id"])

    # ---- reading

    async def records(self, collection_id: str) -> list[dict[str, Any]]:
        page = await self._json("GET", self._path(f"/collections/{collection_id}/records"), params={"limit": 500})
        return page["items"]

    async def record(self, collection_id: str, record_id: str) -> dict[str, Any]:
        return await self._json("GET", self._path(f"/collections/{collection_id}/records/{record_id}"))

    async def query(self, collection_id: str, body: dict[str, Any]) -> list[dict[str, Any]]:
        """A record query, returned as dicts rather than the wire's columns + rows."""
        payload = await self._json("POST", self._path(f"/collections/{collection_id}/query"), json=body)
        return [dict(zip(payload["columns"], row, strict=True)) for row in payload["rows"]]

    async def document(self, doc_id: str) -> dict[str, Any]:
        return await self._json("GET", self._path(f"/documents/{doc_id}"))

    # ---- operations

    async def await_operation(self, operation_id: str) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + OPERATION_TIMEOUT_S
        while True:
            operation = await self._json("GET", self._path(f"/operations/{operation_id}"))
            status = operation.get("status")
            if status == "completed":
                return operation
            if status in ("failed", "cancelled"):
                raise RuntimeError(f"operation {operation_id} {status}: {operation.get('error_message')}")
            if asyncio.get_running_loop().time() > deadline:
                raise TimeoutError(f"operation {operation_id} still {status} after {OPERATION_TIMEOUT_S}s")
            await asyncio.sleep(_POLL_S)
