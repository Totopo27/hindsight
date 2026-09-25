"""HTTP API for knowledge banks: /v1/default/knowledge-banks.

Its own tree beside /banks: a memory bank's retain/recall/reflect surface means nothing
for documents, and sharing it would put a bank-kind check on every route.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from ..models import RequestContext
from .service import MAX_BATCH_DOCUMENTS, DocumentInput, KnowledgeBankError, KnowledgeService


class CreateKnowledgeBank(BaseModel):
    id: str = Field(min_length=1, max_length=255, description="Bank id, unique among knowledge banks")
    name: str | None = Field(default=None, description="Display name; defaults to the id")


class WriteDocument(BaseModel):
    id: str = Field(min_length=1, max_length=512, description="Caller's document id; writing it again replaces it")
    # Empty text is accepted and stored with zero chunks: real corpora carry records
    # whose body extracted to nothing, and one of them must not fail the whole batch.
    text: str
    title: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WriteRequest(BaseModel):
    documents: list[WriteDocument] = Field(min_length=1, max_length=MAX_BATCH_DOCUMENTS)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=200)
    mode: Literal["hybrid", "vector", "keyword"] = "hybrid"
    tags: list[str] | None = None
    rerank: bool | None = Field(default=None, description="Override the bank's rerank setting")
    collapse_documents: bool = Field(
        default=False, description="Best chunk per document, so top_k means k distinct documents"
    )


def build_router(get_request_context: Any) -> APIRouter:
    router = APIRouter(prefix="/v1/default/knowledge-banks", tags=["Knowledge Banks"])

    async def service(request: Request, ctx: RequestContext = Depends(get_request_context)) -> KnowledgeService:
        memory = request.app.state.memory
        await memory._authenticate_tenant(ctx)  # resolves the tenant schema for fq_table
        cached = getattr(request.app.state, "knowledge_service", None)
        if cached is None:
            cached = request.app.state.knowledge_service = KnowledgeService(memory)
        return cached

    async def run(coro: Any) -> Any:
        try:
            return await coro
        except KnowledgeBankError as e:
            raise HTTPException(status_code=e.status_code, detail=e.detail)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.post("", summary="Create a knowledge bank", status_code=201)
    async def create_bank(body: CreateKnowledgeBank, svc: KnowledgeService = Depends(service)):
        return await run(svc.create_bank(body.id, body.name))

    @router.get("", summary="List knowledge banks")
    async def list_banks(
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        q: str | None = Query(default=None, description="Filter by id or name"),
        svc: KnowledgeService = Depends(service),
    ):
        return await run(svc.list_banks(limit, offset, q))

    @router.get("/{kb}", summary="Knowledge bank stats")
    async def get_bank(kb: str, svc: KnowledgeService = Depends(service)):
        return await run(svc.get_bank(kb))

    @router.delete("/{kb}", summary="Delete a knowledge bank and everything in it")
    async def delete_bank(kb: str, svc: KnowledgeService = Depends(service)):
        return await run(svc.delete_bank(kb))

    @router.post(
        "/{kb}/documents",
        summary="Write a batch of documents (async)",
        description="Queues one knowledge_write_batch operation and returns its id. Track it through "
        "the bank's operations endpoints. Writing a document id again replaces it; identical text is a no-op.",
        status_code=202,
    )
    async def write_documents(kb: str, body: WriteRequest, svc: KnowledgeService = Depends(service)):
        documents = [
            DocumentInput(doc_id=d.id, text=d.text, title=d.title, tags=d.tags, metadata=d.metadata)
            for d in body.documents
        ]
        return await run(svc.submit_write(kb, documents))

    @router.get("/{kb}/documents", summary="List documents")
    async def list_documents(
        kb: str,
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        q: str | None = Query(default=None, description="Filter by document id or title"),
        svc: KnowledgeService = Depends(service),
    ):
        return await run(svc.list_documents(kb, limit, offset, q))

    @router.get("/{kb}/documents/{doc_id:path}", summary="One document with its chunks")
    async def get_document(kb: str, doc_id: str, svc: KnowledgeService = Depends(service)):
        return await run(svc.get_document(kb, doc_id))

    @router.delete("/{kb}/documents/{doc_id:path}", summary="Delete a document and its chunks")
    async def delete_document(kb: str, doc_id: str, svc: KnowledgeService = Depends(service)):
        return await run(svc.delete_document(kb, doc_id))

    @router.get("/{kb}/operations", summary="Write operations for this bank")
    async def list_operations(
        kb: str,
        status: str | None = Query(default=None, description="pending, processing, completed, failed"),
        limit: int = Query(default=20, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        ctx: RequestContext = Depends(get_request_context),
        svc: KnowledgeService = Depends(service),
    ):
        # The operations substrate is the memory banks': a knowledge bank is a bank row,
        # so its write batches are ordinary async_operations of one new type.
        await run(svc.get_bank(kb))
        return await svc.memory.list_operations(kb, status=status, limit=limit, offset=offset, request_context=ctx)

    @router.get("/{kb}/operations/{operation_id}", summary="One operation")
    async def get_operation(
        kb: str,
        operation_id: str,
        ctx: RequestContext = Depends(get_request_context),
        svc: KnowledgeService = Depends(service),
    ):
        await run(svc.get_bank(kb))
        result = await svc.memory.get_operation_status(kb, operation_id, request_context=ctx)
        if result is None:
            raise HTTPException(status_code=404, detail="operation not found")
        return result

    @router.post("/{kb}/search", summary="Hybrid search over the documents")
    async def search(
        kb: str,
        body: SearchRequest,
        ctx: RequestContext = Depends(get_request_context),
        svc: KnowledgeService = Depends(service),
    ):
        hits = await run(
            svc.search(
                kb,
                body.query,
                top_k=body.top_k,
                mode=body.mode,
                tags=body.tags,
                rerank=body.rerank,
                collapse_documents=body.collapse_documents,
                request_context=ctx,
            )
        )
        return {
            "results": [
                {
                    "document_id": hit.doc_id,
                    "chunk_index": hit.chunk_index,
                    "text": hit.text,
                    "score": hit.score,
                    "ranks": hit.ranks,
                }
                for hit in hits
            ]
        }

    return router
