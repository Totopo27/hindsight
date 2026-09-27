"""HTTP API for knowledge banks: /v1/default/knowledge-banks.

Its own tree beside /banks: a memory bank's retain/recall/reflect surface means nothing
for documents, and sharing it would put a bank-kind check on every route.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from ..extensions import OperationValidationError
from ..models import RequestContext
from .service import MAX_BATCH_DOCUMENTS, DocumentInput, KnowledgeBankError, KnowledgeService


class CreateKnowledgeBank(BaseModel):
    id: str = Field(min_length=1, max_length=255, description="Bank id, unique among knowledge banks")
    name: str | None = Field(default=None, description="Display name; defaults to the id")


class WriteDocument(BaseModel):
    id: str = Field(min_length=1, max_length=512, description="Caller's document id; writing it again replaces it")
    # Empty text is accepted and stored with zero passages: real corpora carry records
    # whose body extracted to nothing, and one of them must not fail the whole batch.
    text: str
    title: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict, description="Free-form metadata, stored as given")
    fields: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Values for the schema's document fields, supplied instead of extracted. They land in the same "
            "object extraction fills, and a field supplied here is not sent to the LLM for this document."
        ),
    )
    passage_fields: dict[int, dict[str, Any]] = Field(
        default_factory=dict, description="The same, per passage index, for the schema's passage fields"
    )
    schema_id: str | None = Field(
        default=None, description="Which schema fills this document's fields; a bank with one schema uses it"
    )


class WriteRequest(BaseModel):
    documents: list[WriteDocument] = Field(min_length=1, max_length=MAX_BATCH_DOCUMENTS)


class SchemaRequest(BaseModel):
    """A schema: the fields a kind of document has, and the fields its passages have."""

    name: str | None = Field(default=None, description="Human name, e.g. 'Supplier contract'")
    description: str | None = Field(default=None, description="What this kind of document is")
    document_fields: dict[str, Any] = Field(
        default_factory=dict, description="Field name -> {type, description, values, items, source}"
    )
    passage_fields: dict[str, Any] = Field(default_factory=dict, description="The same, filled per passage")


class ExtractRequest(BaseModel):
    doc_ids: list[str] = Field(default_factory=list, description="Only these documents; empty means all")
    only_missing: bool = Field(default=True, description="Skip documents whose fields are already filled")
    schema_id: str | None = Field(default=None, description="Which schema to extract with")


class CollectionRequest(BaseModel):
    """A collection: what one record is, and which field identifies it."""

    name: str | None = Field(default=None, description="Human name, e.g. 'Vendors'")
    description: str | None = Field(default=None, description="What one record of this collection is")
    fields: dict[str, Any] = Field(
        description="Field name -> {type, description, values, items} or {collection: '<other>'} for a relationship"
    )
    identity: str | None = Field(
        default=None,
        description="The field that identifies one record, so the same thing found twice is one row",
    )
    derive_on_write: bool = Field(
        default=False, description="Re-derive this collection's records whenever a document is written"
    )


class RecordsRequest(BaseModel):
    records: list[dict[str, Any]] = Field(
        min_length=1, description="[{record_id?, values, evidence?, doc_ids?}] — written as given, no LLM"
    )


class DeriveRequest(BaseModel):
    doc_ids: list[str] = Field(default_factory=list, description="Only these documents; empty means all")
    replace: bool = Field(default=True, description="Drop what those documents contributed before re-reading them")


class MergeRequest(BaseModel):
    into: str = Field(description="The record that survives; this one is folded into it")


class PinRequest(BaseModel):
    values: dict[str, Any] = Field(min_length=1, description="Values that outrank what the documents say")


class RecordQueryRequest(BaseModel):
    """The document query DSL, over records, with joins across relationship fields."""

    join: list[dict[str, Any]] | None = Field(default=None, description="[{on: <relationship field>, as: <alias>}]")
    select: list[Any] = Field(min_length=1)
    where: dict[str, Any] | None = None
    group_by: list[Any] | None = None
    having: dict[str, Any] | None = None
    order_by: list[Any] | None = None
    limit: int = Field(default=100, ge=0, le=1000)
    offset: int = Field(default=0, ge=0)


class QueryRequest(BaseModel):
    """A query in the knowledge-bank DSL. See hindsight_api/knowledge/query.py."""

    from_: Literal["documents", "passages"] = Field(default="passages", alias="from")
    select: list[Any] = Field(min_length=1, description="Fields, aggregates and expressions over them")
    where: dict[str, Any] | None = Field(default=None, description="The same metadata filter search takes")
    group_by: list[Any] | None = None
    having: dict[str, Any] | None = Field(default=None, description="Conditions on select's aggregate columns")
    order_by: list[Any] | None = None
    limit: int = Field(default=100, ge=0, le=1000)
    offset: int = Field(default=0, ge=0)

    model_config = {"populate_by_name": True}


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=200)
    mode: Literal["hybrid", "vector", "keyword"] = "hybrid"
    tags: list[str] | None = None
    fields: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Field filter: {field: value} or {field: {$gte: 1, $in: [...], $contains: x, $exists: true}}. "
            "Matches passage-level values, then document-level values, then metadata written with the document."
        ),
    )
    rerank: bool | None = Field(default=None, description="Override the bank's rerank setting")
    collapse_documents: bool = Field(
        default=False, description="Best passage per document, so top_k means k distinct documents"
    )


def build_router(get_request_context: Any) -> APIRouter:
    router = APIRouter(prefix="/v1/default/knowledge-banks", tags=["Knowledge Banks"])

    async def service(request: Request, ctx: RequestContext = Depends(get_request_context)) -> KnowledgeService:
        memory = request.app.state.memory
        await memory._authenticate_tenant(ctx)  # resolves the tenant schema for fq_table
        cached = getattr(request.app.state, "knowledge_service", None)
        if cached is None:
            cached = request.app.state.knowledge_service = KnowledgeService(memory)
        bank = request.path_params.get("kb")
        if bank:
            # Gated here, not per route: every route below depends on this, so one added
            # later is gated without anyone remembering to. Search and the query DSL are
            # POSTs that only read, so the direction comes from the path, not the method.
            reads = request.method in ("GET", "HEAD") or request.url.path.endswith(("/search", "/query"))
            try:
                await cached.authorize(bank, write=not reads, request_context=ctx)
            except OperationValidationError as e:
                raise HTTPException(status_code=e.status_code, detail=e.reason)
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
        ctx: RequestContext = Depends(get_request_context),
    ):
        return await run(svc.list_banks(limit, offset, q, request_context=ctx))

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
            DocumentInput(
                doc_id=d.id,
                text=d.text,
                title=d.title,
                tags=d.tags,
                metadata=d.metadata,
                fields=d.fields,
                passage_fields=d.passage_fields,
                schema_id=d.schema_id,
            )
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

    @router.get("/{kb}/documents/{doc_id:path}", summary="One document with its passages")
    async def get_document(kb: str, doc_id: str, svc: KnowledgeService = Depends(service)):
        return await run(svc.get_document(kb, doc_id))

    @router.delete("/{kb}/documents/{doc_id:path}", summary="Delete a document and its passages")
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

    @router.get("/{kb}/schemas", summary="The schemas of this bank")
    async def list_schemas(
        kb: str,
        ctx: RequestContext = Depends(get_request_context),
        svc: KnowledgeService = Depends(service),
    ):
        return await run(svc.list_schemas(kb))

    @router.post("/{kb}/schemas", status_code=201, summary="Define a schema and its fields")
    async def create_schema(
        kb: str,
        body: SchemaRequest,
        schema_id: str = Query(default=KnowledgeService.DEFAULT_SCHEMA_ID, description="Id for this schema"),
        ctx: RequestContext = Depends(get_request_context),
        svc: KnowledgeService = Depends(service),
    ):
        return await run(
            svc.put_schema(
                kb,
                schema_id,
                name=body.name,
                description=body.description,
                document_fields=body.document_fields,
                passage_fields=body.passage_fields,
            )
        )

    @router.get("/{kb}/schemas/{schema_id}", summary="One schema, and how much it has filled")
    async def get_schema(
        kb: str,
        schema_id: str,
        ctx: RequestContext = Depends(get_request_context),
        svc: KnowledgeService = Depends(service),
    ):
        return await run(svc.get_schema(kb, schema_id))

    @router.put("/{kb}/schemas/{schema_id}", summary="Define or redefine a schema")
    async def put_schema(
        kb: str,
        schema_id: str,
        body: SchemaRequest,
        ctx: RequestContext = Depends(get_request_context),
        svc: KnowledgeService = Depends(service),
    ):
        # Takes effect on the next write; documents already written are re-extracted on
        # request (POST fields/extract), because re-reading a corpus costs LLM calls.
        return await run(
            svc.put_schema(
                kb,
                schema_id,
                name=body.name,
                description=body.description,
                document_fields=body.document_fields,
                passage_fields=body.passage_fields,
            )
        )

    @router.delete("/{kb}/schemas/{schema_id}", summary="Delete a schema")
    async def delete_schema(
        kb: str,
        schema_id: str,
        ctx: RequestContext = Depends(get_request_context),
        svc: KnowledgeService = Depends(service),
    ):
        return await run(svc.delete_schema(kb, schema_id))

    @router.get("/{kb}/field-values/{field_name}", summary="How the corpus splits per value of a field")
    async def field_values(
        kb: str,
        field_name: str,
        level: Literal["document", "passages"] = Query(default="document"),
        ctx: RequestContext = Depends(get_request_context),
        svc: KnowledgeService = Depends(service),
    ):
        return await run(svc.field_values(kb, field_name, level))

    @router.post("/{kb}/fields/extract", status_code=202, summary="Fill fields for documents already stored")
    async def extract_fields(
        kb: str,
        body: ExtractRequest,
        ctx: RequestContext = Depends(get_request_context),
        svc: KnowledgeService = Depends(service),
    ):
        return await run(svc.submit_extract(kb, body.doc_ids, body.only_missing, body.schema_id))

    @router.get("/{kb}/collections", summary="The collections of this bank")
    async def list_collections(
        kb: str,
        ctx: RequestContext = Depends(get_request_context),
        svc: KnowledgeService = Depends(service),
    ):
        return await run(svc.list_collections(kb))

    @router.post("/{kb}/collections", status_code=201, summary="Define a collection and its record fields")
    async def create_collection(
        kb: str,
        body: CollectionRequest,
        collection_id: str = Query(description="Id for this collection, e.g. 'vendors'"),
        reprocess: bool = Query(default=False, description="Re-derive the records after the change"),
        ctx: RequestContext = Depends(get_request_context),
        svc: KnowledgeService = Depends(service),
    ):
        return await run(
            svc.put_collection(
                kb,
                collection_id,
                name=body.name,
                description=body.description,
                fields=body.fields,
                identity=body.identity,
                derive_on_write=body.derive_on_write,
                reprocess=reprocess,
            )
        )

    @router.get("/{kb}/collections/{collection_id}", summary="One collection")
    async def get_collection(
        kb: str,
        collection_id: str,
        ctx: RequestContext = Depends(get_request_context),
        svc: KnowledgeService = Depends(service),
    ):
        return await run(svc.get_collection(kb, collection_id))

    @router.put("/{kb}/collections/{collection_id}", summary="Define or redefine a collection")
    async def put_collection(
        kb: str,
        collection_id: str,
        body: CollectionRequest,
        reprocess: bool = Query(default=False, description="Re-derive the records from the documents after the change"),
        ctx: RequestContext = Depends(get_request_context),
        svc: KnowledgeService = Depends(service),
    ):
        return await run(
            svc.put_collection(
                kb,
                collection_id,
                name=body.name,
                description=body.description,
                fields=body.fields,
                identity=body.identity,
                derive_on_write=body.derive_on_write,
                reprocess=reprocess,
            )
        )

    @router.delete("/{kb}/collections/{collection_id}", summary="Delete a collection and its records")
    async def delete_collection(
        kb: str,
        collection_id: str,
        ctx: RequestContext = Depends(get_request_context),
        svc: KnowledgeService = Depends(service),
    ):
        return await run(svc.delete_collection(kb, collection_id))

    @router.post("/{kb}/collections/{collection_id}/records", summary="Write records directly (no LLM)")
    async def put_records(
        kb: str,
        collection_id: str,
        body: RecordsRequest,
        ctx: RequestContext = Depends(get_request_context),
        svc: KnowledgeService = Depends(service),
    ):
        return await run(svc.put_records(kb, collection_id, body.records))

    @router.post(
        "/{kb}/collections/{collection_id}/derive",
        status_code=202,
        summary="Derive this collection's records from the documents",
    )
    async def derive_records(
        kb: str,
        collection_id: str,
        body: DeriveRequest,
        ctx: RequestContext = Depends(get_request_context),
        svc: KnowledgeService = Depends(service),
    ):
        return await run(svc.submit_derive_records(kb, collection_id, body.doc_ids, replace=body.replace))

    @router.post("/{kb}/collections/{collection_id}/query", summary="Query records, joins included")
    async def query_records(
        kb: str,
        collection_id: str,
        body: RecordQueryRequest,
        ctx: RequestContext = Depends(get_request_context),
        svc: KnowledgeService = Depends(service),
    ):
        return await run(svc.query_records(kb, collection_id, body.model_dump(exclude_none=True), request_context=ctx))

    @router.get("/{kb}/collections/{collection_id}/records/{record_id}", summary="One record, with its evidence")
    async def get_record(
        kb: str,
        collection_id: str,
        record_id: str,
        ctx: RequestContext = Depends(get_request_context),
        svc: KnowledgeService = Depends(service),
    ):
        return await run(svc.get_record(kb, collection_id, record_id))

    @router.post(
        "/{kb}/collections/{collection_id}/records/{record_id}/merge",
        summary="Merge this record into another — they are the same thing",
    )
    async def merge_records(
        kb: str,
        collection_id: str,
        record_id: str,
        body: MergeRequest,
        ctx: RequestContext = Depends(get_request_context),
        svc: KnowledgeService = Depends(service),
    ):
        return await run(svc.merge_records(kb, collection_id, record_id, body.into))

    @router.delete("/{kb}/collections/{collection_id}/records/{record_id}", summary="Delete one record")
    async def delete_record(
        kb: str,
        collection_id: str,
        record_id: str,
        ctx: RequestContext = Depends(get_request_context),
        svc: KnowledgeService = Depends(service),
    ):
        return await run(svc.delete_record(kb, collection_id, record_id))

    @router.put(
        "/{kb}/collections/{collection_id}/records/{record_id}/pins",
        summary="Pin corrected values, which outrank the documents",
    )
    async def pin_record(
        kb: str,
        collection_id: str,
        record_id: str,
        body: PinRequest,
        ctx: RequestContext = Depends(get_request_context),
        svc: KnowledgeService = Depends(service),
    ):
        return await run(svc.pin_record_values(kb, collection_id, record_id, body.values))

    @router.post("/{kb}/query", summary="Aggregate and filter over documents and passages")
    async def query(
        kb: str,
        body: QueryRequest,
        ctx: RequestContext = Depends(get_request_context),
        svc: KnowledgeService = Depends(service),
    ):
        # The DSL is compiled to one parameterised statement; nothing the caller sends
        # reaches it as text. See knowledge/query.py for the language and its limits.
        payload = body.model_dump(by_alias=True, exclude_none=True)
        return await run(svc.query(kb, payload, request_context=ctx))

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
                fields=body.fields,
                rerank=body.rerank,
                collapse_documents=body.collapse_documents,
                request_context=ctx,
            )
        )
        return {
            "results": [
                {
                    "document_id": hit.doc_id,
                    "passage_index": hit.passage_index,
                    "text": hit.text,
                    "score": hit.score,
                    "ranks": hit.ranks,
                }
                for hit in hits
            ]
        }

    return router
