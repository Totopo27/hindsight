"""Knowledge banks v1 over HTTP: create, async batch write, search, delete.

The write path is the real one — a `knowledge_write_batch` operation goes through
`async_operations` and the engine's task dispatch (SyncTaskBackend runs it inline in
tests), so this covers the operation record as well as the chunking and the search.
"""

import json
import uuid

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app
from hindsight_api.knowledge.passages import split_into_passages

DOCS = [
    {
        "id": "milan",
        "title": "Milan",
        "text": "Milan is a city in northern Italy, the capital of Lombardy. "
        "It is known for fashion, design and the Duomo cathedral. "
        "The city hosts the Borsa Italiana, Italy's stock exchange.",
    },
    {
        "id": "turin",
        "title": "Turin",
        "text": "Turin is a city in the Piedmont region of Italy. "
        "It was the first capital of unified Italy and is home to the Fiat car company.",
    },
    {
        "id": "espresso",
        "title": "Espresso",
        "text": "Espresso is brewed by forcing hot water under pressure through finely ground coffee. "
        "It originated in Italy in the early twentieth century.",
        "metadata": {"topic": "food"},
    },
]


@pytest_asyncio.fixture
async def kb_client(memory):
    app = create_app(memory, initialize_memory=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield client


async def _bank(client) -> str:
    kb = f"kb-{uuid.uuid4().hex[:8]}"
    created = await client.post("/v1/default/knowledge-banks", json={"id": kb, "name": "Test"})
    assert created.status_code == 201, created.text
    assert created.json()["documents"] == 0
    return kb


async def _write(client, kb: str, documents: list[dict]) -> dict:
    response = await client.post(f"/v1/default/knowledge-banks/{kb}/documents", json={"documents": documents})
    assert response.status_code == 202, response.text
    return response.json()


@pytest.mark.asyncio
async def test_write_batch_runs_as_an_operation_and_becomes_searchable(kb_client):
    kb = await _bank(kb_client)
    submitted = await _write(kb_client, kb, DOCS)
    assert submitted["documents"] == 3 and submitted["status"] == "pending"

    # The write is an ordinary async operation of the new type, and it completed.
    operation = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/operations/{submitted['operation_id']}")).json()
    assert operation["status"] == "completed"
    listed = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/operations")).json()
    assert [o["task_type"] for o in listed["operations"]] == ["knowledge_write_batch"]

    stats = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}")).json()
    assert stats["documents"] == 3 and stats["passages"] >= 3

    hits = (
        await kb_client.post(f"/v1/default/knowledge-banks/{kb}/search", json={"query": "stock exchange", "top_k": 3})
    ).json()["results"]
    assert hits and hits[0]["document_id"] == "milan"
    assert "Borsa" in hits[0]["text"]

    # Each arm on its own. A hit is a document, a passage index and its text — the
    # fusion score and the per-arm ranks are how the order was reached, not part of the
    # answer, and a score that means nothing across two queries is worse than none.
    assert set(hits[0]) == {"document_id", "passage_index", "text"}
    keyword = (
        await kb_client.post(
            f"/v1/default/knowledge-banks/{kb}/search", json={"query": "Fiat", "mode": "keyword", "top_k": 2}
        )
    ).json()["results"]
    assert keyword[0]["document_id"] == "turin"
    vector = (
        await kb_client.post(
            f"/v1/default/knowledge-banks/{kb}/search",
            json={"query": "car manufacturer", "mode": "vector", "top_k": 2},
        )
    ).json()["results"]
    assert vector[0]["document_id"] == "turin"

    # A field written with the document narrows the search to the documents that carry it.
    # (Metadata is matched by the same filter as a schema field, so this works with no
    # schema at all — which is what a document written with `metadata` gets.)
    food = (
        await kb_client.post(
            f"/v1/default/knowledge-banks/{kb}/search",
            json={"query": "Italy", "fields": {"topic": "food"}, "top_k": 5},
        )
    ).json()["results"]
    assert {h["document_id"] for h in food} == {"espresso"}


@pytest.mark.asyncio
async def test_rewriting_a_document_replaces_it_and_identical_text_is_a_no_op(kb_client):
    kb = await _bank(kb_client)
    await _write(kb_client, kb, DOCS[:1])
    again = await _write(kb_client, kb, DOCS[:1])
    result = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/operations/{again['operation_id']}")).json()
    assert result["result_metadata"]["documents_unchanged"] == 1
    assert result["result_metadata"]["documents_written"] == 0

    changed = [{**DOCS[0], "text": "Milan hosts the Salone del Mobile furniture fair every April."}]
    await _write(kb_client, kb, changed)
    document = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/documents/milan")).json()
    assert "Salone" in document["text"]
    assert len(document["passages"]) == document["passage_count"]
    stale = (
        await kb_client.post(f"/v1/default/knowledge-banks/{kb}/search", json={"query": "Duomo cathedral"})
    ).json()["results"]
    assert all("Duomo" not in hit["text"] for hit in stale)  # the old passages are gone

    deleted = await kb_client.delete(f"/v1/default/knowledge-banks/{kb}/documents/milan")
    assert deleted.status_code == 200
    assert (await kb_client.get(f"/v1/default/knowledge-banks/{kb}")).json()["documents"] == 0
    assert (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/documents/milan")).status_code == 404


@pytest.mark.asyncio
async def test_chunking_follows_the_banks_configured_size(kb_client, memory):
    kb = await _bank(kb_client)
    long_text = " ".join(f"Sentence number {i} about retrieval quality." for i in range(400))
    await _write(kb_client, kb, [{"id": "long", "text": long_text}])
    default_chunks = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/documents/long")).json()["passages"]
    assert max(c["token_count"] for c in default_chunks) <= 512

    patched = await kb_client.patch(f"/v1/default/banks/{kb}/config", json={"updates": {"kb_passage_size": 128}})
    assert patched.status_code == 200, patched.text
    await _write(kb_client, kb, [{"id": "long2", "text": long_text}])
    small_chunks = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/documents/long2")).json()["passages"]
    assert max(c["token_count"] for c in small_chunks) <= 128
    assert len(small_chunks) > len(default_chunks)


@pytest.mark.asyncio
async def test_search_vector_weight_moves_the_fused_order(kb_client):
    """The fusion weight decides which arm wins a disagreement. The query names Turin's
    company but asks about it in Milan's words, so the two arms rank the documents in
    opposite orders; the weight is what picks between them."""
    kb = await _bank(kb_client)
    await _write(kb_client, kb, DOCS)

    async def ranked(weight: float) -> list[str]:
        patched = await kb_client.patch(
            f"/v1/default/banks/{kb}/config", json={"updates": {"kb_search_vector_weight": weight}}
        )
        assert patched.status_code == 200, patched.text
        results = (
            await kb_client.post(
                f"/v1/default/knowledge-banks/{kb}/search",
                json={"query": "Fiat", "top_k": 3, "rerank": False, "collapse_documents": True},
            )
        ).json()["results"]
        return [hit["document_id"] for hit in results]

    # All the weight on the keyword arm: the document that literally says "Fiat" wins.
    assert (await ranked(0.0))[0] == "turin"
    # A single-arm search ignores the weight — the arm it uses is the only voter.
    vector_only = (
        await kb_client.post(
            f"/v1/default/knowledge-banks/{kb}/search",
            json={"query": "Fiat", "top_k": 3, "mode": "vector", "rerank": False, "collapse_documents": True},
        )
    ).json()["results"]
    assert vector_only, "vector-only search returned nothing"


@pytest.mark.asyncio
async def test_bad_requests_are_refused(kb_client):
    kb = await _bank(kb_client)
    assert (await kb_client.post("/v1/default/knowledge-banks", json={"id": kb})).status_code == 409
    assert (await kb_client.get("/v1/default/knowledge-banks/nope")).status_code == 404
    assert (await kb_client.post("/v1/default/knowledge-banks/nope/search", json={"query": "x"})).status_code == 404
    duplicate = await kb_client.post(
        f"/v1/default/knowledge-banks/{kb}/documents",
        json={"documents": [{"id": "a", "text": "one"}, {"id": "a", "text": "two"}]},
    )
    assert duplicate.status_code == 400 and "duplicate" in duplicate.json()["detail"]
    assert (
        await kb_client.post(f"/v1/default/knowledge-banks/{kb}/documents", json={"documents": []})
    ).status_code == 422


@pytest.mark.asyncio
async def test_a_document_with_no_text_is_stored_with_no_chunks(kb_client):
    """Corpora carry records whose body extracted to nothing. Writing one must not fail the
    batch around it: the document is stored, contributes no passages, and is never a hit."""
    kb = await _bank(kb_client)
    await _write(kb_client, kb, [{"id": "empty", "text": ""}, {"id": "real", "text": "Turin has Fiat."}])

    listed = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/documents")).json()["items"]
    assert {d["doc_id"]: d["passage_count"] for d in listed} == {"empty": 0, "real": 1}
    hits = (
        await kb_client.post(f"/v1/default/knowledge-banks/{kb}/search", json={"query": "Fiat", "top_k": 5})
    ).json()["results"]
    assert [hit["document_id"] for hit in hits] == ["real"]


def test_chunk_overlap_repeats_the_tail_of_the_previous_chunk():
    text = " ".join(f"word{i}" for i in range(2000))
    passages = split_into_passages(text, passage_size=64, passage_overlap=16)
    assert len(passages) > 1
    assert all(c.token_count <= 64 for c in passages)
    tail = passages[0].text.split()[-3:]
    assert " ".join(tail) in passages[1].text


def test_embedding_text_does_not_repeat_a_title_the_chunk_already_opens_with():
    from hindsight_api.knowledge.service import embedding_text

    assert embedding_text("Milan", "Milan is in Lombardy.") == "Milan is in Lombardy."
    assert embedding_text("Milan", "It is in Lombardy.") == "Milan\n\nIt is in Lombardy."
    assert embedding_text(None, "It is in Lombardy.") == "It is in Lombardy."
    # Case and leading whitespace are not a reason to repeat it.
    assert embedding_text("MILAN", "  Milan is in Lombardy.") == "  Milan is in Lombardy."


@pytest.mark.asyncio
async def test_a_documents_title_is_searchable_from_every_chunk(kb_client):
    kb = await _bank(kb_client)
    # The body never names its subject — only the title does, as in encyclopedia text.
    await _write(
        kb_client,
        kb,
        [{"id": "p1", "title": "Barack Obama", "text": "He served as the 44th president of the United States."}],
    )
    keyword = (
        await kb_client.post(
            f"/v1/default/knowledge-banks/{kb}/search", json={"query": "Obama", "mode": "keyword", "top_k": 3}
        )
    ).json()["results"]
    assert [hit["document_id"] for hit in keyword] == ["p1"]
    assert "Obama" not in keyword[0]["text"]  # the text is returned as written


@pytest.mark.asyncio
async def test_collapse_documents_returns_one_chunk_per_document(kb_client):
    kb = await _bank(kb_client)
    long_text = " ".join(f"Milan hosts fair number {i} for design and fashion." for i in range(300))
    await _write(kb_client, kb, [{"id": "milan", "text": long_text}, {"id": "turin", "text": "Turin has Fiat."}])
    plain = (
        await kb_client.post(
            f"/v1/default/knowledge-banks/{kb}/search", json={"query": "Milan design fair", "top_k": 3}
        )
    ).json()["results"]
    assert len({hit["document_id"] for hit in plain}) < len(plain)  # several passages of one document
    collapsed = (
        await kb_client.post(
            f"/v1/default/knowledge-banks/{kb}/search",
            json={"query": "Milan design fair", "top_k": 3, "collapse_documents": True},
        )
    ).json()["results"]
    assert len({hit["document_id"] for hit in collapsed}) == len(collapsed)


def _kb_mcp_tools(memory):
    """The knowledge-bank MCP tools, registered on a real engine."""
    from fastmcp import FastMCP

    from hindsight_api.mcp_tools import MCPToolsConfig, register_mcp_tools

    mcp = FastMCP("test")
    tools = {
        "list_knowledge_banks",
        "list_knowledge_schemas",
        "search_knowledge_bank",
        "query_knowledge_bank",
        "list_knowledge_collections",
        "query_knowledge_records",
    }
    register_mcp_tools(mcp, memory, MCPToolsConfig(bank_id_resolver=lambda: None, tools=tools))
    return {
        k.split(":")[1].split("@")[0]: v for k, v in mcp._local_provider._components.items() if k.startswith("tool:")
    }


@pytest.mark.asyncio
async def test_mcp_tools_read_a_knowledge_bank(kb_client, memory):
    """The MCP surface answers the same questions the HTTP one does.

    A knowledge bank is addressed by its own id, not by the session bank, so these tools
    take it as an argument — which is also why a caller needs list_knowledge_banks to
    find one at all.
    """
    kb = await _bank(kb_client)
    await _write(kb_client, kb, DOCS)
    tools = _kb_mcp_tools(memory)

    banks = await tools["list_knowledge_banks"].fn()
    assert kb in [bank["bank_id"] for bank in banks["items"]]

    hits = await tools["search_knowledge_bank"].fn(knowledge_bank_id=kb, query="Italian stock exchange", top_k=3)
    assert hits["results"][0]["document_id"] == "milan"

    counted = await tools["query_knowledge_bank"].fn(knowledge_bank_id=kb, source="documents", select=[{"count": "*"}])
    assert counted["rows"] == [[len(DOCS)]]

    # No schemas and no collections is an empty answer, not an error.
    assert (await tools["list_knowledge_schemas"].fn(knowledge_bank_id=kb))["items"] == []
    assert (await tools["list_knowledge_collections"].fn(knowledge_bank_id=kb))["items"] == []

    # An unknown bank comes back as a message the model can read, not a stack trace.
    missing = await tools["search_knowledge_bank"].fn(knowledge_bank_id="nope", query="x")
    assert "error" in missing


@pytest.mark.asyncio
async def test_a_tenant_extension_can_refuse_a_knowledge_bank(kb_client, memory):
    """The validator gates the knowledge surface the way it gates a memory bank's.

    Reads and writes are separate names, and the direction comes from the path rather
    than the method: search and the query DSL are POSTs that only read.
    """
    from unittest.mock import AsyncMock, MagicMock

    from hindsight_api.extensions import BankListResult, BankReadOperation, ValidationResult

    kb = await _bank(kb_client)
    await _write(kb_client, kb, DOCS)

    validator = MagicMock()
    validator.validate_bank_read = AsyncMock(
        side_effect=lambda ctx: (
            ValidationResult.reject("not yours")
            if ctx.operation is BankReadOperation.KNOWLEDGE_BANK_READ
            else ValidationResult.accept()
        )
    )
    validator.validate_bank_write = AsyncMock(return_value=ValidationResult.accept())
    validator.filter_bank_list = AsyncMock(side_effect=lambda ctx: BankListResult(banks=ctx.banks))
    memory._operation_validator = validator
    try:
        search = await kb_client.post(f"/v1/default/knowledge-banks/{kb}/search", json={"query": "Milan"})
        assert search.status_code == 403
        assert search.json()["detail"] == "not yours"
        assert (await kb_client.get(f"/v1/default/knowledge-banks/{kb}")).status_code == 403
        # A write is a different name, so refusing reads does not refuse it.
        assert (await kb_client.get("/v1/default/knowledge-banks")).status_code == 200
        assert (
            await kb_client.post(
                f"/v1/default/knowledge-banks/{kb}/documents",
                json={"documents": [{"id": "x", "text": "Rome is a city."}]},
            )
        ).status_code == 202

        # MCP goes through the same gate, or it would be a way around the extension.
        refused = await _kb_mcp_tools(memory)["search_knowledge_bank"].fn(knowledge_bank_id=kb, query="Milan")
        assert "not yours" in refused["error"]
    finally:
        memory._operation_validator = None


@pytest.mark.asyncio
async def test_a_document_written_without_an_id_gets_one(kb_client):
    """An id is the caller's handle for replacing a document, not a thing they must invent."""
    kb = await _bank(kb_client)
    response = await kb_client.post(
        f"/v1/default/knowledge-banks/{kb}/documents",
        json={"documents": [{"text": "Genoa is a port city in Liguria."}]},
    )
    assert response.status_code == 202
    documents = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/documents")).json()["items"]
    assert len(documents) == 1
    uuid.UUID(documents[0]["doc_id"])  # a uuid, not an empty string or the title


@pytest.mark.asyncio
async def test_an_uploaded_file_becomes_a_searchable_document(kb_client):
    """A file is converted and then written, so it lands as an ordinary document.

    Two operations, like the memory side: the conversion, then the write it queues.
    """
    kb = await _bank(kb_client)
    response = await kb_client.post(
        f"/v1/default/knowledge-banks/{kb}/files",
        files={"files": ("liguria.txt", b"Genoa is the capital of Liguria and its largest port.", "text/plain")},
        data={"request": json.dumps({"metadata": {"source": "upload"}})},
    )
    assert response.status_code == 202, response.text
    assert len(response.json()["operation_ids"]) == 1

    documents = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/documents")).json()["items"]
    assert len(documents) == 1
    assert documents[0]["title"] == "liguria.txt"
    assert documents[0]["metadata"]["source"] == "upload"
    # The file it came from stays on the document, which is the only way back to the bytes.
    assert documents[0]["metadata"]["file_original_name"] == "liguria.txt"

    hits = (await kb_client.post(f"/v1/default/knowledge-banks/{kb}/search", json={"query": "port of Liguria"})).json()[
        "results"
    ]
    assert hits[0]["document_id"] == documents[0]["doc_id"]
