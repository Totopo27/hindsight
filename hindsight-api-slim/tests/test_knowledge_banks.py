"""Knowledge banks v1 over HTTP: create, async batch write, search, delete.

The write path is the real one — a `knowledge_write_batch` operation goes through
`async_operations` and the engine's task dispatch (SyncTaskBackend runs it inline in
tests), so this covers the operation record as well as the chunking and the search.
"""

import uuid

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app
from hindsight_api.knowledge.chunking import chunk_document

DOCS = [
    {
        "id": "milan",
        "title": "Milan",
        "text": "Milan is a city in northern Italy, the capital of Lombardy. "
        "It is known for fashion, design and the Duomo cathedral. "
        "The city hosts the Borsa Italiana, Italy's stock exchange.",
        "tags": ["cities", "italy"],
    },
    {
        "id": "turin",
        "title": "Turin",
        "text": "Turin is a city in the Piedmont region of Italy. "
        "It was the first capital of unified Italy and is home to the Fiat car company.",
        "tags": ["cities", "italy"],
    },
    {
        "id": "espresso",
        "title": "Espresso",
        "text": "Espresso is brewed by forcing hot water under pressure through finely ground coffee. "
        "It originated in Italy in the early twentieth century.",
        "tags": ["food"],
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
    assert stats["documents"] == 3 and stats["chunks"] >= 3

    hits = (
        await kb_client.post(f"/v1/default/knowledge-banks/{kb}/search", json={"query": "stock exchange", "top_k": 3})
    ).json()["results"]
    assert hits and hits[0]["document_id"] == "milan"
    assert "Borsa" in hits[0]["text"]

    # Each arm on its own, and the ranks that produced the fused order.
    keyword = (
        await kb_client.post(
            f"/v1/default/knowledge-banks/{kb}/search", json={"query": "Fiat", "mode": "keyword", "top_k": 2}
        )
    ).json()["results"]
    assert keyword[0]["document_id"] == "turin" and set(keyword[0]["ranks"]) == {"keyword"}
    vector = (
        await kb_client.post(
            f"/v1/default/knowledge-banks/{kb}/search", json={"query": "car manufacturer", "mode": "vector", "top_k": 2}
        )
    ).json()["results"]
    assert set(vector[0]["ranks"]) == {"vector"}

    # Tags narrow the search to the documents that carry them.
    food = (
        await kb_client.post(
            f"/v1/default/knowledge-banks/{kb}/search", json={"query": "Italy", "tags": ["food"], "top_k": 5}
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
    assert len(document["chunks"]) == document["chunk_count"]
    stale = (
        await kb_client.post(f"/v1/default/knowledge-banks/{kb}/search", json={"query": "Duomo cathedral"})
    ).json()["results"]
    assert all("Duomo" not in hit["text"] for hit in stale)  # the old chunks are gone

    deleted = await kb_client.delete(f"/v1/default/knowledge-banks/{kb}/documents/milan")
    assert deleted.status_code == 200
    assert (await kb_client.get(f"/v1/default/knowledge-banks/{kb}")).json()["documents"] == 0
    assert (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/documents/milan")).status_code == 404


@pytest.mark.asyncio
async def test_chunking_follows_the_banks_configured_size(kb_client, memory):
    kb = await _bank(kb_client)
    long_text = " ".join(f"Sentence number {i} about retrieval quality." for i in range(400))
    await _write(kb_client, kb, [{"id": "long", "text": long_text}])
    default_chunks = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/documents/long")).json()["chunks"]
    assert max(c["token_count"] for c in default_chunks) <= 512

    patched = await kb_client.patch(f"/v1/default/banks/{kb}/config", json={"updates": {"kb_chunk_size": 128}})
    assert patched.status_code == 200, patched.text
    await _write(kb_client, kb, [{"id": "long2", "text": long_text}])
    small_chunks = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/documents/long2")).json()["chunks"]
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
    batch around it: the document is stored, contributes no chunks, and is never a hit."""
    kb = await _bank(kb_client)
    await _write(kb_client, kb, [{"id": "empty", "text": ""}, {"id": "real", "text": "Turin has Fiat."}])

    listed = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/documents")).json()["items"]
    assert {d["doc_id"]: d["chunk_count"] for d in listed} == {"empty": 0, "real": 1}
    hits = (
        await kb_client.post(f"/v1/default/knowledge-banks/{kb}/search", json={"query": "Fiat", "top_k": 5})
    ).json()["results"]
    assert [hit["document_id"] for hit in hits] == ["real"]


def test_chunk_overlap_repeats_the_tail_of_the_previous_chunk():
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = chunk_document(text, chunk_size=64, chunk_overlap=16)
    assert len(chunks) > 1
    assert all(c.token_count <= 64 for c in chunks)
    tail = chunks[0].text.split()[-3:]
    assert " ".join(tail) in chunks[1].text


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
    assert len({hit["document_id"] for hit in plain}) < len(plain)  # several chunks of one document
    collapsed = (
        await kb_client.post(
            f"/v1/default/knowledge-banks/{kb}/search",
            json={"query": "Milan design fair", "top_k": 3, "collapse_documents": True},
        )
    ).json()["results"]
    assert len({hit["document_id"] for hit in collapsed}) == len(collapsed)
