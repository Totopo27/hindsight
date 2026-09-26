"""Knowledge-bank metadata: the schema, LLM extraction on write, and filtered search.

The extraction itself is one LLM call per document (and per chunk) through the bank's own
configured provider, so here it runs against the MockLLM the ``memory`` fixture installs:
what is asserted is that the schema reaches the model, that the values it returns are
stored where filters can find them, and that the filters mean what they say. Whether a
real model classifies a contract correctly is a different test and a different budget.
"""

import uuid

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app
from hindsight_api.knowledge.filters import FilterError, compile_filters
from hindsight_api.knowledge.metadata import MetadataSchemaError, extraction_model, validate_schema

SCHEMA = {
    "document": {
        "doc_type": {"type": "string", "values": ["invoice", "contract", "memo"], "description": "Kind of document"},
        "vendor": {"type": "string"},
        "total": {"type": "number"},
        "signed": {"type": "boolean"},
        "signed_on": {"type": "date"},
        "parties": {"type": "array", "items": "string"},
        "terms": {"type": "object"},
    },
    "chunks": {"clause": {"type": "string", "values": ["payment", "termination", "liability"]}},
}


@pytest_asyncio.fixture
async def kb_client(memory):
    app = create_app(memory, initialize_memory=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield client


async def _bank(client) -> str:
    kb = f"kb-{uuid.uuid4().hex[:8]}"
    assert (await client.post("/v1/default/knowledge-banks", json={"id": kb})).status_code == 201
    return kb


def _mock(memory, values: dict):
    """Make the mock LLM answer every extraction call with these values."""
    provider = memory._llm_config._provider_impl
    provider.set_response_callback(lambda messages, scope: dict(values))
    return provider


# ---- schema


def test_schema_validation_accepts_the_documented_shapes():
    document = validate_schema(SCHEMA["document"], level="document")
    assert document["doc_type"]["values"] == ["invoice", "contract", "memo"]
    assert document["parties"] == {"type": "array", "source": "extract", "items": "string"}
    assert document["doc_type"]["source"] == "extract", "extraction is the default source"
    assert validate_schema(None, level="chunk") == {}


@pytest.mark.parametrize(
    "bad",
    [
        {"x": {"type": "float"}},  # not a type we have
        {"x": {"type": "array", "items": "object"}},  # arrays of objects are v3's job
        {"x": {"type": "string", "values": []}},  # an empty classification list
        {"x": {"type": "object", "values": ["a"]}},  # an object cannot be one of a set
        {"x": {"type": "string", "values": ["a", "a"]}},
        {"bad name!": {"type": "string"}},
        {"x": "string"},  # the spec must be an object
    ],
)
def test_schema_validation_refuses_what_a_caller_cannot_have_meant(bad):
    with pytest.raises(MetadataSchemaError):
        validate_schema(bad, level="document")


def test_extraction_model_makes_every_property_optional_and_enforces_fixed_values():
    model = extraction_model(validate_schema(SCHEMA["document"], level="document"), name="M")
    assert model is not None
    empty = model()  # nothing extracted at all is a legitimate answer
    assert empty.doc_type is None and empty.total is None
    filled = model.model_validate({"doc_type": "invoice", "total": 12.5, "signed_on": "2026-01-31"})
    assert filled.doc_type == "invoice" and filled.total == 12.5
    assert filled.signed_on.isoformat() == "2026-01-31"
    with pytest.raises(Exception):  # noqa: B017 - pydantic's own error type is not the point
        model.model_validate({"doc_type": "receipt"})
    assert extraction_model({}, name="Empty") is None


# ---- filters


def test_filter_compiler_covers_the_operators():
    params: list = []
    sql = compile_filters({"a": "x", "b": {"$gte": 3}, "c": {"$in": ["p", "q"]}, "d": {"$exists": True}}, params)
    assert sql.startswith(" AND")
    assert "numeric" in sql and "IS NOT NULL" in sql
    # Each property name and each value is a bound parameter, never inlined.
    assert "x" in [p.strip('"') for p in params]
    assert compile_filters(None, []) == ""


def test_filter_compiler_refuses_nonsense():
    with pytest.raises(FilterError):
        compile_filters({"a": {"$like": "x"}}, [])
    with pytest.raises(FilterError):
        compile_filters({"a": {"$in": []}}, [])
    with pytest.raises(FilterError):
        compile_filters({"a": {"$exists": "yes"}}, [])


# ---- end to end


@pytest.mark.asyncio
async def test_schema_is_stored_and_returned_with_counts(kb_client):
    kb = await _bank(kb_client)
    put = await kb_client.put(f"/v1/default/knowledge-banks/{kb}/metadata-schema", json=SCHEMA)
    assert put.status_code == 200, put.text
    body = put.json()
    assert body["document"]["doc_type"]["values"] == ["invoice", "contract", "memo"]
    assert body["chunks"]["clause"]["type"] == "string"
    assert body["documents_extracted"] == 0

    got = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/metadata-schema")).json()
    assert got["document"] == body["document"]

    bad = await kb_client.put(
        f"/v1/default/knowledge-banks/{kb}/metadata-schema", json={"document": {"x": {"type": "float"}}}
    )
    assert bad.status_code == 400 and "float" in bad.json()["detail"]


@pytest.mark.asyncio
async def test_writing_a_document_extracts_the_schema_and_search_filters_on_it(kb_client, memory):
    kb = await _bank(kb_client)
    await kb_client.put(f"/v1/default/knowledge-banks/{kb}/metadata-schema", json=SCHEMA)
    _mock(
        memory,
        {
            "doc_type": "invoice",
            "vendor": "Acme",
            "total": 1200,
            "signed": True,
            "signed_on": "2026-01-31",
            "parties": ["Acme", "Globex"],
            "terms": {"net": 30},
            "clause": "payment",
        },
    )

    write = await kb_client.post(
        f"/v1/default/knowledge-banks/{kb}/documents",
        json={"documents": [{"id": "inv-1", "text": "Acme invoice for 1200 EUR, payable in 30 days."}]},
    )
    assert write.status_code == 202, write.text

    document = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/documents/inv-1")).json()
    assert document["extracted_metadata"]["doc_type"] == "invoice"
    assert document["extracted_metadata"]["total"] == 1200
    assert document["extracted_metadata"]["parties"] == ["Acme", "Globex"]
    assert document["extracted_metadata"]["signed_on"] == "2026-01-31"
    assert document["chunks"][0]["metadata"] == {"clause": "payment"}

    schema = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/metadata-schema")).json()
    assert schema["documents_extracted"] == 1 and schema["chunks_extracted"] == 1

    async def search(metadata):
        response = await kb_client.post(
            f"/v1/default/knowledge-banks/{kb}/search",
            json={"query": "invoice", "top_k": 5, "metadata": metadata},
        )
        assert response.status_code == 200, response.text
        return [hit["document_id"] for hit in response.json()["results"]]

    assert await search({"doc_type": "invoice"}) == ["inv-1"]
    assert await search({"doc_type": "contract"}) == []
    assert await search({"total": {"$gte": 1000, "$lt": 2000}}) == ["inv-1"]
    assert await search({"total": {"$lt": 1000}}) == []
    assert await search({"vendor": {"$in": ["Acme", "Globex"]}}) == ["inv-1"]
    assert await search({"parties": {"$contains": "Globex"}}) == ["inv-1"]
    assert await search({"signed_on": {"$gte": "2026-01-01"}}) == ["inv-1"]
    assert await search({"signed_on": {"$gte": "2027-01-01"}}) == []
    assert await search({"clause": "payment"}) == ["inv-1"]  # a chunk-level property
    assert await search({"clause": "termination"}) == []
    assert await search({"missing": {"$exists": False}}) == ["inv-1"]

    facets = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/metadata-values/doc_type")).json()
    assert facets["values"] == [{"value": "invoice", "count": 1}]

    bad = await kb_client.post(
        f"/v1/default/knowledge-banks/{kb}/search", json={"query": "x", "metadata": {"a": {"$like": "y"}}}
    )
    assert bad.status_code == 400


@pytest.mark.asyncio
async def test_filters_also_match_metadata_the_caller_wrote(kb_client, memory):
    """No schema, no LLM: a filter still works on the metadata supplied with the document."""
    kb = await _bank(kb_client)
    await kb_client.post(
        f"/v1/default/knowledge-banks/{kb}/documents",
        json={
            "documents": [
                {"id": "a", "text": "Milan is in Lombardy.", "metadata": {"region": "lombardy", "year": 2024}},
                {"id": "b", "text": "Turin is in Piedmont.", "metadata": {"region": "piedmont", "year": 2019}},
            ]
        },
    )
    results = await kb_client.post(
        f"/v1/default/knowledge-banks/{kb}/search",
        json={"query": "Italy region", "top_k": 5, "metadata": {"region": "piedmont"}},
    )
    assert [hit["document_id"] for hit in results.json()["results"]] == ["b"]

    recent = await kb_client.post(
        f"/v1/default/knowledge-banks/{kb}/search",
        json={"query": "Italy region", "top_k": 5, "metadata": {"year": {"$gte": 2020}}},
    )
    assert [hit["document_id"] for hit in recent.json()["results"]] == ["a"]


@pytest.mark.asyncio
async def test_extraction_is_skipped_when_the_bank_turns_it_off(kb_client, memory):
    kb = await _bank(kb_client)
    await kb_client.put(f"/v1/default/knowledge-banks/{kb}/metadata-schema", json=SCHEMA)
    patched = await kb_client.patch(
        f"/v1/default/banks/{kb}/config", json={"updates": {"kb_metadata_extraction": False}}
    )
    assert patched.status_code == 200, patched.text
    provider = _mock(memory, {"doc_type": "invoice"})
    before = len(provider.get_mock_calls())

    await kb_client.post(
        f"/v1/default/knowledge-banks/{kb}/documents",
        json={"documents": [{"id": "d1", "text": "Acme invoice for 1200 EUR."}]},
    )
    document = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/documents/d1")).json()
    assert document["extracted_metadata"] == {}
    assert len(provider.get_mock_calls()) == before, "the LLM was called with extraction off"


@pytest.mark.asyncio
async def test_re_extraction_fills_documents_written_before_the_schema(kb_client, memory):
    kb = await _bank(kb_client)
    await kb_client.post(
        f"/v1/default/knowledge-banks/{kb}/documents",
        json={"documents": [{"id": "old", "text": "Globex contract, terminating in 2027."}]},
    )
    document = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/documents/old")).json()
    assert document["extracted_metadata"] == {}

    await kb_client.put(f"/v1/default/knowledge-banks/{kb}/metadata-schema", json=SCHEMA)
    _mock(memory, {"doc_type": "contract", "vendor": "Globex", "clause": "termination"})
    extract = await kb_client.post(f"/v1/default/knowledge-banks/{kb}/metadata/extract", json={"only_missing": True})
    assert extract.status_code == 202, extract.text

    document = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/documents/old")).json()
    assert document["extracted_metadata"]["doc_type"] == "contract"
    assert document["chunks"][0]["metadata"] == {"clause": "termination"}

    operation = (
        await kb_client.get(f"/v1/default/knowledge-banks/{kb}/operations/{extract.json()['operation_id']}")
    ).json()
    assert operation["status"] == "completed"


@pytest.mark.asyncio
async def test_a_failing_extraction_does_not_fail_the_write(kb_client, memory):
    kb = await _bank(kb_client)
    await kb_client.put(f"/v1/default/knowledge-banks/{kb}/metadata-schema", json=SCHEMA)
    memory._llm_config._provider_impl.set_mock_exception(RuntimeError("model is down"))

    write = await kb_client.post(
        f"/v1/default/knowledge-banks/{kb}/documents",
        json={"documents": [{"id": "d1", "text": "Acme invoice for 1200 EUR."}]},
    )
    assert write.status_code == 202
    document = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/documents/d1")).json()
    assert document["extracted_metadata"] == {}
    assert document["chunk_count"] == 1  # the document is stored and searchable regardless
    hits = (
        await kb_client.post(f"/v1/default/knowledge-banks/{kb}/search", json={"query": "Acme invoice", "top_k": 3})
    ).json()["results"]
    assert [hit["document_id"] for hit in hits] == ["d1"]
