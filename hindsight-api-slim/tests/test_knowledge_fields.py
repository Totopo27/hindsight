"""Knowledge-bank metadata: the schema, LLM extraction on write, and filtered search.

The extraction itself is one LLM call per document (and per passage) through the bank's own
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
from hindsight_api.knowledge.fields import SchemaError, extraction_model, validate_field_schema

SCHEMA = {
    "document_fields": {
        "doc_type": {"type": "string", "values": ["invoice", "contract", "memo"], "description": "Kind of document"},
        "vendor": {"type": "string"},
        "total": {"type": "number"},
        "signed": {"type": "boolean"},
        "signed_on": {"type": "date"},
        "parties": {"type": "array", "items": "string"},
        "terms": {"type": "object"},
    },
    "passage_fields": {"clause": {"type": "string", "values": ["payment", "termination", "liability"]}},
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
    document = validate_field_schema(SCHEMA["document_fields"], level="document")
    assert document["doc_type"]["values"] == ["invoice", "contract", "memo"]
    assert document["parties"] == {
        "type": "array",
        "source": "extract",
        "filterable": True,
        "indexed": False,
        "items": "string",
    }
    assert document["doc_type"]["source"] == "extract", "extraction is the default source"
    assert validate_field_schema(None, level="passage") == {}


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
    with pytest.raises(SchemaError):
        validate_field_schema(bad, level="document")


def test_extraction_model_makes_every_property_optional_and_enforces_fixed_values():
    model = extraction_model(validate_field_schema(SCHEMA["document_fields"], level="document"), name="M")
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
    # Each property name and each value is a bound parameter, never inlined; an $in list
    # binds as one array parameter rather than one comparison per value.
    flat = [item for p in params for item in (p if isinstance(p, list) else [p])]
    assert "x" in [str(item).strip('"') for item in flat]
    assert any(isinstance(p, list) and len(p) == 2 for p in params)
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
    put = await kb_client.put(f"/v1/default/knowledge-banks/{kb}/schemas/default", json=SCHEMA)
    assert put.status_code == 200, put.text
    body = put.json()
    assert body["document_fields"]["doc_type"]["values"] == ["invoice", "contract", "memo"]
    assert body["passage_fields"]["clause"]["type"] == "string"
    assert body["documents_with_fields"] == 0

    got = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/schemas/default")).json()
    assert got["document_fields"] == body["document_fields"]

    bad = await kb_client.put(
        f"/v1/default/knowledge-banks/{kb}/schemas/default", json={"document_fields": {"x": {"type": "float"}}}
    )
    assert bad.status_code == 400 and "float" in bad.json()["detail"]


@pytest.mark.asyncio
async def test_writing_a_document_extracts_the_schema_and_search_filters_on_it(kb_client, memory):
    kb = await _bank(kb_client)
    await kb_client.put(f"/v1/default/knowledge-banks/{kb}/schemas/default", json=SCHEMA)
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
    assert document["fields"]["doc_type"] == "invoice"
    assert document["fields"]["total"] == 1200
    assert document["fields"]["parties"] == ["Acme", "Globex"]
    assert document["fields"]["signed_on"] == "2026-01-31"
    assert document["passages"][0]["fields"] == {"clause": "payment"}

    schema = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/schemas/default")).json()
    assert schema["documents_with_fields"] == 1 and schema["passages_with_fields"] == 1

    async def search(fields):
        response = await kb_client.post(
            f"/v1/default/knowledge-banks/{kb}/search",
            json={"query": "invoice", "top_k": 5, "fields": fields},
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
    assert await search({"clause": "payment"}) == ["inv-1"]  # a passage-level property
    assert await search({"clause": "termination"}) == []
    assert await search({"missing": {"$exists": False}}) == ["inv-1"]

    facets = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/field-values/doc_type")).json()
    assert facets["values"] == [{"value": "invoice", "count": 1}]

    bad = await kb_client.post(
        f"/v1/default/knowledge-banks/{kb}/search", json={"query": "x", "fields": {"a": {"$like": "y"}}}
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
        json={"query": "Italy region", "top_k": 5, "fields": {"region": "piedmont"}},
    )
    assert [hit["document_id"] for hit in results.json()["results"]] == ["b"]

    recent = await kb_client.post(
        f"/v1/default/knowledge-banks/{kb}/search",
        json={"query": "Italy region", "top_k": 5, "fields": {"year": {"$gte": 2020}}},
    )
    assert [hit["document_id"] for hit in recent.json()["results"]] == ["a"]


@pytest.mark.asyncio
async def test_extraction_is_skipped_when_the_bank_turns_it_off(kb_client, memory):
    kb = await _bank(kb_client)
    await kb_client.put(f"/v1/default/knowledge-banks/{kb}/schemas/default", json=SCHEMA)
    patched = await kb_client.patch(f"/v1/default/banks/{kb}/config", json={"updates": {"kb_field_extraction": False}})
    assert patched.status_code == 200, patched.text
    provider = _mock(memory, {"doc_type": "invoice"})
    before = len(provider.get_mock_calls())

    await kb_client.post(
        f"/v1/default/knowledge-banks/{kb}/documents",
        json={"documents": [{"id": "d1", "text": "Acme invoice for 1200 EUR."}]},
    )
    document = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/documents/d1")).json()
    assert document["fields"] == {}
    assert len(provider.get_mock_calls()) == before, "the LLM was called with extraction off"


@pytest.mark.asyncio
async def test_re_extraction_fills_documents_written_before_the_schema(kb_client, memory):
    kb = await _bank(kb_client)
    await kb_client.post(
        f"/v1/default/knowledge-banks/{kb}/documents",
        json={"documents": [{"id": "old", "text": "Globex contract, terminating in 2027."}]},
    )
    document = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/documents/old")).json()
    assert document["fields"] == {}

    await kb_client.put(f"/v1/default/knowledge-banks/{kb}/schemas/default", json=SCHEMA)
    _mock(memory, {"doc_type": "contract", "vendor": "Globex", "clause": "termination"})
    extract = await kb_client.post(f"/v1/default/knowledge-banks/{kb}/fields/extract", json={"only_missing": True})
    assert extract.status_code == 202, extract.text

    document = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/documents/old")).json()
    assert document["fields"]["doc_type"] == "contract"
    assert document["passages"][0]["fields"] == {"clause": "termination"}

    operation = (
        await kb_client.get(f"/v1/default/knowledge-banks/{kb}/operations/{extract.json()['operation_id']}")
    ).json()
    assert operation["status"] == "completed"


@pytest.mark.asyncio
async def test_a_failing_extraction_does_not_fail_the_write(kb_client, memory):
    kb = await _bank(kb_client)
    await kb_client.put(f"/v1/default/knowledge-banks/{kb}/schemas/default", json=SCHEMA)
    memory._llm_config._provider_impl.set_mock_exception(RuntimeError("model is down"))

    write = await kb_client.post(
        f"/v1/default/knowledge-banks/{kb}/documents",
        json={"documents": [{"id": "d1", "text": "Acme invoice for 1200 EUR."}]},
    )
    assert write.status_code == 202
    document = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/documents/d1")).json()
    assert document["fields"] == {}
    assert document["passage_count"] == 1  # the document is stored and searchable regardless
    hits = (
        await kb_client.post(f"/v1/default/knowledge-banks/{kb}/search", json={"query": "Acme invoice", "top_k": 3})
    ).json()["results"]
    assert [hit["document_id"] for hit in hits] == ["d1"]


@pytest.mark.asyncio
async def test_documents_can_be_listed_by_the_schema_they_were_read_with(kb_client):
    """A schema is only as good as what it filled, so its documents have to be findable.

    'none' is a real filter value, not a missing one: the documents no schema applied to
    are the ones worth finding, because nothing was extracted from them.
    """
    bank = await _bank(kb_client)
    for schema_id, fields in [("invoice", {"vendor": {"type": "string"}}), ("memo", {"topic": {"type": "string"}})]:
        response = await kb_client.put(
            f"/v1/default/knowledge-banks/{bank}/schemas/{schema_id}",
            json={"document_fields": fields, "passage_fields": {}},
        )
        assert response.status_code == 200, response.text
    written = await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/documents",
        json={
            "documents": [
                {"id": "a", "text": "Acme invoice.", "schema_id": "invoice", "fields": {"vendor": "acme"}},
                {"id": "b", "text": "Globex invoice.", "schema_id": "invoice", "fields": {"vendor": "globex"}},
                {"id": "c", "text": "A memo.", "schema_id": "memo", "fields": {"topic": "hiring"}},
            ]
        },
    )
    assert written.status_code == 202, written.text

    async def ids(schema_id: str) -> set[str]:
        page = await kb_client.get(f"/v1/default/knowledge-banks/{bank}/documents", params={"schema_id": schema_id})
        assert page.status_code == 200, page.text
        return {d["doc_id"] for d in page.json()["items"]}

    assert await ids("invoice") == {"a", "b"}
    assert await ids("memo") == {"c"}
    assert await ids("none") == set()


@pytest.mark.asyncio
async def test_search_can_be_scoped_to_one_schema(kb_client):
    """Which schema read a document is a filter, not just a label.

    A bank with several kinds of document is the case knowledge banks exist for, and
    "search the invoices" is the first thing anyone asks of one.
    """
    bank = await _bank(kb_client)
    for schema_id in ("invoice", "memo"):
        response = await kb_client.put(
            f"/v1/default/knowledge-banks/{bank}/schemas/{schema_id}",
            json={"document_fields": {}, "passage_fields": {}},
        )
        assert response.status_code == 200, response.text
    written = await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/documents",
        json={
            "documents": [
                {"id": "inv", "text": "Acme charged us for cloud hosting.", "schema_id": "invoice"},
                {"id": "memo", "text": "Acme is our preferred cloud hosting vendor.", "schema_id": "memo"},
                {"id": "loose", "text": "Cloud hosting costs are rising."},
            ]
        },
    )
    assert written.status_code == 202, written.text

    async def hits(**body) -> set[str]:
        response = await kb_client.post(
            f"/v1/default/knowledge-banks/{bank}/search", json={"query": "cloud hosting", **body}
        )
        assert response.status_code == 200, response.text
        return {hit["document_id"] for hit in response.json()["results"]}

    assert await hits() == {"inv", "memo", "loose"}
    assert await hits(schema_id="invoice") == {"inv"}
    assert await hits(schema_id="memo") == {"memo"}
    # The documents no schema applied to are findable too — that is what "none" is for.
    assert await hits(schema_id="none") == {"loose"}


@pytest.mark.asyncio
async def test_re_extraction_leaves_another_schemas_documents_alone(kb_client, memory):
    """Re-reading one schema must not re-read the whole bank with it.

    A bank with several schemas is the case they exist for; extracting "invoice" over a
    memo would overwrite the memo's fields with an invoice's, which no one asked for.
    """
    bank = await _bank(kb_client)
    for schema_id, fields in [("invoice", {"vendor": {"type": "string"}}), ("memo", {"topic": {"type": "string"}})]:
        response = await kb_client.put(
            f"/v1/default/knowledge-banks/{bank}/schemas/{schema_id}",
            json={"document_fields": fields, "passage_fields": {}},
        )
        assert response.status_code == 200, response.text
    written = await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/documents",
        json={
            "documents": [
                {"id": "inv", "text": "Acme invoice.", "schema_id": "invoice", "fields": {"vendor": "acme"}},
                {"id": "memo", "text": "A memo.", "schema_id": "memo", "fields": {"topic": "hiring"}},
            ]
        },
    )
    assert written.status_code == 202, written.text

    _mock(memory, {"vendor": "re-read"})
    extracted = await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/fields/extract",
        json={"schema_id": "invoice", "only_missing": False},
    )
    assert extracted.status_code == 202, extracted.text

    async def document(doc_id: str) -> dict:
        response = await kb_client.get(f"/v1/default/knowledge-banks/{bank}/documents/{doc_id}")
        assert response.status_code == 200, response.text
        return response.json()

    assert (await document("inv"))["fields"]["vendor"] == "re-read"
    memo = await document("memo")
    assert memo["fields"] == {"topic": "hiring"}, "the memo kept what its own schema read"
    assert memo["schema_id"] == "memo"
