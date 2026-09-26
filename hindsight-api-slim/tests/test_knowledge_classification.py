"""Choosing a document's schema, and the two flags a field carries.

`filterable` and `indexed` are the two things a field definition says about *use* rather
than shape: whether search may filter on it, and whether its value joins the text that
gets embedded. Classification is what picks the schema in the first place when a bank
holds more than one and the write does not say which.
"""

import uuid

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app
from hindsight_api.knowledge.fields import filterable_names, validate_field_schema
from hindsight_api.knowledge.service import embedding_text, indexed_values

CONTRACT = {
    "name": "Contract",
    "description": "An agreement between two parties, with a term and a counterparty",
    "document_fields": {
        "counterparty": {"type": "string", "indexed": True},
        "term_years": {"type": "integer"},
        "internal_ref": {"type": "string", "filterable": False},
    },
}
INVOICE = {
    "name": "Invoice",
    "description": "A bill for goods or services, with an amount due",
    "document_fields": {"amount": {"type": "number"}, "vendor": {"type": "string", "indexed": True}},
}


@pytest_asyncio.fixture
async def kb_client(memory):
    app = create_app(memory, initialize_memory=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield client


async def _bank(client, *schemas: tuple[str, dict]) -> str:
    kb = f"kb-{uuid.uuid4().hex[:8]}"
    assert (await client.post("/v1/default/knowledge-banks", json={"id": kb})).status_code == 201
    for schema_id, body in schemas:
        response = await client.put(f"/v1/default/knowledge-banks/{kb}/schemas/{schema_id}", json=body)
        assert response.status_code == 200, response.text
    return kb


# ---- the flags, as definitions


def test_flags_default_to_filterable_and_not_indexed():
    schema = validate_field_schema({"a": {"type": "string"}}, level="document")
    assert schema["a"]["filterable"] is True
    assert schema["a"]["indexed"] is False, "indexing dilutes the passage vector; it has to be asked for"
    assert filterable_names({"document_fields": schema, "passage_fields": {}}) == {"a"}


def test_a_non_filterable_field_is_not_offered_to_filters():
    schema = validate_field_schema(
        {"a": {"type": "string"}, "b": {"type": "string", "filterable": False}}, level="document"
    )
    assert filterable_names({"document_fields": schema, "passage_fields": {}}) == {"a"}
    assert filterable_names(None) is None, "no schema means no opinion about what filters"


def test_indexed_values_join_the_embedded_text():
    schema = {
        "document_fields": validate_field_schema(
            {"region": {"type": "string", "indexed": True}, "ref": {"type": "string"}}, level="document"
        ),
        "passage_fields": validate_field_schema({"clause": {"type": "string", "indexed": True}}, level="passage"),
    }
    values = indexed_values(schema, {"region": "emea", "ref": "X-1"}, {"clause": "payment"})
    assert values == {"region": "emea", "clause": "payment"}

    text = embedding_text("Supplier agreement", "The party shall pay within 30 days.", values)
    assert text.startswith("clause: payment, region: emea")
    assert "Supplier agreement" in text and "30 days" in text
    # Without indexed values it is the title-then-passage text it always was.
    assert embedding_text("Supplier agreement", "Body.", {}) == "Supplier agreement\n\nBody."


def test_an_object_field_cannot_be_indexed():
    from hindsight_api.knowledge.fields import SchemaError

    with pytest.raises(SchemaError, match="indexed"):
        validate_field_schema({"a": {"type": "object", "indexed": True}}, level="document")


# ---- the flags, in the API


@pytest.mark.asyncio
async def test_search_refuses_a_filter_on_a_field_the_schema_does_not_expose(kb_client, memory):
    kb = await _bank(kb_client, ("contract", CONTRACT))
    memory._llm_config._provider_impl.set_response_callback(
        lambda messages, scope: {"counterparty": "Acme", "term_years": 3, "internal_ref": "X-1"}
    )
    await kb_client.post(
        f"/v1/default/knowledge-banks/{kb}/documents",
        json={"documents": [{"id": "c1", "text": "Agreement with Acme for three years."}]},
    )

    ok = await kb_client.post(
        f"/v1/default/knowledge-banks/{kb}/search",
        json={"query": "agreement", "fields": {"counterparty": "Acme"}},
    )
    assert ok.status_code == 200 and [h["document_id"] for h in ok.json()["results"]] == ["c1"]

    refused = await kb_client.post(
        f"/v1/default/knowledge-banks/{kb}/search",
        json={"query": "agreement", "fields": {"internal_ref": "X-1"}},
    )
    assert refused.status_code == 400 and "internal_ref" in refused.json()["detail"]

    # A name the schema never defined is the caller's own document metadata, still filterable.
    unknown = await kb_client.post(
        f"/v1/default/knowledge-banks/{kb}/search", json={"query": "agreement", "fields": {"anything": "x"}}
    )
    assert unknown.status_code == 200


@pytest.mark.asyncio
async def test_an_indexed_field_makes_a_passage_findable_by_a_word_it_never_says(kb_client, memory):
    """The point of `indexed`: the value is in the vector, not just in the filter."""
    kb = await _bank(
        kb_client,
        ("contract", {"document_fields": {"counterparty": {"type": "string", "indexed": True, "source": "request"}}}),
    )
    await kb_client.post(
        f"/v1/default/knowledge-banks/{kb}/documents",
        json={
            "documents": [
                {
                    "id": "c1",
                    "text": "The parties agree to a three year term with annual renewal.",
                    "fields": {"counterparty": "Zalando"},
                },
                {"id": "c2", "text": "The parties agree to a two year term with annual renewal."},
            ]
        },
    )
    hits = (
        await kb_client.post(
            f"/v1/default/knowledge-banks/{kb}/search",
            json={"query": "Zalando", "top_k": 2, "mode": "vector", "rerank": False},
        )
    ).json()["results"]
    assert hits[0]["document_id"] == "c1", "the indexed value should pull its passage up on a vector search"


# ---- classification


@pytest.mark.asyncio
async def test_one_schema_needs_no_naming_and_no_classification(kb_client, memory):
    kb = await _bank(kb_client, ("invoice", INVOICE))
    provider = memory._llm_config._provider_impl
    provider.set_response_callback(lambda messages, scope: {"amount": 99, "vendor": "Acme"})

    await kb_client.post(
        f"/v1/default/knowledge-banks/{kb}/documents",
        json={"documents": [{"id": "d1", "text": "Invoice from Acme for 99 EUR."}]},
    )
    document = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/documents/d1")).json()
    assert document["fields"] == {"amount": 99, "vendor": "Acme"}
    assert document["schema_id"] == "invoice"
    assert all(call["scope"] != "knowledge_classify" for call in provider.get_mock_calls())


@pytest.mark.asyncio
async def test_several_schemas_are_classified_by_the_llm(kb_client, memory):
    kb = await _bank(kb_client, ("contract", CONTRACT), ("invoice", INVOICE))
    provider = memory._llm_config._provider_impl

    def answer(messages, scope):
        if scope == "knowledge_classify":
            # The prompt carries the schema catalogue *and* the document, so the document's
            # own words are what this stub keys on — the catalogue names both schemas.
            return {"schema_id": "invoice" if "due in 30 days" in messages[-1]["content"] else "contract"}
        return {"amount": 120, "vendor": "Globex", "counterparty": "Globex", "term_years": 2}

    provider.set_response_callback(answer)
    await kb_client.post(
        f"/v1/default/knowledge-banks/{kb}/documents",
        json={
            "documents": [
                {"id": "inv", "text": "Invoice from Globex for 120 EUR, due in 30 days."},
                {"id": "con", "text": "This agreement between Globex and us runs for two years."},
            ]
        },
    )
    invoice = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/documents/inv")).json()
    contract = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/documents/con")).json()
    assert invoice["schema_id"] == "invoice" and contract["schema_id"] == "contract"
    # Each document was classified, and each got only its own schema's fields.
    assert set(invoice["fields"]) <= {"amount", "vendor"}
    assert set(contract["fields"]) <= {"counterparty", "term_years", "internal_ref"}
    assert sum(call["scope"] == "knowledge_classify" for call in provider.get_mock_calls()) == 2


@pytest.mark.asyncio
async def test_a_named_schema_skips_classification(kb_client, memory):
    kb = await _bank(kb_client, ("contract", CONTRACT), ("invoice", INVOICE))
    provider = memory._llm_config._provider_impl
    provider.set_response_callback(lambda messages, scope: {"amount": 10, "vendor": "Acme"})

    await kb_client.post(
        f"/v1/default/knowledge-banks/{kb}/documents",
        json={"documents": [{"id": "d1", "text": "Anything at all.", "schema_id": "invoice"}]},
    )
    document = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/documents/d1")).json()
    assert document["schema_id"] == "invoice"
    assert all(call["scope"] != "knowledge_classify" for call in provider.get_mock_calls())


@pytest.mark.asyncio
async def test_classification_off_means_no_fields_rather_than_the_wrong_ones(kb_client, memory):
    kb = await _bank(kb_client, ("contract", CONTRACT), ("invoice", INVOICE))
    patched = await kb_client.patch(
        f"/v1/default/banks/{kb}/config", json={"updates": {"kb_schema_classification": False}}
    )
    assert patched.status_code == 200, patched.text
    provider = memory._llm_config._provider_impl
    provider.set_response_callback(lambda messages, scope: {"amount": 10})

    await kb_client.post(
        f"/v1/default/knowledge-banks/{kb}/documents",
        json={"documents": [{"id": "d1", "text": "Invoice from Acme."}]},
    )
    document = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/documents/d1")).json()
    assert document["fields"] == {} and document["schema_id"] is None
    assert provider.get_mock_calls() == []


@pytest.mark.asyncio
async def test_a_document_that_is_none_of_the_schemas_keeps_its_fields_empty(kb_client, memory):
    kb = await _bank(kb_client, ("contract", CONTRACT), ("invoice", INVOICE))
    memory._llm_config._provider_impl.set_response_callback(
        lambda messages, scope: {"schema_id": "none"} if scope == "knowledge_classify" else {}
    )
    await kb_client.post(
        f"/v1/default/knowledge-banks/{kb}/documents",
        json={"documents": [{"id": "d1", "text": "Lunch menu for Thursday."}]},
    )
    document = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/documents/d1")).json()
    assert document["fields"] == {} and document["schema_id"] is None
    assert document["passage_count"] == 1, "it is still a stored, searchable document"
