"""Collections: records derived from documents, and the query DSL with joins.

Most of this runs with records written directly — the deterministic path — because the
query language, the joins and the merge rule are what is under test, and none of them
are questions about a model. Derivation itself gets its own test against the stub.
"""

import uuid

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app
from hindsight_api.knowledge.collections import relationships, validate_collection_fields
from hindsight_api.knowledge.fields import SchemaError
from hindsight_api.knowledge.filters import FilterError
from hindsight_api.knowledge.records_query import compile_record_query

VENDORS = {
    "name": "Vendors",
    "identity": "name",
    "fields": {
        "name": {"type": "string"},
        "country": {"type": "string", "values": ["de", "fr", "it", "us"]},
        "tier": {"type": "string", "values": ["gold", "silver"]},
    },
}
CONTRACTS = {
    "name": "Contracts",
    "identity": "reference",
    "fields": {
        "reference": {"type": "string"},
        "value": {"type": "number"},
        "status": {"type": "string", "values": ["active", "expired"]},
        "signed_on": {"type": "date"},
        "vendor": {"collection": "vendors", "description": "The vendor this contract is with"},
    },
}

VENDOR_ROWS = [
    {"values": {"name": "acme", "country": "de", "tier": "gold"}},
    {"values": {"name": "globex", "country": "de", "tier": "silver"}},
    {"values": {"name": "initech", "country": "us", "tier": "gold"}},
]
CONTRACT_ROWS = [
    {"values": {"reference": "c-1", "value": 12000, "status": "active", "signed_on": "2026-01-10", "vendor": "acme"}},
    {"values": {"reference": "c-2", "value": 8000, "status": "active", "signed_on": "2026-02-01", "vendor": "acme"}},
    {"values": {"reference": "c-3", "value": 3000, "status": "expired", "signed_on": "2025-06-01", "vendor": "globex"}},
    {
        "values": {
            "reference": "c-4",
            "value": 25000,
            "status": "active",
            "signed_on": "2026-03-01",
            "vendor": "initech",
        }
    },
    {"values": {"reference": "c-5", "value": 500, "status": "active", "signed_on": "2026-03-02", "vendor": None}},
]


@pytest_asyncio.fixture
async def kb_client(memory):
    app = create_app(memory, initialize_memory=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture
async def bank(kb_client):
    kb = f"kb-{uuid.uuid4().hex[:8]}"
    assert (await kb_client.post("/v1/default/knowledge-banks", json={"id": kb})).status_code == 201
    for collection_id, body in (("vendors", VENDORS), ("contracts", CONTRACTS)):
        response = await kb_client.put(f"/v1/default/knowledge-banks/{kb}/collections/{collection_id}", json=body)
        assert response.status_code == 200, response.text
    for collection_id, rows in (("vendors", VENDOR_ROWS), ("contracts", CONTRACT_ROWS)):
        written = await kb_client.post(
            f"/v1/default/knowledge-banks/{kb}/collections/{collection_id}/records", json={"records": rows}
        )
        assert written.status_code == 200, written.text
    return kb


async def client_get(client, kb: str, collection: str, record_id: str) -> dict:
    response = await client.get(f"/v1/default/knowledge-banks/{kb}/collections/{collection}/records/{record_id}")
    assert response.status_code == 200, response.text
    return response.json()


async def _query(client, kb: str, collection: str, body: dict) -> list[dict]:
    response = await client.post(f"/v1/default/knowledge-banks/{kb}/collections/{collection}/query", json=body)
    assert response.status_code == 200, response.text
    payload = response.json()
    return [dict(zip(payload["columns"], row, strict=True)) for row in payload["rows"]]


# ---- definitions


def test_a_relationship_must_point_at_a_collection_that_exists():
    fields = validate_collection_fields(
        {"name": {"type": "string"}, "vendor": {"collection": "vendors"}}, known_collections={"vendors"}
    )
    assert fields["vendor"] == {"type": "string", "collection": "vendors", "description": None}
    assert relationships(fields) == {"vendor": "vendors"}

    with pytest.raises(SchemaError, match="not a collection"):
        validate_collection_fields({"vendor": {"collection": "nope"}}, known_collections={"vendors"})
    with pytest.raises(SchemaError, match="at least one field"):
        validate_collection_fields({}, known_collections=set())


@pytest.mark.asyncio
async def test_identity_must_be_one_of_the_fields(kb_client, bank):
    bad = await kb_client.put(
        f"/v1/default/knowledge-banks/{bank}/collections/x",
        json={"fields": {"a": {"type": "string"}}, "identity": "b"},
    )
    assert bad.status_code == 400 and "identity" in bad.json()["detail"]


# ---- records


@pytest.mark.asyncio
async def test_the_identity_field_makes_one_record_of_two_writes(kb_client, bank):
    """Two documents about the same vendor are one row, each filling part of it."""
    await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/records",
        json={"records": [{"values": {"name": "acme", "tier": "silver"}, "doc_ids": ["d2"]}]},
    )
    record = (await kb_client.get(f"/v1/default/knowledge-banks/{bank}/collections/vendors/records/acme")).json()
    assert record["values"] == {"name": "acme", "country": "de", "tier": "silver"}, "the later write wins a conflict"
    assert record["doc_ids"] == ["d2"]

    rows = await _query(kb_client, bank, "vendors", {"select": [{"count": "*", "as": "n"}]})
    assert rows == [{"n": 3}], "still three vendors, not four"


@pytest.mark.asyncio
async def test_a_pinned_value_outranks_later_writes(kb_client, bank):
    pinned = await kb_client.put(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/records/acme/pins",
        json={"values": {"tier": "gold"}},
    )
    assert pinned.status_code == 200 and pinned.json()["values"]["tier"] == "gold"

    await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/records",
        json={"records": [{"values": {"name": "acme", "tier": "silver", "country": "fr"}}]},
    )
    record = (await kb_client.get(f"/v1/default/knowledge-banks/{bank}/collections/vendors/records/acme")).json()
    assert record["values"]["tier"] == "gold", "a human's correction is not overwritten by a document"
    assert record["values"]["country"] == "fr", "but everything else still updates"


# ---- the query DSL over records


@pytest.mark.asyncio
async def test_aggregates_and_filters_over_records(kb_client, bank):
    rows = await _query(
        kb_client,
        bank,
        "contracts",
        {
            "select": [
                {"field": "status", "as": "status"},
                {"count": "*", "as": "contracts"},
                {"sum": "value", "as": "total"},
                {"round": [{"avg": "value"}, 2], "as": "average"},
            ],
            "group_by": ["status"],
            "order_by": [{"field": "total", "direction": "desc"}],
        },
    )
    assert rows == [
        {"status": "active", "contracts": 4, "total": 45500, "average": 11375},
        {"status": "expired", "contracts": 1, "total": 3000, "average": 3000},
    ]

    filtered = await _query(
        kb_client,
        bank,
        "contracts",
        {
            "select": ["record_id", {"field": "value", "as": "value"}],
            "where": {"status": "active", "value": {"$gte": 8000}},
            "order_by": [{"field": "value", "direction": "desc"}],
        },
    )
    # Record ids are the normalised identity — "c-4" is stored as "c 4" — and a caller can
    # still fetch one by the reference their own data uses.
    assert [row["record_id"] for row in filtered] == ["c 4", "c 1", "c 2"]
    fetched = await client_get(kb_client, bank, "contracts", "c-4")
    assert fetched["values"]["reference"] == "c-4"


@pytest.mark.asyncio
async def test_a_join_follows_a_relationship_field(kb_client, bank):
    rows = await _query(
        kb_client,
        bank,
        "contracts",
        {
            "join": [{"collection": "vendors", "on": "vendor", "as": "vendor"}],
            "select": [
                {"field": "vendor.country", "as": "country"},
                {"count": "*", "as": "contracts"},
                {"sum": "value", "as": "total"},
            ],
            "where": {"status": "active"},
            "group_by": ["vendor.country"],
            "order_by": [{"field": "total", "direction": "desc"}],
        },
    )
    # de: c-1 + c-2 (acme) = 20000; us: c-4 = 25000; c-5 has no vendor and still counts.
    assert rows == [
        {"country": "us", "contracts": 1, "total": 25000},
        {"country": "de", "contracts": 2, "total": 20000},
        {"country": None, "contracts": 1, "total": 500},
    ]


@pytest.mark.asyncio
async def test_a_join_filters_and_haves_on_the_joined_collection(kb_client, bank):
    rows = await _query(
        kb_client,
        bank,
        "contracts",
        {
            "join": [{"on": "vendor", "as": "v"}],
            "select": [{"field": "v.name", "as": "vendor"}, {"count": "*", "as": "contracts"}],
            "where": {"v.tier": "gold", "status": "active"},
            "group_by": ["v.name"],
            "having": {"contracts": {"$gte": 2}},
        },
    )
    assert rows == [{"vendor": "acme", "contracts": 2}], "initech is gold but has one contract"


@pytest.mark.asyncio
async def test_a_query_cannot_join_something_that_is_not_a_relationship(kb_client, bank):
    response = await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/contracts/query",
        json={"join": [{"on": "status", "as": "s"}], "select": [{"count": "*", "as": "n"}]},
    )
    assert response.status_code == 400 and "relationship" in response.json()["detail"]


def test_the_record_compiler_binds_everything_and_scopes_to_one_collection():
    collection = {
        "collection_id": "contracts",
        "fields": {"vendor": {"type": "string", "collection": "vendors"}, "status": {"type": "string"}},
    }
    joined_fields = {"vendors": frozenset({"name", "country"}), "contracts": frozenset(collection["fields"])}
    compiled = compile_record_query(
        {
            "join": [{"on": "vendor", "as": "v"}],
            "select": [{"field": "v.name", "as": "vendor"}, {"count": "*", "as": "n"}],
            "where": {"status": "'; DROP TABLE kb_records; --"},
            "group_by": ["v.name"],
        },
        "bank-1",
        collection,
        joined_fields=joined_fields,
    )
    assert "DROP TABLE" not in compiled.sql
    assert compiled.params[0] == "bank-1" and compiled.params[1] == "contracts"
    assert "LEFT JOIN" in compiled.sql and compiled.grouped

    with pytest.raises(FilterError, match="unknown join alias"):
        compile_record_query({"select": [{"field": "nope.x", "as": "x"}]}, "bank-1", collection)


# ---- derivation


@pytest.mark.asyncio
async def test_records_are_derived_from_documents_with_their_evidence(kb_client, memory):
    kb = f"kb-{uuid.uuid4().hex[:8]}"
    await kb_client.post("/v1/default/knowledge-banks", json={"id": kb})
    await kb_client.put(f"/v1/default/knowledge-banks/{kb}/collections/vendors", json=VENDORS)
    await kb_client.post(
        f"/v1/default/knowledge-banks/{kb}/documents",
        json={
            "documents": [
                {"id": "d1", "text": "Acme Ltd is a German supplier. We treat them as a gold tier partner."},
                {"id": "d2", "text": "Acme Ltd shipped late twice this quarter."},
            ]
        },
    )

    def answer(messages, scope):
        if scope != "knowledge_records":
            return {}
        return {
            "records": [
                {
                    "values": {"name": "acme", "country": "de", "tier": "gold"},
                    "evidence": {"country": "Acme Ltd is a German supplier."},
                }
            ]
        }

    memory._llm_config._provider_impl.set_response_callback(answer)
    derive = await kb_client.post(f"/v1/default/knowledge-banks/{kb}/collections/vendors/derive", json={})
    assert derive.status_code == 202, derive.text

    record = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/collections/vendors/records/acme")).json()
    assert record["values"] == {"name": "acme", "country": "de", "tier": "gold"}
    assert record["evidence"]["country"][0]["doc_id"] in ("d1", "d2")
    assert record["evidence"]["country"][0]["quote"] == "Acme Ltd is a German supplier."
    # Both documents mention Acme, and both are on the one record.
    assert sorted(record["doc_ids"]) == ["d1", "d2"]

    operation = (
        await kb_client.get(f"/v1/default/knowledge-banks/{kb}/operations/{derive.json()['operation_id']}")
    ).json()
    assert operation["status"] == "completed"


@pytest.mark.asyncio
async def test_records_of_one_bank_are_invisible_to_another(kb_client, bank):
    other = f"kb-{uuid.uuid4().hex[:8]}"
    await kb_client.post("/v1/default/knowledge-banks", json={"id": other})
    await kb_client.put(f"/v1/default/knowledge-banks/{other}/collections/vendors", json=VENDORS)
    await kb_client.post(
        f"/v1/default/knowledge-banks/{other}/collections/vendors/records",
        json={"records": [{"values": {"name": "acme", "country": "fr"}}]},
    )
    mine = await _query(kb_client, bank, "vendors", {"select": [{"count": "*", "as": "n"}]})
    theirs = await _query(kb_client, other, "vendors", {"select": [{"count": "*", "as": "n"}]})
    assert mine == [{"n": 3}] and theirs == [{"n": 1}]
    record = (await kb_client.get(f"/v1/default/knowledge-banks/{other}/collections/vendors/records/acme")).json()
    assert record["values"]["country"] == "fr", "same record id, different bank, different row"
