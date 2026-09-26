"""The knowledge-bank query DSL, and metadata supplied by the write instead of the LLM.

These two belong together: supplying properties on write is what makes a corpus queryable
without an LLM anywhere, and every query test here runs against a bank built that way — no
model, no stubbed model, nothing to script.
"""

import uuid

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app
from hindsight_api.knowledge.filters import FilterError
from hindsight_api.knowledge.query import Limits, compile_query

SCHEMA = {
    "document": {
        "doc_type": {"type": "string", "values": ["invoice", "contract", "memo"], "source": "request"},
        "vendor": {"type": "string", "source": "request"},
        "total": {"type": "number", "source": "request"},
        "region": {"type": "string", "source": "request"},
        "signed_on": {"type": "date", "source": "request"},
    },
    "chunks": {"clause": {"type": "string", "values": ["payment", "termination", "liability"], "source": "request"}},
}

CORPUS = [
    ("inv-1", "invoice", "Acme", 1200, "emea", "2026-01-31", "payment"),
    ("inv-2", "invoice", "Acme", 800, "emea", "2026-02-15", "payment"),
    ("inv-3", "invoice", "Globex", 5000, "amer", "2026-03-01", "payment"),
    ("con-1", "contract", "Acme", 20000, "emea", "2025-11-01", "termination"),
    ("con-2", "contract", "Initech", 7500, "apac", "2026-01-05", "liability"),
    ("memo-1", "memo", "Acme", None, "emea", None, None),
]


@pytest_asyncio.fixture
async def kb_client(memory):
    app = create_app(memory, initialize_memory=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture
async def bank(kb_client, memory):
    """A bank whose whole schema is request-served, so nothing here calls an LLM."""
    kb = f"kb-{uuid.uuid4().hex[:8]}"
    assert (await kb_client.post("/v1/default/knowledge-banks", json={"id": kb})).status_code == 201
    assert (await kb_client.put(f"/v1/default/knowledge-banks/{kb}/metadata-schema", json=SCHEMA)).status_code == 200

    documents = []
    for doc_id, doc_type, vendor, total, region, signed_on, clause in CORPUS:
        properties = {"doc_type": doc_type, "vendor": vendor, "region": region}
        if total is not None:
            properties["total"] = total
        if signed_on is not None:
            properties["signed_on"] = signed_on
        documents.append(
            {
                "id": doc_id,
                "title": f"{doc_type} {doc_id}",
                "text": f"{doc_type} from {vendor} in {region}. " * 20,
                "properties": properties,
                "chunk_properties": {"0": {"clause": clause}} if clause else {},
            }
        )
    write = await kb_client.post(f"/v1/default/knowledge-banks/{kb}/documents", json={"documents": documents})
    assert write.status_code == 202, write.text
    return kb


async def _query(client, kb: str, body: dict) -> dict:
    response = await client.post(f"/v1/default/knowledge-banks/{kb}/query", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def _rows(result: dict) -> list[dict]:
    return [dict(zip(result["columns"], row, strict=True)) for row in result["rows"]]


# ---- metadata supplied by the write


@pytest.mark.asyncio
async def test_supplied_properties_land_where_extraction_would_and_cost_no_llm_call(kb_client, memory, bank):
    provider = memory._llm_config._provider_impl
    assert provider.get_mock_calls() == [], "a fully request-served schema must not call the LLM"

    document = (await kb_client.get(f"/v1/default/knowledge-banks/{bank}/documents/inv-1")).json()
    assert document["extracted_metadata"] == {
        "doc_type": "invoice",
        "vendor": "Acme",
        "total": 1200,
        "region": "emea",
        "signed_on": "2026-01-31",
    }
    assert document["chunks"][0]["metadata"] == {"clause": "payment"}

    # And the values filter exactly like extracted ones, because they are the same object.
    hits = (
        await kb_client.post(
            f"/v1/default/knowledge-banks/{bank}/search",
            json={"query": "invoice", "top_k": 10, "metadata": {"vendor": "Globex"}},
        )
    ).json()["results"]
    assert {hit["document_id"] for hit in hits} == {"inv-3"}


@pytest.mark.asyncio
async def test_a_supplied_value_overrides_extraction_for_that_document(kb_client, memory):
    """The schema says extract, but this write already knows the answer: no call for it."""
    kb = f"kb-{uuid.uuid4().hex[:8]}"
    await kb_client.post("/v1/default/knowledge-banks", json={"id": kb})
    await kb_client.put(
        f"/v1/default/knowledge-banks/{kb}/metadata-schema",
        json={"document": {"doc_type": {"type": "string", "values": ["invoice", "memo"]}}},
    )
    provider = memory._llm_config._provider_impl
    provider.set_response_callback(lambda messages, scope: {"doc_type": "memo"})

    await kb_client.post(
        f"/v1/default/knowledge-banks/{kb}/documents",
        json={"documents": [{"id": "d1", "text": "Acme invoice.", "properties": {"doc_type": "invoice"}}]},
    )
    document = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/documents/d1")).json()
    assert document["extracted_metadata"] == {"doc_type": "invoice"}
    assert provider.get_mock_calls() == [], "the supplied value should have skipped the call"

    # A document that supplies nothing still gets extracted.
    await kb_client.post(
        f"/v1/default/knowledge-banks/{kb}/documents",
        json={"documents": [{"id": "d2", "text": "Internal note."}]},
    )
    other = (await kb_client.get(f"/v1/default/knowledge-banks/{kb}/documents/d2")).json()
    assert other["extracted_metadata"] == {"doc_type": "memo"}
    assert len(provider.get_mock_calls()) == 1


@pytest.mark.asyncio
async def test_supplied_values_are_held_to_the_schema(kb_client, bank):
    bad_value = await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/documents",
        json={"documents": [{"id": "x", "text": "t", "properties": {"doc_type": "receipt"}}]},
    )
    assert bad_value.status_code == 400 and "schema" in bad_value.json()["detail"]

    unknown = await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/documents",
        json={"documents": [{"id": "x", "text": "t", "properties": {"nope": 1}}]},
    )
    assert unknown.status_code == 400 and "nope" in unknown.json()["detail"]


# ---- the DSL, compiled


def test_compiler_binds_every_value_and_never_interpolates_input():
    compiled = compile_query(
        {
            "from": "chunks",
            "select": [{"field": "metadata.doc_type", "as": "kind"}, {"count": "*", "as": "n"}],
            "where": {"vendor": "Acme'; DROP TABLE kb_chunks; --"},
            "group_by": ["metadata.doc_type"],
        },
        "bank-1",
    )
    assert "DROP TABLE" not in compiled.sql
    assert compiled.params[0] == "bank-1"
    assert any("DROP TABLE" in str(p) for p in compiled.params)
    assert compiled.columns == ["kind", "n"] and compiled.grouped


@pytest.mark.parametrize(
    "body,message",
    [
        ({"select": []}, "non-empty"),
        ({"select": ["nope"]}, "unknown field"),
        ({"select": [{"sum": "token_count"}, "doc_id"]}, "group_by is empty"),
        ({"select": [{"count": "*", "as": "n"}], "having": {"missing": {"$gte": 1}}}, "select does not define"),
        ({"select": [{"count": "*", "as": "n"}], "order_by": [{"field": "n", "direction": "sideways"}]}, "asc or desc"),
        ({"select": [{"sum": "*"}]}, "only count takes"),
        ({"select": [{"count": "*", "as": "n"}], "limit": 100000}, "exceeds"),
        ({"select": [{"count": "*", "as": "a"}, {"count": "*", "as": "a"}]}, "duplicate"),
        ({"select": [{"date_trunc": ["century", "created_at"]}]}, "date_trunc"),
        ({"select": [{"nonsense": "doc_id"}]}, "unknown operator"),
        ({"from": "documents", "select": [{"field": "chunk_metadata.x"}]}, "not available"),
        ({"from": "nothing", "select": ["doc_id"]}, "documents"),
    ],
)
def test_compiler_refuses_what_a_caller_cannot_have_meant(body, message):
    with pytest.raises(FilterError, match=message):
        compile_query(body, "bank-1")


def test_compiler_bounds_the_work_a_single_query_can_ask_for():
    limits = Limits()
    with pytest.raises(FilterError, match="deeper than"):
        deep = "token_count"
        for _ in range(limits.max_depth + 2):
            deep = {"add": [deep, 1]}
        compile_query({"select": [{"sum": deep}]}, "bank-1")

    with pytest.raises(FilterError, match="at most"):
        compile_query({"select": [{"count": "*", "as": f"c{i}"} for i in range(limits.max_select + 1)]}, "bank-1")

    with pytest.raises(FilterError, match="at most"):
        compile_query(
            {"select": ["doc_id"], "group_by": [f"metadata.p{i}" for i in range(limits.max_group_by + 1)]},
            "bank-1",
        )

    with pytest.raises(FilterError, match="at most"):
        compile_query(
            {"select": [{"count": "*", "as": "n"}], "where": {"vendor": {"$in": [str(i) for i in range(600)]}}},
            "bank-1",
        )


# ---- the DSL, run


@pytest.mark.asyncio
async def test_counts_and_sums_group_the_way_sql_would(kb_client, bank):
    result = await _query(
        kb_client,
        bank,
        {
            "from": "documents",
            "select": [
                {"field": "metadata.doc_type", "as": "doc_type"},
                {"count": "*", "as": "documents"},
                {"sum": "metadata.total", "as": "total"},
                {"round": [{"avg": "metadata.total"}, 2], "as": "avg_total"},
                {"max": "metadata.total", "as": "biggest"},
            ],
            "group_by": ["metadata.doc_type"],
            "order_by": [{"field": "total", "direction": "desc"}],
        },
    )
    rows = _rows(result)
    assert rows[0] == {"doc_type": "contract", "documents": 2, "total": 27500, "avg_total": 13750, "biggest": 20000}
    assert [row["doc_type"] for row in rows] == ["contract", "invoice", "memo"]
    # The memo has no total: a missing value is not a zero.
    assert rows[-1] == {"doc_type": "memo", "documents": 1, "total": None, "avg_total": None, "biggest": None}


@pytest.mark.asyncio
async def test_where_having_and_expressions_over_aggregates(kb_client, bank):
    result = await _query(
        kb_client,
        bank,
        {
            "from": "documents",
            "select": [
                {"field": "metadata.vendor", "as": "vendor"},
                {"count": "*", "as": "documents"},
                {"sum": "metadata.total", "as": "total"},
                {
                    "round": [{"divide": [{"sum": "metadata.total"}, {"count": "metadata.total"}]}, 1],
                    "as": "per_document",
                },
            ],
            "where": {"region": "emea"},
            "group_by": ["metadata.vendor"],
            "having": {"documents": {"$gte": 2}},
        },
    )
    rows = _rows(result)
    assert rows == [{"vendor": "Acme", "documents": 4, "total": 22000, "per_document": 7333.3}]


@pytest.mark.asyncio
async def test_chunk_level_aggregation_joins_its_document(kb_client, bank):
    result = await _query(
        kb_client,
        bank,
        {
            "from": "chunks",
            "select": [
                {"field": "chunk_metadata.clause", "as": "clause"},
                {"count": "*", "as": "chunks"},
                {"count_distinct": "doc_id", "as": "documents"},
                {"sum": "token_count", "as": "tokens"},
            ],
            "group_by": ["chunk_metadata.clause"],
            "order_by": [{"field": "chunks", "direction": "desc"}, "clause"],
        },
    )
    rows = _rows(result)
    payment = next(row for row in rows if row["clause"] == "payment")
    assert payment["documents"] == 3 and payment["chunks"] >= 3 and payment["tokens"] > 0


@pytest.mark.asyncio
async def test_a_query_with_no_aggregate_returns_rows(kb_client, bank):
    result = await _query(
        kb_client,
        bank,
        {
            "from": "documents",
            "select": ["doc_id", {"field": "metadata.total", "as": "total"}, {"length": "title", "as": "title_length"}],
            "where": {"total": {"$gte": 5000}},
            "order_by": [{"field": "total", "direction": "desc"}],
            "limit": 10,
        },
    )
    assert not result["grouped"]
    assert [row["doc_id"] for row in _rows(result)] == ["con-1", "con-2", "inv-3"]


@pytest.mark.asyncio
async def test_date_functions_group_by_period(kb_client, bank):
    result = await _query(
        kb_client,
        bank,
        {
            "from": "documents",
            "select": [
                {"extract": ["year", "metadata.signed_on"], "as": "year"},
                {"count": "*", "as": "documents"},
            ],
            "where": {"signed_on": {"$exists": True}},
            "group_by": [{"extract": ["year", "metadata.signed_on"]}],
            "order_by": ["year"],
        },
    )
    assert _rows(result) == [{"year": 2025, "documents": 1}, {"year": 2026, "documents": 4}]


@pytest.mark.asyncio
async def test_a_query_postgres_rejects_is_a_400_not_a_500(kb_client, bank):
    # A non-numeric property summed is NULL, not an error — but an unorderable comparison
    # still has to come back as the caller's problem.
    response = await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/query",
        json={"select": [{"date_trunc": ["day", "metadata.vendor"]}, {"count": "*", "as": "n"}]},
    )
    assert response.status_code == 400, response.text


@pytest.mark.asyncio
async def test_text_values_never_break_a_numeric_aggregate(kb_client, bank):
    """A property holds whatever was extracted; summing it must not abort the statement."""
    result = await _query(
        kb_client,
        bank,
        {"from": "documents", "select": [{"sum": "metadata.vendor", "as": "nonsense"}, {"count": "*", "as": "n"}]},
    )
    assert _rows(result) == [{"nonsense": None, "n": len(CORPUS)}]


@pytest.mark.asyncio
async def test_a_query_cannot_reach_another_bank(kb_client, bank, memory):
    other = f"kb-{uuid.uuid4().hex[:8]}"
    await kb_client.post("/v1/default/knowledge-banks", json={"id": other})
    await kb_client.post(
        f"/v1/default/knowledge-banks/{other}/documents",
        json={"documents": [{"id": "inv-1", "text": "A different bank's document with the same id."}]},
    )
    result = await _query(kb_client, other, {"from": "documents", "select": [{"count": "*", "as": "n"}]})
    assert _rows(result) == [{"n": 1}]
    mine = await _query(kb_client, bank, {"from": "documents", "select": [{"count": "*", "as": "n"}]})
    assert _rows(mine) == [{"n": len(CORPUS)}]
