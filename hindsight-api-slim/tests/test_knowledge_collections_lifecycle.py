"""Collections over time: what happens to a record when the corpus and the schema move.

The easy half of a derived dataset is filling it once. The hard half is everything after:
a second document about the same thing, a document rewritten, a document deleted, a field
added, a field dropped, a relationship introduced, a human correction that must survive
all of it. Each of those is a way for a record to end up wrong and stay wrong silently,
so each gets a case here.

The LLM is the stub throughout: what is under test is the bookkeeping — which contribution
a value came from, and what happens to the record when that contribution changes — not
whether a model can read a contract.
"""

import uuid

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app

VENDORS = {"name": "Vendors", "identity": "name", "fields": {"name": {"type": "string"}, "country": {"type": "string"}}}


@pytest_asyncio.fixture
async def kb_client(memory):
    app = create_app(memory, initialize_memory=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield client


@pytest.fixture
def memory_engine_service(memory):
    """The service itself, for the cases that need two operations in flight at once."""
    from hindsight_api.knowledge.service import KnowledgeService

    return KnowledgeService(memory)


@pytest_asyncio.fixture
async def bank(kb_client) -> str:
    kb = f"kb-{uuid.uuid4().hex[:8]}"
    assert (await kb_client.post("/v1/default/knowledge-banks", json={"id": kb})).status_code == 201
    return kb


class Derivation:
    """Scripts what the model 'finds' in each document, keyed by a marker in its text."""

    def __init__(self, memory):
        self.provider = memory._llm_config._provider_impl
        self.by_marker: dict[str, list[dict]] = {}
        self.provider.set_response_callback(self._answer)

    def says(self, marker: str, records: list[dict]) -> None:
        self.by_marker[marker] = records

    def _answer(self, messages: list[dict], scope: str):
        if scope != "knowledge_records":
            return {}
        content = messages[-1]["content"]
        for marker, records in self.by_marker.items():
            if marker in content:
                return {"records": records}
        return {"records": []}


async def _write(client, kb: str, documents: list[dict]) -> None:
    response = await client.post(f"/v1/default/knowledge-banks/{kb}/documents", json={"documents": documents})
    assert response.status_code == 202, response.text


async def _collection(client, kb: str, collection_id: str, body: dict, **params) -> dict:
    response = await client.put(
        f"/v1/default/knowledge-banks/{kb}/collections/{collection_id}", json=body, params=params
    )
    assert response.status_code == 200, response.text
    return response.json()


async def _derive(client, kb: str, collection_id: str, **body) -> None:
    response = await client.post(f"/v1/default/knowledge-banks/{kb}/collections/{collection_id}/derive", json=body)
    assert response.status_code == 202, response.text


async def _record(client, kb: str, collection_id: str, record_id: str) -> dict | None:
    response = await client.get(f"/v1/default/knowledge-banks/{kb}/collections/{collection_id}/records/{record_id}")
    return response.json() if response.status_code == 200 else None


async def _count(client, kb: str, collection_id: str) -> int:
    response = await client.post(
        f"/v1/default/knowledge-banks/{kb}/collections/{collection_id}/query",
        json={"select": [{"count": "*", "as": "n"}]},
    )
    assert response.status_code == 200, response.text
    return response.json()["rows"][0][0]


# ---- a corpus that grows


@pytest.mark.asyncio
async def test_a_second_document_completes_a_record_without_duplicating_it(kb_client, memory, bank):
    llm = Derivation(memory)
    await _collection(kb_client, bank, "vendors", VENDORS)

    llm.says("first", [{"values": {"name": "acme"}, "evidence": {"name": "Acme Ltd"}}])
    await _write(kb_client, bank, [{"id": "d1", "text": "first: Acme Ltd supplies us."}])
    await _derive(kb_client, bank, "vendors")
    assert (await _record(kb_client, bank, "vendors", "acme"))["values"] == {"name": "acme"}

    llm.says("second", [{"values": {"name": "acme", "country": "de"}, "evidence": {"country": "based in Germany"}}])
    await _write(kb_client, bank, [{"id": "d2", "text": "second: Acme Ltd, based in Germany."}])
    await _derive(kb_client, bank, "vendors", doc_ids=["d2"])

    record = await _record(kb_client, bank, "vendors", "acme")
    assert record["values"] == {"name": "acme", "country": "de"}, "the second document filled the gap"
    assert sorted(record["doc_ids"]) == ["d1", "d2"], "and both documents are credited"
    assert await _count(kb_client, bank, "vendors") == 1, "one vendor, not two"
    assert {e["doc_id"] for e in record["evidence"]["country"]} == {"d2"}


@pytest.mark.asyncio
async def test_re_deriving_the_same_document_is_idempotent(kb_client, memory, bank):
    llm = Derivation(memory)
    await _collection(kb_client, bank, "vendors", VENDORS)
    llm.says("acme", [{"values": {"name": "acme", "country": "de"}, "evidence": {"country": "German"}}])
    await _write(kb_client, bank, [{"id": "d1", "text": "acme: a German supplier."}])

    for _ in range(3):
        await _derive(kb_client, bank, "vendors")

    record = await _record(kb_client, bank, "vendors", "acme")
    assert record["values"] == {"name": "acme", "country": "de"}
    assert record["doc_ids"] == ["d1"]
    assert len(record["evidence"]["country"]) == 1, "three runs, one piece of evidence — not three"
    assert await _count(kb_client, bank, "vendors") == 1


@pytest.mark.asyncio
async def test_a_corrected_document_replaces_what_it_said_before(kb_client, memory, bank):
    """The hard one: a document that used to say Germany now says France."""
    llm = Derivation(memory)
    await _collection(kb_client, bank, "vendors", VENDORS)
    llm.says("supplier", [{"values": {"name": "acme", "country": "de"}, "evidence": {"country": "German supplier"}}])
    await _write(kb_client, bank, [{"id": "d1", "text": "Acme is a German supplier."}])
    await _derive(kb_client, bank, "vendors")
    assert (await _record(kb_client, bank, "vendors", "acme"))["values"]["country"] == "de"

    llm.says("supplier", [{"values": {"name": "acme", "country": "fr"}, "evidence": {"country": "French supplier"}}])
    await _derive(kb_client, bank, "vendors", replace=True)

    record = await _record(kb_client, bank, "vendors", "acme")
    assert record["values"]["country"] == "fr", "the document's old value must not survive its own correction"
    assert len(record["evidence"]["country"]) == 1
    assert record["evidence"]["country"][0]["quote"] == "French supplier"


@pytest.mark.asyncio
async def test_a_document_that_stops_mentioning_a_record_releases_it(kb_client, memory, bank):
    llm = Derivation(memory)
    await _collection(kb_client, bank, "vendors", VENDORS)
    llm.says(
        "both",
        [{"values": {"name": "acme"}, "evidence": {}}, {"values": {"name": "globex"}, "evidence": {}}],
    )
    await _write(kb_client, bank, [{"id": "d1", "text": "both: Acme and Globex."}])
    await _derive(kb_client, bank, "vendors")
    assert await _count(kb_client, bank, "vendors") == 2

    llm.says("both", [{"values": {"name": "acme"}, "evidence": {}}])
    await _derive(kb_client, bank, "vendors", replace=True)

    assert await _count(kb_client, bank, "vendors") == 1, "globex had no other source and should be gone"
    assert await _record(kb_client, bank, "vendors", "globex") is None
    assert await _record(kb_client, bank, "vendors", "acme") is not None


# ---- a corpus that shrinks


@pytest.mark.asyncio
async def test_deleting_a_document_takes_back_exactly_what_it_contributed(kb_client, memory, bank):
    llm = Derivation(memory)
    await _collection(kb_client, bank, "vendors", VENDORS)
    llm.says("one", [{"values": {"name": "acme"}, "evidence": {"name": "Acme"}}])
    llm.says("two", [{"values": {"name": "acme", "country": "de"}, "evidence": {"country": "Germany"}}])
    await _write(kb_client, bank, [{"id": "d1", "text": "one: Acme."}, {"id": "d2", "text": "two: Acme, Germany."}])
    await _derive(kb_client, bank, "vendors")
    assert (await _record(kb_client, bank, "vendors", "acme"))["values"] == {"name": "acme", "country": "de"}

    assert (await kb_client.delete(f"/v1/default/knowledge-banks/{bank}/documents/d2")).status_code == 200

    record = await _record(kb_client, bank, "vendors", "acme")
    assert record is not None, "d1 still supports the record"
    assert record["values"] == {"name": "acme"}, "but the country came only from d2 and goes with it"
    assert record["doc_ids"] == ["d1"]
    assert "country" not in record["evidence"]

    assert (await kb_client.delete(f"/v1/default/knowledge-banks/{bank}/documents/d1")).status_code == 200
    assert await _record(kb_client, bank, "vendors", "acme") is None, "nothing is behind it any more"
    assert await _count(kb_client, bank, "vendors") == 0


@pytest.mark.asyncio
async def test_a_record_can_be_deleted_outright(kb_client, memory, bank):
    await _collection(kb_client, bank, "vendors", VENDORS)
    await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/records",
        json={"records": [{"values": {"name": "acme", "country": "de"}}]},
    )
    deleted = await kb_client.delete(f"/v1/default/knowledge-banks/{bank}/collections/vendors/records/acme")
    assert deleted.status_code == 200 and deleted.json()["deleted"] is True
    assert await _count(kb_client, bank, "vendors") == 0
    assert (
        await kb_client.delete(f"/v1/default/knowledge-banks/{bank}/collections/vendors/records/acme")
    ).status_code == 404


@pytest.mark.asyncio
async def test_a_pin_survives_new_documents_deletions_and_re_derivation(kb_client, memory, bank):
    llm = Derivation(memory)
    await _collection(kb_client, bank, "vendors", VENDORS)
    llm.says("acme", [{"values": {"name": "acme", "country": "de"}, "evidence": {}}])
    await _write(kb_client, bank, [{"id": "d1", "text": "acme: German."}])
    await _derive(kb_client, bank, "vendors")

    pinned = await kb_client.put(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/records/acme/pins",
        json={"values": {"country": "fr"}},
    )
    assert pinned.status_code == 200 and pinned.json()["values"]["country"] == "fr"

    # Another document says Germany again; re-derivation runs; the pin holds.
    llm.says("also", [{"values": {"name": "acme", "country": "de"}, "evidence": {}}])
    await _write(kb_client, bank, [{"id": "d2", "text": "also: Acme is German."}])
    await _derive(kb_client, bank, "vendors", replace=True)
    assert (await _record(kb_client, bank, "vendors", "acme"))["values"]["country"] == "fr"

    # Even when every document behind the record is gone, the correction keeps it alive.
    for doc_id in ("d1", "d2"):
        await kb_client.delete(f"/v1/default/knowledge-banks/{bank}/documents/{doc_id}")
    record = await _record(kb_client, bank, "vendors", "acme")
    assert record is not None and record["values"]["country"] == "fr"
    assert record["doc_ids"] == []


# ---- a schema that changes


@pytest.mark.asyncio
async def test_adding_a_field_leaves_records_alone_until_they_are_reprocessed(kb_client, memory, bank):
    llm = Derivation(memory)
    await _collection(kb_client, bank, "vendors", VENDORS)
    llm.says("acme", [{"values": {"name": "acme", "country": "de"}, "evidence": {}}])
    await _write(kb_client, bank, [{"id": "d1", "text": "acme: German, gold tier."}])
    await _derive(kb_client, bank, "vendors")

    with_tier = {**VENDORS, "fields": {**VENDORS["fields"], "tier": {"type": "string"}}}
    changed = await _collection(kb_client, bank, "vendors", with_tier)
    assert changed["changes"]["added"] == ["tier"] and changed["changes"]["removed"] == []
    assert (await _record(kb_client, bank, "vendors", "acme"))["values"] == {"name": "acme", "country": "de"}

    # Reprocessing is what fills it, and it is a decision because it costs LLM calls.
    llm.says("acme", [{"values": {"name": "acme", "country": "de", "tier": "gold"}, "evidence": {}}])
    reprocessed = await _collection(kb_client, bank, "vendors", with_tier, reprocess=True)
    assert reprocessed["reprocess"]["status"] == "pending"
    assert (await _record(kb_client, bank, "vendors", "acme"))["values"]["tier"] == "gold"


@pytest.mark.asyncio
async def test_dropping_a_field_takes_its_values_out_of_the_records(kb_client, memory, bank):
    await _collection(kb_client, bank, "vendors", VENDORS)
    await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/records",
        json={"records": [{"values": {"name": "acme", "country": "de"}, "evidence": {"country": "Germany"}}]},
    )
    changed = await _collection(
        kb_client, bank, "vendors", {"name": "Vendors", "identity": "name", "fields": {"name": {"type": "string"}}}
    )
    assert changed["changes"]["removed"] == ["country"]
    assert changed["changes"]["records_pruned"] == 1

    record = await _record(kb_client, bank, "vendors", "acme")
    assert record["values"] == {"name": "acme"}, "a column the schema dropped must not survive in the rows"
    assert "country" not in record["evidence"]

    # And the field is gone from the collection, so querying it is an error rather than
    # an empty column: the schema is the contract, and it no longer has that column.
    response = await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/query",
        json={"select": [{"count": "*", "as": "n"}], "where": {"country": {"$exists": True}}},
    )
    assert response.status_code == 400 and "country" in response.json()["detail"]


@pytest.mark.asyncio
async def test_retyping_a_field_clears_the_values_of_the_old_type(kb_client, memory, bank):
    await _collection(
        kb_client,
        bank,
        "contracts",
        {"identity": "reference", "fields": {"reference": {"type": "string"}, "value": {"type": "string"}}},
    )
    await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/contracts/records",
        json={"records": [{"values": {"reference": "c-1", "value": "about twelve thousand"}}]},
    )
    changed = await _collection(
        kb_client,
        bank,
        "contracts",
        {"identity": "reference", "fields": {"reference": {"type": "string"}, "value": {"type": "number"}}},
    )
    assert changed["changes"]["retyped"] == ["value"]
    record = await _record(kb_client, bank, "contracts", "c-1")
    assert record["values"] == {"reference": "c-1"}, "prose is not silently kept as a number"


@pytest.mark.asyncio
async def test_a_relationship_added_later_is_resolved_by_reprocessing(kb_client, memory, bank):
    llm = Derivation(memory)
    await _collection(kb_client, bank, "vendors", VENDORS)
    await _collection(
        kb_client,
        bank,
        "contracts",
        {"identity": "reference", "fields": {"reference": {"type": "string"}, "value": {"type": "number"}}},
    )
    llm.says(
        "c-1",
        [{"values": {"reference": "c-1", "value": 1000}, "evidence": {}}],
    )
    await _write(kb_client, bank, [{"id": "d1", "text": "c-1: a contract with Acme worth 1000."}])
    await _derive(kb_client, bank, "contracts")
    await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/records",
        json={"records": [{"values": {"name": "acme", "country": "de"}}]},
    )

    # The relationship did not exist when the contract was derived; it does now.
    with_vendor = {
        "identity": "reference",
        "fields": {
            "reference": {"type": "string"},
            "value": {"type": "number"},
            "vendor": {"collection": "vendors"},
        },
    }
    llm.says("c-1", [{"values": {"reference": "c-1", "value": 1000, "vendor": "Acme"}, "evidence": {}}])
    changed = await _collection(kb_client, bank, "contracts", with_vendor, reprocess=True)
    assert changed["changes"]["added"] == ["vendor"]

    joined = await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/contracts/query",
        json={
            "join": [{"on": "vendor", "as": "v"}],
            "select": [{"field": "reference", "as": "reference"}, {"field": "v.country", "as": "country"}],
        },
    )
    assert joined.json()["rows"] == [["c-1", "de"]], "the join resolves a relationship the model named by hand"


@pytest.mark.asyncio
async def test_derive_on_write_keeps_a_collection_current_with_the_corpus(kb_client, memory, bank):
    """No job to run: writing a document updates the records it belongs to."""
    llm = Derivation(memory)
    await _collection(kb_client, bank, "vendors", {**VENDORS, "derive_on_write": True})

    llm.says("acme", [{"values": {"name": "acme", "country": "de"}, "evidence": {}}])
    await _write(kb_client, bank, [{"id": "d1", "text": "acme: German supplier."}])
    assert (await _record(kb_client, bank, "vendors", "acme"))["values"]["country"] == "de"

    llm.says("globex", [{"values": {"name": "globex", "country": "fr"}, "evidence": {}}])
    await _write(kb_client, bank, [{"id": "d2", "text": "globex: French supplier."}])
    assert await _count(kb_client, bank, "vendors") == 2

    # Rewriting a document re-derives it rather than contributing twice.
    llm.says("acme", [{"values": {"name": "acme", "country": "it"}, "evidence": {}}])
    await _write(kb_client, bank, [{"id": "d1", "text": "acme: Italian supplier now."}])
    record = await _record(kb_client, bank, "vendors", "acme")
    assert record["values"]["country"] == "it" and record["doc_ids"] == ["d1"]
    assert await _count(kb_client, bank, "vendors") == 2


@pytest.mark.asyncio
async def test_deleting_a_collection_leaves_the_documents_alone(kb_client, memory, bank):
    llm = Derivation(memory)
    await _collection(kb_client, bank, "vendors", VENDORS)
    llm.says("acme", [{"values": {"name": "acme"}, "evidence": {}}])
    await _write(kb_client, bank, [{"id": "d1", "text": "acme: a supplier."}])
    await _derive(kb_client, bank, "vendors")

    assert (await kb_client.delete(f"/v1/default/knowledge-banks/{bank}/collections/vendors")).status_code == 200
    assert (await kb_client.get(f"/v1/default/knowledge-banks/{bank}/collections/vendors")).status_code == 404
    document = await kb_client.get(f"/v1/default/knowledge-banks/{bank}/documents/d1")
    assert document.status_code == 200 and document.json()["passage_count"] == 1

    # Re-creating the collection starts empty; the documents are still there to re-read.
    await _collection(kb_client, bank, "vendors", VENDORS)
    assert await _count(kb_client, bank, "vendors") == 0
    await _derive(kb_client, bank, "vendors")
    assert await _count(kb_client, bank, "vendors") == 1


# ---- the awkward ones


@pytest.mark.asyncio
async def test_two_documents_that_disagree_keep_both_sides_visible(kb_client, memory, bank):
    """A later document wins the value, but the disagreement must not disappear."""
    llm = Derivation(memory)
    await _collection(kb_client, bank, "vendors", VENDORS)
    llm.says("older", [{"values": {"name": "acme", "country": "de"}, "evidence": {"country": "German company"}}])
    llm.says("newer", [{"values": {"name": "acme", "country": "fr"}, "evidence": {"country": "French company"}}])
    await _write(kb_client, bank, [{"id": "d1", "text": "older: Acme, a German company."}])
    await _derive(kb_client, bank, "vendors", doc_ids=["d1"])
    await _write(kb_client, bank, [{"id": "d2", "text": "newer: Acme, a French company."}])
    await _derive(kb_client, bank, "vendors", doc_ids=["d2"])

    record = await _record(kb_client, bank, "vendors", "acme")
    assert record["values"]["country"] == "fr", "the later document decides the value"
    quotes = {(e["doc_id"], e["quote"]) for e in record["evidence"]["country"]}
    assert quotes == {("d1", "German company"), ("d2", "French company")}, "both sides stay on the record"


@pytest.mark.asyncio
async def test_a_record_whose_identity_value_changes_does_not_leave_a_ghost(kb_client, memory, bank):
    llm = Derivation(memory)
    await _collection(kb_client, bank, "vendors", VENDORS)
    llm.says("vendor", [{"values": {"name": "acme", "country": "de"}, "evidence": {}}])
    await _write(kb_client, bank, [{"id": "d1", "text": "vendor: Acme."}])
    await _derive(kb_client, bank, "vendors")
    assert await _count(kb_client, bank, "vendors") == 1

    # The document is corrected: it was never Acme, it was Northwind. (A suffix would not
    # do here — "Acme Inc" resolves to the same record as "Acme", which is the point of
    # identity resolution.)
    llm.says("vendor", [{"values": {"name": "northwind", "country": "de"}, "evidence": {}}])
    await _derive(kb_client, bank, "vendors", replace=True)

    assert await _record(kb_client, bank, "vendors", "acme") is None, "the old id has nothing behind it"
    assert (await _record(kb_client, bank, "vendors", "northwind"))["values"]["country"] == "de"
    assert await _count(kb_client, bank, "vendors") == 1


@pytest.mark.asyncio
async def test_changing_the_identity_field_re_keys_every_record(kb_client, memory, bank):
    llm = Derivation(memory)
    await _collection(
        kb_client,
        bank,
        "vendors",
        {"identity": "name", "fields": {"name": {"type": "string"}, "code": {"type": "string"}}},
    )
    llm.says("acme", [{"values": {"name": "acme", "code": "v-001"}, "evidence": {}}])
    await _write(kb_client, bank, [{"id": "d1", "text": "acme: code v-001."}])
    await _derive(kb_client, bank, "vendors")
    assert await _record(kb_client, bank, "vendors", "acme") is not None

    changed = await _collection(
        kb_client,
        bank,
        "vendors",
        {"identity": "code", "fields": {"name": {"type": "string"}, "code": {"type": "string"}}},
        reprocess=True,
    )
    assert changed["changes"].get("identity_changed") is True
    assert await _record(kb_client, bank, "vendors", "v-001") is not None, "re-keyed to the new identity"
    assert await _record(kb_client, bank, "vendors", "acme") is None, "and the old key is gone"
    assert await _count(kb_client, bank, "vendors") == 1


@pytest.mark.asyncio
async def test_a_record_with_no_identity_field_is_replaced_not_duplicated(kb_client, memory, bank):
    """With no identity the id is a hash of the values, so a changed value is a new id."""
    llm = Derivation(memory)
    await _collection(kb_client, bank, "notes", {"fields": {"subject": {"type": "string"}}})
    llm.says("note", [{"values": {"subject": "delivery delays"}, "evidence": {}}])
    await _write(kb_client, bank, [{"id": "d1", "text": "note: about delivery delays."}])
    await _derive(kb_client, bank, "notes")
    assert await _count(kb_client, bank, "notes") == 1

    llm.says("note", [{"values": {"subject": "payment terms"}, "evidence": {}}])
    await _derive(kb_client, bank, "notes", replace=True)
    assert await _count(kb_client, bank, "notes") == 1, "the old hash had nothing left behind it"


@pytest.mark.asyncio
async def test_a_dangling_relationship_survives_the_other_collection_disappearing(kb_client, memory, bank):
    await _collection(kb_client, bank, "vendors", VENDORS)
    await _collection(
        kb_client,
        bank,
        "contracts",
        {"identity": "reference", "fields": {"reference": {"type": "string"}, "vendor": {"collection": "vendors"}}},
    )
    await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/records",
        json={"records": [{"values": {"name": "acme", "country": "de"}}]},
    )
    await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/contracts/records",
        json={"records": [{"values": {"reference": "c-1", "vendor": "acme"}}]},
    )
    assert (await kb_client.delete(f"/v1/default/knowledge-banks/{bank}/collections/vendors")).status_code == 200

    # The contract is still a contract and still queryable on its own fields...
    alone = await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/contracts/query",
        json={"select": [{"field": "reference", "as": "reference"}, {"field": "vendor", "as": "vendor"}]},
    )
    assert alone.status_code == 200 and alone.json()["rows"] == [["c-1", "acme"]]

    # ...but joining a collection that no longer exists is an error, not a null column.
    joined = await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/contracts/query",
        json={
            "join": [{"on": "vendor", "as": "v"}],
            "select": [{"field": "reference", "as": "reference"}, {"field": "v.country", "as": "country"}],
        },
    )
    assert joined.status_code == 400 and "no longer exists" in joined.json()["detail"]


@pytest.mark.asyncio
async def test_a_pin_on_a_field_that_is_later_dropped_goes_with_it(kb_client, memory, bank):
    await _collection(kb_client, bank, "vendors", VENDORS)
    await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/records",
        json={"records": [{"values": {"name": "acme", "country": "de"}}]},
    )
    await kb_client.put(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/records/acme/pins",
        json={"values": {"country": "fr"}},
    )
    await _collection(kb_client, bank, "vendors", {"identity": "name", "fields": {"name": {"type": "string"}}})
    record = await _record(kb_client, bank, "vendors", "acme")
    assert record["values"] == {"name": "acme"}
    assert record["pinned"] == {}, "a pin on a field the schema dropped cannot outlive it"


@pytest.mark.asyncio
async def test_writing_a_record_twice_updates_it_rather_than_stacking_evidence(kb_client, bank):
    await _collection(kb_client, bank, "vendors", VENDORS)
    for country in ("de", "fr", "it"):
        await kb_client.post(
            f"/v1/default/knowledge-banks/{bank}/collections/vendors/records",
            json={"records": [{"values": {"name": "acme", "country": country}, "doc_ids": ["d1"]}]},
        )
    record = await _record(kb_client, bank, "vendors", "acme")
    assert record["values"]["country"] == "it"
    assert record["doc_ids"] == ["d1"]
    assert await _count(kb_client, bank, "vendors") == 1


@pytest.mark.asyncio
async def test_a_query_naming_a_field_the_collection_does_not_have_is_refused(kb_client, bank):
    """A typo must not come back as a column of nulls: the collection knows its fields."""
    await _collection(kb_client, bank, "vendors", VENDORS)
    await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/records",
        json={"records": [{"values": {"name": "acme", "country": "de"}}]},
    )
    misspelled = await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/query",
        json={"select": [{"field": "countrry", "as": "c"}]},
    )
    assert misspelled.status_code == 400 and "countrry" in misspelled.json()["detail"]

    filtered = await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/query",
        json={"select": [{"count": "*", "as": "n"}], "where": {"nope": "x"}},
    )
    assert filtered.status_code == 400 and "nope" in filtered.json()["detail"]


@pytest.mark.asyncio
async def test_a_query_naming_a_field_the_joined_collection_does_not_have_is_refused(kb_client, bank):
    await _collection(kb_client, bank, "vendors", VENDORS)
    await _collection(
        kb_client,
        bank,
        "contracts",
        {"identity": "reference", "fields": {"reference": {"type": "string"}, "vendor": {"collection": "vendors"}}},
    )
    response = await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/contracts/query",
        json={
            "join": [{"on": "vendor", "as": "v"}],
            "select": [{"field": "v.turnover", "as": "turnover"}],
        },
    )
    assert response.status_code == 400 and "turnover" in response.json()["detail"]


@pytest.mark.asyncio
async def test_one_document_feeding_two_collections_stays_consistent_in_both(kb_client, memory, bank):
    llm = Derivation(memory)
    await _collection(kb_client, bank, "vendors", {**VENDORS, "derive_on_write": True})
    await _collection(
        kb_client,
        bank,
        "contracts",
        {
            "identity": "reference",
            "derive_on_write": True,
            "fields": {"reference": {"type": "string"}, "value": {"type": "number"}},
        },
    )

    def answer(messages, scope):
        if scope != "knowledge_records":
            return {}
        content = messages[-1]["content"]
        if "Collection: Vendors" in content:
            return {"records": [{"values": {"name": "acme", "country": "de"}, "evidence": {}}]}
        return {"records": [{"values": {"reference": "c-1", "value": 5000}, "evidence": {}}]}

    llm.provider.set_response_callback(answer)
    await _write(kb_client, bank, [{"id": "d1", "text": "Acme, German, contract c-1 worth 5000."}])

    assert (await _record(kb_client, bank, "vendors", "acme"))["values"]["country"] == "de"
    assert (await _record(kb_client, bank, "contracts", "c-1"))["values"]["value"] == 5000

    await kb_client.delete(f"/v1/default/knowledge-banks/{bank}/documents/d1")
    assert await _count(kb_client, bank, "vendors") == 0, "both collections forget it"
    assert await _count(kb_client, bank, "contracts") == 0


@pytest.mark.asyncio
async def test_the_same_thing_named_untidily_is_one_record(kb_client, memory, bank):
    llm = Derivation(memory)
    await _collection(kb_client, bank, "vendors", VENDORS)
    llm.says("one", [{"values": {"name": "Acme "}, "evidence": {}}])
    llm.says("two", [{"values": {"name": "ACME"}, "evidence": {}}])
    llm.says("three", [{"values": {"name": "acme"}, "evidence": {}}])
    await _write(
        kb_client,
        bank,
        [
            {"id": "d1", "text": "one: Acme."},
            {"id": "d2", "text": "two: ACME."},
            {"id": "d3", "text": "three: acme."},
        ],
    )
    await _derive(kb_client, bank, "vendors")
    assert await _count(kb_client, bank, "vendors") == 1, "case and spacing are not three different companies"
    assert sorted((await _record(kb_client, bank, "vendors", "acme"))["doc_ids"]) == ["d1", "d2", "d3"]


@pytest.mark.asyncio
async def test_a_record_fed_by_many_documents_does_not_grow_an_unbounded_payload(kb_client, memory, bank):
    """Evidence is an audit trail, not a log: the record keeps the recent ones."""
    llm = Derivation(memory)
    await _collection(kb_client, bank, "vendors", VENDORS)
    llm.says("acme", [{"values": {"name": "acme", "country": "de"}, "evidence": {"country": "German"}}])
    await _write(kb_client, bank, [{"id": f"d{i}", "text": f"acme: mention {i}"} for i in range(60)])
    await _derive(kb_client, bank, "vendors")

    record = await _record(kb_client, bank, "vendors", "acme")
    assert len(record["doc_ids"]) == 60, "every document is still credited"
    assert len(record["evidence"]["country"]) <= 20, "the quotes are capped"
    assert record["values"] == {"name": "acme", "country": "de"}


@pytest.mark.asyncio
async def test_two_derivations_at_once_do_not_lose_a_value(kb_client, memory, bank, memory_engine_service):
    """Two documents about one record, derived concurrently: both halves must land."""
    import asyncio

    llm = Derivation(memory)
    await _collection(kb_client, bank, "vendors", VENDORS)
    llm.says("left", [{"values": {"name": "acme", "country": "de"}, "evidence": {}}])
    llm.says("right", [{"values": {"name": "acme"}, "evidence": {"name": "Acme Ltd"}}])
    await _write(
        kb_client, bank, [{"id": "d1", "text": "left: Acme, German."}, {"id": "d2", "text": "right: Acme Ltd."}]
    )

    await asyncio.gather(
        memory_engine_service.run_derive_records(
            {"bank_id": bank, "collection_id": "vendors", "doc_ids": ["d1"], "replace": True}
        ),
        memory_engine_service.run_derive_records(
            {"bank_id": bank, "collection_id": "vendors", "doc_ids": ["d2"], "replace": True}
        ),
    )
    record = await _record(kb_client, bank, "vendors", "acme")
    assert record["values"] == {"name": "acme", "country": "de"}
    assert sorted(record["doc_ids"]) == ["d1", "d2"], "neither derivation overwrote the other's contribution"


@pytest.mark.asyncio
async def test_a_failing_derivation_does_not_lose_the_document_or_the_record(kb_client, memory, bank):
    llm = Derivation(memory)
    await _collection(kb_client, bank, "vendors", {**VENDORS, "derive_on_write": True})
    llm.says("acme", [{"values": {"name": "acme", "country": "de"}, "evidence": {}}])
    await _write(kb_client, bank, [{"id": "d1", "text": "acme: German."}])
    assert (await _record(kb_client, bank, "vendors", "acme"))["values"]["country"] == "de"

    memory._llm_config._provider_impl.set_mock_exception(RuntimeError("model is down"))
    await _write(kb_client, bank, [{"id": "d2", "text": "acme: also ships to France."}])

    document = await kb_client.get(f"/v1/default/knowledge-banks/{bank}/documents/d2")
    assert document.status_code == 200, "the document is stored even when derivation fails"
    record = await _record(kb_client, bank, "vendors", "acme")
    assert record is not None and record["values"]["country"] == "de", "and the existing record is untouched"


@pytest.mark.asyncio
async def test_dropping_a_relationship_field_refuses_the_join_that_used_it(kb_client, bank):
    await _collection(kb_client, bank, "vendors", VENDORS)
    await _collection(
        kb_client,
        bank,
        "contracts",
        {"identity": "reference", "fields": {"reference": {"type": "string"}, "vendor": {"collection": "vendors"}}},
    )
    await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/contracts/records",
        json={"records": [{"values": {"reference": "c-1", "vendor": "acme"}}]},
    )
    await _collection(
        kb_client, bank, "contracts", {"identity": "reference", "fields": {"reference": {"type": "string"}}}
    )
    response = await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/contracts/query",
        json={"join": [{"on": "vendor", "as": "v"}], "select": [{"count": "*", "as": "n"}]},
    )
    assert response.status_code == 400 and "relationship" in response.json()["detail"]
    record = await _record(kb_client, bank, "contracts", "c-1")
    assert record["values"] == {"reference": "c-1"}


@pytest.mark.asyncio
async def test_a_relationship_to_a_deleted_record_reads_as_missing_not_stale(kb_client, bank):
    await _collection(kb_client, bank, "vendors", VENDORS)
    await _collection(
        kb_client,
        bank,
        "contracts",
        {"identity": "reference", "fields": {"reference": {"type": "string"}, "vendor": {"collection": "vendors"}}},
    )
    await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/records",
        json={"records": [{"values": {"name": "acme", "country": "de"}}]},
    )
    await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/contracts/records",
        json={"records": [{"values": {"reference": "c-1", "vendor": "acme"}}]},
    )
    await kb_client.delete(f"/v1/default/knowledge-banks/{bank}/collections/vendors/records/acme")

    joined = await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/contracts/query",
        json={
            "join": [{"on": "vendor", "as": "v"}],
            "select": [{"field": "reference", "as": "reference"}, {"field": "v.country", "as": "country"}],
        },
    )
    assert joined.json()["rows"] == [["c-1", None]], "the contract stands; the vendor is simply gone"


@pytest.mark.asyncio
async def test_filtering_and_ordering_on_a_joined_collection(kb_client, bank):
    await _collection(kb_client, bank, "vendors", VENDORS)
    await _collection(
        kb_client,
        bank,
        "contracts",
        {
            "identity": "reference",
            "fields": {
                "reference": {"type": "string"},
                "value": {"type": "number"},
                "vendor": {"collection": "vendors"},
            },
        },
    )
    await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/records",
        json={
            "records": [
                {"values": {"name": "acme", "country": "de"}},
                {"values": {"name": "globex", "country": "fr"}},
                {"values": {"name": "initech", "country": "us"}},
            ]
        },
    )
    await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/contracts/records",
        json={
            "records": [
                {"values": {"reference": "c-1", "value": 100, "vendor": "acme"}},
                {"values": {"reference": "c-2", "value": 300, "vendor": "globex"}},
                {"values": {"reference": "c-3", "value": 200, "vendor": "initech"}},
            ]
        },
    )
    response = await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/contracts/query",
        json={
            "join": [{"on": "vendor", "as": "v"}],
            "select": [{"field": "reference", "as": "reference"}, {"field": "v.country", "as": "country"}],
            "where": {"v.country": {"$in": ["de", "fr"]}},
            "order_by": [{"field": "value", "direction": "desc"}],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["rows"] == [["c-2", "fr"], ["c-1", "de"]]


@pytest.mark.asyncio
async def test_derivation_updates_a_record_a_caller_wrote_by_hand(kb_client, memory, bank):
    """A written record is data, not a correction: only a pin outranks the documents."""
    llm = Derivation(memory)
    await _collection(kb_client, bank, "vendors", VENDORS)
    await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/records",
        json={"records": [{"values": {"name": "acme", "country": "de"}}]},
    )
    llm.says("acme", [{"values": {"name": "acme", "country": "fr"}, "evidence": {}}])
    await _write(kb_client, bank, [{"id": "d1", "text": "acme: a French supplier."}])
    await _derive(kb_client, bank, "vendors")

    record = await _record(kb_client, bank, "vendors", "acme")
    assert record["values"]["country"] == "fr", "the document is newer, so it wins"
    assert record["doc_ids"] == ["d1"]

    # Pin it, and the next derivation cannot move it.
    await kb_client.put(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/records/acme/pins",
        json={"values": {"country": "de"}},
    )
    llm.says("acme", [{"values": {"name": "acme", "country": "it"}, "evidence": {}}])
    await _derive(kb_client, bank, "vendors", replace=True)
    assert (await _record(kb_client, bank, "vendors", "acme"))["values"]["country"] == "de"
