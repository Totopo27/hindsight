"""Moving a knowledge bank: export it, import it somewhere else, get the same bank.

The interesting parts are the ones a count would not catch: a document keeps the fields
the source extracted without the importer paying for an LLM call, a record comes back
with the evidence that justified it, and a config-only archive is a template — shapes
with nothing in them.
"""

import uuid

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app

SCHEMA = {
    "name": "Contract",
    "document_fields": {
        "vendor": {"type": "string", "description": "Who the contract is with"},
        "value": {"type": "number", "description": "What it is worth"},
    },
    "passage_fields": {},
}
VENDORS = {
    "name": "Vendors",
    "identity": "name",
    "fields": {"name": {"type": "string"}, "country": {"type": "string"}},
}
DOCUMENTS = [
    {
        "id": "c-1",
        "title": "Acme agreement",
        "text": "This agreement is between the buyer and Acme Ltd, for a total of 12000 EUR per year.",
        "schema_id": "contract",
        "fields": {"vendor": "Acme Ltd", "value": 12000},
        "metadata": {"source": "crm"},
    },
    {
        "id": "c-2",
        "title": "Globex agreement",
        "text": "This agreement is between the buyer and Globex, for a total of 8000 EUR per year.",
        "schema_id": "contract",
        "fields": {"vendor": "Globex", "value": 8000},
    },
]


@pytest_asyncio.fixture
async def kb_client(memory):
    app = create_app(memory, initialize_memory=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield client


async def _bank(client, *, with_data: bool = True) -> str:
    kb = f"kb-{uuid.uuid4().hex[:8]}"
    assert (await client.post("/v1/default/knowledge-banks", json={"id": kb, "name": "Contracts"})).status_code == 201
    assert (await client.put(f"/v1/default/knowledge-banks/{kb}/schemas/contract", json=SCHEMA)).status_code == 200
    assert (await client.put(f"/v1/default/knowledge-banks/{kb}/collections/vendors", json=VENDORS)).status_code == 200
    if with_data:
        written = await client.post(f"/v1/default/knowledge-banks/{kb}/documents", json={"documents": DOCUMENTS})
        assert written.status_code == 202, written.text
        records = await client.post(
            f"/v1/default/knowledge-banks/{kb}/collections/vendors/records",
            json={
                "records": [
                    {
                        "values": {"name": "acme", "country": "de"},
                        "doc_ids": ["c-1"],
                        "evidence": {"country": [{"doc_id": "c-1", "quote": "Acme Ltd"}]},
                    }
                ]
            },
        )
        assert records.status_code == 200, records.text
    return kb


async def _export(client, memory, kb: str, **flags) -> bytes:
    """Export the bank and fetch the archive the operation points at."""
    response = await client.post(f"/v1/default/knowledge-banks/{kb}/transfer/export", params=flags)
    assert response.status_code == 202, response.text
    operation_id = response.json()["operation_id"]
    operation = (await client.get(f"/v1/default/knowledge-banks/{kb}/operations/{operation_id}")).json()
    assert operation["status"] == "completed", operation
    return await memory._file_storage.retrieve(operation["result_metadata"]["storage_key"])


async def _import(client, kb: str, archive: bytes, **params) -> dict:
    response = await client.post(
        f"/v1/default/knowledge-banks/{kb}/transfer/import",
        params=params,
        files={"file": ("knowledge.zip", archive, "application/zip")},
    )
    assert response.status_code == 202, response.text
    body = response.json()
    operation = (await client.get(f"/v1/default/knowledge-banks/{kb}/operations/{body['operation_id']}")).json()
    assert operation["status"] == "completed", operation
    return {**body, "result": operation["result_metadata"]}


@pytest.mark.asyncio
async def test_a_bank_survives_the_round_trip(kb_client, memory):
    """Export a bank, restore it under a new id, and find the same bank there."""
    source = await _bank(kb_client)
    archive = await _export(kb_client, memory, source)
    target = f"{source}-copy"
    imported = await _import(kb_client, source, archive, target_bank_id=target)
    assert imported["result"]["documents"] == 2

    bank = (await kb_client.get(f"/v1/default/knowledge-banks/{target}")).json()
    assert bank["documents"] == 2
    assert bank["passages"] > 0, "the importer re-passaged and re-embedded"

    schemas = (await kb_client.get(f"/v1/default/knowledge-banks/{target}/schemas")).json()
    assert [s["schema_id"] for s in schemas["items"]] == ["contract"]
    assert schemas["items"][0]["document_fields"]["value"]["type"] == "number"

    document = (await kb_client.get(f"/v1/default/knowledge-banks/{target}/documents/c-1")).json()
    # The fields came over with the document: nothing was extracted again, which is what
    # makes an import free. MockLLM would have answered, so this is about cost, not truth.
    assert document["fields"] == {"vendor": "Acme Ltd", "value": 12000}
    assert document["metadata"] == {"source": "crm"}
    assert document["schema_id"] == "contract"

    record = (await kb_client.get(f"/v1/default/knowledge-banks/{target}/collections/vendors/records/acme")).json()
    assert record["values"] == {"name": "acme", "country": "de"}
    # Evidence is what makes a derived value auditable, so it has to survive the move.
    assert record["evidence"]["country"][0]["doc_id"] == "c-1"

    # And the copy is searchable, which is the point of re-embedding rather than
    # carrying vectors that belong to another model.
    found = await kb_client.post(
        f"/v1/default/knowledge-banks/{target}/search", json={"query": "agreement with Globex", "top_k": 5}
    )
    assert found.status_code == 200, found.text
    assert any(hit["document_id"] == "c-2" for hit in found.json()["results"])


@pytest.mark.asyncio
async def test_a_config_only_export_is_a_template(kb_client, memory):
    """Shapes without content: the schemas and collections, and not one document."""
    source = await _bank(kb_client)
    template = await _export(kb_client, memory, source, include_data=False)

    target = f"{source}-from-template"
    await _import(kb_client, source, template, target_bank_id=target)

    bank = (await kb_client.get(f"/v1/default/knowledge-banks/{target}")).json()
    assert bank["documents"] == 0, "a template carries no documents"
    assert [s["schema_id"] for s in (await kb_client.get(f"/v1/default/knowledge-banks/{target}/schemas")).json()["items"]] == ["contract"]  # fmt: skip
    collections = (await kb_client.get(f"/v1/default/knowledge-banks/{target}/collections")).json()
    assert [c["collection_id"] for c in collections["items"]] == ["vendors"]
    assert (await kb_client.get(f"/v1/default/knowledge-banks/{target}/collections/vendors/records")).json()[
        "total"
    ] == 0


@pytest.mark.asyncio
async def test_a_template_can_be_applied_to_a_bank_that_already_exists(kb_client, memory):
    """mode=merge is how a template is used on a bank someone is already filling."""
    source = await _bank(kb_client)
    template = await _export(kb_client, memory, source, include_data=False)

    target = await _bank(kb_client, with_data=False)
    await kb_client.delete(f"/v1/default/knowledge-banks/{target}/schemas/contract")
    await kb_client.delete(f"/v1/default/knowledge-banks/{target}/collections/vendors")

    refused = await kb_client.post(
        f"/v1/default/knowledge-banks/{source}/transfer/import",
        params={"target_bank_id": target},
        files={"file": ("knowledge.zip", template, "application/zip")},
    )
    assert refused.status_code == 409, "a restore writes into a fresh bank"

    await _import(kb_client, source, template, target_bank_id=target, mode="merge")
    schemas = (await kb_client.get(f"/v1/default/knowledge-banks/{target}/schemas")).json()
    assert [s["schema_id"] for s in schemas["items"]] == ["contract"]


@pytest.mark.asyncio
async def test_an_archive_from_a_memory_bank_is_refused(kb_client):
    """A zip is a zip. The manifest says which kind of bank it came from."""
    import io
    import json
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"schema_version": 1, "source_bank_id": "m", "archive_type": "bank"}))
    kb = await _bank(kb_client, with_data=False)
    response = await kb_client.post(
        f"/v1/default/knowledge-banks/{kb}/transfer/import",
        files={"file": ("transfer.zip", buffer.getvalue(), "application/zip")},
    )
    assert response.status_code == 400
    assert "not a knowledge bank" in response.json()["detail"]

    not_a_zip = await kb_client.post(
        f"/v1/default/knowledge-banks/{kb}/transfer/import",
        files={"file": ("transfer.zip", b"nope", "application/zip")},
    )
    assert not_a_zip.status_code == 400
