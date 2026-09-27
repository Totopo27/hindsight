"""Identity resolution: when two documents name the same thing differently.

This is the hardest part of a derived dataset and the one that fails quietly. "Apple" and
"Apple Inc." are one vendor; "Apple" and "Apple Bank" are two, and a system that gets the
first right by being generous gets the second wrong. So the cases come in pairs: what must
merge, and what must not.

The second half is the escape hatch. Resolution will sometimes be wrong, so a person has
to be able to say "these two are the same" — and that decision has to *stick*, or the next
document re-creates the duplicate it just took someone a minute to fix.
"""

import uuid

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app
from hindsight_api.knowledge.identity import edit_distance, is_typo_of, normalise

VENDORS = {
    "name": "Vendors",
    "identity": "name",
    "fields": {"name": {"type": "string"}, "country": {"type": "string"}, "tier": {"type": "string"}},
}


@pytest_asyncio.fixture
async def kb_client(memory):
    app = create_app(memory, initialize_memory=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture
async def bank(kb_client) -> str:
    kb = f"kb-{uuid.uuid4().hex[:8]}"
    assert (await kb_client.post("/v1/default/knowledge-banks", json={"id": kb})).status_code == 201
    response = await kb_client.put(f"/v1/default/knowledge-banks/{kb}/collections/vendors", json=VENDORS)
    assert response.status_code == 200, response.text
    return kb


async def _write_records(client, kb: str, rows: list[dict], collection: str = "vendors") -> None:
    response = await client.post(
        f"/v1/default/knowledge-banks/{kb}/collections/{collection}/records", json={"records": rows}
    )
    assert response.status_code == 200, response.text


async def _records(client, kb: str, collection: str = "vendors") -> list[dict]:
    response = await client.post(
        f"/v1/default/knowledge-banks/{kb}/collections/{collection}/query",
        json={"select": ["record_id", {"field": "name", "as": "name"}, {"field": "country", "as": "country"}]},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    return [dict(zip(payload["columns"], row, strict=True)) for row in payload["rows"]]


async def _record(client, kb: str, record_id: str, collection: str = "vendors") -> dict | None:
    response = await client.get(f"/v1/default/knowledge-banks/{kb}/collections/{collection}/records/{record_id}")
    return response.json() if response.status_code == 200 else None


# ---- what must resolve to one record


@pytest.mark.parametrize(
    "variants",
    [
        pytest.param(["Apple", "Apple Inc.", "Apple, Inc.", "APPLE INC"], id="legal-suffix-inc"),
        pytest.param(["Acme Ltd", "Acme Limited", "ACME LTD."], id="ltd-spelled-out"),
        pytest.param(["Globex GmbH", "Globex  gmbh", "GLOBEX GMBH"], id="gmbh"),
        pytest.param(["Initech LLC", "Initech, L.L.C.", "initech llc"], id="llc-dotted"),
        pytest.param(["Umbrella Corp", "Umbrella Corporation", "Umbrella Corp."], id="corp-spelled-out"),
        pytest.param(["Nestlé", "Nestle", "NESTLE"], id="accents"),
        pytest.param(["Vandelay Industries", "Vandelay Industries,", " Vandelay  Industries "], id="punctuation"),
        pytest.param(["Soylent Co", "Soylent Company", "Soylent co."], id="co-spelled-out"),
        pytest.param(["Stark S.p.A.", "Stark SpA", "STARK S.P.A"], id="spa-italian"),
        pytest.param(["Wayne plc", "Wayne PLC", "Wayne Plc."], id="plc"),
    ],
)
@pytest.mark.asyncio
async def test_the_same_company_written_differently_is_one_record(kb_client, bank, variants):
    for index, name in enumerate(variants):
        await _write_records(kb_client, bank, [{"values": {"name": name, "country": f"c{index}"}}])
    records = await _records(kb_client, bank)
    assert len(records) == 1, f"{variants} produced {[r['name'] for r in records]}"


@pytest.mark.asyncio
async def test_a_typo_is_still_the_same_vendor(kb_client, bank):
    """One transposed letter in a long name is a typo, not a second company."""
    await _write_records(kb_client, bank, [{"values": {"name": "Vandelay Industries"}}])
    await _write_records(kb_client, bank, [{"values": {"name": "Vandaley Industries"}}])
    assert len(await _records(kb_client, bank)) == 1


# ---- what must stay apart


@pytest.mark.parametrize(
    "left,right",
    [
        pytest.param("Apple", "Apple Bank", id="one-is-a-qualifier"),
        pytest.param("Delta Air Lines", "Delta Faucet", id="shared-first-word"),
        pytest.param("Allianz", "Allianz Technology", id="parent-and-subsidiary"),
        pytest.param("Acme Ltd", "Acme Holdings Ltd", id="same-suffix-different-body"),
        pytest.param("Globex", "Globus", id="close-but-different"),
        pytest.param("BP", "HP", id="short-names-one-letter-apart"),
    ],
)
@pytest.mark.asyncio
async def test_different_companies_stay_different(kb_client, bank, left, right):
    await _write_records(kb_client, bank, [{"values": {"name": left}}, {"values": {"name": right}}])
    records = await _records(kb_client, bank)
    assert len(records) == 2, f"{left!r} and {right!r} were merged into {[r['name'] for r in records]}"


# ---- resolution keeps the data whole


@pytest.mark.asyncio
async def test_resolved_variants_share_one_record_and_keep_every_source(kb_client, bank):
    await _write_records(
        kb_client,
        bank,
        [
            {"values": {"name": "Apple Inc.", "country": "us"}, "doc_ids": ["d1"]},
            {"values": {"name": "Apple", "tier": "gold"}, "doc_ids": ["d2"]},
        ],
    )
    records = await _records(kb_client, bank)
    assert len(records) == 1
    record = await _record(kb_client, bank, records[0]["record_id"])
    assert record["values"]["country"] == "us" and record["values"]["tier"] == "gold"
    assert sorted(record["doc_ids"]) == ["d1", "d2"]


@pytest.mark.asyncio
async def test_the_record_keeps_a_readable_name_not_a_normalised_key(kb_client, bank):
    """Normalisation decides *which* record; it must not become the value people read."""
    await _write_records(kb_client, bank, [{"values": {"name": "Apple Inc."}}])
    records = await _records(kb_client, bank)
    assert records[0]["name"] == "Apple Inc.", "the stored value is what the document said"


# ---- merging by hand, when resolution got it wrong


@pytest.mark.asyncio
async def test_a_person_can_merge_two_records_that_should_have_been_one(kb_client, bank):
    await _write_records(
        kb_client,
        bank,
        [
            {"values": {"name": "Big Blue", "country": "us"}, "doc_ids": ["d1"]},
            {"values": {"name": "IBM", "tier": "gold"}, "doc_ids": ["d2"]},
        ],
    )
    assert len(await _records(kb_client, bank)) == 2, "nothing could have told us these are the same"

    merged = await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/records/big blue/merge",
        json={"into": "ibm"},
    )
    assert merged.status_code == 200, merged.text

    records = await _records(kb_client, bank)
    assert len(records) == 1 and records[0]["record_id"] == "ibm"
    record = await _record(kb_client, bank, "ibm")
    assert record["values"]["country"] == "us" and record["values"]["tier"] == "gold", "both halves survive"
    assert sorted(record["doc_ids"]) == ["d1", "d2"]
    # The retired id is not an error: a link someone saved before the merge still lands
    # on the record that now holds the data.
    stale = await _record(kb_client, bank, "big blue")
    assert stale is not None and stale["record_id"] == "ibm"


@pytest.mark.asyncio
async def test_a_merge_sticks_when_the_documents_are_read_again(kb_client, bank):
    """The point of merging: the next document must not re-create the duplicate."""
    await _write_records(
        kb_client,
        bank,
        [{"values": {"name": "Big Blue"}, "doc_ids": ["d1"]}, {"values": {"name": "IBM"}, "doc_ids": ["d2"]}],
    )
    await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/records/big blue/merge", json={"into": "ibm"}
    )
    # A later document calls it Big Blue again.
    await _write_records(kb_client, bank, [{"values": {"name": "Big Blue", "tier": "silver"}, "doc_ids": ["d3"]}])

    records = await _records(kb_client, bank)
    assert len(records) == 1 and records[0]["record_id"] == "ibm", "the alias holds"
    record = await _record(kb_client, bank, "ibm")
    assert record["values"]["tier"] == "silver"
    assert sorted(record["doc_ids"]) == ["d1", "d2", "d3"]


@pytest.mark.asyncio
async def test_merging_repoints_what_referred_to_the_old_record(kb_client, bank):
    await kb_client.put(
        f"/v1/default/knowledge-banks/{bank}/collections/contracts",
        json={
            "identity": "reference",
            "fields": {"reference": {"type": "string"}, "vendor": {"collection": "vendors"}},
        },
    )
    await _write_records(
        kb_client, bank, [{"values": {"name": "Big Blue"}}, {"values": {"name": "IBM", "country": "us"}}]
    )
    await _write_records(
        kb_client,
        bank,
        [{"values": {"reference": "c-1", "vendor": "big blue"}}],
        collection="contracts",
    )
    await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/records/big blue/merge", json={"into": "ibm"}
    )

    joined = await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/contracts/query",
        json={
            "join": [{"on": "vendor", "as": "v"}],
            "select": [{"field": "reference", "as": "reference"}, {"field": "v.country", "as": "country"}],
        },
    )
    assert joined.json()["rows"] == [["c-1", "us"]], "the contract follows the surviving vendor"


@pytest.mark.asyncio
async def test_a_merge_keeps_both_records_evidence(kb_client, bank):
    await _write_records(
        kb_client,
        bank,
        [
            {
                "values": {"name": "Big Blue", "country": "us"},
                "evidence": {"country": "headquartered in Armonk"},
                "doc_ids": ["d1"],
            },
            {"values": {"name": "IBM", "tier": "gold"}, "evidence": {"tier": "a strategic partner"}, "doc_ids": ["d2"]},
        ],
    )
    await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/records/big blue/merge", json={"into": "ibm"}
    )
    record = await _record(kb_client, bank, "ibm")
    assert record["evidence"]["country"][0]["quote"] == "headquartered in Armonk"
    assert record["evidence"]["tier"][0]["quote"] == "a strategic partner"


@pytest.mark.asyncio
async def test_a_pin_on_either_side_survives_a_merge(kb_client, bank):
    await _write_records(
        kb_client, bank, [{"values": {"name": "Big Blue", "country": "de"}}, {"values": {"name": "IBM"}}]
    )
    await kb_client.put(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/records/big blue/pins",
        json={"values": {"country": "us"}},
    )
    await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/records/big blue/merge", json={"into": "ibm"}
    )
    record = await _record(kb_client, bank, "ibm")
    assert record["values"]["country"] == "us", "a correction is not lost by merging the record it was made on"
    assert record["pinned"]["country"] == "us"


@pytest.mark.asyncio
async def test_merging_refuses_what_it_cannot_do(kb_client, bank):
    await _write_records(kb_client, bank, [{"values": {"name": "IBM"}}])
    missing = await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/records/nope/merge", json={"into": "ibm"}
    )
    assert missing.status_code == 404
    into_missing = await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/records/ibm/merge", json={"into": "nope"}
    )
    assert into_missing.status_code == 404
    itself = await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/records/ibm/merge", json={"into": "ibm"}
    )
    assert itself.status_code == 400


@pytest.mark.asyncio
async def test_resolution_is_per_bank_and_per_collection(kb_client, bank):
    """An alias someone set in one bank must not decide another bank's records."""
    other = f"kb-{uuid.uuid4().hex[:8]}"
    await kb_client.post("/v1/default/knowledge-banks", json={"id": other})
    await kb_client.put(f"/v1/default/knowledge-banks/{other}/collections/vendors", json=VENDORS)

    for kb in (bank, other):
        await _write_records(kb_client, kb, [{"values": {"name": "Big Blue"}}, {"values": {"name": "IBM"}}])
    await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/records/big blue/merge", json={"into": "ibm"}
    )
    assert len(await _records(kb_client, bank)) == 1
    assert len(await _records(kb_client, other)) == 2, "the other bank is untouched"


# ---- the adversarial half


@pytest.mark.asyncio
async def test_a_chain_of_merges_stays_one_hop_deep(kb_client, bank):
    await _write_records(
        kb_client,
        bank,
        [
            {"values": {"name": "Big Blue"}, "doc_ids": ["d1"]},
            {"values": {"name": "IBM"}, "doc_ids": ["d2"]},
            {"values": {"name": "International Business Machines"}, "doc_ids": ["d3"]},
        ],
    )
    await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/records/big blue/merge", json={"into": "ibm"}
    )
    await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/records/ibm/merge",
        json={"into": "international business machines"},
    )
    records = await _records(kb_client, bank)
    assert len(records) == 1 and records[0]["record_id"] == "international business machines"

    # Both retired spellings resolve to the survivor, not to a record that no longer exists.
    await _write_records(kb_client, bank, [{"values": {"name": "Big Blue", "tier": "gold"}, "doc_ids": ["d4"]}])
    await _write_records(kb_client, bank, [{"values": {"name": "IBM", "country": "us"}, "doc_ids": ["d5"]}])
    records = await _records(kb_client, bank)
    assert len(records) == 1
    record = await _record(kb_client, bank, "international business machines")
    assert record["values"]["tier"] == "gold" and record["values"]["country"] == "us"
    assert sorted(record["doc_ids"]) == ["d1", "d2", "d3", "d4", "d5"]


@pytest.mark.asyncio
async def test_merging_into_a_record_that_was_itself_merged_away_follows_the_chain(kb_client, bank):
    await _write_records(
        kb_client,
        bank,
        [{"values": {"name": "Alpha"}}, {"values": {"name": "Beta"}}, {"values": {"name": "Gamma"}}],
    )
    await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/records/beta/merge", json={"into": "gamma"}
    )
    # "beta" is gone; merging into it must land on gamma rather than failing or resurrecting it.
    merged = await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/records/alpha/merge", json={"into": "beta"}
    )
    assert merged.status_code == 200, merged.text
    records = await _records(kb_client, bank)
    assert [r["record_id"] for r in records] == ["gamma"]


@pytest.mark.asyncio
async def test_resolution_applies_to_records_the_llm_derives_too(kb_client, memory, bank):
    """The write path and the derivation path must group names the same way."""
    answers = {
        "d1": [{"values": {"name": "Apple Inc."}, "evidence": {}}],
        "d2": [{"values": {"name": "apple"}, "evidence": {}}],
        "d3": [{"values": {"name": "APPLE, INC."}, "evidence": {}}],
    }

    def answer(messages, scope):
        if scope != "knowledge_records":
            return {}
        content = messages[-1]["content"]
        for marker, records in answers.items():
            if marker in content:
                return {"records": records}
        return {"records": []}

    memory._llm_config._provider_impl.set_response_callback(answer)
    await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/documents",
        json={"documents": [{"id": doc_id, "text": f"{doc_id}: a document about the vendor."} for doc_id in answers]},
    )
    derive = await kb_client.post(f"/v1/default/knowledge-banks/{bank}/collections/vendors/derive", json={})
    assert derive.status_code == 202, derive.text

    records = await _records(kb_client, bank)
    assert len(records) == 1, f"derivation produced {[r['name'] for r in records]}"
    assert sorted((await _record(kb_client, bank, records[0]["record_id"]))["doc_ids"]) == ["d1", "d2", "d3"]


@pytest.mark.asyncio
async def test_a_transliteration_is_left_for_a_person_to_decide(kb_client, bank):
    """ "Müller" and "Mueller" are probably one company — but nothing here can know that."""
    await _write_records(kb_client, bank, [{"values": {"name": "Müller"}}, {"values": {"name": "Mueller"}}])
    assert len(await _records(kb_client, bank)) == 2, "guessing this would be guessing"

    await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/records/mueller/merge", json={"into": "muller"}
    )
    assert len(await _records(kb_client, bank)) == 1
    # And the decision holds for the next document, which is the whole point.
    await _write_records(kb_client, bank, [{"values": {"name": "Mueller", "country": "de"}}])
    records = await _records(kb_client, bank)
    assert len(records) == 1 and records[0]["country"] == "de"


@pytest.mark.asyncio
async def test_a_company_whose_whole_name_is_a_legal_form_keeps_it(kb_client, bank):
    """Stripping suffixes must never strip a name to nothing."""
    await _write_records(kb_client, bank, [{"values": {"name": "Co"}}, {"values": {"name": "Ltd"}}])
    records = await _records(kb_client, bank)
    assert len(records) == 2 and {r["record_id"] for r in records} == {"co", "ltd"}


@pytest.mark.asyncio
async def test_a_name_that_normalises_to_nothing_does_not_collect_every_record(kb_client, bank):
    await _write_records(
        kb_client,
        bank,
        [{"values": {"name": "???", "country": "de"}}, {"values": {"name": "---", "country": "fr"}}],
    )
    records = await _records(kb_client, bank)
    assert len(records) == 2, "two nameless things are not one thing"


@pytest.mark.asyncio
async def test_a_merge_survives_re_derivation_of_the_documents(kb_client, memory, bank):
    llm_records = {"d1": [{"values": {"name": "Big Blue", "country": "us"}, "evidence": {}}]}

    def answer(messages, scope):
        if scope != "knowledge_records":
            return {}
        content = messages[-1]["content"]
        for marker, records in llm_records.items():
            if marker in content:
                return {"records": records}
        return {"records": []}

    memory._llm_config._provider_impl.set_response_callback(answer)
    await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/documents",
        json={"documents": [{"id": "d1", "text": "d1: about Big Blue."}]},
    )
    await kb_client.post(f"/v1/default/knowledge-banks/{bank}/collections/vendors/derive", json={})
    await _write_records(kb_client, bank, [{"values": {"name": "IBM", "tier": "gold"}, "doc_ids": ["d2"]}])
    await kb_client.post(
        f"/v1/default/knowledge-banks/{bank}/collections/vendors/records/big blue/merge", json={"into": "ibm"}
    )
    assert len(await _records(kb_client, bank)) == 1

    # Re-derive: the document still says "Big Blue", and the merge must still hold.
    await kb_client.post(f"/v1/default/knowledge-banks/{bank}/collections/vendors/derive", json={"replace": True})
    records = await _records(kb_client, bank)
    assert len(records) == 1 and records[0]["record_id"] == "ibm"
    record = await _record(kb_client, bank, "ibm")
    assert record["values"]["country"] == "us" and record["values"]["tier"] == "gold"


@pytest.mark.asyncio
async def test_turning_typo_matching_off_leaves_only_exact_keys_and_merges(kb_client, bank):
    patched = await kb_client.patch(
        f"/v1/default/banks/{bank}/config", json={"updates": {"kb_record_identity_similarity": 0}}
    )
    assert patched.status_code == 200, patched.text
    await _write_records(
        kb_client, bank, [{"values": {"name": "Vandelay Industries"}}, {"values": {"name": "Vandaley Industries"}}]
    )
    assert len(await _records(kb_client, bank)) == 2, "with matching off, only exact keys group"
    # Normalisation still applies: it is exact, not a guess.
    await _write_records(kb_client, bank, [{"values": {"name": "Vandelay Industries, Inc."}}])
    assert len(await _records(kb_client, bank)) == 2


# ---- the normalisation itself, without a database in the way


@pytest.mark.parametrize(
    "written,key",
    [
        ("Apple Inc.", "apple"),
        ("APPLE, INC.", "apple"),
        ("Acme Limited", "acme"),
        ("Globex  GmbH", "globex"),
        ("Initech, L.L.C.", "initech"),
        ("Stark S.p.A.", "stark"),
        ("Nestlé", "nestle"),
        ("Acme Holdings Ltd Co", "acme holdings"),
        ("  Spaced   Out  ", "spaced out"),
        ("Co", "co"),
        ("", ""),
        ("???", ""),
    ],
)
def test_normalisation_is_a_tidy_key_not_a_guess(written, key):
    assert normalise(written) == key


def test_normalisation_keeps_what_distinguishes_two_companies():
    assert normalise("Apple") != normalise("Apple Bank")
    assert normalise("Allianz") != normalise("Allianz Technology")
    assert normalise("Delta Air Lines") != normalise("Delta Faucet")


def test_the_typo_rule_is_about_small_changes_to_long_names():
    assert is_typo_of("vandelay industries", "vandaley industries")
    assert is_typo_of("international business machines", "internatonal business machines")
    # Short names: one character is a different company, not a slip.
    assert not is_typo_of("bp", "hp")
    assert not is_typo_of("globex", "globus")
    # Long names that differ by a word, not a letter.
    assert not is_typo_of("johnson controls", "johnson matthey")
    assert not is_typo_of("acme industries", "acme holdings")
    assert not is_typo_of("apple", "apple bank")


def test_edit_distance_stops_counting_once_the_answer_is_no():
    assert edit_distance("kitten", "sitting", cap=5) == 3
    assert edit_distance("kitten", "sitting", cap=2) == 3, "the cap only bounds the work, not the verdict"
    assert edit_distance("abc", "xyz", cap=1) > 1
    assert edit_distance("same", "same", cap=0) == 0


@pytest.mark.parametrize(
    "left,right",
    [
        pytest.param("The Coca-Cola Company", "Coca-Cola", id="leading-article"),
        pytest.param("The Walt Disney Company", "Walt Disney", id="leading-article-suffix"),
        pytest.param("Procter & Gamble", "Procter and Gamble", id="ampersand-as-word"),
        pytest.param("AT&T", "AT and T", id="ampersand-no-spaces"),
        pytest.param("Marks & Spencer plc", "Marks and Spencer", id="ampersand-and-suffix"),
    ],
)
def test_the_ways_a_company_writes_its_own_name(left, right):
    assert normalise(left) == normalise(right), f"{normalise(left)!r} != {normalise(right)!r}"


def test_an_article_or_conjunction_that_is_the_name_is_kept():
    assert normalise("The") == "the"
    assert normalise("Bread & Butter") == normalise("Bread and Butter") == "bread and butter"
    # "The" inside a name is part of it, not an article to strip.
    assert normalise("Save The Children") == "save the children"
    assert normalise("Save Children") != normalise("Save The Children")
