"""A table derived from a folder, graded on what the folder does to it.

Deriving records from documents is the claim every document-AI product makes, and
the demo is always the easy half: one schema, clean files, extract once. This
suite is the other half, on one story — a procurement folder — and it grades the
four things that decide whether the table is usable a month later:

* identity — the same vendor written three ways is one row;
* supersession — an amendment beats the agreement it amends, and the evidence
  says so;
* absence — a field nobody wrote stays empty;
* change — a corrected document moves its row, a deleted one takes its
  contribution with it.

The corpus and gold live in ``hindsight_system_evals/collections.py``. Most of the
grading is a direct assert, because "one record" and "185000" are not matters of
opinion; the judge is used for the one question that is — whether the document a
value cites actually says it.

What the first runs found, all of it open at the time of writing — a red
``--full`` run here is the product, not the suite:

1. **Supersession is decided by filename.** Contributions are folded
   ``ORDER BY updated_at, doc_id`` and the last writer wins, so Amendment No. 2
   (185,000) loses to the agreement it amends (120,000) because
   ``amendment-acme-2`` sorts before ``msa-acme``. Nothing reads the dates the
   documents state. Across three runs C-1041 came back 185,000, then 120,000,
   then empty — the value is whichever document the model happened to fill, and
   when both are filled, whichever filename sorts later.
2. **Rows with no identity.** A document that yields half a record — ``{"value":
   120000}`` with no reference — becomes a row of its own under a random hex id.
   One run produced 14 contract rows for 3 contracts, and the buyer,
   Northbridge Manufacturing, was filed as one of its own vendors.
3. **Relationships do not resolve.** ``contracts.vendor`` is stored as
   ``str(value).strip().lower()`` while a vendor record's id comes from
   ``identity.normalise``, which also strips legal suffixes: the contract points
   at ``"acme ltd"`` and the record is ``"acme"``, so every join returns null.
4. **No citations.** ``evidence`` is an optional field on the derivation schema
   and the model omits it, so every derived value arrives with nothing to trace
   it to — the one thing every competitor in this space leads with.

Cost shapes the layout. Derivation is an LLM call per document per collection, so
the folder is built **once** and every read-only case is graded against that one
table; the two cases that mutate the bank build their own small one rather than
running in an order the others depend on.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid

import httpx
import pytest

from hindsight_system_evals import evaluate
from hindsight_system_evals.collections import (
    CONTRACT_GOLD,
    CONTRACTS,
    CORRECTED_DOCUMENT,
    DELETED_DOCUMENT_ID,
    MINIMUM_ACCEPTANCE,
    VENDOR_GOLD,
    VENDORS,
    Document,
    cases,
    correction_documents,
    documents,
)
from hindsight_system_evals.knowledge_banks import KnowledgeBank
from hindsight_system_evals.report import RECORDED, EvalRecord

log = logging.getLogger(__name__)

_CASES = {case.id: case for case in cases()}

#: Built once for the whole module: bank id plus the two tables as derived. Every
#: read-only case reads this. Guarded because pytest-asyncio gives each test its
#: own loop, so the *data* is shared and the HTTP client never is.
_FOLDER: dict[str, object] = {}
_BUILD_LOCK = asyncio.Lock()

#: Every bank this suite makes, so a remote target does not collect one per run.
_BANKS: list[str] = []


@pytest.fixture(scope="session")
def bank_prefix(request: pytest.FixtureRequest, target) -> str:
    """One prefix per run. The folder is built once and read by every case, so the
    bank cannot be per-test the way the answer suites' is."""
    prefix = f"syseval-col-{uuid.uuid4().hex[:8]}-"
    yield prefix
    if target.is_remote and not request.config.getoption("--keep-banks"):
        for bank_id in _BANKS:
            with contextlib.suppress(Exception):
                httpx.delete(f"{target.url}/v1/default/knowledge-banks/{bank_id}", timeout=30.0)


def _payload(document: Document) -> dict[str, str]:
    return {"id": document.doc_id, "title": document.title, "text": document.text}


async def _seed(bank: KnowledgeBank, corpus: list[Document]) -> None:
    """Create the bank, define both collections, write the folder, derive both tables."""
    await bank.create(name="Procurement folder")
    # Vendors first: a contract's `vendor` field points at a vendor record, and a
    # relationship cannot be defined against a collection that does not exist.
    await bank.put_collection("vendors", VENDORS)
    await bank.put_collection("contracts", CONTRACTS)
    await bank.write([_payload(document) for document in corpus])
    await bank.derive("vendors")
    await bank.derive("contracts")


async def _folder(target, bank_prefix: str) -> dict[str, object]:
    async with _BUILD_LOCK:
        if _FOLDER:
            return _FOLDER
        bank = KnowledgeBank(target.url, f"{bank_prefix}folder", target.api_key)
        _BANKS.append(bank.bank_id)
        try:
            await _seed(bank, documents())
            _FOLDER.update(
                {
                    "bank_id": bank.bank_id,
                    "vendors": await bank.records("vendors"),
                    "contracts": await bank.records("contracts"),
                }
            )
        finally:
            await bank.aclose()
    return _FOLDER


def _by_id(records: list[dict]) -> dict[str, dict]:
    return {record["record_id"]: record for record in records}


def _record(kind: str, case_id: str, bank_id: str, correct: bool, hit_trap: bool, reason: str) -> None:
    case = _CASES[case_id]
    RECORDED.append(
        EvalRecord(
            kind="collections",
            question_id=case_id,
            category=case.category,
            correct=correct,
            hit_trap=hit_trap,
            bank_id=bank_id,
            reason=reason,
        )
    )
    log.info("%s (%s): correct=%s trap=%s — %s", case_id, kind, correct, hit_trap, reason)


class _Graded:
    """Collects a case's verdict so the trap is reported before correctness fails it."""

    def __init__(self, case_id: str, bank_id: str) -> None:
        self.case_id, self.bank_id = case_id, bank_id
        self.problems: list[str] = []
        self.trap = ""

    def fails(self, message: str) -> None:
        self.problems.append(message)

    def trapped(self, message: str) -> None:
        self.trap = message
        self.problems.append(message)

    def finish(self) -> None:
        reason = self.trap or ("; ".join(self.problems) if self.problems else "as expected")
        _record(self.case_id, self.case_id, self.bank_id, not self.problems, bool(self.trap), reason)
        case = _CASES[self.case_id]
        assert not self.trap, f"{self.case_id}: {case.must_not} — {self.trap}\nbank {self.bank_id}"
        assert not self.problems, f"{self.case_id}: {'; '.join(self.problems)}\nbank {self.bank_id}"


# ---- the read-only cases, all against the one derived table


async def _grade(case_id: str, target, bank_prefix: str) -> None:
    folder = await _folder(target, bank_prefix)
    bank_id = str(folder["bank_id"])
    vendors = _by_id(folder["vendors"])  # type: ignore[arg-type]
    contracts = _by_id(folder["contracts"])  # type: ignore[arg-type]
    graded = _Graded(case_id, bank_id)

    if case_id == "col-identity":
        # Every document that names Acme names it differently. One row, or the table
        # is a worse index of the folder than the folder is.
        acme = [rid for rid in vendors if "acme" in rid]
        if len(acme) > 1:
            graded.trapped(f"Acme became {len(acme)} records: {acme}")
        elif not acme:
            graded.fails(f"no Acme record at all; vendors are {sorted(vendors)}")

    elif case_id == "col-fields":
        # The plain extraction score every vendor quotes, on the whole gold: each field
        # the folder states, and each field it does not, counted once.
        for gold, table in ((VENDOR_GOLD, vendors), (CONTRACT_GOLD, contracts)):
            for expectation in gold:
                record = table.get(expectation.record)
                if record is None:
                    graded.fails(f"no record {expectation.record!r}")
                    continue
                for field_name, expected in expectation.values.items():
                    got = record["values"].get(field_name)
                    # A date may come back as a date or as its ISO text, and a number as
                    # int or float; neither difference is a wrong answer.
                    if isinstance(expected, str) and isinstance(got, str):
                        ok = got.startswith(expected)
                    elif isinstance(expected, (int, float)) and isinstance(got, (int, float)):
                        ok = float(got) == float(expected)
                    else:
                        ok = got == expected
                    if not ok:
                        graded.fails(f"{expectation.record}.{field_name} is {got!r}, expected {expected!r}")
                for field_name in expectation.empty:
                    if record["values"].get(field_name) not in (None, "", []):
                        graded.trapped(
                            f"{expectation.record}.{field_name} was filled with "
                            f"{record['values'][field_name]!r} — no document states it"
                        )

    elif case_id == "col-fragments":
        # A row the model produced from half a sentence — no reference, no name — is
        # not a contract; it is the sentence, filed as though it were a thing.
        anonymous = [record["record_id"] for record in folder["contracts"] if not record["values"].get("reference")]
        if anonymous:
            graded.trapped(
                f"{len(anonymous)} contract rows carry no reference: {anonymous[:4]} — "
                f"{len(contracts)} rows for the 3 contracts in the folder"
            )
        # The buyer is the other side of every agreement in the folder, and is not a vendor.
        if any("northbridge" in rid for rid in vendors):
            graded.trapped("Northbridge Manufacturing, the buyer, was filed as one of its own vendors")

    elif case_id == "col-supersession":
        contract = contracts.get("c 1041")
        if contract is None:
            graded.fails(f"no record for C-1041; contracts are {sorted(contracts)}")
        else:
            value = contract["values"].get("value")
            if value == 120000:
                graded.trapped("C-1041 still reads the superseded 120,000")
            elif value != 185000:
                graded.fails(f"C-1041 value is {value!r}, expected 185000")
            cited = [item.get("doc_id") for item in (contract.get("evidence") or {}).get("value", [])]
            if cited and "amendment-acme-2" not in cited:
                graded.fails(f"the amended value cites {cited}, not the amendment")

    elif case_id == "col-absence":
        northwind = vendors.get("northwind packaging")
        if northwind is None:
            graded.fails(f"no Northwind Packaging record; vendors are {sorted(vendors)}")
        else:
            country = northwind["values"].get("country")
            if country not in (None, "", []):
                graded.trapped(f"Northwind Packaging was given a country: {country!r} — no document states one")

    elif case_id == "col-status":
        expired = contracts.get("c 3115", {}).get("values", {}).get("status")
        if expired != "expired":
            graded.trapped(f"C-3115 reads {expired!r}, but a notice says it expired on 8 June 2025")
        for reference in ("c 1041", "c 2207"):
            status = contracts.get(reference, {}).get("values", {}).get("status")
            if status != "active":
                graded.fails(f"{reference} reads {status!r}, expected active")

    elif case_id == "col-join":
        bank = KnowledgeBank(target.url, bank_id, target.api_key)
        try:
            rows = await bank.query(
                "contracts",
                {
                    "join": [{"collection": "vendors", "on": "vendor", "as": "v"}],
                    "select": ["record_id", {"field": "v.country", "as": "country"}],
                },
            )
        finally:
            await bank.aclose()
        unresolved = [row["record_id"] for row in rows if not row.get("country")]
        # Only the three contracts whose vendor has a country in the folder are
        # gradeable here; a contract nobody wrote a vendor country for is absence.
        expected = {"c 1041": "Ireland", "c 2207": "France", "c 3115": "Germany"}
        for reference, country in expected.items():
            got = next((row.get("country") for row in rows if row["record_id"] == reference), None)
            if got != country:
                graded.fails(f"{reference} joined to country {got!r}, expected {country!r}")
        if len(unresolved) == len(rows):
            graded.trapped("no contract resolved to a vendor record at all")

    elif case_id == "col-evidence":
        bank = KnowledgeBank(target.url, bank_id, target.api_key)
        try:
            checks = [
                ("contracts", "c 1041", "value", "the committed annual value of contract C-1041 is EUR 185,000"),
                ("contracts", "c 2207", "value", "the committed annual value of contract C-2207 is EUR 64,000"),
                ("vendors", "acme", "country", "Acme is registered in Ireland"),
            ]
            for collection, record_id, field_name, claim in checks:
                record = (contracts if collection == "contracts" else vendors).get(record_id)
                if record is None:
                    graded.fails(f"{collection}/{record_id} is missing")
                    continue
                cited = [item.get("doc_id") for item in (record.get("evidence") or {}).get(field_name, [])]
                if not cited:
                    graded.fails(f"{collection}/{record_id}.{field_name} cites no document")
                    continue
                text = (await bank.document(cited[0]))["text"]
                verdict = await evaluate(
                    text,
                    f"The document states that {claim}.",
                    context="This is the document cited as the evidence for that value.",
                )
                if not verdict.meets_criteria:
                    graded.trapped(
                        f"{collection}/{record_id}.{field_name} cites {cited[0]!r}, which does not say it "
                        f"— {verdict.reasoning}"
                    )
        finally:
            await bank.aclose()

    elif case_id == "col-aggregate":
        bank = KnowledgeBank(target.url, bank_id, target.api_key)
        try:
            rows = await bank.query(
                "contracts",
                {"select": [{"sum": "value", "as": "total"}], "where": {"status": "active"}},
            )
        finally:
            await bank.aclose()
        total = rows[0]["total"] if rows else None
        if total in (280500, 249000 + 31500):
            graded.trapped(f"the total {total} counts the expired contract")
        elif total != 249000:
            graded.fails(f"committed value in force is {total!r}, expected 249000")

    else:  # pragma: no cover - a case id with no grader is a bug in this file
        raise AssertionError(f"no grader for {case_id}")

    graded.finish()


@pytest.mark.parametrize("case_id", MINIMUM_ACCEPTANCE)
async def test_collections_minimum_acceptance(target, bank_prefix: str, case_id: str) -> None:
    await _grade(case_id, target, bank_prefix)


@pytest.mark.full
@pytest.mark.parametrize(
    "case_id", [case.id for case in cases() if case.id not in MINIMUM_ACCEPTANCE and case.category != "change"]
)
async def test_collections_full(target, bank_prefix: str, case_id: str) -> None:
    await _grade(case_id, target, bank_prefix)


# ---- the two that change the folder, each on a bank of its own


@pytest.mark.full
async def test_a_corrected_document_moves_its_row(target, bank_prefix: str) -> None:
    """The figure was transcribed wrong, the document is rewritten, the row follows.

    A table derived once and never rechecked is a snapshot. Writing the same
    document id again replaces it, and the record has to be re-derived from what
    the document says now — not hold both figures, and not keep the withdrawn one.
    """
    corpus = correction_documents()
    bank = KnowledgeBank(target.url, f"{bank_prefix}correction", target.api_key)
    _BANKS.append(bank.bank_id)
    graded = _Graded("col-correction", bank.bank_id)
    try:
        await _seed(bank, corpus)
        await bank.write([_payload(CORRECTED_DOCUMENT)])
        await bank.derive("contracts", doc_ids=[CORRECTED_DOCUMENT.doc_id], replace=True)
        contract = _by_id(await bank.records("contracts")).get("c 4402")
        if contract is None:
            graded.fails("no record for C-4402 after the correction")
        else:
            value = contract["values"].get("value")
            if value == 52000:
                graded.trapped("C-4402 still reads the withdrawn 52,000")
            elif value != 57500:
                graded.fails(f"C-4402 reads {value!r}, expected 57500")
    finally:
        await bank.aclose()
    graded.finish()


@pytest.mark.full
async def test_deleting_a_document_takes_its_contribution(target, bank_prefix: str) -> None:
    """Delete the invoice and nothing in the table may still rest on it.

    A record is the sum of what each document said about it, so removing a
    document has to remove its part. The failure this catches is a value that
    outlives its only source — the row still says Lyon when the only document
    that said Lyon is gone.
    """
    corpus = [document for document in documents() if document.doc_id in ("msa-globex", DELETED_DOCUMENT_ID)]
    bank = KnowledgeBank(target.url, f"{bank_prefix}deletion", target.api_key)
    _BANKS.append(bank.bank_id)
    graded = _Graded("col-deletion", bank.bank_id)
    try:
        await _seed(bank, corpus)
        before = _by_id(await bank.records("contracts"))
        if "c 2207" not in before:
            graded.fails("C-2207 was never derived, so its deletion proves nothing")
        await bank.delete_document(DELETED_DOCUMENT_ID)
        after = _by_id(await bank.records("contracts"))
        for record in after.values():
            citing = [
                field_name
                for field_name, items in (record.get("evidence") or {}).items()
                if any(item.get("doc_id") == DELETED_DOCUMENT_ID for item in items)
            ]
            if citing:
                graded.trapped(f"{record['record_id']} still cites the deleted document for {citing}")
            if DELETED_DOCUMENT_ID in (record.get("doc_ids") or []):
                graded.fails(f"{record['record_id']} still lists the deleted document among its sources")
    finally:
        await bank.aclose()
    graded.finish()
