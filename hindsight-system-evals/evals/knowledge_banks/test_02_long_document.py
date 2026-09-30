"""A document longer than one prompt, and a table built from all of it.

The procurement folder in ``test_01`` is thirteen paragraphs: every document fits in
one extraction call, which is the shape a demo has and a corpus does not. This suite
is one document of sixteen thousand characters — Act 1 of a play — against a limit
that cuts a prompt at twelve thousand.

That gap is the whole point. Before derivation read a document in slices it sent the
document whole and truncated it, so everything past the cut was never read and nothing
said so: on the full thirty pages the cast simply stopped.

**What can and cannot prove that.** An earlier version of this suite asserted that
Paris — who speaks only in Scene 2, past the cut — is in the table. That proves
nothing, for two reasons the fixture makes plain: the play's cast list names Paris at
character 901, and every slice after the first is prefixed with the document's opening
two thousand characters by design, so the cast list is in front of the model on every
call. No character in this fixture is reachable only from the second slice. Worse, the
probe is a coin flip on the model's own recall: consecutive runs of the identical bank
did and did not return Paris, with both slices answered.

So the run reports the fact directly instead — ``slices_read`` — and the cases here
grade arithmetic over what came back:

* **the document was read whole**, i.e. as many slices as its length demands;
* **the cast list places people the scenes do not**, so a character's house has to
  reach a slice that never states it;
* **two collections, one pass**, so ``slices_read`` says whether a slice was read once
  or once per collection.

Gold comes from the play's cast list, which the document contains: Romeo, Benvolio and
Abram are Montagues, Tybalt, Sampson and Gregory are Capulets. It is not a judgement
call and nothing here needs a judge.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import pathlib
import uuid

import httpx
import pytest

from hindsight_system_evals.knowledge_banks import KnowledgeBank
from hindsight_system_evals.report import RECORDED, EvalRecord

log = logging.getLogger(__name__)

PLAY = pathlib.Path(__file__).resolve().parents[2] / "fixtures" / "romeo-and-juliet-act-1.md"

HOUSES = {
    "name": "Houses",
    "description": "A noble household of Verona",
    "identity": "name",
    "fields": {"name": {"type": "string", "description": "The house's name"}},
}
CHARACTERS = {
    "name": "Characters",
    "description": "A person who appears or speaks in the play",
    "identity": "name",
    "fields": {
        "name": {"type": "string", "description": "The character's name, as the cast list writes it"},
        "house": {"collection": "houses", "description": "The household they belong to or serve"},
    },
}

#: ``kb_field_extraction_max_chars``: where one extraction prompt is cut, and so how
#: many slices a document of a given length is read in.
EXTRACTION_LIMIT = 12_000

#: The cast list places these, in the document's own words.
MONTAGUES = ("romeo", "benvolio", "abram")
CAPULETS = ("tybalt", "sampson", "gregory")

_BUILT: dict[str, object] = {}
_LOCK = asyncio.Lock()
_BANKS: list[str] = []


@pytest.fixture(scope="session")
def play_bank_prefix(request: pytest.FixtureRequest, target) -> str:
    prefix = f"syseval-play-{uuid.uuid4().hex[:8]}-"
    yield prefix
    if target.is_remote and not request.config.getoption("--keep-banks"):
        for bank_id in _BANKS:
            with contextlib.suppress(Exception):
                httpx.delete(f"{target.url}/v1/default/knowledge-banks/{bank_id}", timeout=30.0)


async def _built(target, prefix: str) -> dict[str, object]:
    """Write the play once, derive once, and let every case read the result."""
    async with _LOCK:
        if _BUILT:
            return _BUILT
        bank = KnowledgeBank(target.url, f"{prefix}play", target.api_key)
        _BANKS.append(bank.bank_id)
        try:
            await bank.create(name="Romeo and Juliet, Act 1")
            await bank.put_collection("houses", HOUSES)
            await bank.put_collection("characters", CHARACTERS)
            # The two houses are known before the play is read, the way a dimension
            # table is: what the derivation has to do is place people in them.
            await bank._json(
                "POST",
                bank._path("/collections/houses/records"),
                json={"records": [{"values": {"name": "Montague"}}, {"values": {"name": "Capulet"}}]},
            )
            text = PLAY.read_text()
            await bank.write([{"id": "act-1", "title": "Romeo and Juliet, Act 1", "text": text}])
            submitted = await bank._json(
                "POST", bank._path("/collections/derive"), json={"doc_ids": [], "replace": True}
            )
            operation = await bank.await_operation(submitted["operation_id"])
            _BUILT.update(
                {
                    "bank_id": bank.bank_id,
                    "characters": {r["record_id"]: r for r in await bank.records("characters")},
                    "counts": operation.get("result_metadata") or {},
                    "length": len(text),
                }
            )
        finally:
            await bank.aclose()
    return _BUILT


def _record(case_id: str, category: str, bank_id: str, problems: list[str], trap: str) -> None:
    RECORDED.append(
        EvalRecord(
            kind="collections",
            question_id=case_id,
            category=category,
            correct=not problems,
            hit_trap=bool(trap),
            bank_id=bank_id,
            reason=trap or ("; ".join(problems) if problems else "as expected"),
        )
    )


def _slices_for(length: int) -> int:
    return -(-length // EXTRACTION_LIMIT)


async def test_the_whole_document_is_read_not_just_the_first_prompt(target, play_bank_prefix: str) -> None:
    """A 16,000-character document against a 12,000-character limit is two slices.

    The run says how many it answered, so this is the one thing a truncated
    derivation can never report — and it reports it deterministically, which a
    character-name probe could not.
    """
    built = await _built(target, play_bank_prefix)
    counts = built["counts"]  # type: ignore[assignment]
    bank_id = str(built["bank_id"])
    length = int(built["length"])
    assert length > EXTRACTION_LIMIT, "the fixture must not fit in one prompt"

    expected = _slices_for(length)
    read = int(counts.get("slices_read", 0))
    failed = int(counts.get("slices_failed", 0))

    problems: list[str] = []
    if read < expected:
        problems.append(f"{read} of {expected} slices were read")
    if failed:
        problems.append(f"{failed} slice(s) failed, so part of the document reached nothing")
    log.info("derivation counts: %s", counts)
    _record("play-window", "windows", bank_id, problems, "; ".join(problems))
    assert not problems, (
        f"{'; '.join(problems)}\nA {length}-character document against a {EXTRACTION_LIMIT}-character "
        f"limit is {expected} slices.\ncounts: {counts}\nbank {bank_id}"
    )


@pytest.mark.full
async def test_the_cast_list_places_people_the_scenes_do_not(target, play_bank_prefix: str) -> None:
    """A character's house is stated once, in the cast list, pages from their scenes."""
    built = await _built(target, play_bank_prefix)
    characters = built["characters"]  # type: ignore[assignment]
    bank_id = str(built["bank_id"])

    problems: list[str] = []
    for expected, house in ((MONTAGUES, "montague"), (CAPULETS, "capulet")):
        for name in expected:
            record = characters.get(name)
            if record is None:
                problems.append(f"{name} is not in the table")
                continue
            got = record["values"].get("house")
            if got != house:
                problems.append(f"{name}.house is {got!r}, the cast list says {house!r}")
    placed = len(MONTAGUES) + len(CAPULETS) - len(problems)
    log.info("houses placed: %d/%d", placed, len(MONTAGUES) + len(CAPULETS))
    _record("play-houses", "relationship", bank_id, problems, "")
    assert not problems, "; ".join(problems) + f"\nbank {bank_id}"


@pytest.mark.full
async def test_a_slice_is_read_once_for_every_collection(target, play_bank_prefix: str) -> None:
    """Two collections over one document must not cost two passes over its slices.

    ``slices_read`` is the count to assert on, not ``llm_calls``: the latter also
    carries the adjudication and link passes, so it grew when those were added and
    said nothing about whether a slice had been read twice.
    """
    built = await _built(target, play_bank_prefix)
    counts = built["counts"]  # type: ignore[assignment]
    bank_id = str(built["bank_id"])
    slices = _slices_for(int(built["length"]))
    read = int(counts.get("slices_read", 0))

    problems = (
        []
        if read == slices
        else [f"{read} slice reads for {slices} slices and 2 collections; one pass over each is {slices}"]
    )
    log.info("derivation counts: %s", counts)
    _record("play-cost", "cost", bank_id, problems, "")
    assert problems == [], (
        f"{problems}\nA slice read once per collection would be {slices * 2} reads.\n"
        f"counts: {counts}\nbank {bank_id}"
    )
