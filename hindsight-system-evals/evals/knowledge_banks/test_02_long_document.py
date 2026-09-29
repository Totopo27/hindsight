"""A document longer than one prompt, and a table built from all of it.

The procurement folder in ``test_01`` is thirteen paragraphs: every document fits in
one extraction call, which is the shape a demo has and a corpus does not. This suite
is one document of sixteen thousand characters — Act 1 of a play — against a limit
that cuts a prompt at twelve thousand.

That gap is the whole point. Before derivation read a document in slices it sent the
document whole and truncated it, so everything past the cut was never read and nothing
said so: on the full thirty pages the cast simply stopped. Here Paris first appears at
character 13,599. If the table has Paris, the second slice was read.

The rest of what it grades follows from the same shape:

* **the cast list is in the first slice and the scenes are in the second**, so a
  character's house has to reach a slice that never states it;
* **two collections, one pass**, so the run's own ``llm_calls`` says whether a slice
  was read once or once per collection.

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

#: Where each one first appears in the fixture. Paris is the case: past the point a
#: single extraction prompt would have stopped.
PARIS_AT = 13_599
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


async def test_a_character_past_the_prompt_limit_is_in_the_table(target, play_bank_prefix: str) -> None:
    """Paris appears once, at character 13,599 of a 16,000-character document.

    A single prompt stops at 12,000. So this asserts the one thing a truncated
    derivation can never do, on a document short enough to run in a minute.
    """
    built = await _built(target, play_bank_prefix)
    characters = built["characters"]  # type: ignore[assignment]
    bank_id = str(built["bank_id"])
    assert int(built["length"]) > EXTRACTION_LIMIT, "the fixture must not fit in one prompt"

    found = [rid for rid in characters if "paris" in rid]
    problems = [] if found else [f"Paris is missing; the table has {sorted(characters)}"]
    trap = "" if found else f"nothing past character {PARIS_AT} reached the table"
    _record("play-window", "windows", bank_id, problems, trap)
    assert found, (
        f"Paris first appears at character {PARIS_AT}, past the {EXTRACTION_LIMIT}-character "
        f"extraction limit, and is not in the table — the document was read to the cut and no "
        f"further.\ntable: {sorted(characters)}\nbank {bank_id}"
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

    The run reports what it spent. A 16,000-character document is two slices, so
    reading it once per collection would be four extraction calls before any linking;
    reading it once for both is two. The ceiling here is deliberately loose — it is
    guarding the shape, not a particular model's behaviour.
    """
    built = await _built(target, play_bank_prefix)
    counts = built["counts"]  # type: ignore[assignment]
    bank_id = str(built["bank_id"])
    calls = int(counts.get("llm_calls", 0))
    slices = -(-int(built["length"]) // EXTRACTION_LIMIT)
    ceiling = slices + 2  # the slices, plus one link pass per relationship, plus slack

    problems = (
        []
        if 0 < calls <= ceiling
        else [f"{calls} LLM calls for {slices} slices and 2 collections; one pass would be at most {ceiling}"]
    )
    log.info("derivation counts: %s", counts)
    _record("play-cost", "cost", bank_id, problems, "")
    assert problems == [], (
        f"{problems}\nA slice read once per collection would cost {slices * 2} calls before linking.\n"
        f"counts: {counts}\nbank {bank_id}"
    )
