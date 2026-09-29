"""Chunking for knowledge banks: recursive character splitting, sized in tokens.

The same shape as LangChain's RecursiveCharacterTextSplitter — try to split on the
biggest natural boundary that fits (paragraphs, then lines, then sentences, then
words, then characters) — but the budget is counted in *tokens* with the engine's
tokenizer, so `passage_size = 512` means 512 tokens, the unit every other budget in
Hindsight uses and the unit the baselines we compare against use.

No new dependency: the separators are a handful of strings and the counting is
``engine/token_encoding.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..engine.token_encoding import count_tokens

#: Biggest boundary first. The empty string is the last resort: split anywhere.
SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", "")


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    token_count: int
    #: The markdown heading the passage sits under, when the document has headings. A
    #: table's rows rarely repeat the title of the statement they belong to ("Consolidated
    #: Statements of Cash Flows"), so the heading is what makes them findable by it.
    section: str | None = None


_HEADING = re.compile(r"^#{1,6}[ \t]+(.+?)[ \t#]*$", re.MULTILINE)


def _heading_text(raw: str) -> str:
    text = re.sub(r"[*_`]+", "", raw).strip()
    return re.sub(r"\s+", " ", text)[:200]


def _split_on(text: str, separator: str) -> list[str]:
    if separator == "":
        return list(text)
    # Keep the separator on the left piece so joining pieces reproduces the text.
    parts = text.split(separator)
    out = [p + separator for p in parts[:-1]]
    if parts[-1]:
        out.append(parts[-1])
    return out


def _split_recursive(text: str, max_tokens: int, separators: tuple[str, ...]) -> list[str]:
    """Pieces that each fit in ``max_tokens``, splitting on the first separator that helps."""
    if not text:
        return []
    if count_tokens(text) <= max_tokens:
        return [text]
    for i, separator in enumerate(separators):
        pieces = _split_on(text, separator)
        if len(pieces) < 2:
            continue
        out: list[str] = []
        for piece in pieces:
            if count_tokens(piece) <= max_tokens:
                out.append(piece)
            else:
                out.extend(_split_recursive(piece, max_tokens, separators[i + 1 :]))
        return out
    return [text]


def _merge(pieces: list[str], max_tokens: int, overlap_tokens: int) -> list[str]:
    """Greedily fill passages up to the budget, then carry ``overlap_tokens`` into the next.

    The budget is measured on the joined text, not as a sum of the pieces' counts: a
    tokenizer merges across a join, so summing parts underfills badly on text without
    natural boundaries (5000 unbroken characters became 8-token passages).
    """
    passages: list[str] = []
    current: list[str] = []
    current_text = ""
    for piece in pieces:
        candidate = current_text + piece
        if current and count_tokens(candidate) > max_tokens:
            passages.append(current_text)
            carried: list[str] = []
            if overlap_tokens > 0:
                for previous in reversed(current):
                    if count_tokens("".join([previous, *carried])) > overlap_tokens:
                        break
                    carried.insert(0, previous)
            current = carried
            current_text = "".join(carried)
            candidate = current_text + piece
        current.append(piece)
        current_text = candidate
    if current_text:
        passages.append(current_text)
    return passages


def split_into_passages(text: str, *, passage_size: int, passage_overlap: int = 0) -> list[Chunk]:
    """Split a document into passages of at most ``passage_size`` tokens.

    ``passage_overlap`` repeats the tail of a passage at the head of the next, which keeps
    a fact that straddles a boundary retrievable from both sides.
    """
    if passage_size <= 0:
        raise ValueError("passage_size must be positive")
    if not 0 <= passage_overlap < passage_size:
        raise ValueError("passage_overlap must be >= 0 and smaller than passage_size")
    cleaned = text.strip()
    if not cleaned:
        return []
    pieces = _split_recursive(cleaned, passage_size, SEPARATORS)
    merged = _merge(pieces, passage_size, passage_overlap)
    headings = [(m.start(), _heading_text(m.group(1))) for m in _HEADING.finditer(cleaned)]
    passages: list[Chunk] = []
    cursor = 0
    for raw in merged:
        stripped = re.sub(r"\s+\Z", "", raw).lstrip()
        if not stripped:
            continue
        # Passages are slices of the text in order; overlap means the next one starts
        # before the previous one ends, so the search starts at the previous start.
        found = cleaned.find(stripped, cursor)
        start = found if found >= 0 else cursor
        cursor = start
        section = next((text for pos, text in reversed(headings) if pos < start and text), None)
        passages.append(Chunk(index=len(passages), text=stripped, token_count=count_tokens(stripped), section=section))
    return passages
