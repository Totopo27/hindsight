"""Chunking for knowledge banks: recursive character splitting, sized in tokens.

The same shape as LangChain's RecursiveCharacterTextSplitter — try to split on the
biggest natural boundary that fits (paragraphs, then lines, then sentences, then
words, then characters) — but the budget is counted in *tokens* with the engine's
tokenizer, so `chunk_size = 512` means 512 tokens, the unit every other budget in
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
    """Greedily fill chunks up to the budget, then carry ``overlap_tokens`` into the next.

    The budget is measured on the joined text, not as a sum of the pieces' counts: a
    tokenizer merges across a join, so summing parts underfills badly on text without
    natural boundaries (5000 unbroken characters became 8-token chunks).
    """
    chunks: list[str] = []
    current: list[str] = []
    current_text = ""
    for piece in pieces:
        candidate = current_text + piece
        if current and count_tokens(candidate) > max_tokens:
            chunks.append(current_text)
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
        chunks.append(current_text)
    return chunks


def chunk_document(text: str, *, chunk_size: int, chunk_overlap: int = 0) -> list[Chunk]:
    """Split a document into chunks of at most ``chunk_size`` tokens.

    ``chunk_overlap`` repeats the tail of a chunk at the head of the next, which keeps
    a fact that straddles a boundary retrievable from both sides.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
    cleaned = text.strip()
    if not cleaned:
        return []
    pieces = _split_recursive(cleaned, chunk_size, SEPARATORS)
    merged = _merge(pieces, chunk_size, chunk_overlap)
    chunks: list[Chunk] = []
    for raw in merged:
        stripped = re.sub(r"\s+\Z", "", raw).lstrip()
        if stripped:
            chunks.append(Chunk(index=len(chunks), text=stripped, token_count=count_tokens(stripped)))
    return chunks
