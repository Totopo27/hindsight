"""Deciding whether two names are the same thing.

Records are grouped by an identity field, and documents do not agree on how to write a
name: "Apple", "Apple Inc.", "APPLE, INC." are one vendor. Three steps, cheapest first,
each one a decision someone can audit:

1. **Normalise** — case, accents, punctuation, spacing, and the legal suffix that says
   what kind of company it is rather than which one. This is exact matching on a tidied
   key, so it never guesses.
2. **Alias** — a merge someone performed, or a variant already resolved. This is the only
   step that can encode a judgement no string comparison could reach ("Big Blue" = IBM),
   and it is set by a person, not inferred.
3. **Typo matching** — for "Vandaley Industries". Two stages, because one is not enough:
   a trigram prefilter that an index can answer, then an edit-distance check that decides.
   Trigram alone cannot do this job — measured on our own cases, a transposition in
   "Vandelay Industries" scores 0.600 while "Apple" against "Apple Bank" scores 0.545, so
   any threshold that catches the typo also merges two different companies. Edit distance
   separates them cleanly, because a typo is a *small* change to a name of *nearly the
   same length*, and "Apple Bank" is neither.

What normalisation must *not* do is decide that "Apple" and "Apple Bank" are one company.
Dropping a suffix is safe because a suffix says nothing about identity; dropping a word is
not, because "Bank" is exactly what distinguishes them.
"""

from __future__ import annotations

import re
import unicodedata

#: Legal forms, in the spellings documents actually use. Stripped from the end of a name
#: only: "Corp" at the end is a suffix, "Corp" at the start is part of the name.
LEGAL_SUFFIXES: tuple[str, ...] = (
    "incorporated",
    "inc",
    "corporation",
    "corp",
    "company",
    "co",
    "limited",
    "ltd",
    "llc",
    "l l c",
    "llp",
    "lllp",
    "lp",
    "plc",
    "gmbh",
    "mbh",
    "ug",
    "kg",
    "ag",
    "spa",
    "s p a",
    "srl",
    "s r l",
    "sarl",
    "sa",
    "sas",
    "bv",
    "nv",
    "oy",
    "oyj",
    "ab",
    "as",
    "asa",
    "aps",
    "pty",
    "pte",
    "kk",
    "kabushiki kaisha",
    "sl",
    "sp z oo",
    "doo",
    "dba",
)

#: Below this many characters, one edit is the difference between two real companies
#: ("BP" and "HP"), so typo matching does not run at all.
MIN_SIMILARITY_LENGTH = 12

#: How much of a name a typo may change. 0.12 accepts two errors in nineteen characters
#: (the transposition case) and rejects seven in sixteen ("Johnson Controls" against
#: "Johnson Matthey").
MAX_RELATIVE_EDITS = 0.12

#: A typo barely changes a name's length; a different company often does. Checked before
#: the distance because it is free and it is what separates "Apple" from "Apple Bank".
MAX_LENGTH_DIFFERENCE = 2

_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def normalise(value: object) -> str:
    """The key two spellings of one name have to agree on.

    Lowercased, unaccented, stripped of punctuation and repeated spacing, with any
    trailing legal suffixes removed. Returns "" for anything that normalises away, which
    the caller must treat as "no identity" rather than as a record everything falls into.
    """
    text = str(value if value is not None else "").strip()
    if not text:
        return ""
    # NFKD splits an accented letter into letter + mark, so dropping the marks leaves the
    # ASCII letter: "Nestlé" and "Nestle" become the same key.
    text = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))
    text = _PUNCTUATION.sub(" ", text.lower())
    text = _WHITESPACE.sub(" ", text).strip()

    # Suffixes come off one at a time, because "Acme Holdings Ltd Co" is a real shape and
    # one pass would leave the inner one behind.
    changed = True
    while changed and text:
        changed = False
        for suffix in LEGAL_SUFFIXES:
            if text.endswith(f" {suffix}"):
                candidate = text[: -len(suffix) - 1].strip()
                # Never strip a name down to nothing: a company literally called "Co" is
                # its own key, not every company's key.
                if candidate:
                    text = candidate
                    changed = True
                    break
    return text


def can_compare_fuzzily(key: str) -> bool:
    """Whether similarity matching is meaningful for this key at all."""
    return len(key) >= MIN_SIMILARITY_LENGTH


def edit_distance(left: str, right: str, *, cap: int) -> int:
    """Levenshtein distance, abandoned once it exceeds ``cap``.

    Only the "is this a typo" question is being asked, so a distance of 3 and a distance
    of 30 are the same answer, and the cap keeps the comparison cheap on long names.
    """
    if left == right:
        return 0
    if abs(len(left) - len(right)) > cap:
        return cap + 1
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        best = i
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            current.append(value)
            best = min(best, value)
        if best > cap:
            return cap + 1
        previous = current
    return previous[-1]


def is_typo_of(candidate: str, existing: str) -> bool:
    """Whether these two names differ only by the kind of mistake people make typing.

    Both must be long enough for the question to be meaningful, nearly the same length,
    and within a small edit distance proportional to that length.
    """
    if not can_compare_fuzzily(candidate) or not can_compare_fuzzily(existing):
        return False
    if abs(len(candidate) - len(existing)) > MAX_LENGTH_DIFFERENCE:
        return False
    allowed = max(1, round(min(len(candidate), len(existing)) * MAX_RELATIVE_EDITS))
    return edit_distance(candidate, existing, cap=allowed) <= allowed
