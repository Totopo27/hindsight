"""Metadata filters: a small JSON DSL compiled to SQL over the passage and document JSONB.

    {"doc_type": "invoice",                  # equality, the common case
     "total": {"$gte": 1000, "$lt": 5000},   # numeric range
     "party": {"$in": ["Acme", "Globex"]},
     "tags_list": {"$contains": "urgent"},   # array holds this value
     "signed_on": {"$exists": true}}

A name is looked up in the passage's extracted values first, then the document's extracted
values, then the metadata the caller wrote — so a filter works the same whether the
property was extracted per passage, per document, or supplied at write time.
"""

from __future__ import annotations

import json
from typing import Any

OPERATORS = ("$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin", "$contains", "$exists")

_COMPARISONS = {"$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}


class FilterError(ValueError):
    """A filter the caller cannot have meant: unknown operator, wrong value shape."""


def _field(name: str, params: list[Any], aliases: tuple[str, str]) -> str:
    """The JSONB value for this field, wherever it lives.

    ``aliases`` is (passage, document). They are the same table when the caller is
    querying documents, which is why this is a parameter: a self-join added only to give
    this function two names to read cost a second scan of the whole table.
    """
    passage, document = aliases
    params.append(name)
    placeholder = f"${len(params)}"
    return (
        f"COALESCE({passage}.fields -> {placeholder}, {document}.fields -> {placeholder}, "
        f"{document}.metadata -> {placeholder})"
    )


def _numeric(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _literal(value: Any, params: list[Any]) -> str:
    """A JSONB literal for the comparison's right-hand side."""
    params.append(json.dumps(value))
    return f"${len(params)}::jsonb"


def _condition(name: str, spec: Any, params: list[Any], aliases: tuple[str, str]) -> str:
    field = _field(name, params, aliases)

    if not isinstance(spec, dict):
        # Bare value: equality. A list means "the array property holds all of these".
        if isinstance(spec, list):
            return f"{field} @> {_literal(spec, params)}"
        return f"{field} = {_literal(spec, params)}"

    unknown = [key for key in spec if key not in OPERATORS]
    if unknown:
        raise FilterError(f"property {name!r}: unknown operator(s) {unknown}; one of {list(OPERATORS)}")

    clauses: list[str] = []
    for operator, value in spec.items():
        if operator == "$eq":
            clauses.append(f"{field} = {_literal(value, params)}")
        elif operator == "$ne":
            clauses.append(f"({field} IS NULL OR {field} <> {_literal(value, params)})")
        elif operator in _COMPARISONS:
            if not _numeric(value):
                # Strings and dates compare as text, numbers as numbers: comparing
                # "1000" to "900" as text would say 1000 < 900.
                clauses.append(f"({field} #>> '{{}}') {_COMPARISONS[operator]} {_literal(value, params)} #>> '{{}}'")
            else:
                clauses.append(f"({field} #>> '{{}}')::numeric {_COMPARISONS[operator]} {_literal(value, params)}::text::numeric")  # fmt: skip
        elif operator in ("$in", "$nin"):
            if not isinstance(value, list) or not value:
                raise FilterError(f"property {name!r}: {operator} needs a non-empty list")
            # One array parameter, not one comparison per value: a 300-value list compiled
            # to 300 OR-ed equality tests cost 480ms on a 22k-document bank, where `= ANY`
            # over a single bound array is a hash lookup.
            params.append([json.dumps(v) for v in value])
            options = f"{field} = ANY(${len(params)}::jsonb[])"
            clauses.append(options if operator == "$in" else f"({field} IS NULL OR NOT ({options}))")
        elif operator == "$contains":
            # Works for an array property (holds the value) and for a string property
            # (contains the substring), because those are the two things "contains" means.
            text = _literal(value, params)
            clauses.append(
                f"(({field} @> {text}) OR (jsonb_typeof({field}) = 'string' "
                f"AND ({field} #>> '{{}}') ILIKE '%' || ({text} #>> '{{}}') || '%'))"
            )
        elif operator == "$exists":
            if not isinstance(value, bool):
                raise FilterError(f"property {name!r}: $exists needs true or false")
            clauses.append(f"{field} IS NOT NULL" if value else f"{field} IS NULL")
    return " AND ".join(clauses) if clauses else "TRUE"


def compile_filters(
    metadata: dict[str, Any] | None, params: list[Any], aliases: tuple[str, str] = ("c", "d")
) -> str:
    """Return a SQL fragment starting with AND, appending its parameters to ``params``.

    The caller must have joined ``kb_documents d`` to ``kb_passages c`` when this returns
    anything but the empty string.
    """
    if not metadata:
        return ""
    if not isinstance(metadata, dict):
        raise FilterError("metadata filter must be an object of property name -> value or operators")
    conditions = [_condition(name, spec, params, aliases) for name, spec in metadata.items()]
    return "".join(f" AND ({condition})" for condition in conditions)
