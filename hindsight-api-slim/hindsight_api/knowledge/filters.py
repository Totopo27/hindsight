"""Metadata filters: a small JSON DSL compiled to SQL over the chunk and document JSONB.

    {"doc_type": "invoice",                  # equality, the common case
     "total": {"$gte": 1000, "$lt": 5000},   # numeric range
     "party": {"$in": ["Acme", "Globex"]},
     "tags_list": {"$contains": "urgent"},   # array holds this value
     "signed_on": {"$exists": true}}

A name is looked up in the chunk's extracted values first, then the document's extracted
values, then the metadata the caller wrote — so a filter works the same whether the
property was extracted per chunk, per document, or supplied at write time.
"""

from __future__ import annotations

from typing import Any

OPERATORS = ("$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin", "$contains", "$exists")

_COMPARISONS = {"$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}


class FilterError(ValueError):
    """A filter the caller cannot have meant: unknown operator, wrong value shape."""


def _field(name: str, params: list[Any]) -> str:
    """The JSONB value for this property, wherever it lives."""
    params.append(name)
    placeholder = f"${len(params)}"
    return f"COALESCE(c.metadata -> {placeholder}, d.extracted_metadata -> {placeholder}, d.metadata -> {placeholder})"


def _numeric(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _literal(value: Any, params: list[Any]) -> str:
    """A JSONB literal for the comparison's right-hand side."""
    import json

    params.append(json.dumps(value))
    return f"${len(params)}::jsonb"


def _condition(name: str, spec: Any, params: list[Any]) -> str:
    field = _field(name, params)

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
            options = " OR ".join(f"{field} = {_literal(v, params)}" for v in value)
            clauses.append(f"({options})" if operator == "$in" else f"({field} IS NULL OR NOT ({options}))")
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


def compile_filters(metadata: dict[str, Any] | None, params: list[Any]) -> str:
    """Return a SQL fragment starting with AND, appending its parameters to ``params``.

    The caller must have joined ``kb_documents d`` to ``kb_chunks c`` when this returns
    anything but the empty string.
    """
    if not metadata:
        return ""
    if not isinstance(metadata, dict):
        raise FilterError("metadata filter must be an object of property name -> value or operators")
    conditions = [_condition(name, spec, params) for name, spec in metadata.items()]
    return "".join(f" AND ({condition})" for condition in conditions)
