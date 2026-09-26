"""Querying records: the passage/document DSL, over collections, with joins.

Same language as ``query.py`` — select, where, group_by, having, order_by, aggregates,
arithmetic and functions over them — and one addition that only makes sense here:

    {"from": "contracts",
     "join": [{"collection": "vendors", "on": "vendor", "as": "vendor"}],
     "select": [{"field": "vendor.country", "as": "country"},
                {"count": "*", "as": "contracts"},
                {"sum": "value", "as": "total"}],
     "where": {"status": "active"},
     "group_by": ["vendor.country"],
     "having": {"contracts": {"$gte": 3}},
     "order_by": [{"field": "total", "direction": "desc"}]}

``on`` names a **relationship field** of the collection being queried: its value is a
record id in the joined collection, which is what the join follows. A joined collection's
fields are then addressed as ``<alias>.<field>``. Nothing else changes — the same limits,
the same numeric handling, the same refusal to let an identifier reach SQL as text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..engine.schema import fq_table
from .filters import FilterError
from .query import AGGREGATES, ARITHMETIC, FUNCTIONS, CompiledQuery, Limits, _Compiler

#: Columns of a record itself, as opposed to its fields.
_RECORD_COLUMNS: dict[str, str] = {
    "record_id": "r.record_id",
    "created_at": "r.created_at",
    "updated_at": "r.updated_at",
    "doc_count": "cardinality(r.doc_ids)",
}

MAX_JOINS = 4


@dataclass
class _Join:
    alias: str
    collection_id: str
    on_field: str
    #: The joined collection's field names, so a typo there is refused too.
    field_names: frozenset[str] = frozenset()

    @property
    def table_alias(self) -> str:
        # The alias a caller picked is validated, but it is not what reaches SQL: the
        # statement uses positional names so a field called "select" cannot end up as one.
        return f"j{abs(hash(self.alias)) % 1000}_{self.on_field}"


@dataclass
class _RecordCompiler(_Compiler):
    """The document/passage compiler, with record columns and joined collections."""

    joins: dict[str, _Join] = field(default_factory=dict)
    #: The queried collection's own field names. A collection knows its fields, so a name
    #: it does not define is a typo, and answering a typo with a column of nulls is worse
    #: than answering it with an error.
    field_names: frozenset[str] = frozenset()

    @property
    def columns(self) -> dict[str, str]:
        return _RECORD_COLUMNS

    def field(self, name: str) -> str:
        if name in _RECORD_COLUMNS:
            return _RECORD_COLUMNS[name]
        alias, _, rest = name.partition(".")
        if rest and alias in self.joins:
            join = self.joins[alias]
            if rest not in join.field_names:
                raise FilterError(
                    f"collection {join.collection_id!r} has no field {rest!r}; "
                    f"its fields are {sorted(join.field_names)}"
                )
            return self.value_expression(join.table_alias, rest)
        if "." in name:
            raise FilterError(f"unknown join alias in {name!r}; joined as {sorted(self.joins) or 'nothing'}")
        if name not in self.field_names:
            raise FilterError(
                f"unknown field {name!r}; this collection's fields are {sorted(self.field_names)} "
                f"plus {sorted(_RECORD_COLUMNS)}"
            )
        return self.value_expression("r", name)

    def value_expression(self, table: str, field_name: str) -> str:
        """One field of a record, as JSONB text — shared by select, where and group_by."""
        key = f"{table}.{field_name}"
        placeholder = self.field_placeholders.get(key)
        if placeholder is None:
            placeholder = self.field_placeholders[key] = self.bind(field_name)
        expression = f"({table}.values -> {placeholder} #>> '{{}}')"
        self.field_expressions.add(expression)
        return expression


def _validate_joins(
    raw: Any, collection: dict[str, Any], limits: Limits, known: dict[str, frozenset[str]]
) -> dict[str, _Join]:
    if raw is None:
        return {}
    if not isinstance(raw, list):
        raise FilterError("join must be a list of {collection, on, as}")
    if len(raw) > MAX_JOINS:
        raise FilterError(f"at most {MAX_JOINS} joins per query")

    from .collections import relationships

    available = relationships(collection.get("fields") or {})
    joins: dict[str, _Join] = {}
    for spec in raw:
        if not isinstance(spec, dict):
            raise FilterError("each join is {collection, on, as}")
        on_field = spec.get("on")
        if on_field not in available:
            raise FilterError(
                f"join on {on_field!r}: not a relationship field of this collection; "
                f"relationships are {sorted(available) or 'none'}"
            )
        collection_id = spec.get("collection", available[on_field])
        if collection_id != available[on_field]:
            raise FilterError(f"join on {on_field!r} points at {available[on_field]!r}, not {collection_id!r}")
        if collection_id not in known:
            # The relationship still names it, but the collection is gone: a join that
            # would quietly return nulls is worse than one that says what happened.
            raise FilterError(f"join on {on_field!r}: collection {collection_id!r} no longer exists in this bank")
        alias = spec.get("as", on_field)
        if not isinstance(alias, str) or not alias.replace("_", "").isalnum():
            raise FilterError(f"join alias {alias!r} must be alphanumeric with _")
        if alias in joins:
            raise FilterError(f"duplicate join alias {alias!r}")
        joins[alias] = _Join(
            alias=alias,
            collection_id=collection_id,
            on_field=on_field,
            field_names=known.get(collection_id, frozenset()),
        )
    return joins


def compile_record_query(
    body: dict[str, Any],
    bank_id: str,
    collection: dict[str, Any],
    *,
    limits: Limits | None = None,
    joined_fields: dict[str, frozenset[str]] | None = None,
) -> CompiledQuery:
    """Compile one record query for one collection of one bank.

    ``joined_fields`` is each other collection's field names, so a name a joined
    collection does not define is refused here rather than answered with nulls.
    """
    limits = limits or Limits()
    compiler = _RecordCompiler(source="documents", limits=limits)
    compiler.field_names = frozenset(collection.get("fields") or {})
    # $1 bank, $2 collection: every row this can reach belongs to both.
    compiler.params.extend([bank_id, collection["collection_id"]])
    compiler.joins = _validate_joins(body.get("join"), collection, limits, joined_fields or {})

    select_parts = compiler.select(body.get("select"))
    group_parts = compiler.group_by(body.get("group_by"))
    where = _compile_where(body.get("where"), compiler)
    having = compiler.having(body.get("having"))
    if having and not compiler.has_aggregate:
        raise FilterError("having needs an aggregate in select")
    order_parts = compiler.order_by(body.get("order_by"))

    limit = body.get("limit", 100)
    offset = body.get("offset", 0)
    for name, value in (("limit", limit), ("offset", offset)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise FilterError(f"{name} must be a non-negative integer")
    if limit > limits.max_limit:
        raise FilterError(f"limit {limit} exceeds {limits.max_limit}")

    sql = f"SELECT {', '.join(select_parts)} FROM {fq_table('kb_records')} r"
    for join in compiler.joins.values():
        # A relationship holds the joined record's id, so the join is that id against the
        # other collection's primary key — LEFT, because a record with nothing on the
        # other side is still a record and should not vanish from a count.
        compiler.params.append(join.collection_id)
        collection_placeholder = f"${len(compiler.params)}"
        compiler.params.append(join.on_field)
        field_placeholder = f"${len(compiler.params)}"
        sql += (
            f" LEFT JOIN {fq_table('kb_records')} {join.table_alias}"
            f" ON {join.table_alias}.bank_id = r.bank_id"
            f" AND {join.table_alias}.collection_id = {collection_placeholder}"
            f" AND {join.table_alias}.record_id = (r.values -> {field_placeholder} #>> '{{}}')"
        )
    sql += " WHERE r.bank_id = $1 AND r.collection_id = $2" + where
    if group_parts:
        sql += " GROUP BY " + ", ".join(group_parts)
    if having:
        sql += " HAVING " + having
    if order_parts:
        sql += " ORDER BY " + ", ".join(order_parts)
    sql += f" LIMIT {int(limit)} OFFSET {int(offset)}"

    return CompiledQuery(
        sql=sql,
        params=compiler.params,
        columns=list(compiler.aliases),
        grouped=bool(group_parts) or compiler.has_aggregate,
    )


def _compile_where(raw: Any, compiler: _RecordCompiler) -> str:
    """Filters over record fields, including a joined collection's.

    The document filter DSL reads two fixed JSONB columns; a record query has one per
    joined table, so the conditions are built here off the compiler's own field
    resolution. The operators are the same, deliberately: one filter language.
    """
    if not raw:
        return ""
    if not isinstance(raw, dict):
        raise FilterError("where must be an object of field name -> value or operators")

    clauses: list[str] = []
    for name, condition in raw.items():
        sql = compiler.field(name)
        if not isinstance(condition, dict):
            clauses.append(f"{sql} = {compiler.bind(_as_text(condition))}")
            continue
        for operator, value in condition.items():
            if operator == "$eq":
                clauses.append(f"{sql} = {compiler.bind(_as_text(value))}")
            elif operator == "$ne":
                clauses.append(f"({sql} IS NULL OR {sql} <> {compiler.bind(_as_text(value))})")
            elif operator in ("$gt", "$gte", "$lt", "$lte"):
                symbol = {"$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}[operator]
                if isinstance(value, int | float) and not isinstance(value, bool):
                    clauses.append(f"{compiler.numeric(sql)} {symbol} {compiler.bind(value)}::numeric")
                else:
                    clauses.append(f"{sql} {symbol} {compiler.bind(_as_text(value))}")
            elif operator in ("$in", "$nin"):
                if not isinstance(value, list) or not value:
                    raise FilterError(f"field {name!r}: {operator} needs a non-empty list")
                if len(value) > compiler.limits.max_in_values:
                    raise FilterError(f"field {name!r}: {operator} has {len(value)} values; at most {compiler.limits.max_in_values}")  # fmt: skip
                placeholder = compiler.bind([_as_text(item) for item in value])
                clauses.append(
                    f"{sql} = ANY({placeholder}::text[])"
                    if operator == "$in"
                    else f"({sql} IS NULL OR NOT ({sql} = ANY({placeholder}::text[])))"
                )
            elif operator == "$contains":
                clauses.append(f"{sql} ILIKE '%' || {compiler.bind(_as_text(value))} || '%'")
            elif operator == "$exists":
                if not isinstance(value, bool):
                    raise FilterError(f"field {name!r}: $exists needs true or false")
                clauses.append(f"{sql} IS NOT NULL" if value else f"{sql} IS NULL")
            else:
                raise FilterError(f"field {name!r}: unknown operator {operator!r}")
    return "".join(f" AND ({clause})" for clause in clauses)


def _as_text(value: Any) -> str:
    """Record values are compared as the JSONB text they are stored as."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


__all__ = ["AGGREGATES", "ARITHMETIC", "FUNCTIONS", "MAX_JOINS", "compile_record_query"]
