"""The knowledge-bank query language: a JSON DSL with SQL's shape, compiled to one query.

Not SQL over the wire — a caller never writes SQL, and nothing they send reaches the
statement as text. Every identifier is matched against a fixed column list or bound as a
JSONB path parameter, every literal is a bind parameter, and the shape of the query is
built here. What the DSL keeps from SQL is its *expressiveness*: projection, filtering,
grouping, aggregates, arithmetic and functions over aggregates, HAVING, ORDER BY, paging.

    {"from": "passages",
     "select": [{"field": "fields.doc_type", "as": "kind"},
                {"count": "*", "as": "passages"},
                {"count_distinct": "doc_id", "as": "documents"},
                {"sum": "token_count", "as": "tokens"},
                {"round": [{"divide": [{"sum": "token_count"}, {"count": "*"}]}, 1],
                 "as": "avg_tokens"}],
     "where": {"doc_type": {"$in": ["invoice", "contract"]}},
     "group_by": ["fields.doc_type"],
     "having": {"passages": {"$gte": 10}},
     "order_by": [{"field": "passages", "direction": "desc"}],
     "limit": 50}

Aggregate-free queries are row queries; a query with any aggregate is grouped by whatever
``group_by`` names (SQL's rule, enforced here with a clearer error than Postgres gives).

Every limit in ``Limits`` exists because this endpoint takes a *structure* from the caller:
without them a request can ask for a query that is cheap to write and ruinous to run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from .filters import FilterError, compile_filters

Source = Literal["documents", "passages"]

#: Aggregates, and the SQL each one becomes. ``count`` is the only one that takes "*".
AGGREGATES = ("count", "count_distinct", "sum", "avg", "min", "max")
#: Arithmetic over anything numeric — including over aggregates, which is the point.
ARITHMETIC = ("add", "subtract", "multiply", "divide")
#: Scalar functions on a field or expression.
FUNCTIONS = ("round", "abs", "ceil", "floor", "length", "lower", "upper", "date_trunc", "extract", "coalesce")

#: Columns a caller may name, per source, mapped to SQL. Anything not in here is either a
#: ``metadata.*`` path (bound as a parameter, never interpolated) or an error.
_DOCUMENT_COLUMNS: dict[str, str] = {
    "doc_id": "d.doc_id",
    "title": "d.title",
    "tags": "d.tags",
    "passage_count": "d.passage_count",
    "text_length": "length(d.text)",
    "created_at": "d.created_at",
    "updated_at": "d.updated_at",
}
_CHUNK_COLUMNS: dict[str, str] = {
    **_DOCUMENT_COLUMNS,
    "passage_index": "c.passage_index",
    "token_count": "c.token_count",
    "passage_text_length": "length(c.text)",
    "heading": "c.heading",
}

#: date_trunc / extract units, whitelisted because they are SQL keywords, not values.
_UNITS = ("microseconds", "milliseconds", "second", "minute", "hour", "day", "week", "month", "quarter", "year")
_EXTRACT_FIELDS = ("year", "quarter", "month", "week", "day", "dow", "doy", "hour", "minute", "second", "epoch")


@dataclass(frozen=True)
class Limits:
    """What a single query may ask for. Every field is a cost the server pays."""

    max_select: int = 32
    max_group_by: int = 8
    max_order_by: int = 8
    max_depth: int = 8
    max_parameters: int = 400
    max_limit: int = 1000
    max_in_values: int = 500


@dataclass(frozen=True)
class CompiledQuery:
    """The SQL and parameters for one DSL query, plus the column names it returns."""

    sql: str
    params: list[Any]
    columns: list[str]
    grouped: bool


@dataclass
class _Compiler:
    source: Source
    limits: Limits
    params: list[Any] = field(default_factory=list)
    aliases: dict[str, str] = field(default_factory=dict)  # alias -> SQL expression
    # A property named twice (in select and in group_by, say) must compile to the *same*
    # SQL, or Postgres cannot match the grouping expression to the selected one. Binding a
    # fresh parameter each time produced `... -> $2` in one place and `... -> $5` in the
    # other, which failed with "must appear in the GROUP BY clause".
    field_placeholders: dict[str, str] = field(default_factory=dict)
    #: Which compiled expressions came from a field. Those arrive as JSONB text,
    #: so "7500" sorts above "20000" and max() returns the wrong row: in a numeric context
    #: they have to be cast, and ordering has to try the number before the text.
    field_expressions: set[str] = field(default_factory=set)
    has_aggregate: bool = False

    # ---- leaves

    @property
    def columns(self) -> dict[str, str]:
        return _CHUNK_COLUMNS if self.source == "passages" else _DOCUMENT_COLUMNS

    def bind(self, value: Any) -> str:
        if len(self.params) >= self.limits.max_parameters:
            raise FilterError(f"query binds more than {self.limits.max_parameters} values")
        self.params.append(value)
        return f"${len(self.params)}"

    def field(self, name: str) -> str:
        """A column, or a field as JSONB text."""
        if not isinstance(name, str) or not name:
            raise FilterError("a field must be a non-empty string")
        if name in self.columns:
            return self.columns[name]
        prefix, _, field_name = name.partition(".")
        if prefix in ("fields", "document_fields", "passage_fields") and field_name:
            return self.field_expression(prefix, field_name)
        raise FilterError(
            f"unknown field {name!r}; columns are {sorted(self.columns)} plus fields.<name> "
            "(or document_fields.<name> / passage_fields.<name> to pin the level)"
        )

    def field_expression(self, prefix: str, field_name: str) -> str:
        """The JSONB value of a field, as text.

        ``fields.x`` follows the same precedence filters use — the passage's extracted
        value, then the document's, then what the caller wrote — so a query and a search
        filter never disagree about where a property lives. The explicit prefixes exist for
        the case where that matters.
        """
        placeholder = self.field_placeholders.get(field_name)
        if placeholder is None:
            placeholder = self.field_placeholders[field_name] = self.bind(field_name)
        if prefix == "document_fields" or self.source == "documents":
            expression = f"COALESCE(d.fields -> {placeholder}, d.metadata -> {placeholder})"
            if prefix == "passage_fields":
                raise FilterError("passage_fields is not available when querying documents")
        elif prefix == "passage_fields":
            expression = f"c.fields -> {placeholder}"
        else:
            expression = f"COALESCE(c.fields -> {placeholder}, d.fields -> {placeholder}, d.metadata -> {placeholder})"
        self.field_expressions.add(f"({expression} #>> '{{}}')")
        return f"({expression} #>> '{{}}')"

    def numeric(self, sql: str) -> str:
        """``sql`` as a number, or NULL when it is not one.

        Postgres has no try_cast: casting 'abc' to numeric aborts the whole statement, and
        a field holds whatever the LLM extracted. The regex guard turns a
        non-numeric value into NULL, which every aggregate already ignores.
        """
        return f"CASE WHEN ({sql})::text ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN ({sql})::numeric END"

    # ---- expressions

    def expression(self, spec: Any, depth: int = 0) -> str:
        if depth > self.limits.max_depth:
            raise FilterError(f"expression nests deeper than {self.limits.max_depth}")
        if isinstance(spec, str):
            return self.field(spec)
        if isinstance(spec, bool) or spec is None:
            raise FilterError(f"{spec!r} is not an expression")
        if isinstance(spec, int | float):
            return f"{self.bind(spec)}::numeric"
        if not isinstance(spec, dict):
            raise FilterError(f"{spec!r} is not an expression")

        keys = [key for key in spec if key != "as"]
        if len(keys) != 1:
            raise FilterError(f"an expression needs exactly one operator, got {keys or 'none'}")
        operator = keys[0]
        operand = spec[operator]

        if operator == "field":
            return self.field(operand)
        if operator in AGGREGATES:
            return self.aggregate(operator, operand, depth)
        if operator in ARITHMETIC:
            return self.arithmetic(operator, operand, depth)
        if operator in FUNCTIONS:
            return self.function(operator, operand, depth)
        raise FilterError(
            f"unknown operator {operator!r}; aggregates {AGGREGATES}, arithmetic {ARITHMETIC}, functions {FUNCTIONS}"
        )

    def aggregate(self, operator: str, operand: Any, depth: int) -> str:
        self.has_aggregate = True
        if operand == "*":
            if operator != "count":
                raise FilterError(f"{operator} needs a field, only count takes '*'")
            return "count(*)"
        inner = self.expression(operand, depth + 1)
        if operator == "count":
            return f"count({inner})"
        if operator == "count_distinct":
            return f"count(DISTINCT {inner})"
        if operator in ("min", "max"):
            # On a real column min/max keep the column's type — they are as useful on a
            # timestamp as on a number. On a field the value is JSONB text, so
            # without the cast max() would answer "7500" for a bank whose biggest total is
            # 20000. A non-numeric property aggregates to NULL, as sum and avg already do;
            # wrap it in lower()/upper() to get the text extreme instead.
            if inner in self.field_expressions:
                return f"{operator}({self.numeric(inner)})"
            return f"{operator}({inner})"
        return f"{operator}({self.numeric(inner)})"

    def arithmetic(self, operator: str, operand: Any, depth: int) -> str:
        if not isinstance(operand, list) or len(operand) < 2:
            raise FilterError(f"{operator} needs a list of at least two operands")
        parts = [self.numeric(self.expression(item, depth + 1)) for item in operand]
        symbol = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}[operator]
        if operator == "divide":
            # Dividing by zero aborts the statement; a ratio with no denominator is NULL.
            return "(" + f" {symbol} ".join(f"NULLIF({part}, 0)" if i else part for i, part in enumerate(parts)) + ")"
        return "(" + f" {symbol} ".join(parts) + ")"

    def function(self, operator: str, operand: Any, depth: int) -> str:
        args = operand if isinstance(operand, list) else [operand]
        if not args:
            raise FilterError(f"{operator} needs an argument")

        if operator == "round":
            value = self.numeric(self.expression(args[0], depth + 1))
            digits = args[1] if len(args) > 1 else 0
            if not isinstance(digits, int) or isinstance(digits, bool) or not 0 <= digits <= 10:
                raise FilterError("round's second argument must be an integer between 0 and 10")
            return f"round({value}, {digits})"
        if operator in ("abs", "ceil", "floor"):
            return f"{operator}({self.numeric(self.expression(args[0], depth + 1))})"
        if operator in ("length", "lower", "upper"):
            return f"{operator}(({self.expression(args[0], depth + 1)})::text)"
        if operator == "coalesce":
            return "COALESCE(" + ", ".join(f"({self.expression(a, depth + 1)})::text" for a in args) + ")"
        if operator == "date_trunc":
            if len(args) != 2 or args[0] not in _UNITS:
                raise FilterError(f"date_trunc takes [unit, field] with unit one of {_UNITS}")
            return f"date_trunc('{args[0]}', ({self.expression(args[1], depth + 1)})::timestamptz)"
        if operator == "extract":
            if len(args) != 2 or args[0] not in _EXTRACT_FIELDS:
                raise FilterError(f"extract takes [field, value] with field one of {_EXTRACT_FIELDS}")
            return f"extract({args[0]} FROM ({self.expression(args[1], depth + 1)})::timestamptz)"
        raise FilterError(f"unknown function {operator!r}")

    # ---- clauses

    def select(self, specs: Any) -> list[str]:
        if not isinstance(specs, list) or not specs:
            raise FilterError("select must be a non-empty list")
        if len(specs) > self.limits.max_select:
            raise FilterError(f"select has {len(specs)} items; at most {self.limits.max_select}")
        parts: list[str] = []
        for index, spec in enumerate(specs):
            sql = self.expression(spec)
            alias = spec.get("as") if isinstance(spec, dict) else None
            if alias is None:
                alias = spec if isinstance(spec, str) else f"column_{index + 1}"
            if not isinstance(alias, str) or not alias.replace("_", "").replace(".", "").isalnum():
                raise FilterError(f"alias {alias!r} must be alphanumeric with _ or .")
            if alias in self.aliases:
                raise FilterError(f"duplicate column name {alias!r}")
            self.aliases[alias] = sql
            # The alias is validated against a charset above, never interpolated from raw
            # input, so quoting it here is safe and lets a caller use dotted names.
            parts.append(f'{sql} AS "{alias}"')
        return parts

    def group_by(self, specs: Any) -> list[str]:
        if specs is None:
            return []
        if not isinstance(specs, list):
            raise FilterError("group_by must be a list")
        if len(specs) > self.limits.max_group_by:
            raise FilterError(f"group_by has {len(specs)} items; at most {self.limits.max_group_by}")
        return [self.expression(spec) for spec in specs]

    def having(self, spec: Any) -> str:
        """HAVING over the aliases select defined — the aggregates, by the name given."""
        if not spec:
            return ""
        if not isinstance(spec, dict):
            raise FilterError("having must be an object of column name -> condition")
        clauses: list[str] = []
        for name, condition in spec.items():
            if name not in self.aliases:
                raise FilterError(f"having refers to {name!r}, which select does not define")
            clauses.append(self.condition(self.aliases[name], condition))
        return " AND ".join(clauses)

    def condition(self, sql: str, condition: Any) -> str:
        """One comparison against an already-compiled expression (HAVING's shape)."""
        if not isinstance(condition, dict):
            return f"{self.numeric(sql)} = {self.bind(condition)}::numeric"
        operators = {"$eq": "=", "$ne": "<>", "$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}
        parts: list[str] = []
        for operator, value in condition.items():
            if operator not in operators:
                raise FilterError(f"having supports {sorted(operators)}, not {operator!r}")
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise FilterError(f"having's {operator} needs a number")
            parts.append(f"{self.numeric(sql)} {operators[operator]} {self.bind(value)}::numeric")
        return "(" + " AND ".join(parts) + ")"

    def order_by(self, specs: Any) -> list[str]:
        if specs is None:
            return []
        if not isinstance(specs, list):
            raise FilterError("order_by must be a list")
        if len(specs) > self.limits.max_order_by:
            raise FilterError(f"order_by has {len(specs)} items; at most {self.limits.max_order_by}")
        parts: list[str] = []
        for spec in specs:
            if isinstance(spec, str):
                name, direction = spec, "asc"
            elif isinstance(spec, dict):
                name = spec.get("field") or spec.get("column")
                direction = str(spec.get("direction", "asc")).lower()
            else:
                raise FilterError("order_by items are a column name or {field, direction}")
            if direction not in ("asc", "desc"):
                raise FilterError(f"order_by direction must be asc or desc, not {direction!r}")
            if not isinstance(name, str) or not name:
                raise FilterError("order_by needs a field")
            # A select alias wins over a column of the same name: ordering by the thing
            # just computed is what a caller means, and it is also what SQL does.
            sql = self.aliases.get(name) or self.field(name)
            keys = [f"{self.numeric(sql)} {direction.upper()} NULLS LAST"] if sql in self.field_expressions else []
            keys.append(f"{sql} {direction.upper()} NULLS LAST")
            parts.extend(keys)
        return parts


def compile_query(body: dict[str, Any], bank_id: str, *, limits: Limits | None = None) -> CompiledQuery:
    """Compile one DSL query for one bank. Raises ``FilterError`` on anything malformed."""
    limits = limits or Limits()
    source = body.get("from", "passages")
    if source not in ("documents", "passages"):
        raise FilterError("from must be 'documents' or 'passages'")

    compiler = _Compiler(source=source, limits=limits)
    # $1 is always the bank: no query this endpoint builds can read another bank's rows.
    compiler.params.append(bank_id)

    select_parts = compiler.select(body.get("select"))
    group_parts = compiler.group_by(body.get("group_by"))

    where = ""
    metadata_filter = body.get("where")
    for name, condition in (metadata_filter or {}).items():
        if isinstance(condition, dict):
            for operator, value in condition.items():
                if operator in ("$in", "$nin") and isinstance(value, list) and len(value) > limits.max_in_values:
                    raise FilterError(f"{name}: {operator} has {len(value)} values; at most {limits.max_in_values}")
    if metadata_filter:
        # The same filter DSL search takes, so one mental model covers both.
        # A document query reads both halves off the document row: same alias twice.
        aliases = ("c", "d") if source == "passages" else ("d", "d")
        where = compile_filters(metadata_filter, compiler.params, aliases)
    if len(compiler.params) > limits.max_parameters:
        raise FilterError(f"query binds more than {limits.max_parameters} values")

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

    if compiler.has_aggregate and not group_parts:
        # One row for the whole bank is a legitimate query ("how many passages?"), so an
        # aggregate with no group_by is allowed — but then nothing else may be selected
        # raw, exactly as SQL requires. Postgres would say it in its own words; this says
        # which column is the problem.
        raw = [part for part in select_parts if "count(" not in part and not _is_aggregate(part)]
        if raw:
            raise FilterError("select mixes plain fields with aggregates but group_by is empty")

    # Documents are the left table in both shapes: a passage query joins its document so a
    # document-level property filters and groups the same way a passage-level one does.
    if source == "passages":
        from_clause = (
            f"FROM {_fq('kb_passages')} c JOIN {_fq('kb_documents')} d ON d.bank_id = c.bank_id AND d.doc_id = c.doc_id"
        )
        bank_predicate = "c.bank_id = $1"
    else:
        # No self-join: the filter compiler is told to read both halves off ``d``. The
        # join that used to give it a ``c`` scanned the whole table a second time.
        from_clause = f"FROM {_fq('kb_documents')} d"
        bank_predicate = "d.bank_id = $1"

    sql = f"SELECT {', '.join(select_parts)} {from_clause} WHERE {bank_predicate}{where}"
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


def _is_aggregate(sql: str) -> bool:
    return any(f"{name}(" in sql for name in ("count", "sum", "avg", "min", "max"))


def _fq(table: str) -> str:
    from ..engine.schema import fq_table

    return fq_table(table)


def json_safe(value: Any) -> Any:
    """Row values as JSON: Decimal and datetime are what Postgres hands back here."""
    import datetime
    import decimal

    if isinstance(value, decimal.Decimal):
        # A float is what a JSON caller can use; the exactness is gone either way once it
        # crosses the wire, and int keeps count()/sum() looking like counts and sums.
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, datetime.datetime | datetime.date):
        return value.isoformat()
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    return value


def explain_json(compiled: CompiledQuery) -> str:
    """The compiled SQL, for the response's ``sql`` field — read-only, for debugging."""
    return json.dumps({"sql": compiled.sql, "parameters": len(compiled.params)})
