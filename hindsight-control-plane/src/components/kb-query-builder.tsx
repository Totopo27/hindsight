"use client";

// Building a record query. The language is small — select, where, group by, order by,
// limit — but writing it as JSON means knowing the spelling of six keys and the shape of
// an aggregate before you can ask "how many contracts per vendor". The builder is those
// five parts as rows; the JSON view is the same query, for the shapes the rows cannot say
// (joins, arithmetic, having) and for pasting one in.

import * as React from "react";
import { Plus, Trash2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";

/** The aggregates the DSL takes, plus the plain field. */
const AGGREGATES = [
  { id: "field", label: "value" },
  { id: "count", label: "count" },
  { id: "count_distinct", label: "count distinct" },
  { id: "sum", label: "sum" },
  { id: "avg", label: "average" },
  { id: "min", label: "min" },
  { id: "max", label: "max" },
] as const;

type Aggregate = (typeof AGGREGATES)[number]["id"];

const OPERATORS = [
  { id: "eq", label: "is" },
  { id: "$ne", label: "is not" },
  { id: "$gte", label: "≥" },
  { id: "$gt", label: ">" },
  { id: "$lte", label: "≤" },
  { id: "$lt", label: "<" },
  { id: "$in", label: "one of" },
  { id: "$contains", label: "contains" },
  { id: "$exists", label: "exists" },
] as const;

type Operator = (typeof OPERATORS)[number]["id"];

interface SelectRow {
  id: number;
  aggregate: Aggregate;
  /** "*" is only meaningful for count. */
  field: string;
  as: string;
}

interface WhereRow {
  id: number;
  field: string;
  operator: Operator;
  value: string;
}

export interface QueryField {
  name: string;
  /** Used to type a value: a number field filtered with "1000" matches nothing. */
  type?: string;
  values?: (string | number | boolean)[];
}

let nextId = 1;

function typedValue(raw: string, type: string | undefined, operator: Operator): unknown {
  if (operator === "$exists") return raw !== "false";
  const one = (text: string): unknown => {
    const trimmed = text.trim();
    if (type === "integer" || type === "number") {
      const parsed = Number(trimmed);
      return Number.isNaN(parsed) ? trimmed : parsed;
    }
    if (type === "boolean") return trimmed === "true";
    return trimmed;
  };
  if (operator === "$in")
    return raw
      .split(",")
      .map(one)
      .filter((v) => v !== "");
  return one(raw);
}

/** The rows as the DSL takes them. */
export function buildQuery(
  selects: SelectRow[],
  wheres: WhereRow[],
  fields: QueryField[],
  groupBy: string[],
  orderBy: { field: string; direction: "asc" | "desc" } | null,
  limit: number
): Record<string, unknown> {
  const select = selects
    .filter((row) => row.aggregate === "count" || row.field)
    .map((row) => {
      const term: Record<string, unknown> =
        row.aggregate === "field"
          ? { field: row.field }
          : { [row.aggregate]: row.aggregate === "count" && !row.field ? "*" : row.field };
      // Always aliased. An unaliased column comes back as "column_2", which is not a
      // heading anyone can read — and a result whose column is named record_id is what
      // makes its rows open as records.
      term.as =
        row.as.trim() ||
        (row.aggregate === "field"
          ? row.field
          : row.field
            ? `${row.aggregate}_${row.field}`
            : row.aggregate);
      return term;
    });

  const where: Record<string, unknown> = {};
  for (const row of wheres) {
    if (!row.field) continue;
    const type = fields.find((f) => f.name === row.field)?.type;
    const value = typedValue(row.value, type, row.operator);
    if (row.operator === "eq") where[row.field] = value;
    else {
      const existing = where[row.field];
      const previous = typeof existing === "object" && existing !== null ? existing : {};
      where[row.field] = { ...previous, [row.operator]: value };
    }
  }

  const query: Record<string, unknown> = { select: select.length > 0 ? select : [{ count: "*" }] };
  if (Object.keys(where).length > 0) query.where = where;
  if (groupBy.length > 0) query.group_by = groupBy.map((name) => ({ field: name }));
  if (orderBy?.field) {
    query.order_by = [{ field: orderBy.field, direction: orderBy.direction }];
  }
  query.limit = limit;
  return query;
}

export function QueryBuilder({
  fields,
  onChange,
}: {
  /** The collection's own fields, plus the record columns the DSL always has. */
  fields: QueryField[];
  /** The query as the API takes it; called on every edit and on every view switch. */
  onChange: (query: Record<string, unknown>) => void;
}) {
  const [view, setView] = React.useState<"builder" | "json">("builder");
  const [selects, setSelects] = React.useState<SelectRow[]>([
    { id: nextId++, aggregate: "field", field: "record_id", as: "" },
  ]);
  const [wheres, setWheres] = React.useState<WhereRow[]>([]);
  const [groupBy, setGroupBy] = React.useState<string[]>([]);
  const [orderBy, setOrderBy] = React.useState<{
    field: string;
    direction: "asc" | "desc";
  } | null>(null);
  const [limit, setLimit] = React.useState(50);
  const [json, setJson] = React.useState("");
  const [jsonError, setJsonError] = React.useState<string | null>(null);

  const built = buildQuery(selects, wheres, fields, groupBy, orderBy, limit);

  // The parent always holds the query the visible view describes, so Run never sends
  // something other than what is on screen.
  React.useEffect(() => {
    if (view === "builder") onChange(built);
  }, [view, JSON.stringify(built)]);

  const showJson = () => {
    setJson(JSON.stringify(built, null, 2));
    setJsonError(null);
    setView("json");
    onChange(built);
  };

  const editJson = (text: string) => {
    setJson(text);
    try {
      const parsed = JSON.parse(text);
      setJsonError(null);
      onChange(parsed);
    } catch (e) {
      setJsonError((e as Error).message);
    }
  };

  const selectClass = "h-9 rounded-md border border-input bg-background px-2 text-sm";

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <div className="inline-flex rounded-md border border-border bg-muted/40 p-0.5">
          {(["builder", "json"] as const).map((option) => (
            <button
              key={option}
              type="button"
              onClick={() => (option === "json" ? showJson() : setView("builder"))}
              className={`px-3 py-1 text-xs font-medium rounded-[5px] transition-colors ${
                view === option
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {option === "builder" ? "Builder" : "JSON"}
            </button>
          ))}
        </div>
        {jsonError && <span className="text-xs text-destructive">{jsonError}</span>}
      </div>

      {view === "json" ? (
        <Textarea
          rows={12}
          className="font-mono text-xs"
          spellCheck={false}
          value={json}
          onChange={(e) => editJson(e.target.value)}
        />
      ) : (
        <div className="space-y-4">
          <div className="space-y-2">
            <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Select
            </div>
            {selects.map((row) => (
              <div key={row.id} className="flex flex-wrap items-center gap-2">
                <select
                  className={`${selectClass} w-36`}
                  value={row.aggregate}
                  onChange={(e) =>
                    setSelects((prev) =>
                      prev.map((r) =>
                        r.id === row.id ? { ...r, aggregate: e.target.value as Aggregate } : r
                      )
                    )
                  }
                >
                  {AGGREGATES.map((option) => (
                    <option key={option.id} value={option.id}>
                      {option.label}
                    </option>
                  ))}
                </select>
                <select
                  className={`${selectClass} w-48`}
                  value={row.field}
                  onChange={(e) =>
                    setSelects((prev) =>
                      prev.map((r) => (r.id === row.id ? { ...r, field: e.target.value } : r))
                    )
                  }
                >
                  {/* count is the one aggregate that means something without a field. */}
                  {row.aggregate === "count" && <option value="">every record</option>}
                  {fields.map((field) => (
                    <option key={field.name} value={field.name}>
                      {field.name}
                    </option>
                  ))}
                </select>
                <Input
                  className="h-9 w-40"
                  placeholder="as…"
                  value={row.as}
                  onChange={(e) =>
                    setSelects((prev) =>
                      prev.map((r) => (r.id === row.id ? { ...r, as: e.target.value } : r))
                    )
                  }
                />
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-9 w-9 p-0"
                  title="Remove"
                  onClick={() => setSelects((prev) => prev.filter((r) => r.id !== row.id))}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            ))}
            <Button
              variant="outline"
              size="sm"
              onClick={() =>
                setSelects((prev) => [
                  ...prev,
                  { id: nextId++, aggregate: "count", field: "", as: "" },
                ])
              }
            >
              <Plus className="h-4 w-4 mr-1" /> Add column
            </Button>
          </div>

          <div className="space-y-2">
            <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Where
            </div>
            {wheres.map((row) => {
              const field = fields.find((f) => f.name === row.field);
              return (
                <div key={row.id} className="flex flex-wrap items-center gap-2">
                  <select
                    className={`${selectClass} w-48`}
                    value={row.field}
                    onChange={(e) =>
                      setWheres((prev) =>
                        prev.map((r) => (r.id === row.id ? { ...r, field: e.target.value } : r))
                      )
                    }
                  >
                    <option value="">Field…</option>
                    {fields.map((f) => (
                      <option key={f.name} value={f.name}>
                        {f.name}
                      </option>
                    ))}
                  </select>
                  <select
                    className={`${selectClass} w-32`}
                    value={row.operator}
                    onChange={(e) =>
                      setWheres((prev) =>
                        prev.map((r) =>
                          r.id === row.id ? { ...r, operator: e.target.value as Operator } : r
                        )
                      )
                    }
                  >
                    {OPERATORS.map((option) => (
                      <option key={option.id} value={option.id}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                  {field?.values && field.values.length > 0 && row.operator !== "$in" ? (
                    <select
                      className={`${selectClass} flex-1`}
                      value={row.value}
                      onChange={(e) =>
                        setWheres((prev) =>
                          prev.map((r) => (r.id === row.id ? { ...r, value: e.target.value } : r))
                        )
                      }
                    >
                      <option value="">Value…</option>
                      {field.values.map((value) => (
                        <option key={String(value)} value={String(value)}>
                          {String(value)}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <Input
                      className="h-9 flex-1"
                      placeholder={row.operator === "$in" ? "a, b, c" : "Value"}
                      value={row.value}
                      onChange={(e) =>
                        setWheres((prev) =>
                          prev.map((r) => (r.id === row.id ? { ...r, value: e.target.value } : r))
                        )
                      }
                    />
                  )}
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-9 w-9 p-0"
                    title="Remove"
                    onClick={() => setWheres((prev) => prev.filter((r) => r.id !== row.id))}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              );
            })}
            <Button
              variant="outline"
              size="sm"
              onClick={() =>
                setWheres((prev) => [
                  ...prev,
                  { id: nextId++, field: "", operator: "eq", value: "" },
                ])
              }
            >
              <Plus className="h-4 w-4 mr-1" /> Add condition
            </Button>
          </div>

          <div className="grid gap-3 sm:grid-cols-3">
            <div>
              <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-1">
                Group by
              </div>
              <select
                className={`${selectClass} w-full`}
                value={groupBy[0] ?? ""}
                onChange={(e) => setGroupBy(e.target.value ? [e.target.value] : [])}
              >
                <option value="">Nothing</option>
                {fields.map((field) => (
                  <option key={field.name} value={field.name}>
                    {field.name}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-1">
                Order by
              </div>
              <div className="flex gap-2">
                <select
                  className={`${selectClass} flex-1`}
                  value={orderBy?.field ?? ""}
                  onChange={(e) =>
                    setOrderBy(
                      e.target.value
                        ? { field: e.target.value, direction: orderBy?.direction ?? "desc" }
                        : null
                    )
                  }
                >
                  <option value="">Nothing</option>
                  {fields.map((field) => (
                    <option key={field.name} value={field.name}>
                      {field.name}
                    </option>
                  ))}
                  {/* An aggregate is ordered by its alias, which only the select knows. */}
                  {selects
                    .filter((row) => row.as.trim())
                    .map((row) => (
                      <option key={`as-${row.id}`} value={row.as.trim()}>
                        {row.as.trim()}
                      </option>
                    ))}
                </select>
                <select
                  className={`${selectClass} w-28`}
                  value={orderBy?.direction ?? "desc"}
                  disabled={!orderBy}
                  onChange={(e) =>
                    setOrderBy((prev) =>
                      prev ? { ...prev, direction: e.target.value as "asc" | "desc" } : prev
                    )
                  }
                >
                  <option value="desc">descending</option>
                  <option value="asc">ascending</option>
                </select>
              </div>
            </div>
            <div>
              <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-1">
                Limit
              </div>
              <Input
                type="number"
                min={1}
                max={1000}
                className="h-9"
                value={limit}
                onChange={(e) => setLimit(Math.max(1, Number(e.target.value) || 1))}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
