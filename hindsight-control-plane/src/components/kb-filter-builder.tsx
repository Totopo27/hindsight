"use client";

// Building a field filter for a knowledge-bank search. Same two views as the schema
// editor: a form that only offers fields the schema marked filterable (a filter on any
// other field is refused by the API, and finding that out from a 400 is no way to learn
// it), and the JSON, which is what the API actually takes.

import * as React from "react";
import { Plus, Trash2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import type { SchemaField } from "@/components/knowledge-bank-api";

export interface FilterableField {
  name: string;
  /** Which schema defines it — two schemas may each define "total", and they are not
   *  the same field to a reader even though the filter matches on the name. */
  schemaId: string;
  level: "document" | "passage";
  spec: SchemaField;
}

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

interface Condition {
  id: number;
  field: string;
  operator: Operator;
  value: string;
}

let nextId = 1;

/** A condition's value, typed the way the field is: a number field filtered with the
 *  string "1000" matches nothing, which looks like "there is no such document". */
function typedValue(raw: string, spec: SchemaField | undefined, operator: Operator): unknown {
  if (operator === "$exists") return raw !== "false";
  const one = (text: string): unknown => {
    const trimmed = text.trim();
    if (spec?.type === "integer" || spec?.type === "number") {
      const parsed = Number(trimmed);
      return Number.isNaN(parsed) ? trimmed : parsed;
    }
    if (spec?.type === "boolean") return trimmed === "true";
    return trimmed;
  };
  if (operator === "$in") {
    return raw
      .split(",")
      .map((part) => one(part))
      .filter((v) => v !== "");
  }
  return one(raw);
}

function toFilter(conditions: Condition[], fields: FilterableField[]): Record<string, unknown> {
  const filter: Record<string, unknown> = {};
  for (const condition of conditions) {
    if (!condition.field) continue;
    const spec = fields.find((f) => f.name === condition.field)?.spec;
    const value = typedValue(condition.value, spec, condition.operator);
    if (condition.operator === "eq") {
      filter[condition.field] = value;
    } else {
      // Two conditions on one field merge into one object: {"total": {"$gte": 1, "$lt": 9}}.
      const existing = filter[condition.field];
      const previous = typeof existing === "object" && existing !== null ? existing : {};
      filter[condition.field] = { ...previous, [condition.operator]: value };
    }
  }
  return filter;
}

export function FilterBuilder({
  fields,
  schemas,
  schemaId,
  onSchemaChange,
  onChange,
}: {
  fields: FilterableField[];
  /** The bank's schema ids, for scoping the search to one kind of document. */
  schemas: string[];
  schemaId: string | null;
  onSchemaChange: (schemaId: string | null) => void;
  /** The filter as the API takes it, or null when nothing is set. */
  onChange: (filter: Record<string, unknown> | null) => void;
}) {
  const [view, setView] = React.useState<"form" | "json">("form");
  const [conditions, setConditions] = React.useState<Condition[]>([]);
  const [json, setJson] = React.useState("");
  const [jsonError, setJsonError] = React.useState<string | null>(null);

  // The new list is computed from the current one and both the state and the parent are
  // told about it here — never from inside a setState updater, which React may run
  // during a render, where setting a parent's state is a warning and a lost update.
  const apply = (next: Condition[]) => {
    setConditions(next);
    const filter = toFilter(next, fields);
    onChange(Object.keys(filter).length > 0 ? filter : null);
  };

  const update = (id: number, change: Partial<Condition>) =>
    apply(conditions.map((c) => (c.id === id ? { ...c, ...change } : c)));

  const remove = (id: number) => apply(conditions.filter((c) => c.id !== id));

  const showJson = () => {
    setJson(JSON.stringify(toFilter(conditions, fields), null, 2));
    setJsonError(null);
    setView("json");
  };

  const editJson = (text: string) => {
    setJson(text);
    if (!text.trim()) {
      setJsonError(null);
      onChange(null);
      return;
    }
    try {
      const parsed = JSON.parse(text);
      setJsonError(null);
      onChange(parsed);
    } catch (e) {
      setJsonError((e as Error).message);
    }
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-muted-foreground">Field filter</span>
        <div className="inline-flex rounded-md border border-border bg-muted/40 p-0.5">
          {(["form", "json"] as const).map((option) => (
            <button
              key={option}
              type="button"
              onClick={() => {
                if (option === "json") showJson();
                else {
                  // The form cannot express every filter the JSON can, so it keeps the
                  // conditions it had rather than trying to read arbitrary JSON back.
                  setView("form");
                  apply(conditions);
                }
              }}
              className={`px-3 py-1 text-xs font-medium rounded-[5px] transition-colors ${
                view === option
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {option === "form" ? "Form" : "JSON"}
            </button>
          ))}
        </div>
        {jsonError && <span className="text-xs text-destructive">{jsonError}</span>}
      </div>

      {view === "form" && (
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground w-20">Schema</span>
          <select
            className="h-9 w-56 rounded-md border border-input bg-background px-2 text-sm"
            value={schemaId ?? ""}
            onChange={(e) => onSchemaChange(e.target.value || null)}
          >
            <option value="">Any schema</option>
            {schemas.map((schema) => (
              <option key={schema} value={schema}>
                {schema}
              </option>
            ))}
            {/* Not a missing filter: the documents no schema applied to are the ones
                nothing was extracted from, and finding them is the point. */}
            <option value="none">No schema</option>
          </select>
        </div>
      )}

      {view === "json" ? (
        <Textarea
          rows={6}
          className="font-mono text-xs"
          spellCheck={false}
          placeholder={'{"vendor": "Acme", "total": {"$gte": 1000}}'}
          value={json}
          onChange={(e) => editJson(e.target.value)}
        />
      ) : (
        <div className="space-y-2">
          {conditions.map((condition) => {
            const spec = fields.find((f) => f.name === condition.field)?.spec;
            return (
              <div key={condition.id} className="flex items-center gap-2">
                <select
                  className="h-9 w-56 rounded-md border border-input bg-background px-2 text-sm"
                  value={condition.field}
                  onChange={(e) => update(condition.id, { field: e.target.value })}
                >
                  <option value="">Field…</option>
                  {Array.from(new Set(fields.map((f) => f.schemaId))).map((schema) => (
                    <optgroup key={schema} label={schema}>
                      {fields
                        .filter((f) => f.schemaId === schema)
                        .map((field) => (
                          <option key={`${schema}.${field.name}`} value={field.name}>
                            {field.name}
                            {field.level === "passage" ? " (passage)" : ""}
                          </option>
                        ))}
                    </optgroup>
                  ))}
                </select>
                {spec && (
                  <span className="shrink-0 rounded bg-muted px-1.5 py-0.5 text-[11px] text-muted-foreground">
                    {spec.type}
                    {spec.type === "array" && spec.items ? ` of ${spec.items}` : ""}
                  </span>
                )}
                <select
                  className="h-9 w-32 rounded-md border border-input bg-background px-2 text-sm"
                  value={condition.operator}
                  onChange={(e) => update(condition.id, { operator: e.target.value as Operator })}
                >
                  {OPERATORS.map((operator) => (
                    <option key={operator.id} value={operator.id}>
                      {operator.label}
                    </option>
                  ))}
                </select>
                {condition.operator === "$exists" ? (
                  <select
                    className="h-9 w-32 rounded-md border border-input bg-background px-2 text-sm"
                    value={condition.value === "false" ? "false" : "true"}
                    onChange={(e) => update(condition.id, { value: e.target.value })}
                  >
                    <option value="true">yes</option>
                    <option value="false">no</option>
                  </select>
                ) : spec?.values && spec.values.length > 0 && condition.operator !== "$in" ? (
                  <select
                    className="h-9 flex-1 rounded-md border border-input bg-background px-2 text-sm"
                    value={condition.value}
                    onChange={(e) => update(condition.id, { value: e.target.value })}
                  >
                    <option value="">Value…</option>
                    {spec.values.map((value) => (
                      <option key={String(value)} value={String(value)}>
                        {String(value)}
                      </option>
                    ))}
                  </select>
                ) : (
                  <Input
                    className="h-9 flex-1"
                    placeholder={condition.operator === "$in" ? "a, b, c" : "Value"}
                    value={condition.value}
                    onChange={(e) => update(condition.id, { value: e.target.value })}
                  />
                )}
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-9 w-9 p-0"
                  title="Remove condition"
                  onClick={() => remove(condition.id)}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            );
          })}
          <div className="flex items-center gap-3">
            <Button
              variant="outline"
              size="sm"
              disabled={fields.length === 0}
              onClick={() =>
                apply([...conditions, { id: nextId++, field: "", operator: "eq", value: "" }])
              }
            >
              <Plus className="h-4 w-4 mr-1" /> Add condition
            </Button>
            {fields.length === 0 && (
              <span className="text-xs text-muted-foreground">
                No filterable fields yet — mark a schema field filterable to filter on it.
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
