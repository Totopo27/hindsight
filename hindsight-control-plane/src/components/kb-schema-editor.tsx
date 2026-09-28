"use client";

// Building a schema. Two views of the same thing: a form, because writing
// {"type": "string", "filterable": true} by hand is how you find out an hour later that
// you typed "filterable" wrong; and the JSON, because a schema of thirty fields is
// faster to paste than to click. The form is what opens, and either view can be saved.
//
// The editor owns its draft and is remounted (key={schemaId}) when another schema is
// selected, so switching never carries one schema's edits into another's.

import * as React from "react";
import { Plus, Trash2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Spinner } from "@/components/ui/spinner";
import type { SchemaField } from "@/components/knowledge-bank-api";

export interface SchemaDraft {
  name: string | null;
  description: string | null;
  document_fields: Record<string, SchemaField>;
  passage_fields: Record<string, SchemaField>;
}

const TYPES: SchemaField["type"][] = [
  "string",
  "integer",
  "number",
  "boolean",
  "date",
  "datetime",
  "array",
  "object",
];

type Level = "document" | "passage";

/** One row of the form. The name is held beside the spec rather than as its key, so
 *  renaming a field does not reorder the list or lose focus on every keystroke. */
interface FieldRow {
  id: number;
  name: string;
  level: Level;
  spec: SchemaField;
}

let nextRowId = 1;

function toRows(draft: SchemaDraft): FieldRow[] {
  const rows: FieldRow[] = [];
  for (const [level, fields] of [
    ["document", draft.document_fields],
    ["passage", draft.passage_fields],
  ] as [Level, Record<string, SchemaField>][]) {
    for (const [name, spec] of Object.entries(fields || {})) {
      rows.push({ id: nextRowId++, name, level, spec: { ...spec, type: spec.type ?? "string" } });
    }
  }
  return rows;
}

function toDraft(rows: FieldRow[], name: string, description: string): SchemaDraft {
  const document_fields: Record<string, SchemaField> = {};
  const passage_fields: Record<string, SchemaField> = {};
  for (const row of rows) {
    const field = row.name.trim();
    if (!field) continue;
    // Drop the flags that are false and the lists that are empty: a schema reads better
    // as what it asks for than as every option spelled out.
    const spec: SchemaField = { type: row.spec.type };
    if (row.spec.description) spec.description = row.spec.description;
    if (row.spec.items) spec.items = row.spec.items;
    if (row.spec.values && row.spec.values.length > 0) spec.values = row.spec.values;
    if (row.spec.filterable) spec.filterable = true;
    if (row.spec.indexed) spec.indexed = true;
    (row.level === "document" ? document_fields : passage_fields)[field] = spec;
  }
  return {
    name: name.trim() || null,
    description: description.trim() || null,
    document_fields,
    passage_fields,
  };
}

export function SchemaEditor({
  initial,
  saving,
  onSave,
}: {
  initial: SchemaDraft;
  saving: boolean;
  onSave: (draft: SchemaDraft) => void;
}) {
  const [view, setView] = React.useState<"form" | "json">("form");
  const [name, setName] = React.useState(initial.name ?? "");
  const [description, setDescription] = React.useState(initial.description ?? "");
  const [rows, setRows] = React.useState<FieldRow[]>(() => toRows(initial));
  const [json, setJson] = React.useState("");
  const [jsonError, setJsonError] = React.useState<string | null>(null);

  const draft = toDraft(rows, name, description);

  const showJson = () => {
    setJson(JSON.stringify(draft, null, 2));
    setJsonError(null);
    setView("json");
  };

  /** Parse the JSON back into the form. A bad document keeps the JSON view open with
   *  the parser's complaint rather than silently discarding what was typed. */
  const parsedJson = (): SchemaDraft | null => {
    try {
      const parsed = JSON.parse(json) as SchemaDraft;
      if (typeof parsed !== "object" || parsed === null) throw new Error("expected an object");
      return {
        name: parsed.name ?? null,
        description: parsed.description ?? null,
        document_fields: parsed.document_fields ?? {},
        passage_fields: parsed.passage_fields ?? {},
      };
    } catch (e) {
      setJsonError((e as Error).message);
      return null;
    }
  };

  const showForm = () => {
    const parsed = parsedJson();
    if (!parsed) return;
    setName(parsed.name ?? "");
    setDescription(parsed.description ?? "");
    setRows(toRows(parsed));
    setJsonError(null);
    setView("form");
  };

  const save = () => {
    if (view === "json") {
      const parsed = parsedJson();
      if (parsed) onSave(parsed);
      return;
    }
    onSave(draft);
  };

  const update = (
    id: number,
    change: Partial<Omit<FieldRow, "spec">> & { spec?: Partial<SchemaField> }
  ) =>
    setRows((prev) =>
      prev.map((row) =>
        row.id === id ? { ...row, ...change, spec: { ...row.spec, ...(change.spec ?? {}) } } : row
      )
    );

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <div className="inline-flex rounded-md border border-border bg-muted/40 p-0.5">
          {(["form", "json"] as const).map((option) => (
            <button
              key={option}
              type="button"
              onClick={() => (option === "json" ? showJson() : showForm())}
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
        <div className="flex-1" />
        <Button size="sm" onClick={save} disabled={saving}>
          {saving ? <Spinner size="sm" /> : "Save schema"}
        </Button>
      </div>

      {view === "json" ? (
        <Textarea
          rows={20}
          className="font-mono text-xs"
          value={json}
          spellCheck={false}
          onChange={(e) => {
            setJson(e.target.value);
            setJsonError(null);
          }}
        />
      ) : (
        <div className="space-y-4">
          <div className="grid gap-3 md:grid-cols-2">
            <div>
              <label className="text-xs font-medium text-muted-foreground">Name</label>
              <Input
                placeholder="Supplier contract"
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
            </div>
            <div>
              <label className="text-xs font-medium text-muted-foreground">Description</label>
              <Input
                placeholder="What this kind of document is"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
              />
            </div>
          </div>

          {/* The pane this sits in is a right-hand column, so the columns are given a
              floor and the frame scrolls rather than squeezing "Description" to nothing. */}
          <div className="rounded-lg border border-border overflow-x-auto">
            <table className="w-full min-w-[860px] text-sm">
              <thead className="bg-muted/40 text-xs text-muted-foreground">
                <tr>
                  <th className="text-left font-medium px-3 py-2 w-48">Field</th>
                  <th className="text-left font-medium px-3 py-2 w-32">Level</th>
                  <th className="text-left font-medium px-3 py-2 w-32">Type</th>
                  <th className="text-left font-medium px-3 py-2">Description</th>
                  <th className="text-left font-medium px-3 py-2 w-44">Values</th>
                  <th className="text-left font-medium px-3 py-2 w-28">Flags</th>
                  <th className="w-10" />
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {rows.length === 0 && (
                  <tr>
                    <td colSpan={7} className="px-3 py-6 text-center text-muted-foreground text-xs">
                      No fields yet. Add one to give this kind of document something to filter on.
                    </td>
                  </tr>
                )}
                {rows.map((row) => (
                  <tr key={row.id}>
                    <td className="px-3 py-2">
                      <Input
                        className="h-8 font-mono text-xs"
                        placeholder="vendor"
                        value={row.name}
                        onChange={(e) => update(row.id, { name: e.target.value })}
                      />
                    </td>
                    <td className="px-3 py-2">
                      <select
                        className="h-8 w-full rounded-md border border-input bg-background px-2 text-xs"
                        value={row.level}
                        onChange={(e) => update(row.id, { level: e.target.value as Level })}
                      >
                        <option value="document">Document</option>
                        <option value="passage">Passage</option>
                      </select>
                    </td>
                    <td className="px-3 py-2">
                      <select
                        className="h-8 w-full rounded-md border border-input bg-background px-2 text-xs"
                        value={row.spec.type}
                        onChange={(e) =>
                          update(row.id, { spec: { type: e.target.value as SchemaField["type"] } })
                        }
                      >
                        {TYPES.map((type) => (
                          <option key={type} value={type}>
                            {type}
                          </option>
                        ))}
                      </select>
                      {row.spec.type === "array" && (
                        <select
                          className="mt-1 h-8 w-full rounded-md border border-input bg-background px-2 text-xs"
                          value={row.spec.items ?? "string"}
                          onChange={(e) => update(row.id, { spec: { items: e.target.value } })}
                        >
                          {TYPES.filter((t) => t !== "array" && t !== "object").map((type) => (
                            <option key={type} value={type}>
                              of {type}
                            </option>
                          ))}
                        </select>
                      )}
                    </td>
                    <td className="px-3 py-2">
                      <Input
                        className="h-8 text-xs"
                        placeholder="What this field is, in the document's words"
                        value={row.spec.description ?? ""}
                        onChange={(e) => update(row.id, { spec: { description: e.target.value } })}
                      />
                    </td>
                    <td className="px-3 py-2">
                      <Input
                        className="h-8 text-xs"
                        placeholder="draft, signed, expired"
                        value={(row.spec.values ?? []).join(", ")}
                        onChange={(e) =>
                          update(row.id, {
                            // A fixed list of values IS the classification: the model is
                            // given these and nothing else to choose from.
                            spec: {
                              values: e.target.value
                                .split(",")
                                .map((v) => v.trim())
                                .filter(Boolean),
                            },
                          })
                        }
                      />
                    </td>
                    <td className="px-3 py-2">
                      <label className="flex items-center gap-1 text-xs">
                        <input
                          type="checkbox"
                          checked={Boolean(row.spec.filterable)}
                          onChange={(e) =>
                            update(row.id, { spec: { filterable: e.target.checked } })
                          }
                        />
                        filterable
                      </label>
                      <label className="flex items-center gap-1 text-xs">
                        <input
                          type="checkbox"
                          checked={Boolean(row.spec.indexed)}
                          onChange={(e) => update(row.id, { spec: { indexed: e.target.checked } })}
                        />
                        indexed
                      </label>
                    </td>
                    <td className="px-3 py-2">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-8 w-8 p-0"
                        title="Remove field"
                        onClick={() => setRows((prev) => prev.filter((r) => r.id !== row.id))}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <Button
            variant="outline"
            size="sm"
            onClick={() =>
              setRows((prev) => [
                ...prev,
                { id: nextRowId++, name: "", level: "document", spec: { type: "string" } },
              ])
            }
          >
            <Plus className="h-4 w-4 mr-1" /> Add field
          </Button>

          <p className="text-xs text-muted-foreground">
            <strong>filterable</strong> lets a search filter on the field. <strong>indexed</strong>{" "}
            puts its value into the passage&apos;s embedding, so the value itself is searchable.{" "}
            <strong>Values</strong> turns the field into a classification: the model picks one of
            them or leaves it out.
          </p>
        </div>
      )}
    </div>
  );
}
