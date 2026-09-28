"use client";

// Editing the collections of a bank — all of them, in one sitting.
//
// A collection rarely changes alone: a relationship field points at another collection,
// so adding "contracts.vendor" means "vendors" has to exist, and splitting one
// collection into two is three edits that are only correct together. Saving them one at
// a time through separate dialogs means passing through states the API refuses (a
// relationship to a collection that is not there yet) and states the data refuses (half
// a rename). So the whole set is edited as a draft and applied at the end: nothing is
// sent until Apply, and Apply sends the writes in an order where each one is valid.

import * as React from "react";
import { toast } from "sonner";
import { Plus, Sparkles, Trash2, Undo2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  kbFetch,
  type CollectionProposal,
  type KnowledgeCollection,
} from "@/components/knowledge-bank-api";
import { cn } from "@/lib/utils";

const FIELD_TYPES = [
  "string",
  "integer",
  "number",
  "boolean",
  "date",
  "datetime",
  "array",
] as const;

interface DraftField {
  key: number;
  name: string;
  /** A scalar type, or "→" plus a collection id for a relationship. */
  type: string;
  /** Set when the field points at another collection. */
  collection: string;
  values: string;
  description: string;
}

interface DraftCollection {
  key: number;
  id: string;
  /** The id it had when the dialog opened; absent for one added here. */
  originalId: string | null;
  name: string;
  description: string;
  identity: string;
  deriveOnWrite: boolean;
  fields: DraftField[];
  deleted: boolean;
}

let nextKey = 1;

function toDraft(collection: KnowledgeCollection): DraftCollection {
  return {
    key: nextKey++,
    id: collection.collection_id,
    originalId: collection.collection_id,
    name: collection.name ?? "",
    description: collection.description ?? "",
    identity: collection.identity ?? "",
    deriveOnWrite: collection.derive_on_write ?? false,
    fields: Object.entries(collection.fields ?? {}).map(([name, spec]) => ({
      key: nextKey++,
      name,
      type: spec.collection ? "relationship" : (spec.type ?? "string"),
      collection: spec.collection ?? "",
      values: Array.isArray(spec.values) ? spec.values.map(String).join(", ") : "",
      description: (spec as { description?: string }).description ?? "",
    })),
    deleted: false,
  };
}

/** The body the API takes for one collection. */
function toBody(draft: DraftCollection): Record<string, unknown> {
  const fields: Record<string, unknown> = {};
  for (const field of draft.fields) {
    const name = field.name.trim();
    if (!name) continue;
    if (field.type === "relationship") {
      fields[name] = {
        collection: field.collection,
        ...(field.description ? { description: field.description } : {}),
      };
      continue;
    }
    const values = field.values
      .split(",")
      .map((v) => v.trim())
      .filter(Boolean);
    fields[name] = {
      type: field.type,
      ...(field.description ? { description: field.description } : {}),
      ...(values.length > 0 ? { values } : {}),
    };
  }
  return {
    name: draft.name.trim() || null,
    description: draft.description.trim() || null,
    identity: draft.identity || null,
    derive_on_write: draft.deriveOnWrite,
    fields,
  };
}

function sameAsSaved(draft: DraftCollection, saved: KnowledgeCollection | undefined): boolean {
  if (!saved) return false;
  if (draft.id !== saved.collection_id) return false;
  return JSON.stringify(toBody(draft)) === JSON.stringify(toBody(toDraft(saved)));
}

/** What a draft would do to the bank, in the order it has to happen. */
interface Change {
  kind: "create" | "update" | "delete";
  id: string;
  draft?: DraftCollection;
}

export function pendingChanges(drafts: DraftCollection[], saved: KnowledgeCollection[]): Change[] {
  const savedById = new Map(saved.map((c) => [c.collection_id, c]));
  const writes: Change[] = [];
  const deletes: Change[] = [];
  const keptIds = new Set<string>();

  for (const draft of drafts) {
    if (draft.deleted) {
      if (draft.originalId) deletes.push({ kind: "delete", id: draft.originalId });
      continue;
    }
    if (!draft.id.trim()) continue;
    keptIds.add(draft.id);
    // Renaming is a create plus a delete: the id is the record's key and the API has no
    // rename, so the old one is removed once the new one exists.
    if (draft.originalId && draft.originalId !== draft.id) {
      writes.push({ kind: "create", id: draft.id, draft });
      deletes.push({ kind: "delete", id: draft.originalId });
      continue;
    }
    if (!draft.originalId) writes.push({ kind: "create", id: draft.id, draft });
    else if (!sameAsSaved(draft, savedById.get(draft.originalId)))
      writes.push({ kind: "update", id: draft.id, draft });
  }

  // A relationship's target must exist when the referrer is written, so a collection
  // nothing points at goes first and its referrers follow.
  const targetsOf = (change: Change) =>
    (change.draft?.fields ?? [])
      .filter((f) => f.type === "relationship" && f.collection && f.collection !== change.id)
      .map((f) => f.collection);
  const ordered: Change[] = [];
  const placed = new Set<string>();
  let remaining = [...writes];
  while (remaining.length > 0) {
    const ready = remaining.filter((change) =>
      targetsOf(change).every((target) => placed.has(target) || !keptIds.has(target))
    );
    // A cycle (two collections pointing at each other) cannot be ordered; write them in
    // the order given and let the API refuse if it must.
    const batch = ready.length > 0 ? ready : remaining;
    for (const change of batch) {
      ordered.push(change);
      placed.add(change.id);
    }
    remaining = remaining.filter((change) => !batch.includes(change));
  }
  // Deletes last: a target removed before its referrer is rewritten breaks the referrer.
  return [...ordered, ...deletes];
}

export function CollectionsEditor({
  kbId,
  open,
  collections,
  onOpenChange,
  onApplied,
}: {
  kbId: string;
  open: boolean;
  collections: KnowledgeCollection[];
  onOpenChange: (open: boolean) => void;
  onApplied: () => Promise<void> | void;
}) {
  const [drafts, setDrafts] = React.useState<DraftCollection[]>([]);
  const [selectedKey, setSelectedKey] = React.useState<number | null>(null);
  const [applying, setApplying] = React.useState(false);
  const [proposing, setProposing] = React.useState(false);
  const [instruction, setInstruction] = React.useState("");

  // The draft is taken when the dialog opens, so edits are never overwritten by a
  // refresh happening behind it.
  React.useEffect(() => {
    if (!open) return;
    const next = collections.map(toDraft);
    setDrafts(next);
    setSelectedKey(next[0]?.key ?? null);
  }, [open, collections]);

  const selected = drafts.find((d) => d.key === selectedKey) ?? null;
  const changes = pendingChanges(drafts, collections);

  const patch = (key: number, change: Partial<DraftCollection>) =>
    setDrafts((prev) => prev.map((d) => (d.key === key ? { ...d, ...change } : d)));

  const patchField = (key: number, fieldKey: number, change: Partial<DraftField>) =>
    setDrafts((prev) =>
      prev.map((d) =>
        d.key === key
          ? { ...d, fields: d.fields.map((f) => (f.key === fieldKey ? { ...f, ...change } : f)) }
          : d
      )
    );

  const addCollection = () => {
    const draft: DraftCollection = {
      key: nextKey++,
      id: "",
      originalId: null,
      name: "",
      description: "",
      identity: "",
      deriveOnWrite: false,
      fields: [],
      deleted: false,
    };
    setDrafts((prev) => [...prev, draft]);
    setSelectedKey(draft.key);
  };

  /** Ask the model what this corpus deserves and stage its answer as edits.
   *
   *  The proposal lands in the same draft as everything else and is applied by the same
   *  Apply — the model cannot write to the bank, and a suggestion nobody reads changes
   *  nothing. */
  const propose = async () => {
    setProposing(true);
    try {
      const response = await kbFetch<{
        proposals: CollectionProposal[];
        documents_read: number;
        warnings: string[];
      }>(`/${encodeURIComponent(kbId)}/collections/propose`, {
        method: "POST",
        body: { instruction: instruction.trim() || undefined, sample_documents: 12 },
      });
      if (response.proposals.length === 0) {
        toast.info(`Read ${response.documents_read} documents and had nothing to suggest.`);
        return;
      }
      setDrafts((prev) => {
        const next = [...prev];
        for (const proposal of response.proposals) {
          const index = next.findIndex((d) => d.id === proposal.collection_id);
          if (proposal.action === "delete") {
            if (index >= 0) next[index] = { ...next[index], deleted: true };
            continue;
          }
          const definition = proposal.definition;
          if (!definition) continue;
          const staged: DraftCollection = {
            key: index >= 0 ? next[index].key : nextKey++,
            id: proposal.collection_id,
            originalId: index >= 0 ? next[index].originalId : null,
            name: definition.name ?? "",
            description: definition.description ?? "",
            identity: definition.identity ?? "",
            deriveOnWrite: index >= 0 ? next[index].deriveOnWrite : false,
            deleted: false,
            fields: Object.entries(definition.fields ?? {}).map(([name, spec]) => ({
              key: nextKey++,
              name,
              type: spec.collection ? "relationship" : (spec.type ?? "string"),
              collection: spec.collection ?? "",
              values: Array.isArray(spec.values) ? spec.values.map(String).join(", ") : "",
              description: spec.description ?? "",
            })),
          };
          if (index >= 0) next[index] = staged;
          else next.push(staged);
        }
        return next;
      });
      for (const warning of response.warnings) toast.warning(warning);
      toast.success(
        `Staged ${response.proposals.length} suggestion${response.proposals.length === 1 ? "" : "s"} from ${response.documents_read} documents. Nothing is written until Apply.`
      );
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setProposing(false);
    }
  };

  /** What is wrong with the draft, in words, or nothing. */
  const problems = (): string[] => {
    const out: string[] = [];
    const live = drafts.filter((d) => !d.deleted);
    const ids = live.map((d) => d.id.trim());
    for (const draft of live) {
      const id = draft.id.trim();
      if (!id) out.push("A collection needs an id.");
      if (id && ids.filter((other) => other === id).length > 1)
        out.push(`Two collections are called ${id}.`);
      for (const field of draft.fields) {
        if (field.type !== "relationship") continue;
        if (!field.collection) out.push(`${id}.${field.name} points at nothing.`);
        else if (!ids.includes(field.collection))
          out.push(`${id}.${field.name} points at ${field.collection}, which will not exist.`);
      }
      if (draft.identity && !draft.fields.some((f) => f.name.trim() === draft.identity))
        out.push(`${id} is identified by ${draft.identity}, which is not one of its fields.`);
    }
    return Array.from(new Set(out));
  };

  const apply = async () => {
    const wrong = problems();
    if (wrong.length > 0) {
      toast.error(wrong[0]);
      return;
    }
    setApplying(true);
    try {
      for (const change of changes) {
        const path = `/${encodeURIComponent(kbId)}/collections/${encodeURIComponent(change.id)}`;
        if (change.kind === "delete") await kbFetch(path, { method: "DELETE" });
        else await kbFetch(path, { method: "PUT", body: toBody(change.draft!) });
      }
      toast.success(
        `Applied ${changes.length} change${changes.length === 1 ? "" : "s"} to the collections.`
      );
      await onApplied();
      onOpenChange(false);
    } catch (e) {
      // The writes before the failure stand: they were each valid on their own, and
      // undoing them would need a transaction the API does not offer. The dialog stays
      // open on the draft so the rest can be retried.
      toast.error(`${(e as Error).message} — earlier changes were applied.`);
    } finally {
      setApplying(false);
    }
  };

  const statusOf = (draft: DraftCollection): "new" | "changed" | "deleted" | null => {
    if (draft.deleted) return "deleted";
    if (!draft.originalId) return "new";
    const saved = collections.find((c) => c.collection_id === draft.originalId);
    return sameAsSaved(draft, saved) ? null : "changed";
  };

  const inputClass = "h-9 rounded-md border border-input bg-background px-2 text-sm";

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="w-[95vw] max-w-[95vw] h-[92vh] sm:max-w-[95vw] flex flex-col overflow-hidden">
        <DialogHeader>
          <DialogTitle>Collections</DialogTitle>
          <DialogDescription>
            Add, change and remove collections together. Nothing is written until Apply, so a
            relationship can point at a collection you are creating in the same sitting.
          </DialogDescription>
        </DialogHeader>

        <div className="grid flex-1 gap-5 overflow-hidden md:grid-cols-[minmax(200px,260px)_1fr]">
          <div className="flex flex-col gap-2 overflow-y-auto">
            <div className="rounded-lg border border-border divide-y divide-border">
              {drafts.map((draft) => {
                const status = statusOf(draft);
                return (
                  <button
                    key={draft.key}
                    onClick={() => setSelectedKey(draft.key)}
                    className={cn(
                      "w-full px-3 py-2.5 text-left transition-colors",
                      draft.key === selectedKey ? "bg-accent" : "hover:bg-muted/50",
                      draft.deleted && "opacity-60"
                    )}
                  >
                    <div className="flex items-center gap-2">
                      <span
                        className={cn(
                          "font-mono text-xs truncate",
                          draft.deleted && "line-through"
                        )}
                      >
                        {draft.id || "new collection"}
                      </span>
                      {status && (
                        <span
                          className={cn(
                            "ml-auto shrink-0 rounded px-1.5 py-0.5 text-[10px]",
                            status === "deleted"
                              ? "bg-destructive/15 text-destructive"
                              : "bg-primary/15 text-primary"
                          )}
                        >
                          {status}
                        </span>
                      )}
                    </div>
                    <div className="text-[11px] text-muted-foreground">
                      {draft.fields.length} field{draft.fields.length === 1 ? "" : "s"}
                    </div>
                  </button>
                );
              })}
            </div>
            <Button variant="outline" size="sm" onClick={addCollection}>
              <Plus className="h-4 w-4 mr-1" /> Add collection
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={proposing}
              onClick={propose}
              title="Read some documents and suggest collections"
            >
              {proposing ? <Spinner size="sm" /> : <Sparkles className="h-4 w-4 mr-1" />}
              Suggest from documents
            </Button>
          </div>

          <div className="overflow-y-auto pr-1">
            {!selected ? (
              <p className="text-sm text-muted-foreground">Pick a collection, or add one.</p>
            ) : (
              <div className="space-y-4">
                <div className="flex items-start justify-between gap-3">
                  <div className="grid flex-1 gap-3 sm:grid-cols-3">
                    <div>
                      <label className="text-xs font-medium text-muted-foreground">Id</label>
                      <Input
                        className="font-mono"
                        placeholder="vendors"
                        value={selected.id}
                        onChange={(e) => patch(selected.key, { id: e.target.value })}
                      />
                    </div>
                    <div>
                      <label className="text-xs font-medium text-muted-foreground">Name</label>
                      <Input
                        placeholder="Vendors"
                        value={selected.name}
                        onChange={(e) => patch(selected.key, { name: e.target.value })}
                      />
                    </div>
                    <div>
                      <label className="text-xs font-medium text-muted-foreground">
                        Identified by
                      </label>
                      <select
                        className={`${inputClass} w-full`}
                        value={selected.identity}
                        onChange={(e) => patch(selected.key, { identity: e.target.value })}
                      >
                        <option value="">Nothing — every mention is its own record</option>
                        {selected.fields
                          .filter((f) => f.name.trim())
                          .map((f) => (
                            <option key={f.key} value={f.name.trim()}>
                              {f.name.trim()}
                            </option>
                          ))}
                      </select>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    className={selected.deleted ? "" : "text-destructive"}
                    onClick={() => patch(selected.key, { deleted: !selected.deleted })}
                    title={selected.deleted ? "Keep this collection" : "Delete this collection"}
                  >
                    {selected.deleted ? (
                      <>
                        <Undo2 className="h-4 w-4 mr-1" /> Keep
                      </>
                    ) : (
                      <>
                        <Trash2 className="h-4 w-4 mr-1" /> Delete
                      </>
                    )}
                  </Button>
                </div>

                <div>
                  <label className="text-xs font-medium text-muted-foreground">Description</label>
                  <Textarea
                    rows={2}
                    placeholder="What one record of this collection is"
                    value={selected.description}
                    onChange={(e) => patch(selected.key, { description: e.target.value })}
                  />
                </div>

                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={selected.deriveOnWrite}
                    onChange={(e) => patch(selected.key, { deriveOnWrite: e.target.checked })}
                  />
                  Re-derive this collection whenever a document is written
                </label>

                <div className="rounded-lg border border-border overflow-x-auto">
                  <table className="w-full min-w-[760px] text-sm">
                    <thead className="bg-muted/40 text-xs text-muted-foreground">
                      <tr>
                        <th className="text-left font-medium px-3 py-2 w-48">Field</th>
                        <th className="text-left font-medium px-3 py-2 w-44">Type</th>
                        <th className="text-left font-medium px-3 py-2 w-44">Values</th>
                        <th className="text-left font-medium px-3 py-2">Description</th>
                        <th className="w-10" />
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                      {selected.fields.length === 0 && (
                        <tr>
                          <td
                            colSpan={5}
                            className="px-3 py-6 text-center text-xs text-muted-foreground"
                          >
                            No fields yet. A record is its fields, so add the ones a document can
                            answer.
                          </td>
                        </tr>
                      )}
                      {selected.fields.map((field) => (
                        <tr key={field.key}>
                          <td className="px-3 py-2">
                            <Input
                              className="h-8 font-mono text-xs"
                              placeholder="name"
                              value={field.name}
                              onChange={(e) =>
                                patchField(selected.key, field.key, { name: e.target.value })
                              }
                            />
                          </td>
                          <td className="px-3 py-2">
                            <select
                              className="h-8 w-full rounded-md border border-input bg-background px-2 text-xs"
                              value={field.type}
                              onChange={(e) =>
                                patchField(selected.key, field.key, { type: e.target.value })
                              }
                            >
                              {FIELD_TYPES.map((type) => (
                                <option key={type} value={type}>
                                  {type}
                                </option>
                              ))}
                              <option value="relationship">→ another collection</option>
                            </select>
                            {field.type === "relationship" && (
                              <select
                                className="mt-1 h-8 w-full rounded-md border border-input bg-background px-2 text-xs"
                                value={field.collection}
                                onChange={(e) =>
                                  patchField(selected.key, field.key, {
                                    collection: e.target.value,
                                  })
                                }
                              >
                                <option value="">Which collection…</option>
                                {/* The drafts, not the saved set: a relationship may point
                                    at a collection being created in this same sitting. */}
                                {drafts
                                  .filter((d) => !d.deleted && d.id.trim())
                                  .map((d) => (
                                    <option key={d.key} value={d.id.trim()}>
                                      {d.id.trim()}
                                    </option>
                                  ))}
                              </select>
                            )}
                          </td>
                          <td className="px-3 py-2">
                            <Input
                              className="h-8 text-xs"
                              placeholder="signed, expired"
                              disabled={field.type === "relationship"}
                              value={field.values}
                              onChange={(e) =>
                                patchField(selected.key, field.key, { values: e.target.value })
                              }
                            />
                          </td>
                          <td className="px-3 py-2">
                            <Input
                              className="h-8 text-xs"
                              placeholder="What this field is, in the document's words"
                              value={field.description}
                              onChange={(e) =>
                                patchField(selected.key, field.key, {
                                  description: e.target.value,
                                })
                              }
                            />
                          </td>
                          <td className="px-3 py-2">
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-8 w-8 p-0"
                              title="Remove field"
                              onClick={() =>
                                patch(selected.key, {
                                  fields: selected.fields.filter((f) => f.key !== field.key),
                                })
                              }
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
                    patch(selected.key, {
                      fields: [
                        ...selected.fields,
                        {
                          key: nextKey++,
                          name: "",
                          type: "string",
                          collection: "",
                          values: "",
                          description: "",
                        },
                      ],
                    })
                  }
                >
                  <Plus className="h-4 w-4 mr-1" /> Add field
                </Button>
              </div>
            )}
          </div>
        </div>

        <Input
          className="h-8 text-xs"
          placeholder="What do you want out of this bank? (optional, steers the suggestion)"
          value={instruction}
          onChange={(e) => setInstruction(e.target.value)}
        />

        <DialogFooter className="items-center sm:justify-between">
          <div className="text-xs text-muted-foreground">
            {changes.length === 0
              ? "No changes yet."
              : `${changes.length} change${changes.length === 1 ? "" : "s"} to apply: ` +
                changes.map((c) => `${c.kind} ${c.id}`).join(", ")}
          </div>
          <div className="flex gap-2">
            <Button variant="outline" onClick={() => onOpenChange(false)} disabled={applying}>
              Cancel
            </Button>
            <Button onClick={apply} disabled={applying || changes.length === 0}>
              {applying ? <Spinner size="sm" /> : `Apply ${changes.length || ""}`.trim()}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
