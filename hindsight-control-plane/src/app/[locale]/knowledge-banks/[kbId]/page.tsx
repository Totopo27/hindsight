"use client";

// A knowledge bank: its own page with a left rail (Overview · Documents · Search ·
// Operations · Configuration), the same shell the memory-bank page uses.

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { toast } from "sonner";
import { FileText, Plus, Search as SearchIcon, Trash2, Upload, X } from "lucide-react";
import { BankSelector } from "@/components/bank-selector";
import { KnowledgeBankSidebar, type KbSection } from "@/components/knowledge-bank-sidebar";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Spinner } from "@/components/ui/spinner";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
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
  kbUpload,
  type KnowledgeBank,
  type KnowledgeDocument,
  type KnowledgeCollection,
  type KnowledgeOperation,
  type QueryResult,
  type KnowledgeSchema,
  type SearchResult,
} from "@/components/knowledge-bank-api";
import { withBasePath } from "@/lib/base-path";

const SECTIONS: KbSection[] = [
  "overview",
  "documents",
  "search",
  "schemas",
  "collections",
  "operations",
  "settings",
];

function Card({
  title,
  description,
  action,
  children,
}: {
  title: string;
  description?: string;
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="bg-card border border-border rounded-[16px] overflow-hidden">
      <div className="px-[21px] py-[16px] border-b border-border flex items-start justify-between gap-3">
        <div>
          <div className="text-[14px] font-semibold leading-[18px]">{title}</div>
          {description && (
            <p className="text-[12px] text-muted-foreground mt-[4px]">{description}</p>
          )}
        </div>
        {action}
      </div>
      <div className="p-[21px]">{children}</div>
    </div>
  );
}

export default function KnowledgeBankPage() {
  const params = useParams();
  const router = useRouter();
  const searchParams = useSearchParams();
  const kbId = decodeURIComponent(String(params.kbId));
  const section = (
    SECTIONS.includes(searchParams.get("section") as KbSection)
      ? searchParams.get("section")
      : "overview"
  ) as KbSection;

  const [bank, setBank] = useState<KnowledgeBank | null>(null);
  const [missing, setMissing] = useState(false);

  const loadBank = useCallback(async () => {
    try {
      setBank(await kbFetch<KnowledgeBank>(`/${encodeURIComponent(kbId)}`));
    } catch {
      setMissing(true);
    }
  }, [kbId]);

  useEffect(() => {
    loadBank();
  }, [loadBank]);

  const settingsTab = (
    searchParams.get("settingsTab") === "configuration" ? "configuration" : "general"
  ) as SettingsTab;

  const go = (next: KbSection) =>
    router.push(`/knowledge-banks/${encodeURIComponent(kbId)}?section=${next}`);

  const goSettingsTab = (next: SettingsTab) =>
    router.push(
      `/knowledge-banks/${encodeURIComponent(kbId)}?section=settings&settingsTab=${next}`
    );

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <div className="sticky top-0 z-30">
        <BankSelector />
      </div>
      <div className="flex flex-1">
        <div className="sticky top-14 self-start h-[calc(100vh-3.5rem)] z-20">
          <KnowledgeBankSidebar current={section} onChange={go} />
        </div>
        <main className="flex-1 min-w-0 p-6">
          <div className="max-w-[1024px] xl:max-w-[1280px] 2xl:max-w-[1440px] mx-auto w-full">
            {missing ? (
              <div className="rounded-[16px] border border-border p-[21px]">
                <p className="text-[13px] font-semibold">Knowledge bank not found</p>
                <p className="text-[12px] text-muted-foreground mt-1">It may have been deleted.</p>
              </div>
            ) : !bank ? (
              <Spinner />
            ) : (
              <>
                <div className="mt-3 mb-5 flex items-center gap-3">
                  <h1 className="text-[28px] font-semibold leading-[34px] tracking-[-0.4px] truncate">
                    {bank.bank_id}
                  </h1>
                  <span className="text-[12px] text-muted-foreground">
                    {bank.documents} documents · {bank.passages} passages
                  </span>
                </div>
                <div className="space-y-5">
                  {section === "overview" && <Overview bank={bank} onGo={go} />}
                  {section === "documents" && <Documents kbId={kbId} onChanged={loadBank} />}
                  {section === "search" && <SearchPanel kbId={kbId} />}
                  {section === "schemas" && <SchemaPanel kbId={kbId} />}
                  {section === "collections" && <CollectionsPanel kbId={kbId} />}
                  {section === "operations" && <Operations kbId={kbId} />}
                  {section === "settings" && (
                    <SettingsPanel
                      kbId={kbId}
                      bank={bank}
                      tab={settingsTab}
                      onTab={goSettingsTab}
                    />
                  )}
                </div>
              </>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}

/** The bank's numbers. Shown on Overview and again under Settings > General, which is
 *  where the memory banks put theirs. */
function Stats({ bank }: { bank: KnowledgeBank }) {
  const stats = [
    ["Documents", bank.documents],
    ["Passages", bank.passages],
    ["Writes in flight", bank.operations_in_flight ?? 0],
    ["Last write", bank.last_write_at ? new Date(bank.last_write_at).toLocaleString() : "—"],
  ] as const;
  return (
    <Card title="Overview" description="What is in this bank right now.">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {stats.map(([label, value]) => (
          <div key={label}>
            <div className="text-[12px] text-muted-foreground">{label}</div>
            <div className="text-[20px] font-semibold">{value}</div>
          </div>
        ))}
      </div>
    </Card>
  );
}

function Overview({ bank, onGo }: { bank: KnowledgeBank; onGo: (s: KbSection) => void }) {
  return (
    <>
      <Stats bank={bank} />
      <div className="flex gap-2">
        <Button size="sm" variant="outline" onClick={() => onGo("documents")}>
          <FileText className="w-4 h-4 mr-1" /> Documents
        </Button>
        <Button size="sm" variant="outline" onClick={() => onGo("search")}>
          <SearchIcon className="w-4 h-4 mr-1" /> Search
        </Button>
      </div>
    </>
  );
}

function Documents({ kbId, onChanged }: { kbId: string; onChanged: () => void }) {
  const [documents, setDocuments] = useState<KnowledgeDocument[] | null>(null);
  const [total, setTotal] = useState(0);
  const [adding, setAdding] = useState(false);
  const [tab, setTab] = useState<"text" | "upload">("text");
  const [form, setForm] = useState({ id: "", title: "", text: "", tags: "" });
  const [files, setFiles] = useState<File[]>([]);
  const [saving, setSaving] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    try {
      const page = await kbFetch<{ items: KnowledgeDocument[]; total: number }>(
        `/${encodeURIComponent(kbId)}/documents?limit=200`
      );
      setDocuments(page.items);
      setTotal(page.total);
    } catch (e) {
      toast.error((e as Error).message);
    }
  }, [kbId]);

  useEffect(() => {
    load();
  }, [load]);

  const tagList = () =>
    form.tags
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean);

  const done = (message: string) => {
    toast.success(message);
    setAdding(false);
    setForm({ id: "", title: "", text: "", tags: "" });
    setFiles([]);
    // The write runs in the background, so the list is re-read a moment later rather
    // than immediately, when it would still show the bank as it was.
    setTimeout(() => {
      load();
      onChanged();
    }, 1200);
  };

  const write = async () => {
    setSaving(true);
    try {
      const result = await kbFetch<{ operation_id: string }>(
        `/${encodeURIComponent(kbId)}/documents`,
        {
          method: "POST",
          body: {
            documents: [
              {
                // Omitted, the server assigns a uuid — an id is for replacing this
                // document later, not something the writer has to invent.
                ...(form.id.trim() ? { id: form.id.trim() } : {}),
                text: form.text,
                title: form.title || null,
                tags: tagList(),
              },
            ],
          },
        }
      );
      done(`Queued write ${result.operation_id.slice(0, 8)} — it runs in the background`);
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  const upload = async () => {
    setSaving(true);
    try {
      const result = await kbUpload(kbId, files, { tags: tagList() });
      done(
        `Uploaded ${files.length} file(s) — each is converted and written in the background ` +
          `(${result.operation_ids.length} operation(s))`
      );
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  const remove = async (docId: string) => {
    try {
      await kbFetch(`/${encodeURIComponent(kbId)}/documents/${encodeURIComponent(docId)}`, {
        method: "DELETE",
      });
      await load();
      onChanged();
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  return (
    <Card
      title="Documents"
      description={`${total} documents. Writes are queued as operations and run in the background.`}
      action={
        <Button size="sm" onClick={() => setAdding(true)}>
          <Plus className="w-4 h-4 mr-1" /> Add document
        </Button>
      }
    >
      {!documents ? (
        <Spinner />
      ) : documents.length === 0 ? (
        <p className="text-sm text-muted-foreground">Nothing written yet.</p>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Document</TableHead>
              <TableHead>Tags</TableHead>
              <TableHead className="text-right">Chunks</TableHead>
              <TableHead className="text-right">Characters</TableHead>
              <TableHead />
            </TableRow>
          </TableHeader>
          <TableBody>
            {documents.map((doc) => (
              <TableRow key={doc.doc_id}>
                <TableCell className="font-mono text-sm">
                  {doc.doc_id}
                  {doc.title && <div className="text-xs text-muted-foreground">{doc.title}</div>}
                </TableCell>
                <TableCell className="text-xs text-muted-foreground">
                  {doc.tags.join(", ")}
                </TableCell>
                <TableCell className="text-right">{doc.passage_count}</TableCell>
                <TableCell className="text-right">{doc.chars.toLocaleString()}</TableCell>
                <TableCell className="text-right">
                  <Button
                    size="sm"
                    variant="ghost"
                    title="Delete"
                    onClick={() => remove(doc.doc_id)}
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}

      <Dialog open={adding} onOpenChange={setAdding}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Add document</DialogTitle>
            <DialogDescription>
              It is split into passages and embedded by a background write. Writing the same id
              again replaces it.
            </DialogDescription>
          </DialogHeader>
          <Tabs value={tab} onValueChange={(v) => setTab(v as "text" | "upload")}>
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="text" className="flex items-center gap-2">
                <FileText className="h-4 w-4" /> Text
              </TabsTrigger>
              <TabsTrigger value="upload" className="flex items-center gap-2">
                <Upload className="h-4 w-4" /> Upload files
              </TabsTrigger>
            </TabsList>

            <TabsContent value="text" className="mt-3 space-y-2">
              <div className="flex gap-2">
                <Input
                  placeholder="Document id (optional)"
                  value={form.id}
                  onChange={(e) => setForm({ ...form, id: e.target.value })}
                />
                <Input
                  placeholder="Title (optional)"
                  value={form.title}
                  onChange={(e) => setForm({ ...form, title: e.target.value })}
                />
              </div>
              <Textarea
                rows={12}
                placeholder="Document text"
                value={form.text}
                onChange={(e) => setForm({ ...form, text: e.target.value })}
              />
            </TabsContent>

            <TabsContent value="upload" className="mt-3 space-y-2">
              <input
                ref={fileInput}
                type="file"
                multiple
                className="hidden"
                id="kb-file-upload"
                onChange={(e) => {
                  setFiles((prev) => [...prev, ...Array.from(e.target.files || [])]);
                  if (fileInput.current) fileInput.current.value = "";
                }}
              />
              <label
                htmlFor="kb-file-upload"
                className="flex h-32 w-full cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-muted-foreground/25 transition-colors hover:border-primary/50 hover:bg-accent/50"
              >
                <Upload className="mb-2 h-8 w-8 text-muted-foreground" />
                <span className="text-sm text-muted-foreground">
                  Click to select files (PDF, DOCX, TXT, …)
                </span>
              </label>
              {files.map((file, index) => (
                <div
                  key={`${file.name}-${index}`}
                  className="flex items-center gap-2 rounded-md bg-muted px-2 py-2"
                >
                  <FileText className="h-4 w-4 shrink-0 text-muted-foreground" />
                  <span className="flex-1 truncate text-sm">{file.name}</span>
                  <span className="shrink-0 text-xs text-muted-foreground">
                    {(file.size / 1024).toFixed(1)} KB
                  </span>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-6 w-6 shrink-0 p-0"
                    onClick={() => setFiles((prev) => prev.filter((_, i) => i !== index))}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              ))}
              <p className="text-xs text-muted-foreground">
                Each file is converted to markdown and written as one document, with a generated id.
              </p>
            </TabsContent>
          </Tabs>

          <Input
            placeholder="tags, comma separated"
            value={form.tags}
            onChange={(e) => setForm({ ...form, tags: e.target.value })}
          />

          <DialogFooter>
            {tab === "text" ? (
              <Button onClick={write} disabled={saving || !form.text.trim()}>
                {saving ? <Spinner size="sm" /> : "Write"}
              </Button>
            ) : (
              <Button onClick={upload} disabled={saving || files.length === 0}>
                {saving ? <Spinner size="sm" /> : `Upload ${files.length || ""}`.trim()}
              </Button>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  );
}

function SearchPanel({ kbId }: { kbId: string }) {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<"hybrid" | "vector" | "keyword">("hybrid");
  const [filter, setFilter] = useState("");
  const [results, setResults] = useState<SearchResult[] | null>(null);
  const [loading, setLoading] = useState(false);

  const run = async () => {
    if (!query.trim()) return;
    let filterValue: unknown = undefined;
    if (filter.trim()) {
      try {
        filterValue = JSON.parse(filter);
      } catch {
        toast.error("The field filter is not valid JSON");
        return;
      }
    }
    setLoading(true);
    try {
      const response = await kbFetch<{ results: SearchResult[] }>(
        `/${encodeURIComponent(kbId)}/search`,
        {
          method: "POST",
          body: { query, mode, top_k: 10, fields: filterValue },
        }
      );
      setResults(response.results);
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card title="Search" description="Hybrid vector + keyword search, reranked.">
      <form
        className="flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          run();
        }}
      >
        <Input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask something…"
          autoFocus
        />
        <div className="flex rounded-md border border-input overflow-hidden text-sm">
          {(["hybrid", "vector", "keyword"] as const).map((m) => (
            <button
              type="button"
              key={m}
              onClick={() => setMode(m)}
              className={`px-3 capitalize ${mode === m ? "bg-primary text-primary-foreground" : "hover:bg-muted"}`}
            >
              {m}
            </button>
          ))}
        </div>
        <Button type="submit" disabled={loading}>
          {loading ? <Spinner size="sm" /> : <SearchIcon className="w-4 h-4" />}
        </Button>
      </form>

      <Input
        className="mt-2 font-mono text-xs"
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        placeholder={'Field filter, e.g. {"doc_type": "invoice", "total": {"$gte": 1000}}'}
      />

      {results && (
        <div className="mt-4 space-y-3">
          {results.length === 0 && <p className="text-sm text-muted-foreground">Nothing found.</p>}
          {results.map((hit, i) => (
            <div
              key={`${hit.document_id}#${hit.passage_index}`}
              className="rounded-lg border border-border p-3"
            >
              <div className="flex items-center gap-2 text-sm mb-1">
                <span className="text-muted-foreground">#{i + 1}</span>
                <span className="font-mono">{hit.document_id}</span>
                <span className="text-muted-foreground">passage {hit.passage_index}</span>
                {hit.ranks.vector && (
                  <span className="rounded bg-blue-100 dark:bg-blue-500/20 px-1.5 py-0.5 text-[11px]">
                    vector #{hit.ranks.vector}
                  </span>
                )}
                {hit.ranks.keyword && (
                  <span className="rounded bg-emerald-100 dark:bg-emerald-500/20 px-1.5 py-0.5 text-[11px]">
                    keyword #{hit.ranks.keyword}
                  </span>
                )}
                <span className="ml-auto text-xs text-muted-foreground">
                  {hit.score.toFixed(4)}
                </span>
              </div>
              <p className="text-sm whitespace-pre-wrap">{hit.text}</p>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}

function SchemaPanel({ kbId }: { kbId: string }) {
  const [schemas, setSchemas] = useState<KnowledgeSchema[] | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [saving, setSaving] = useState(false);
  const [newId, setNewId] = useState("");
  const [creating, setCreating] = useState(false);

  const load = useCallback(async () => {
    const response = await kbFetch<{ items: KnowledgeSchema[] }>(
      `/${encodeURIComponent(kbId)}/schemas`
    );
    setSchemas(response.items);
    setSelected((current) => current ?? response.items[0]?.schema_id ?? null);
  }, [kbId]);

  useEffect(() => {
    load().catch((e) => toast.error((e as Error).message));
  }, [load]);

  // The editor follows the selection: the JSON is the schema's own definition, so
  // switching schemas has to reload it rather than keep the previous one's text.
  const current = schemas?.find((schema) => schema.schema_id === selected) ?? null;
  useEffect(() => {
    if (!current) return;
    setDraft(
      JSON.stringify(
        {
          name: current.name,
          description: current.description,
          document_fields: current.document_fields,
          passage_fields: current.passage_fields,
        },
        null,
        2
      )
    );
  }, [current]);

  const save = async (schemaId: string, body: unknown) => {
    setSaving(true);
    try {
      await kbFetch(`/${encodeURIComponent(kbId)}/schemas/${encodeURIComponent(schemaId)}`, {
        method: "PUT",
        body,
      });
      toast.success("Saved. It applies to the next write.");
      setSelected(schemaId);
      await load();
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  const saveCurrent = () => {
    if (!selected) return;
    let body: unknown;
    try {
      body = JSON.parse(draft);
    } catch {
      toast.error("That is not valid JSON");
      return;
    }
    save(selected, body);
  };

  const create = async () => {
    const id = newId.trim();
    if (!id) return;
    setCreating(false);
    setNewId("");
    await save(id, { document_fields: {}, passage_fields: {} });
  };

  const remove = async (schemaId: string) => {
    try {
      await kbFetch(`/${encodeURIComponent(kbId)}/schemas/${encodeURIComponent(schemaId)}`, {
        method: "DELETE",
      });
      setSelected(null);
      await load();
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  const extract = async () => {
    try {
      await kbFetch(`/${encodeURIComponent(kbId)}/fields/extract`, {
        method: "POST",
        body: { only_missing: true },
      });
      toast.success("Extracting in the background — watch it in Operations.");
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  if (!schemas) return <Spinner />;
  return (
    <Card
      title="Schemas"
      description="A schema is the fields one kind of document has. A bank with several lets the LLM classify which one a document is."
      action={
        <Button size="sm" variant="outline" onClick={() => setCreating(true)}>
          <Plus className="w-4 h-4 mr-1" /> New schema
        </Button>
      }
    >
      {schemas.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          No schemas yet. Create one to give documents fields that search and query can filter on.
        </p>
      ) : (
        <>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Schema</TableHead>
                <TableHead>Fields</TableHead>
                <TableHead className="text-right">Filled</TableHead>
                <TableHead />
              </TableRow>
            </TableHeader>
            <TableBody>
              {schemas.map((schema) => (
                <TableRow
                  key={schema.schema_id}
                  onClick={() => setSelected(schema.schema_id)}
                  className={`cursor-pointer ${schema.schema_id === selected ? "bg-accent/40" : ""}`}
                >
                  <TableCell className="font-mono text-xs">
                    {schema.schema_id}
                    {schema.name && (
                      <div className="text-xs text-muted-foreground">{schema.name}</div>
                    )}
                  </TableCell>
                  <TableCell className="text-xs text-muted-foreground">
                    {[
                      ...Object.keys(schema.document_fields),
                      ...Object.keys(schema.passage_fields).map((name) => `${name} (passage)`),
                    ].join(", ") || "—"}
                  </TableCell>
                  <TableCell className="text-right text-xs">
                    {schema.documents_with_fields} doc / {schema.passages_with_fields} psg
                  </TableCell>
                  <TableCell className="text-right" onClick={(e) => e.stopPropagation()}>
                    <Button
                      size="sm"
                      variant="ghost"
                      title="Delete"
                      onClick={() => remove(schema.schema_id)}
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>

          {selected && (
            <>
              <p className="mt-4 mb-2 text-sm font-semibold">{selected}</p>
              <Textarea
                rows={16}
                className="font-mono text-xs"
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                spellCheck={false}
              />
              <p className="mt-2 text-xs text-muted-foreground">
                A field is <code>{'{"type": "string"}'}</code> plus an optional{" "}
                <code>description</code>, an <code>items</code> type for arrays, <code>values</code>{" "}
                — a fixed list, which is how classification is expressed — and the flags{" "}
                <code>filterable</code> (usable in a search filter) and <code>indexed</code> (its
                value joins the passage&apos;s embedding). Types: string, integer, number, boolean,
                date, datetime, array, object.
              </p>
              <div className="mt-3 flex gap-2">
                <Button size="sm" onClick={saveCurrent} disabled={saving}>
                  {saving ? <Spinner size="sm" /> : "Save schema"}
                </Button>
                <Button size="sm" variant="outline" onClick={extract}>
                  Extract documents missing values
                </Button>
              </div>
            </>
          )}
        </>
      )}

      <Dialog open={creating} onOpenChange={setCreating}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>New schema</DialogTitle>
            <DialogDescription>
              The id names this kind of document, e.g. <code>contract</code>. It starts empty; add
              its fields in the editor.
            </DialogDescription>
          </DialogHeader>
          <Input placeholder="contract" value={newId} onChange={(e) => setNewId(e.target.value)} />
          <DialogFooter>
            <Button disabled={!newId.trim()} onClick={create}>
              Create
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  );
}

const NEW_COLLECTION_DEFINITION = `{
  "name": "Vendors",
  "description": "One company we buy from",
  "identity": "name",
  "derive_on_write": false,
  "fields": {
    "name": { "type": "string", "description": "The company's name" },
    "country": { "type": "string" }
  }
}`;

function CollectionsPanel({ kbId }: { kbId: string }) {
  const [collections, setCollections] = useState<KnowledgeCollection[] | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [editing, setEditing] = useState<{ id: string; definition: string } | null>(null);
  const [query, setQuery] = useState('{\n  "select": [{"count": "*", "as": "records"}]\n}');
  const [result, setResult] = useState<QueryResult | null>(null);
  const [records, setRecords] = useState<QueryResult | null>(null);
  const [running, setRunning] = useState(false);

  const load = useCallback(async () => {
    const response = await kbFetch<{ items: KnowledgeCollection[] }>(
      `/${encodeURIComponent(kbId)}/collections`
    );
    setCollections(response.items);
    setSelected((current) => current ?? response.items[0]?.collection_id ?? null);
  }, [kbId]);

  useEffect(() => {
    load().catch((e) => toast.error((e as Error).message));
  }, [load]);

  const current = collections?.find((c) => c.collection_id === selected) ?? null;

  // The records are read through the query DSL rather than a list endpoint: the columns
  // are the collection's own fields, which only the definition knows.
  const loadRecords = useCallback(
    async (collection: KnowledgeCollection) => {
      try {
        setRecords(
          await kbFetch<QueryResult>(
            `/${encodeURIComponent(kbId)}/collections/${encodeURIComponent(collection.collection_id)}/query`,
            {
              method: "POST",
              body: {
                select: [
                  { field: "record_id", as: "record_id" },
                  // Aliased to the field name: an unaliased select column comes back as
                  // "column_2", which is not a heading anyone can read.
                  ...Object.keys(collection.fields).map((name) => ({ field: name, as: name })),
                  { field: "doc_count", as: "documents" },
                ],
                order_by: [{ field: "updated_at", direction: "desc" }],
                limit: 50,
              },
            }
          )
        );
      } catch (e) {
        toast.error((e as Error).message);
        setRecords(null);
      }
    },
    [kbId]
  );

  useEffect(() => {
    if (current) loadRecords(current);
    else setRecords(null);
  }, [current, loadRecords]);

  const run = async () => {
    if (!selected) return;
    let body: unknown;
    try {
      body = JSON.parse(query);
    } catch {
      toast.error("That is not valid JSON");
      return;
    }
    setRunning(true);
    try {
      setResult(
        await kbFetch<QueryResult>(
          `/${encodeURIComponent(kbId)}/collections/${encodeURIComponent(selected)}/query`,
          { method: "POST", body }
        )
      );
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setRunning(false);
    }
  };

  const saveDefinition = async () => {
    if (!editing) return;
    const id = editing.id.trim();
    if (!id) return;
    let body: unknown;
    try {
      body = JSON.parse(editing.definition);
    } catch {
      toast.error("That is not valid JSON");
      return;
    }
    try {
      await kbFetch(`/${encodeURIComponent(kbId)}/collections/${encodeURIComponent(id)}`, {
        method: "PUT",
        body,
      });
      toast.success("Saved. Derive to fill it from the documents.");
      setEditing(null);
      setSelected(id);
      await load();
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  const remove = async (collectionId: string) => {
    try {
      await kbFetch(
        `/${encodeURIComponent(kbId)}/collections/${encodeURIComponent(collectionId)}`,
        {
          method: "DELETE",
        }
      );
      setSelected(null);
      await load();
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  const derive = async (collectionId: string) => {
    try {
      await kbFetch(
        `/${encodeURIComponent(kbId)}/collections/${encodeURIComponent(collectionId)}/derive`,
        { method: "POST", body: {} }
      );
      toast.success("Deriving in the background — watch it in Operations.");
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  const openEditor = (collection?: KnowledgeCollection) =>
    setEditing(
      collection
        ? {
            id: collection.collection_id,
            definition: JSON.stringify(
              {
                name: collection.name,
                description: collection.description,
                identity: collection.identity,
                derive_on_write: collection.derive_on_write ?? false,
                fields: collection.fields,
              },
              null,
              2
            ),
          }
        : { id: "", definition: NEW_COLLECTION_DEFINITION }
    );

  if (!collections) return <Spinner />;
  return (
    <>
      <Card
        title="Collections"
        description="A collection is a kind of thing the documents talk about; each record folds together what every document said about one of them."
        action={
          <Button size="sm" variant="outline" onClick={() => openEditor()}>
            <Plus className="w-4 h-4 mr-1" /> New collection
          </Button>
        }
      >
        {collections.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No collections yet. Define one — its fields are what an LLM reads out of each document,
            and the identity field is what makes the same thing found twice one record.
          </p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Collection</TableHead>
                <TableHead>Identity</TableHead>
                <TableHead>Fields</TableHead>
                <TableHead className="text-right">Records</TableHead>
                <TableHead />
              </TableRow>
            </TableHeader>
            <TableBody>
              {collections.map((collection) => (
                <TableRow
                  key={collection.collection_id}
                  onClick={() => setSelected(collection.collection_id)}
                  className={`cursor-pointer ${
                    collection.collection_id === selected ? "bg-accent/40" : ""
                  }`}
                >
                  <TableCell className="font-mono text-xs">
                    {collection.collection_id}
                    {collection.name && (
                      <div className="text-xs text-muted-foreground">{collection.name}</div>
                    )}
                  </TableCell>
                  <TableCell className="text-xs">{collection.identity ?? "—"}</TableCell>
                  <TableCell className="text-xs text-muted-foreground">
                    {Object.entries(collection.fields)
                      .map(([name, spec]) =>
                        spec.collection ? `${name} → ${spec.collection}` : name
                      )
                      .join(", ")}
                  </TableCell>
                  <TableCell className="text-right text-xs">{collection.records ?? 0}</TableCell>
                  <TableCell className="text-right" onClick={(e) => e.stopPropagation()}>
                    <Button size="sm" variant="ghost" onClick={() => openEditor(collection)}>
                      Edit
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => derive(collection.collection_id)}
                    >
                      Derive
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      title="Delete"
                      onClick={() => remove(collection.collection_id)}
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </Card>

      {current && (
        <Card
          title={`Records · ${current.collection_id}`}
          description="The 50 most recently updated records, with how many documents each one folds together."
        >
          {!records ? (
            <Spinner />
          ) : records.rows.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              No records yet. Derive the collection to read them out of the documents.
            </p>
          ) : (
            <div className="overflow-x-auto">
              <ResultTable result={records} />
            </div>
          )}
        </Card>
      )}

      {current && (
        <Card
          title="Query"
          description="The same language the documents use, over records — aggregates, grouping, and joins across relationship fields."
        >
          <div className="flex gap-2 items-center">
            <select
              className="h-9 rounded-md border border-input bg-background px-2 text-sm"
              value={selected ?? ""}
              onChange={(e) => setSelected(e.target.value)}
            >
              {collections.map((collection) => (
                <option key={collection.collection_id} value={collection.collection_id}>
                  {collection.collection_id}
                </option>
              ))}
            </select>
            <Button size="sm" onClick={run} disabled={running}>
              {running ? <Spinner size="sm" /> : "Run query"}
            </Button>
          </div>
          <Textarea
            rows={10}
            className="mt-2 font-mono text-xs"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            spellCheck={false}
          />
          {result && (
            <div className="mt-4 overflow-x-auto">
              <ResultTable result={result} />
            </div>
          )}
        </Card>
      )}

      <Dialog open={editing !== null} onOpenChange={(open) => !open && setEditing(null)}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>{editing?.id ? `Collection ${editing.id}` : "New collection"}</DialogTitle>
            <DialogDescription>
              A field is <code>{'{"type": "string"}'}</code>, or{" "}
              <code>{'{"collection": "vendors"}'}</code> for a relationship to another
              collection&apos;s record. <code>identity</code> names the field that says which thing
              a record is.
            </DialogDescription>
          </DialogHeader>
          <Input
            placeholder="Collection id, e.g. vendors"
            value={editing?.id ?? ""}
            disabled={Boolean(collections.find((c) => c.collection_id === editing?.id))}
            onChange={(e) => setEditing((prev) => (prev ? { ...prev, id: e.target.value } : prev))}
          />
          <Textarea
            rows={16}
            className="font-mono text-xs"
            value={editing?.definition ?? ""}
            spellCheck={false}
            onChange={(e) =>
              setEditing((prev) => (prev ? { ...prev, definition: e.target.value } : prev))
            }
          />
          <DialogFooter>
            <Button onClick={saveDefinition} disabled={!editing?.id.trim()}>
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

/** Rows of a query result, whatever its columns turn out to be. */
function ResultTable({ result }: { result: QueryResult }) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          {result.columns.map((column) => (
            <TableHead key={column}>{column}</TableHead>
          ))}
        </TableRow>
      </TableHeader>
      <TableBody>
        {result.rows.map((row, i) => (
          <TableRow key={i}>
            {row.map((value, j) => (
              <TableCell key={j} className="text-xs">
                {value === null
                  ? "—"
                  : typeof value === "object"
                    ? JSON.stringify(value)
                    : String(value)}
              </TableCell>
            ))}
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

function Operations({ kbId }: { kbId: string }) {
  const [operations, setOperations] = useState<KnowledgeOperation[] | null>(null);

  useEffect(() => {
    const load = () =>
      kbFetch<{ operations: KnowledgeOperation[] }>(
        `/${encodeURIComponent(kbId)}/operations?limit=50`
      )
        .then((r) => setOperations(r.operations))
        .catch((e) => toast.error((e as Error).message));
    load();
    const timer = setInterval(load, 4000); // writes are async: keep the list live
    return () => clearInterval(timer);
  }, [kbId]);

  return (
    <Card title="Operations" description="Every write is an operation; this list refreshes itself.">
      {!operations ? (
        <Spinner />
      ) : operations.length === 0 ? (
        <p className="text-sm text-muted-foreground">No writes yet.</p>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Type</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Detail</TableHead>
              <TableHead>Updated</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {operations.map((op) => (
              <TableRow key={op.id}>
                <TableCell className="font-mono text-xs">{op.task_type}</TableCell>
                <TableCell>{op.status}</TableCell>
                <TableCell className="text-xs text-muted-foreground">
                  {op.error_message ?? op.details ?? ""}
                </TableCell>
                <TableCell className="text-xs">
                  {op.updated_at ? new Date(op.updated_at).toLocaleTimeString() : ""}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </Card>
  );
}

type SettingsTab = "general" | "configuration";

/** Settings, laid out like a memory bank's: General carries the stats and the write
 *  operations, Configuration carries the per-bank settings. */
function SettingsPanel({
  kbId,
  bank,
  tab,
  onTab,
}: {
  kbId: string;
  bank: KnowledgeBank;
  tab: SettingsTab;
  onTab: (tab: SettingsTab) => void;
}) {
  const tabs: { id: SettingsTab; label: string }[] = [
    { id: "general", label: "General" },
    { id: "configuration", label: "Configuration" },
  ];
  return (
    <div>
      <div className="mb-6">
        <h2 className="text-[20px] font-semibold leading-[26px]">Settings</h2>
        <p className="text-sm text-muted-foreground mt-1">
          What is in this bank, what it is doing, and how it is configured.
        </p>
      </div>
      <div className="border-b border-border mb-6 flex">
        {tabs.map((item) => (
          <button
            key={item.id}
            onClick={() => onTab(item.id)}
            className={`px-6 py-3 font-semibold text-sm transition-all relative ${
              tab === item.id ? "text-primary" : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {item.label}
            {tab === item.id && (
              <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary-gradient" />
            )}
          </button>
        ))}
      </div>
      {tab === "general" ? (
        <div className="space-y-5">
          <Stats bank={bank} />
          <Operations kbId={kbId} />
        </div>
      ) : (
        <Configuration kbId={kbId} />
      )}
    </div>
  );
}

function Configuration({ kbId }: { kbId: string }) {
  const [config, setConfig] = useState<Record<string, unknown> | null>(null);
  const [draft, setDraft] = useState<Record<string, string>>({});
  const fields = [
    "kb_passage_size",
    "kb_passage_overlap",
    "kb_search_candidates",
    "kb_search_rerank",
    "kb_search_vector_weight",
    "kb_field_extraction",
    "kb_schema_classification",
    "kb_record_identity_similarity",
  ];
  const booleans = ["kb_search_rerank", "kb_field_extraction", "kb_schema_classification"];

  const load = useCallback(async () => {
    // Bank config is the memory banks' endpoint: a knowledge bank is a bank row.
    const response = await fetch(withBasePath(`/api/banks/${encodeURIComponent(kbId)}/config`));
    const data = await response.json();
    setConfig(data.config ?? {});
    setDraft(Object.fromEntries(fields.map((f) => [f, String(data.config?.[f] ?? "")])));
  }, [kbId]);

  useEffect(() => {
    load().catch((e) => toast.error((e as Error).message));
  }, [load]);

  const save = async () => {
    const updates: Record<string, unknown> = {};
    for (const field of fields) {
      const value = draft[field];
      updates[field] = booleans.includes(field) ? value === "true" : Number(value);
    }
    const response = await fetch(withBasePath(`/api/banks/${encodeURIComponent(kbId)}/config`), {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ updates }),
    });
    if (!response.ok) {
      toast.error(`Could not save: HTTP ${response.status}`);
      return;
    }
    toast.success("Saved. It applies to the next write and the next search.");
    await load();
  };

  if (!config) return <Spinner />;
  return (
    <Card title="Configuration" description="Chunking and search settings for this bank.">
      <div className="grid gap-3 md:grid-cols-2">
        {fields.map((field) => (
          <label key={field} className="text-sm space-y-1">
            <span className="font-mono text-xs text-muted-foreground">{field}</span>
            {booleans.includes(field) ? (
              <select
                className="h-9 w-full rounded-md border border-input bg-background px-2 text-sm"
                value={draft[field]}
                onChange={(e) => setDraft({ ...draft, [field]: e.target.value })}
              >
                <option value="true">true</option>
                <option value="false">false</option>
              </select>
            ) : (
              <Input
                type="number"
                value={draft[field]}
                onChange={(e) => setDraft({ ...draft, [field]: e.target.value })}
              />
            )}
          </label>
        ))}
      </div>
      <Button className="mt-4" size="sm" onClick={save}>
        Save
      </Button>
    </Card>
  );
}
