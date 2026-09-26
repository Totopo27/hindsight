"use client";

// A knowledge bank: its own page with a left rail (Overview · Documents · Search ·
// Operations · Configuration), the same shell the memory-bank page uses.

import { useCallback, useEffect, useState } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { toast } from "sonner";
import { FileText, Plus, Search as SearchIcon, Trash2 } from "lucide-react";
import { BankSelector } from "@/components/bank-selector";
import { KnowledgeBankSidebar, type KbSection } from "@/components/knowledge-bank-sidebar";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Spinner } from "@/components/ui/spinner";
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
  type KnowledgeBank,
  type KnowledgeDocument,
  type KnowledgeOperation,
  type MetadataSchema,
  type SearchResult,
} from "@/components/knowledge-bank-api";
import { withBasePath } from "@/lib/base-path";

const SECTIONS: KbSection[] = [
  "overview",
  "documents",
  "search",
  "metadata",
  "operations",
  "configuration",
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

  const go = (next: KbSection) =>
    router.push(`/knowledge-banks/${encodeURIComponent(kbId)}?section=${next}`);

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
                    {bank.documents} documents · {bank.chunks} chunks
                  </span>
                </div>
                <div className="space-y-5">
                  {section === "overview" && <Overview bank={bank} onGo={go} />}
                  {section === "documents" && <Documents kbId={kbId} onChanged={loadBank} />}
                  {section === "search" && <SearchPanel kbId={kbId} />}
                  {section === "metadata" && <MetadataPanel kbId={kbId} />}
                  {section === "operations" && <Operations kbId={kbId} />}
                  {section === "configuration" && <Configuration kbId={kbId} />}
                </div>
              </>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}

function Overview({ bank, onGo }: { bank: KnowledgeBank; onGo: (s: KbSection) => void }) {
  const stats = [
    ["Documents", bank.documents],
    ["Chunks", bank.chunks],
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
      <div className="mt-4 flex gap-2">
        <Button size="sm" variant="outline" onClick={() => onGo("documents")}>
          <FileText className="w-4 h-4 mr-1" /> Documents
        </Button>
        <Button size="sm" variant="outline" onClick={() => onGo("search")}>
          <SearchIcon className="w-4 h-4 mr-1" /> Search
        </Button>
      </div>
    </Card>
  );
}

function Documents({ kbId, onChanged }: { kbId: string; onChanged: () => void }) {
  const [documents, setDocuments] = useState<KnowledgeDocument[] | null>(null);
  const [total, setTotal] = useState(0);
  const [adding, setAdding] = useState(false);
  const [form, setForm] = useState({ id: "", title: "", text: "", tags: "" });
  const [saving, setSaving] = useState(false);

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
                id: form.id,
                text: form.text,
                title: form.title || null,
                tags: form.tags
                  .split(",")
                  .map((t) => t.trim())
                  .filter(Boolean),
              },
            ],
          },
        }
      );
      toast.success(`Queued write ${result.operation_id.slice(0, 8)} — it runs in the background`);
      setAdding(false);
      setForm({ id: "", title: "", text: "", tags: "" });
      setTimeout(() => {
        load();
        onChanged();
      }, 1200);
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
                <TableCell className="text-right">{doc.chunk_count}</TableCell>
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
              It is chunked and embedded by a background write. Writing the same id again replaces
              it.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-2">
            <div className="flex gap-2">
              <Input
                placeholder="Document id"
                value={form.id}
                onChange={(e) => setForm({ ...form, id: e.target.value })}
              />
              <Input
                placeholder="Title (optional)"
                value={form.title}
                onChange={(e) => setForm({ ...form, title: e.target.value })}
              />
              <Input
                placeholder="tags, comma separated"
                value={form.tags}
                onChange={(e) => setForm({ ...form, tags: e.target.value })}
              />
            </div>
            <Textarea
              rows={12}
              placeholder="Document text"
              value={form.text}
              onChange={(e) => setForm({ ...form, text: e.target.value })}
            />
          </div>
          <DialogFooter>
            <Button onClick={write} disabled={saving || !form.id || !form.text}>
              {saving ? <Spinner size="sm" /> : "Write"}
            </Button>
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
    let metadata: unknown = undefined;
    if (filter.trim()) {
      try {
        metadata = JSON.parse(filter);
      } catch {
        toast.error("The metadata filter is not valid JSON");
        return;
      }
    }
    setLoading(true);
    try {
      const response = await kbFetch<{ results: SearchResult[] }>(
        `/${encodeURIComponent(kbId)}/search`,
        {
          method: "POST",
          body: { query, mode, top_k: 10, metadata },
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
        placeholder={'Metadata filter, e.g. {"doc_type": "invoice", "total": {"$gte": 1000}}'}
      />

      {results && (
        <div className="mt-4 space-y-3">
          {results.length === 0 && <p className="text-sm text-muted-foreground">Nothing found.</p>}
          {results.map((hit, i) => (
            <div
              key={`${hit.document_id}#${hit.chunk_index}`}
              className="rounded-lg border border-border p-3"
            >
              <div className="flex items-center gap-2 text-sm mb-1">
                <span className="text-muted-foreground">#{i + 1}</span>
                <span className="font-mono">{hit.document_id}</span>
                <span className="text-muted-foreground">chunk {hit.chunk_index}</span>
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

function MetadataPanel({ kbId }: { kbId: string }) {
  const [schema, setSchema] = useState<MetadataSchema | null>(null);
  const [draft, setDraft] = useState("");
  const [saving, setSaving] = useState(false);

  const load = useCallback(async () => {
    const response = await kbFetch<MetadataSchema>(`/${encodeURIComponent(kbId)}/metadata-schema`);
    setSchema(response);
    setDraft(JSON.stringify({ document: response.document, chunks: response.chunks }, null, 2));
  }, [kbId]);

  useEffect(() => {
    load().catch((e) => toast.error((e as Error).message));
  }, [load]);

  const save = async () => {
    let body: unknown;
    try {
      body = JSON.parse(draft);
    } catch {
      toast.error("That is not valid JSON");
      return;
    }
    setSaving(true);
    try {
      await kbFetch(`/${encodeURIComponent(kbId)}/metadata-schema`, { method: "PUT", body });
      toast.success("Saved. It applies to the next write.");
      await load();
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  const extract = async () => {
    try {
      await kbFetch(`/${encodeURIComponent(kbId)}/metadata/extract`, {
        method: "POST",
        body: { only_missing: true },
      });
      toast.success("Extracting in the background — watch it in Operations.");
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  if (!schema) return <Spinner />;
  return (
    <Card
      title="Metadata"
      description="What the LLM extracts from every document and chunk. Search filters on these values."
    >
      <p className="text-sm text-muted-foreground mb-2">
        {schema.documents_extracted} document(s) and {schema.chunks_extracted} chunk(s) carry
        extracted values.
      </p>
      <Textarea
        rows={18}
        className="font-mono text-xs"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        spellCheck={false}
      />
      <p className="mt-2 text-xs text-muted-foreground">
        A property is <code>{'{"type": "string"}'}</code> plus an optional <code>description</code>,
        an <code>items</code> type for arrays, and <code>values</code> — a fixed list, which is how
        classification is expressed. Types: string, integer, number, boolean, date, datetime, array,
        object.
      </p>
      <div className="mt-3 flex gap-2">
        <Button size="sm" onClick={save} disabled={saving}>
          {saving ? <Spinner size="sm" /> : "Save schema"}
        </Button>
        <Button size="sm" variant="outline" onClick={extract}>
          Extract documents missing values
        </Button>
      </div>
    </Card>
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

function Configuration({ kbId }: { kbId: string }) {
  const [config, setConfig] = useState<Record<string, unknown> | null>(null);
  const [draft, setDraft] = useState<Record<string, string>>({});
  const fields = ["kb_chunk_size", "kb_chunk_overlap", "kb_search_candidates", "kb_search_rerank"];

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
      updates[field] = field === "kb_search_rerank" ? value === "true" : Number(value);
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
            {field === "kb_search_rerank" ? (
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
