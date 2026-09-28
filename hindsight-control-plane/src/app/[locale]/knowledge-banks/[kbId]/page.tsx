"use client";

// A knowledge bank: its own page with a left rail (Overview · Documents · Search ·
// Operations · Configuration), the same shell the memory-bank page uses.

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { toast } from "sonner";
import {
  FileText,
  Layers,
  List,
  Network,
  MoreVertical,
  Plus,
  RefreshCw,
  Search as SearchIcon,
  Settings as SettingsIcon,
  Table2,
  Tags,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import { BankSelector } from "@/components/bank-selector";
import { KnowledgeBankSidebar, type KbSection } from "@/components/knowledge-bank-sidebar";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  DisclosureButton,
  Hint,
  Row,
  Section as Section2,
  Segmented,
} from "@/components/form-layout";
import { ErdDiagram, type ErdField, type ErdNode } from "@/components/kb-erd";
import { InfoCard, MetadataRow } from "@/components/ui/info-card";
import { InlineStat, StatStrip } from "@/components/ui/inline-stat";
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
  type SchemaField,
  type SearchResult,
} from "@/components/knowledge-bank-api";
import { FilterBuilder } from "@/components/kb-filter-builder";
import { SchemaEditor, type SchemaDraft } from "@/components/kb-schema-editor";
import { withBasePath } from "@/lib/base-path";

const SECTIONS: KbSection[] = ["overview", "documents", "collections", "settings"];

/** A page header in the memory-bank shape: the view's name, what the view is for, and
 *  its one action on the same row. No card chrome — the memory-bank views don't use any,
 *  and a page of nested boxes reads as a different product. */
function Section({
  title,
  description,
  action,
  sub,
  tab,
  children,
}: {
  title: string;
  description?: string;
  action?: React.ReactNode;
  /** A block within a view rather than the view itself: same shape, quieter heading,
   *  so one page does not read as three pages stacked. */
  sub?: boolean;
  /** The view is a tab, and the tab bar above it is its title — so it prints none. */
  tab?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className={`flex items-start justify-between gap-4 ${sub || tab ? "mb-3" : "mb-6"}`}>
        <div>
          {tab ? null : sub ? (
            <h2 className="text-lg font-semibold text-foreground">{title}</h2>
          ) : (
            <h1 className="text-3xl font-bold mb-2 text-foreground">{title}</h1>
          )}
          {description && (
            <p className={sub || tab ? "text-sm text-muted-foreground" : "text-muted-foreground"}>
              {description}
            </p>
          )}
        </div>
        {action}
      </div>
      {children}
    </div>
  );
}

/** A table's frame, as the memory-bank views draw it. */
function TableFrame({ children }: { children: React.ReactNode }) {
  return <div className="rounded-lg border border-border overflow-hidden">{children}</div>;
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

  // Search first, and so the default: a bank is asked questions far more often than its
  // document list is read.
  const rawDocTab = searchParams.get("docTab");
  const docTab = (
    rawDocTab === "schemas" || rawDocTab === "documents" ? rawDocTab : "search"
  ) as DocTab;
  const goDocTab = (next: DocTab) =>
    router.push(`/knowledge-banks/${encodeURIComponent(kbId)}?section=documents&docTab=${next}`);

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
                <div>
                  {section === "overview" && <Overview bank={bank} onGo={go} />}
                  {section === "documents" && (
                    <DocumentsSection
                      kbId={kbId}
                      onChanged={loadBank}
                      tab={docTab}
                      onTab={goDocTab}
                    />
                  )}
                  {section === "collections" && <CollectionsPanel kbId={kbId} />}
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
function Stats({ bank, sub }: { bank: KnowledgeBank; sub?: boolean }) {
  const stats = [
    ["Documents", bank.documents],
    ["Passages", bank.passages],
    ["Writes in flight", bank.operations_in_flight ?? 0],
    ["Last write", bank.last_write_at ? new Date(bank.last_write_at).toLocaleString() : "—"],
  ] as const;
  return (
    <Section title="Overview" sub={sub} description="What is in this bank right now.">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {stats.map(([label, value]) => (
          <div key={label}>
            <div className="text-[12px] text-muted-foreground">{label}</div>
            <div className="text-[20px] font-semibold">{value}</div>
          </div>
        ))}
      </div>
    </Section>
  );
}

function Overview({ bank, onGo }: { bank: KnowledgeBank; onGo: (s: KbSection) => void }) {
  return (
    <div className="space-y-6">
      <Stats bank={bank} />
      <div className="flex gap-2">
        <Button size="sm" variant="outline" onClick={() => onGo("documents")}>
          <FileText className="w-4 h-4 mr-1" /> Documents
        </Button>
        <Button size="sm" variant="outline" onClick={() => onGo("collections")}>
          <Table2 className="w-4 h-4 mr-1" /> Collections
        </Button>
      </div>
    </div>
  );
}

type DocTab = "documents" | "search" | "schemas";

/** Documents and schemas in one place: a document's fields only mean something next to
 *  the schema that defines them, and a schema is only worth editing to watch it fill. */
function DocumentsSection({
  kbId,
  onChanged,
  tab,
  onTab,
}: {
  kbId: string;
  onChanged: () => void;
  tab: DocTab;
  onTab: (tab: DocTab) => void;
}) {
  const [schemas, setSchemas] = useState<KnowledgeSchema[] | null>(null);

  const loadSchemas = useCallback(async () => {
    try {
      setSchemas(
        (await kbFetch<{ items: KnowledgeSchema[] }>(`/${encodeURIComponent(kbId)}/schemas`)).items
      );
    } catch {
      setSchemas([]);
    }
  }, [kbId]);

  useEffect(() => {
    loadSchemas();
  }, [loadSchemas]);

  const tabs: { id: DocTab; label: string }[] = [
    { id: "search", label: "Search" },
    { id: "documents", label: "Documents" },
    { id: "schemas", label: "Schemas" },
  ];
  return (
    <div>
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2 text-foreground">Knowledge</h1>
        <p className="text-muted-foreground">
          The documents in this bank, the schemas that give them fields, and search over their
          passages.
        </p>
      </div>
      <div className="border-b border-border mb-5 flex">
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
      {tab === "documents" && (
        <Documents kbId={kbId} onChanged={onChanged} schemas={schemas ?? []} />
      )}
      {tab === "search" && <SearchPanel kbId={kbId} schemas={schemas ?? []} />}
      {tab === "schemas" && <SchemaPanel kbId={kbId} onSaved={loadSchemas} />}
    </div>
  );
}

/** List or diagram, the segmented control the entities and memories pages use. */
function ViewToggle({
  value,
  onChange,
}: {
  value: "list" | "diagram";
  onChange: (view: "list" | "diagram") => void;
}) {
  const options = [
    { id: "list" as const, label: "List", Icon: List },
    { id: "diagram" as const, label: "Diagram", Icon: Network },
  ];
  return (
    <div className="flex items-center gap-2 bg-muted rounded-lg p-1">
      {options.map(({ id, label, Icon }) => (
        <button
          key={id}
          onClick={() => onChange(id)}
          className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all flex items-center gap-1.5 ${
            value === id
              ? "bg-background text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          <Icon className="w-4 h-4" />
          {label}
        </button>
      ))}
    </div>
  );
}

function Documents({
  kbId,
  onChanged,
  schemas,
}: {
  kbId: string;
  onChanged: () => void;
  schemas: KnowledgeSchema[];
}) {
  const [open, setOpen] = useState<string | null>(null);
  const [documents, setDocuments] = useState<KnowledgeDocument[] | null>(null);
  const [total, setTotal] = useState(0);
  const [adding, setAdding] = useState(false);
  const [tab, setTab] = useState<"text" | "upload">("text");
  const [form, setForm] = useState({ id: "", title: "", text: "" });
  const [files, setFiles] = useState<File[]>([]);
  const [saving, setSaving] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [schemaFilter, setSchemaFilter] = useState("all");

  // Both filters are the server's, as they are for a memory bank's documents: the list
  // is paginated, so filtering the page in the browser would filter the wrong set.
  const load = useCallback(async () => {
    try {
      const params = new URLSearchParams({ limit: "200" });
      if (searchQuery.trim()) params.set("q", searchQuery.trim());
      if (schemaFilter !== "all") params.set("schema_id", schemaFilter);
      const page = await kbFetch<{ items: KnowledgeDocument[]; total: number }>(
        `/${encodeURIComponent(kbId)}/documents?${params}`
      );
      setDocuments(page.items);
      setTotal(page.total);
    } catch (e) {
      toast.error((e as Error).message);
    }
  }, [kbId, searchQuery, schemaFilter]);

  useEffect(() => {
    const timer = setTimeout(load, searchQuery ? 250 : 0);
    return () => clearTimeout(timer);
  }, [load, searchQuery]);

  const done = (message: string) => {
    toast.success(message);
    setAdding(false);
    setForm({ id: "", title: "", text: "" });
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
      const result = await kbUpload(kbId, files);
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
    <Section
      title="Documents"
      tab
      description={`${total} document${total === 1 ? "" : "s"}. Writes are queued as operations and run in the background.`}
      action={
        <Button size="sm" onClick={() => setAdding(true)}>
          <Plus className="w-4 h-4 mr-1" /> Add document
        </Button>
      }
    >
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <div className="relative min-w-[220px] flex-1">
          <SearchIcon className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search by document id or title…"
            className="pl-8 pr-8 h-9"
          />
          {searchQuery && (
            <button
              type="button"
              onClick={() => setSearchQuery("")}
              aria-label="Clear search"
              className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
        <Select value={schemaFilter} onValueChange={setSchemaFilter}>
          <SelectTrigger className="w-[200px] h-9" aria-label="Filter by schema">
            <SelectValue />
          </SelectTrigger>
          <SelectContent position="popper">
            <SelectItem value="all">Any schema</SelectItem>
            {schemas.map((schema) => (
              <SelectItem key={schema.schema_id} value={schema.schema_id}>
                {schema.schema_id}
              </SelectItem>
            ))}
            {/* Not a missing filter: these are the documents nothing was extracted from. */}
            <SelectItem value="none">No schema</SelectItem>
          </SelectContent>
        </Select>
        {(searchQuery || schemaFilter !== "all") && (
          <Button
            variant="ghost"
            size="sm"
            className="h-9 gap-1 text-xs shrink-0"
            onClick={() => {
              setSearchQuery("");
              setSchemaFilter("all");
            }}
          >
            <X className="h-3.5 w-3.5" />
            Clear filters
          </Button>
        )}
      </div>

      {!documents ? (
        <Spinner />
      ) : documents.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          {searchQuery || schemaFilter !== "all"
            ? "No documents match these filters."
            : "Nothing written yet."}
        </p>
      ) : (
        <TableFrame>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Document</TableHead>
                <TableHead className="w-40">Schema</TableHead>
                <TableHead className="w-48">Updated</TableHead>
                <TableHead className="text-right w-24">Passages</TableHead>
                <TableHead className="w-12" />
              </TableRow>
            </TableHeader>
            <TableBody>
              {documents.map((doc) => (
                <TableRow
                  key={doc.doc_id}
                  className="cursor-pointer"
                  onClick={() => setOpen(doc.doc_id)}
                >
                  <TableCell className="font-mono text-sm">
                    {doc.doc_id}
                    {doc.title && <div className="text-xs text-muted-foreground">{doc.title}</div>}
                  </TableCell>
                  <TableCell className="text-xs">
                    {doc.schema_id ? (
                      <span className="font-mono">{doc.schema_id}</span>
                    ) : (
                      <span className="text-muted-foreground">none</span>
                    )}
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {new Date(doc.updated_at).toLocaleString()}
                  </TableCell>
                  <TableCell className="text-right">{doc.passage_count}</TableCell>
                  <TableCell className="text-right" onClick={(e) => e.stopPropagation()}>
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
        </TableFrame>
      )}

      <DocumentDetail kbId={kbId} docId={open} schemas={schemas} onClose={() => setOpen(null)} />

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
    </Section>
  );
}

/** Field values as chips. Shows what a document or passage actually carries, in the
 *  width a table cell has; the rest is behind the count. */
function FieldChips({ values, limit }: { values: Record<string, unknown>; limit?: number }) {
  const entries = Object.entries(values || {});
  if (entries.length === 0) return <span className="text-xs text-muted-foreground">—</span>;
  const shown = limit ? entries.slice(0, limit) : entries;
  return (
    <div className="flex flex-wrap gap-1">
      {shown.map(([name, value]) => (
        <span
          key={name}
          className="inline-flex items-center gap-1 rounded bg-muted px-1.5 py-0.5 text-[11px]"
          title={`${name}: ${formatFieldValue(value)}`}
        >
          <span className="font-mono text-muted-foreground">{name}</span>
          <span className="max-w-[12rem] truncate">{formatFieldValue(value)}</span>
        </span>
      ))}
      {limit && entries.length > limit && (
        <span className="text-[11px] text-muted-foreground">+{entries.length - limit}</span>
      )}
    </div>
  );
}

function formatFieldValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (Array.isArray(value)) return value.map((v) => formatFieldValue(v)).join(", ");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

interface DocumentWithPassages extends KnowledgeDocument {
  text: string;
  passages: {
    passage_index: number;
    text: string;
    token_count: number;
    fields: Record<string, unknown>;
  }[];
}

/** One document: which schema it was read with, what that schema asks for, what this
 *  document answered — and the same, per passage. A field the schema defines but nothing
 *  filled is shown empty rather than left out, because "we looked and found nothing" and
 *  "we never asked" are different answers. */
function DocumentDetail({
  kbId,
  docId,
  schemas,
  onClose,
}: {
  kbId: string;
  docId: string | null;
  schemas: KnowledgeSchema[];
  onClose: () => void;
}) {
  const [document, setDocument] = useState<DocumentWithPassages | null>(null);

  useEffect(() => {
    if (!docId) {
      setDocument(null);
      return;
    }
    let live = true;
    kbFetch<DocumentWithPassages>(
      `/${encodeURIComponent(kbId)}/documents/${encodeURIComponent(docId)}`
    )
      .then((d) => live && setDocument(d))
      .catch((e) => toast.error((e as Error).message));
    return () => {
      live = false;
    };
  }, [kbId, docId]);

  const schema = schemas.find((item) => item.schema_id === document?.schema_id) ?? null;

  return (
    <Dialog open={docId !== null} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="w-[95vw] max-w-[95vw] h-[92vh] sm:max-w-[95vw] flex flex-col overflow-hidden">
        <DialogHeader className="pr-10">
          <DialogTitle className="truncate font-mono text-sm">{docId}</DialogTitle>
        </DialogHeader>

        {!document ? (
          <div className="flex flex-1 items-center justify-center">
            <Spinner size="xl" variant="jump" />
          </div>
        ) : (
          <Tabs defaultValue="general" className="flex-1 flex flex-col overflow-hidden">
            <TabsList className="grid grid-cols-3 w-full max-w-md">
              <TabsTrigger value="general" className="flex items-center gap-1.5">
                <SettingsIcon className="w-3.5 h-3.5" /> General
              </TabsTrigger>
              <TabsTrigger value="fields" className="flex items-center gap-1.5">
                <Tags className="w-3.5 h-3.5" /> Fields
              </TabsTrigger>
              <TabsTrigger value="passages" className="flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5" /> Passages ({document.passages.length})
              </TabsTrigger>
            </TabsList>

            <TabsContent value="general" className="flex-1 overflow-y-auto mt-4">
              <div className="space-y-4">
                {/* The same InfoCard/MetadataRow furniture the memory-bank document
                    dialog uses, so the two read as one product. */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <InfoCard title="Document" icon={<FileText className="w-3.5 h-3.5" />}>
                    <MetadataRow
                      label="Id"
                      value={<span className="font-mono text-xs">{document.doc_id}</span>}
                    />
                    <MetadataRow label="Title" value={document.title || "—"} />
                    <MetadataRow
                      label="Created"
                      value={new Date(document.created_at).toLocaleString()}
                    />
                    <MetadataRow
                      label="Updated"
                      value={new Date(document.updated_at).toLocaleString()}
                    />
                    <MetadataRow
                      label="Size"
                      value={`${(document.text ?? "").length.toLocaleString()} characters`}
                    />
                  </InfoCard>

                  <InfoCard title="Extraction" icon={<Tags className="w-3.5 h-3.5" />}>
                    <MetadataRow
                      label="Schema"
                      value={
                        document.schema_id ? (
                          <span className="font-mono text-xs">{document.schema_id}</span>
                        ) : (
                          <span className="italic text-muted-foreground">
                            none — nothing was extracted from this document
                          </span>
                        )
                      }
                    />
                    <MetadataRow
                      label="Fields filled"
                      value={Object.keys(document.fields || {}).length}
                    />
                    <MetadataRow label="Passages" value={document.passages.length} />
                    {Object.keys(document.metadata || {}).length > 0 && (
                      <MetadataRow
                        label="Metadata"
                        value={<FieldChips values={document.metadata} />}
                      />
                    )}
                  </InfoCard>
                </div>

                <InfoCard title="Text" icon={<FileText className="w-3.5 h-3.5" />}>
                  <pre className="text-sm whitespace-pre-wrap font-sans leading-relaxed">
                    {document.text}
                  </pre>
                </InfoCard>
              </div>
            </TabsContent>

            <TabsContent value="fields" className="flex-1 overflow-y-auto mt-4">
              <div className="space-y-4">
                <InfoCard title="Document fields" icon={<Tags className="w-3.5 h-3.5" />}>
                  <FieldTable definition={schema?.document_fields} values={document.fields} />
                </InfoCard>
                {schema && Object.keys(schema.passage_fields).length > 0 && (
                  <p className="text-xs text-muted-foreground">
                    This schema also asks each passage for{" "}
                    <span className="font-mono">
                      {Object.keys(schema.passage_fields).join(", ")}
                    </span>
                    ; the answers are on the Passages tab.
                  </p>
                )}
              </div>
            </TabsContent>

            <TabsContent value="passages" className="flex-1 overflow-y-auto mt-4">
              <div className="space-y-4">
                {document.passages.map((passage) => (
                  <InfoCard
                    key={passage.passage_index}
                    title={`Passage ${passage.passage_index}`}
                    icon={<Layers className="w-3.5 h-3.5" />}
                  >
                    {Object.keys(passage.fields || {}).length > 0 && (
                      <MetadataRow label="Fields" value={<FieldChips values={passage.fields} />} />
                    )}
                    <MetadataRow
                      label="Text"
                      value={<p className="whitespace-pre-wrap">{passage.text}</p>}
                    />
                  </InfoCard>
                ))}
              </div>
            </TabsContent>
          </Tabs>
        )}
      </DialogContent>
    </Dialog>
  );
}

/** Every field a schema defines, beside what this document answered for it. */
function FieldTable({
  definition,
  values,
}: {
  definition?: Record<string, SchemaField>;
  values: Record<string, unknown>;
}) {
  // A value with no definition still shows: it was supplied with the write, or the
  // schema changed after extraction, and hiding it would hide what search can filter on.
  const names = Array.from(new Set([...Object.keys(definition ?? {}), ...Object.keys(values)]));
  return (
    <div>
      {names.length === 0 ? (
        <p className="text-xs text-muted-foreground">
          No fields. Define a schema to give this kind of document fields.
        </p>
      ) : (
        <TableFrame>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-56">Field</TableHead>
                <TableHead className="w-32">Type</TableHead>
                <TableHead>Value</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {names.map((name) => {
                const spec = definition?.[name];
                return (
                  <TableRow key={name}>
                    <TableCell className="font-mono text-xs">
                      {name}
                      {spec?.filterable && (
                        <span className="ml-1 text-[10px] text-muted-foreground">filterable</span>
                      )}
                      {spec?.indexed && (
                        <span className="ml-1 text-[10px] text-muted-foreground">indexed</span>
                      )}
                    </TableCell>
                    <TableCell className="text-xs text-muted-foreground">
                      {spec?.type ?? "—"}
                      {/* A fixed value list IS the classification, so it belongs next to
                          the type: it says what an answer may be, not just its shape. */}
                      {spec?.values && spec.values.length > 0 && (
                        <div className="text-[10px]">{spec.values.join(" · ")}</div>
                      )}
                    </TableCell>
                    <TableCell className="text-xs">
                      {name in values ? (
                        formatFieldValue(values[name])
                      ) : (
                        <span className="text-muted-foreground">not filled</span>
                      )}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </TableFrame>
      )}
    </div>
  );
}

function SearchPanel({ kbId, schemas }: { kbId: string; schemas: KnowledgeSchema[] }) {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<"hybrid" | "vector" | "keyword">("hybrid");
  const [topK, setTopK] = useState(10);
  const [collapse, setCollapse] = useState(false);
  const [filter, setFilter] = useState<Record<string, unknown> | null>(null);
  const [schemaId, setSchemaId] = useState<string | null>(null);
  const [optionsOpen, setOptionsOpen] = useState(false);
  const [results, setResults] = useState<SearchResult[] | null>(null);
  const [loading, setLoading] = useState(false);

  const run = async () => {
    if (!query.trim()) return;
    setLoading(true);
    try {
      const response = await kbFetch<{ results: SearchResult[] }>(
        `/${encodeURIComponent(kbId)}/search`,
        {
          method: "POST",
          body: {
            query,
            mode,
            top_k: topK,
            collapse_documents: collapse,
            fields: filter ?? undefined,
            schema_id: schemaId ?? undefined,
          },
        }
      );
      setResults(response.results);
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  // Only a filterable field can be filtered on, so those are the only ones offered —
  // each under the schema that defines it, because two schemas may name the same field.
  const filterable = schemas.flatMap((schema) => [
    ...Object.entries(schema.document_fields)
      .filter(([, spec]) => spec.filterable)
      .map(([name, spec]) => ({
        name,
        spec,
        schemaId: schema.schema_id,
        level: "document" as const,
      })),
    ...Object.entries(schema.passage_fields)
      .filter(([, spec]) => spec.filterable)
      .map(([name, spec]) => ({
        name,
        spec,
        schemaId: schema.schema_id,
        level: "passage" as const,
      })),
  ]);

  // The badge on the collapsed disclosure: how many options differ from the defaults.
  const activeOptions =
    (schemaId ? 1 : 0) + (filter ? 1 : 0) + (topK !== 10 ? 1 : 0) + (collapse ? 1 : 0);

  return (
    <Section title="Search" tab description="Hybrid vector + keyword search, reranked.">
      <div className="flex gap-3">
        <div className="relative flex-1">
          <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && run()}
            placeholder="Ask something…"
            className="pl-10 h-12 text-lg"
            autoFocus
          />
        </div>
        <Button onClick={run} disabled={loading || !query.trim()} className="h-12 px-8">
          {loading ? "Searching…" : "Search"}
        </Button>
      </div>

      {/* The everyday control inline, everything else behind Options — the shape the
          Recall page uses, with the same building blocks. */}
      <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-3">
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">Mode</span>
          <Hint text="Hybrid fuses a vector arm and a keyword arm; the single arms are for seeing what each one finds on its own." />
          <Segmented
            value={mode}
            onChange={setMode}
            ariaLabel="Search mode"
            options={[
              { value: "hybrid", label: "Hybrid" },
              { value: "vector", label: "Vector" },
              { value: "keyword", label: "Keyword" },
            ]}
          />
        </div>
        <div className="ml-auto">
          <DisclosureButton
            open={optionsOpen}
            onToggle={() => setOptionsOpen((open) => !open)}
            label="Options"
            badge={activeOptions}
          />
        </div>
      </div>

      {optionsOpen && (
        <div className="mt-5 grid gap-8 md:grid-cols-2">
          <Section2 title="Retrieval">
            <Row label="Results" description="How many passages to return." htmlFor="kb-top-k">
              <Input
                id="kb-top-k"
                type="number"
                min={1}
                max={200}
                value={topK}
                onChange={(e) => setTopK(Math.max(1, Number(e.target.value) || 1))}
              />
            </Row>
            <Row
              label="One per document"
              description="Keep only each document's best passage, so the count means that many distinct documents."
            >
              <Segmented
                value={collapse ? "yes" : "no"}
                onChange={(v) => setCollapse(v === "yes")}
                ariaLabel="One passage per document"
                options={[
                  { value: "no", label: "Off" },
                  { value: "yes", label: "On" },
                ]}
              />
            </Row>
          </Section2>

          <Section2 title="Filters">
            <FilterBuilder
              fields={filterable}
              schemas={schemas.map((schema) => schema.schema_id)}
              schemaId={schemaId}
              onSchemaChange={setSchemaId}
              onChange={setFilter}
            />
          </Section2>
        </div>
      )}

      {results && (
        <Tabs defaultValue="data" className="mt-6">
          <TabsList className="grid w-full max-w-xs grid-cols-2">
            <TabsTrigger value="data">Data</TabsTrigger>
            <TabsTrigger value="json">JSON</TabsTrigger>
          </TabsList>

          <TabsContent value="data" className="mt-3 space-y-3">
            {results.length === 0 && (
              <p className="text-sm text-muted-foreground">Nothing found.</p>
            )}
            {results.map((hit, i) => (
              <div
                key={`${hit.document_id}#${hit.passage_index}`}
                className="rounded-lg border border-border p-3"
              >
                <div className="flex items-center gap-2 text-sm mb-1">
                  <span className="text-muted-foreground">#{i + 1}</span>
                  <span className="font-mono">{hit.document_id}</span>
                  <span className="text-muted-foreground">passage {hit.passage_index}</span>
                </div>
                <p className="text-sm whitespace-pre-wrap">{hit.text}</p>
              </div>
            ))}
          </TabsContent>

          {/* The response as it came back — what a client would have to parse. */}
          <TabsContent value="json" className="mt-3">
            <pre className="max-h-[600px] overflow-auto rounded-lg border border-border bg-muted/30 p-4 text-xs leading-relaxed">
              {JSON.stringify({ results }, null, 2)}
            </pre>
          </TabsContent>
        </Tabs>
      )}
    </Section>
  );
}

function SchemaPanel({ kbId, onSaved }: { kbId: string; onSaved?: () => void }) {
  const [schemas, setSchemas] = useState<KnowledgeSchema[] | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [newId, setNewId] = useState("");
  const [creating, setCreating] = useState(false);
  // A schema is read far more often than it is changed, so the right pane opens on the
  // fields and the form is one click away.
  const [editing, setEditing] = useState(false);
  const [pane, setPane] = useState<"schema" | "data">("schema");
  // A saved schema changes what the next write extracts; whether it also re-reads what
  // is already stored costs LLM calls, so it is asked rather than assumed.
  const [pendingSave, setPendingSave] = useState<SchemaDraft | null>(null);
  const [view, setView] = useState<"list" | "diagram">("list");

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

  const current = schemas?.find((schema) => schema.schema_id === selected) ?? null;

  const save = async (schemaId: string, body: unknown, reprocess = false) => {
    setSaving(true);
    try {
      await kbFetch(`/${encodeURIComponent(kbId)}/schemas/${encodeURIComponent(schemaId)}`, {
        method: "PUT",
        body,
      });
      if (reprocess) {
        // only_missing false: the point of asking was to re-read documents that already
        // have values, because the schema that produced them just changed.
        await kbFetch(`/${encodeURIComponent(kbId)}/fields/extract`, {
          method: "POST",
          body: { schema_id: schemaId, only_missing: false },
        });
        toast.success("Saved. Re-reading the documents in the background — see Settings.");
      } else {
        toast.success("Saved. It applies to the next write.");
      }
      setSelected(schemaId);
      setEditing(false);
      await load();
      onSaved?.();
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setSaving(false);
    }
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
    <Section
      title="Schemas"
      tab
      description="A schema is the fields one kind of document has. A bank with several lets the LLM classify which one a document is."
      action={
        <Button size="sm" variant="outline" onClick={() => setCreating(true)}>
          <Plus className="w-4 h-4 mr-1" /> New schema
        </Button>
      }
    >
      <div className="mb-6 flex items-center justify-between">
        <div className="text-sm text-muted-foreground">
          {schemas.length} schema{schemas.length === 1 ? "" : "s"}
        </div>
        <ViewToggle value={view} onChange={setView} />
      </div>
      {view === "diagram" && schemas.length > 0 && (
        <div className="mb-6">
          <ErdDiagram
            nodes={schemas.map(schemaCard)}
            edges={[]}
            selected={selected}
            onSelect={(id) => {
              setSelected(id);
              setEditing(false);
              setPane("schema");
            }}
          />
        </div>
      )}

      {schemas.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          No schemas yet. Create one to give documents fields that search and query can filter on.
        </p>
      ) : view === "diagram" ? null : (
        // List on the left, the schema itself on the right: picking one is a move of the
        // eye rather than a scroll past the table you were just reading.
        <div className="grid gap-5 md:grid-cols-[minmax(200px,260px)_1fr] items-start">
          <div className="rounded-lg border border-border overflow-hidden divide-y divide-border">
            {schemas.map((schema) => (
              <button
                key={schema.schema_id}
                onClick={() => {
                  setSelected(schema.schema_id);
                  setEditing(false);
                  setPane("schema");
                }}
                className={`w-full text-left px-3 py-2.5 transition-colors ${
                  schema.schema_id === selected ? "bg-accent" : "hover:bg-muted/50"
                }`}
              >
                <div className="font-mono text-xs">{schema.schema_id}</div>
                {schema.name && (
                  <div className="text-xs text-muted-foreground truncate">{schema.name}</div>
                )}
                <div className="text-[11px] text-muted-foreground mt-0.5">
                  {Object.keys(schema.document_fields).length +
                    Object.keys(schema.passage_fields).length}{" "}
                  fields · {schema.documents_with_fields} filled
                </div>
              </button>
            ))}
          </div>

          <div>
            {!current ? (
              <p className="text-sm text-muted-foreground">Pick a schema to see its fields.</p>
            ) : (
              <>
                <div className="mb-4 flex items-start justify-between gap-3">
                  <div>
                    <div className="font-mono text-sm font-semibold">{current.schema_id}</div>
                    <p className="text-xs text-muted-foreground">
                      {current.description ||
                        "Filled on the next write; run the extraction for documents already stored."}
                    </p>
                  </div>
                  <div className="flex gap-2 shrink-0">
                    {pane === "schema" && !editing && (
                      <Button size="sm" variant="outline" onClick={() => setEditing(true)}>
                        Edit
                      </Button>
                    )}
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="outline" size="sm">
                          Actions
                          <MoreVertical className="w-4 h-4 ml-2" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end" className="w-56">
                        <DropdownMenuItem onClick={extract}>
                          <RefreshCw className="w-4 h-4 mr-2" />
                          Extract missing values
                        </DropdownMenuItem>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem
                          onClick={() => remove(current.schema_id)}
                          className="text-destructive focus:text-destructive"
                        >
                          <Trash2 className="w-4 h-4 mr-2" />
                          Delete schema
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                </div>

                <StatStrip className="mb-4">
                  <InlineStat
                    icon={Tags}
                    label="Fields"
                    value={
                      Object.keys(current.document_fields).length +
                      Object.keys(current.passage_fields).length
                    }
                  />
                  <InlineStat
                    icon={FileText}
                    label="Documents filled"
                    value={current.documents_with_fields}
                  />
                  <InlineStat
                    icon={Layers}
                    label="Passages filled"
                    value={current.passages_with_fields}
                  />
                </StatStrip>

                {/* Its definition, and what that definition actually caught. */}
                <div className="border-b border-border mb-4 flex">
                  {(["schema", "data"] as const).map((item) => (
                    <button
                      key={item}
                      onClick={() => setPane(item)}
                      className={`px-4 py-2 text-sm font-semibold transition-all relative capitalize ${
                        pane === item
                          ? "text-primary"
                          : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      {item}
                      {pane === item && (
                        <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary-gradient" />
                      )}
                    </button>
                  ))}
                </div>

                {pane === "data" ? (
                  <SchemaDocuments kbId={kbId} schemaId={current.schema_id} />
                ) : editing ? (
                  /* Remounted per schema, so switching never carries one schema's edits
                     into another's. */
                  <SchemaEditor
                    key={current.schema_id}
                    initial={{
                      name: current.name,
                      description: current.description,
                      document_fields: current.document_fields,
                      passage_fields: current.passage_fields,
                    }}
                    saving={saving}
                    onSave={(next: SchemaDraft) => setPendingSave(next)}
                  />
                ) : (
                  <ErdDiagram nodes={[schemaCard(current)]} edges={[]} />
                )}
              </>
            )}
          </div>
        </div>
      )}

      <Dialog open={pendingSave !== null} onOpenChange={(open) => !open && setPendingSave(null)}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Apply to the documents already stored?</DialogTitle>
            <DialogDescription>
              A saved schema is read on the next write either way. Re-reading what is already stored
              fills the new fields on those documents too — one LLM call per document, run in the
              background.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              disabled={saving}
              onClick={() => {
                const draft = pendingSave;
                setPendingSave(null);
                if (draft && current) save(current.schema_id, draft);
              }}
            >
              Save only
            </Button>
            <Button
              disabled={saving}
              onClick={() => {
                const draft = pendingSave;
                setPendingSave(null);
                if (draft && current) save(current.schema_id, draft, true);
              }}
            >
              Save and re-read documents
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

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
    </Section>
  );
}

/** The documents this schema was read with — what the definition actually caught. */
function SchemaDocuments({ kbId, schemaId }: { kbId: string; schemaId: string }) {
  const [documents, setDocuments] = useState<KnowledgeDocument[] | null>(null);
  const [total, setTotal] = useState(0);

  useEffect(() => {
    let live = true;
    kbFetch<{ items: KnowledgeDocument[]; total: number }>(
      `/${encodeURIComponent(kbId)}/documents?limit=100&schema_id=${encodeURIComponent(schemaId)}`
    )
      .then((page) => {
        if (!live) return;
        setDocuments(page.items);
        setTotal(page.total);
      })
      .catch((e) => toast.error((e as Error).message));
    return () => {
      live = false;
    };
  }, [kbId, schemaId]);

  if (!documents) return <Spinner />;
  if (documents.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">No documents were read with this schema yet.</p>
    );
  }
  return (
    <div className="space-y-2">
      <p className="text-xs text-muted-foreground">
        {total} document{total === 1 ? "" : "s"} read with this schema.
      </p>
      <TableFrame>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Document</TableHead>
              <TableHead>Fields</TableHead>
              <TableHead className="text-right w-24">Passages</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {documents.map((doc) => (
              <TableRow key={doc.doc_id}>
                <TableCell className="font-mono text-xs">
                  {doc.doc_id}
                  {doc.title && <div className="text-xs text-muted-foreground">{doc.title}</div>}
                </TableCell>
                <TableCell>
                  <FieldChips values={doc.fields} />
                </TableCell>
                <TableCell className="text-right text-xs">{doc.passage_count}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableFrame>
    </div>
  );
}

/** One schema as a diagram card: its document fields, then its passage fields, with the
 *  flags and allowed values that decide what a search can do with them. */
function schemaCard(schema: KnowledgeSchema): ErdNode {
  const field = (name: string, spec: SchemaField, passage: boolean): ErdField => ({
    name,
    type: spec.type + (spec.type === "array" && spec.items ? ` of ${spec.items}` : ""),
    // The card groups by level rather than renaming the field: "clause" is the field's
    // name, and "(passage)" tacked on was a label pretending to be one.
    group: passage ? "passage" : "document",
    badges: [spec.filterable ? "filterable" : null, spec.indexed ? "indexed" : null].filter(
      Boolean
    ) as string[],
    values: spec.values,
    description: spec.description,
  });
  return {
    id: schema.schema_id,
    title: schema.schema_id,
    badge: "schema",
    fields: [
      ...Object.entries(schema.document_fields).map(([name, spec]) => field(name, spec, false)),
      // A passage field is filled per passage, not per document, so it is marked rather
      // than mixed in with the document's own.
      ...Object.entries(schema.passage_fields).map(([name, spec]) => field(name, spec, true)),
    ],
  };
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
  const [view, setView] = useState<"list" | "diagram">("list");

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
    <div className="space-y-8">
      <Section
        title="Collections"
        description="A collection is a kind of thing the documents talk about; each record folds together what every document said about one of them."
        action={
          <Button size="sm" variant="outline" onClick={() => openEditor()}>
            <Plus className="w-4 h-4 mr-1" /> New collection
          </Button>
        }
      >
        <div className="mb-6 flex items-center justify-between">
          <div className="text-sm text-muted-foreground">
            {collections.length} collection{collections.length === 1 ? "" : "s"}
          </div>
          <ViewToggle value={view} onChange={setView} />
        </div>
        {view === "diagram" && collections.length > 0 && (
          <ErdDiagram
            nodes={collections.map((collection) => ({
              id: collection.collection_id,
              title: collection.collection_id,
              badge: "collection",
              fields: Object.entries(collection.fields).map(([name, spec]) => ({
                name,
                // A relationship field holds another collection's record id, so its
                // "type" is that collection — which is also where its line goes.
                type: spec.collection ?? spec.type ?? "string",
                primary: collection.identity === name,
                relation: Boolean(spec.collection),
              })),
            }))}
            edges={collections.flatMap((collection) =>
              Object.entries(collection.fields)
                .filter(([, spec]) => spec.collection)
                .map(([name, spec]) => ({
                  from: collection.collection_id,
                  fromField: name,
                  to: String(spec.collection),
                }))
            )}
            selected={selected}
            onSelect={setSelected}
          />
        )}

        {view === "diagram" && collections.length > 0 ? null : collections.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No collections yet. Define one — its fields are what an LLM reads out of each document,
            and the identity field is what makes the same thing found twice one record.
          </p>
        ) : (
          <TableFrame>
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
          </TableFrame>
        )}
      </Section>

      {current && (
        <Section
          title={`Records · ${current.collection_id}`}
          sub
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
        </Section>
      )}

      {current && (
        <Section
          title="Query"
          sub
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
        </Section>
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
    </div>
  );
}

/** Rows of a query result, whatever its columns turn out to be. */
function ResultTable({ result }: { result: QueryResult }) {
  return (
    <TableFrame>
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
    </TableFrame>
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
    <Section
      title="Operations"
      sub
      description="Every write is an operation; this list refreshes itself."
    >
      {!operations ? (
        <Spinner />
      ) : operations.length === 0 ? (
        <p className="text-sm text-muted-foreground">No writes yet.</p>
      ) : (
        <TableFrame>
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
        </TableFrame>
      )}
    </Section>
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
        <h1 className="text-3xl font-bold mb-2 text-foreground">Settings</h1>
        <p className="text-muted-foreground">
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
        <div className="space-y-8">
          <Stats bank={bank} sub />
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
    <Section title="Configuration" tab description="Chunking and search settings for this bank.">
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
    </Section>
  );
}
