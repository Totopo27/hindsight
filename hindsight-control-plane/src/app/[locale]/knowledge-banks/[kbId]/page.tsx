"use client";

// A knowledge bank: its own page with a left rail (Overview · Documents · Search ·
// Operations · Configuration), the same shell the memory-bank page uses.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { toast } from "sonner";
import {
  FileText,
  Layers,
  List,
  Network,
  MoreVertical,
  Pencil,
  Plus,
  RefreshCw,
  RotateCcw,
  Activity,
  Search as SearchIcon,
  Settings as SettingsIcon,
  Sparkles,
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
import { Constellation } from "@/components/constellation";
import { ErdDiagram, type ErdField, type ErdNode } from "@/components/kb-erd";
import { CollectionsEditor } from "@/components/kb-collections-editor";
import { QueryBuilder } from "@/components/kb-query-builder";
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
  type QueryResult,
  type CollectionStats,
  type KnowledgeRecord,
  type KnowledgeRecordDetail,
  type KnowledgeSchema,
  type PassagePoint,
  type SchemaField,
  type SearchResult,
} from "@/components/knowledge-bank-api";
import { FilterBuilder } from "@/components/kb-filter-builder";
import { SchemaEditor, type SchemaDraft } from "@/components/kb-schema-editor";
import { withBasePath } from "@/lib/base-path";
import { LLMRequestsView } from "@/components/llm-requests-view";
import {
  Distribution,
  ProgressRow,
  OperationsCard,
  SectionHeading as StatsHeading,
  type BankStats,
} from "@/components/bank-stats-view";
import { BankOperationsView } from "@/components/bank-operations-view";
import { ConfigSection, FieldRow } from "@/components/bank-config-view";
import { LlmHealthDialog } from "@/components/llm-health-dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { client } from "@/lib/api";
import { FeatureNotEnabled } from "@/components/feature-not-enabled";
import { useFeatures } from "@/lib/features-context";

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

  // The header's Add document button lives beside the bank selector, as it does for a
  // memory bank; the dialog is this page's, so the button asks for it by event.
  const [addRequested, setAddRequested] = useState(0);
  useEffect(() => {
    const open = () => {
      router.push(`/knowledge-banks/${encodeURIComponent(kbId)}?section=documents`);
      setAddRequested((n) => n + 1);
    };
    window.addEventListener("hindsight:kb-add-document", open);
    return () => window.removeEventListener("hindsight:kb-add-document", open);
  }, [kbId, router]);

  const rawDocTab = searchParams.get("docTab");
  const docTab = (
    rawDocTab === "schemas" || rawDocTab === "search" ? rawDocTab : "documents"
  ) as DocTab;
  const goDocTab = (next: DocTab) =>
    router.push(`/knowledge-banks/${encodeURIComponent(kbId)}?section=documents&docTab=${next}`);

  const rawSettingsTab = searchParams.get("settingsTab");
  const settingsTab = (
    rawSettingsTab === "configuration" || rawSettingsTab === "llm-requests"
      ? rawSettingsTab
      : "general"
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
        <BankSelector knowledgeBankId={kbId} />
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
                  {section === "overview" && <Overview bank={bank} />}
                  {section === "documents" && (
                    <DocumentsSection
                      kbId={kbId}
                      onChanged={loadBank}
                      tab={docTab}
                      onTab={goDocTab}
                      addRequested={addRequested}
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
function Stats({ bank }: { bank: KnowledgeBank }) {
  // Operations by status come from the memory banks' stats endpoint: a knowledge bank's
  // writes are ordinary async operations of its bank row.
  const [byStatus, setByStatus] = useState<Record<string, number>>({});
  useEffect(() => {
    client
      .getBankStats(bank.bank_id)
      .then((s) => setByStatus((s as BankStats).operations_by_status ?? {}))
      .catch(() => setByStatus({}));
  }, [bank.bank_id, bank.operations_in_flight]);

  return (
    <div className="space-y-8">
      <section>
        <StatsHeading>Knowledge store</StatsHeading>
        <StatStrip>
          <InlineStat icon={FileText} label="Documents" value={bank.documents} />
          <InlineStat icon={Layers} label="Passages" value={bank.passages} />
          <InlineStat
            icon={Activity}
            label="Writes in flight"
            value={bank.operations_in_flight ?? 0}
          />
        </StatStrip>
        <p className="text-xs text-muted-foreground mt-2">
          Last write: {bank.last_write_at ? new Date(bank.last_write_at).toLocaleString() : "never"}
        </p>
      </section>
      <section>
        <StatsHeading>Activity</StatsHeading>
        <OperationsCard byStatus={byStatus} />
      </section>
    </div>
  );
}

function Overview({ bank }: { bank: KnowledgeBank }) {
  const router = useRouter();
  const [schemas, setSchemas] = useState<KnowledgeSchema[] | null>(null);
  const [collections, setCollections] = useState<KnowledgeCollection[] | null>(null);
  const [byStatus, setByStatus] = useState<Record<string, number>>({});
  const id = encodeURIComponent(bank.bank_id);

  useEffect(() => {
    kbFetch<{ items: KnowledgeSchema[] }>(`/${id}/schemas`)
      .then((r) => setSchemas(r.items))
      .catch(() => setSchemas([]));
    kbFetch<{ items: KnowledgeCollection[] }>(`/${id}/collections`)
      .then((r) => setCollections(r.items))
      .catch(() => setCollections([]));
    client
      .getBankStats(bank.bank_id)
      .then((s) => setByStatus((s as BankStats).operations_by_status ?? {}))
      .catch(() => setByStatus({}));
  }, [id, bank.bank_id, bank.operations_in_flight]);

  const records = (collections ?? []).reduce((sum, c) => sum + (c.records ?? 0), 0);
  const open = (query: string) => router.push(`/knowledge-banks/${id}?${query}`);
  const links = [
    {
      icon: FileText,
      title: "Documents",
      text: "Browse what was written and the fields pulled from it.",
      query: "section=documents&docTab=documents",
    },
    {
      icon: SearchIcon,
      title: "Search",
      text: "Hybrid search over the passages, filtered by fields.",
      query: "section=documents&docTab=search",
    },
    {
      icon: Tags,
      title: "Schemas",
      text: "The fields each kind of document has.",
      query: "section=documents&docTab=schemas",
    },
    {
      icon: Table2,
      title: "Collections",
      text: "Records derived from documents, queried like tables.",
      query: "section=collections",
    },
  ];

  return (
    <Section
      title="Overview"
      description={
        bank.last_write_at
          ? `What is in this bank right now. Last write ${new Date(bank.last_write_at).toLocaleString()}.`
          : "What is in this bank right now. Nothing written yet."
      }
    >
      <div className="space-y-8">
        <StatStrip className="sm:grid-cols-5">
          <InlineStat icon={FileText} label="Documents" value={bank.documents} />
          <InlineStat icon={Layers} label="Passages" value={bank.passages} />
          <InlineStat icon={Tags} label="Schemas" value={schemas?.length ?? 0} />
          <InlineStat icon={Table2} label="Collections" value={collections?.length ?? 0} />
          <InlineStat icon={List} label="Records" value={records} />
        </StatStrip>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="lg:col-span-2">
            {schemas && schemas.length > 0 ? (
              <SchemaShareCard schemas={schemas} />
            ) : (
              <div className="rounded-lg border border-dashed border-border p-5 text-sm text-muted-foreground h-full flex items-center">
                No schema yet: documents are searchable, but carry no fields. Add one under
                Documents › Schemas.
              </div>
            )}
          </div>
          <OperationsCard byStatus={byStatus} />
        </div>

        <section>
          <StatsHeading>How a knowledge bank works</StatsHeading>
          <div className="rounded-lg border border-border bg-card p-4 overflow-x-auto">
            <img
              src={withBasePath("/img/knowledge/how-knowledge-banks-work.svg")}
              alt="A Document is kept as Passages. A Schema says which Fields a kind of document has; a Collection keeps one Record per thing across documents. Search finds passages; Query filters and counts on fields and records."
              className="w-full max-w-[900px] mx-auto"
            />
          </div>
        </section>

        <section>
          <StatsHeading>Go to</StatsHeading>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {links.map((link) => (
              <button
                key={link.title}
                onClick={() => open(link.query)}
                className="text-left rounded-lg border border-border bg-card p-4 hover:border-primary/50 hover:bg-muted/40 transition-colors"
              >
                <div className="flex items-center gap-2 mb-1">
                  <div className="p-1.5 rounded-md bg-muted">
                    <link.icon className="w-4 h-4 text-muted-foreground" />
                  </div>
                  <span className="font-semibold text-sm">{link.title}</span>
                </div>
                <p className="text-xs text-muted-foreground">{link.text}</p>
              </button>
            ))}
          </div>
        </section>
      </div>
    </Section>
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
  addRequested,
}: {
  kbId: string;
  onChanged: () => void;
  tab: DocTab;
  onTab: (tab: DocTab) => void;
  /** Bumped by the header's Add document button; each bump opens the dialog once. */
  addRequested: number;
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
    { id: "documents", label: "Documents" },
    { id: "search", label: "Search" },
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
        <Documents
          kbId={kbId}
          onChanged={onChanged}
          schemas={schemas ?? []}
          addRequested={addRequested}
        />
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
  options = [
    { id: "list" as const, label: "List", Icon: List },
    { id: "diagram" as const, label: "Diagram", Icon: Network },
  ],
}: {
  value: "list" | "diagram";
  onChange: (view: "list" | "diagram") => void;
  options?: { id: "list" | "diagram"; label: string; Icon: typeof List }[];
}) {
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
  addRequested,
}: {
  kbId: string;
  onChanged: () => void;
  schemas: KnowledgeSchema[];
  addRequested: number;
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
  const [page, setPage] = useState(0);
  // The map is what the tab opens on: a bank of a thousand passages is a shape before it
  // is a list, and the shape is what says whether the schemas caught anything.
  const [view, setView] = useState<"list" | "diagram">("diagram");

  // Both filters are the server's, as they are for a memory bank's documents: the list
  // is paginated, so filtering the page in the browser would filter the wrong set.
  const load = useCallback(async () => {
    try {
      const params = new URLSearchParams({
        limit: String(DOCUMENTS_PER_PAGE),
        offset: String(page * DOCUMENTS_PER_PAGE),
      });
      if (searchQuery.trim()) params.set("q", searchQuery.trim());
      if (schemaFilter !== "all") params.set("schema_id", schemaFilter);
      const result = await kbFetch<{ items: KnowledgeDocument[]; total: number }>(
        `/${encodeURIComponent(kbId)}/documents?${params}`
      );
      setDocuments(result.items);
      setTotal(result.total);
    } catch (e) {
      toast.error((e as Error).message);
    }
  }, [kbId, searchQuery, schemaFilter, page]);

  // A new filter is a new list: start it from its first page.
  useEffect(() => setPage(0), [searchQuery, schemaFilter]);

  useEffect(() => {
    const timer = setTimeout(load, searchQuery ? 250 : 0);
    return () => clearTimeout(timer);
  }, [load, searchQuery]);

  // Zero is the initial value, not a request, so the dialog does not open on mount.
  useEffect(() => {
    if (addRequested > 0) setAdding(true);
  }, [addRequested]);

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
      description="Writes are queued as operations and run in the background."
    >
      <div className="mb-6 flex items-center justify-between">
        <div className="text-sm text-muted-foreground">
          {total} document{total === 1 ? "" : "s"}
        </div>
        <ViewToggle
          value={view}
          onChange={setView}
          options={[
            { id: "diagram", label: "Map", Icon: Sparkles },
            { id: "list", label: "List", Icon: List },
          ]}
        />
      </div>

      {view === "diagram" && <PassageMap kbId={kbId} schemas={schemas} onOpen={setOpen} />}

      {view === "list" && (
        <>
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
                        {doc.title && (
                          <div className="text-xs text-muted-foreground">{doc.title}</div>
                        )}
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
          {documents && documents.length > 0 && (
            <div className="mt-3">
              <Pager
                page={page}
                total={total}
                perPage={DOCUMENTS_PER_PAGE}
                noun="document"
                onPage={setPage}
              />
            </div>
          )}
        </>
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
/** The bank as a star field: one point per passage, clustered by the schema its document
 *  was read with, and the passages of one document strung together. It answers what a
 *  table cannot — is this bank one kind of thing or five, and did the schemas catch them
 *  — in the time it takes to look. */
function PassageMap({
  kbId,
  schemas,
  onOpen,
}: {
  kbId: string;
  schemas: KnowledgeSchema[];
  onOpen: (docId: string) => void;
}) {
  const [points, setPoints] = useState<PassagePoint[] | null>(null);
  const [total, setTotal] = useState(0);

  useEffect(() => {
    let live = true;
    kbFetch<{ items: PassagePoint[]; total: number }>(`/${encodeURIComponent(kbId)}/map?limit=1500`)
      .then((page) => {
        if (!live) return;
        setPoints(page.items);
        setTotal(page.total);
      })
      .catch((e) => toast.error((e as Error).message));
    return () => {
      live = false;
    };
  }, [kbId]);

  const data = useMemo(() => {
    const items = points ?? [];
    const nodes = items.map((point) => ({
      id: `${point.doc_id}#${point.passage_index}`,
      label: point.snippet.slice(0, 60),
      group: point.schema_id ?? "no schema",
      // Size is the passage's length: a map of a corpus should show where its weight is.
      size: point.token_count,
      metadata: { doc_id: point.doc_id, title: point.title },
    }));
    // A document is a thread through its own passages, so a long document reads as a
    // constellation rather than as unrelated dots.
    const links = items.slice(1).flatMap((point, i) => {
      const previous = items[i];
      return previous.doc_id === point.doc_id
        ? [
            {
              source: `${previous.doc_id}#${previous.passage_index}`,
              target: `${point.doc_id}#${point.passage_index}`,
            },
          ]
        : [];
    });
    return { nodes, links };
  }, [points]);

  const palette = useMemo(() => {
    const colors = ["#6366f1", "#10b981", "#f59e0b", "#ec4899", "#06b6d4", "#8b5cf6"];
    const keys = [...schemas.map((schema) => schema.schema_id), "no schema"];
    return new Map(
      keys.map((key, i) => [key, key === "no schema" ? "#94a3b8" : colors[i % colors.length]])
    );
  }, [schemas]);

  if (!points) return <Spinner />;
  if (points.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        Nothing written yet — the map fills in as documents arrive.
      </p>
    );
  }

  const bySchema = new Map<string, number>();
  for (const point of points) {
    const key = point.schema_id ?? "no schema";
    bySchema.set(key, (bySchema.get(key) ?? 0) + 1);
  }

  return (
    <div className="space-y-3">
      <div className="rounded-lg border border-border bg-card overflow-hidden">
        <Constellation
          data={data}
          height={520}
          nodeSizeFn={(node) => 2 + Math.min(6, Math.sqrt((node.size ?? 1) / 20))}
          clusterKeyFn={(node) => node.group ?? null}
          clusterColorFn={(key) => palette.get(key) ?? "#94a3b8"}
          clusterLabelFn={(key) => key}
          sizeLegendLabel="passage length"
          onNodeClick={(node) => onOpen(String(node.metadata?.doc_id ?? ""))}
          compactLabels
        />
      </div>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
        <span>
          {points.length.toLocaleString()} of {total.toLocaleString()} passages
        </span>
        {[...bySchema.entries()].map(([key, count]) => (
          <span key={key} className="inline-flex items-center gap-1.5">
            <span
              className="inline-block h-2 w-2 rounded-full"
              style={{ backgroundColor: palette.get(key) ?? "#94a3b8" }}
            />
            <span className="font-mono">{key}</span>
            <span>{count}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

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

function SearchPanel({
  kbId,
  schemas: allSchemas,
  lockedSchemaId,
}: {
  kbId: string;
  schemas: KnowledgeSchema[];
  /** Embedded in one schema's page: search only that schema, filter only its fields. */
  lockedSchemaId?: string;
}) {
  const schemas = lockedSchemaId
    ? allSchemas.filter((schema) => schema.schema_id === lockedSchemaId)
    : allSchemas;
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<"hybrid" | "vector" | "keyword">("hybrid");
  const [topK, setTopK] = useState(10);
  const [collapse, setCollapse] = useState(false);
  const [filter, setFilter] = useState<Record<string, unknown> | null>(null);
  const [schemaId, setSchemaId] = useState<string | null>(lockedSchemaId ?? null);
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
    (schemaId && !lockedSchemaId ? 1 : 0) +
    (filter ? 1 : 0) +
    (topK !== 10 ? 1 : 0) +
    (collapse ? 1 : 0);

  return (
    <Section
      title="Search"
      tab={!lockedSchemaId}
      sub={!!lockedSchemaId}
      description={
        lockedSchemaId
          ? `Hybrid search over this schema's documents, filtered by its fields.`
          : "Hybrid vector + keyword search, reranked."
      }
    >
      <div className="flex gap-3">
        <div className="relative flex-1">
          <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && run()}
            placeholder="Ask something…"
            className="pl-10 h-12 text-lg"
            autoFocus={!lockedSchemaId}
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
              onSchemaChange={lockedSchemaId ? () => {} : setSchemaId}
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
  // A schema is read far more often than it is changed, so the page opens on the
  // fields and the form is one click away.
  const [editing, setEditing] = useState(false);
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
      {schemas.length > 0 && (
        <div className="mb-6">
          <SchemaShareCard schemas={schemas} />
        </div>
      )}
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
            }}
          />
        </div>
      )}

      {schemas.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          No schemas yet. Create one to give documents fields that search and query can filter on.
        </p>
      ) : view === "diagram" ? null : (
        // The schemas as a row of pills, like the collections: the page below gets the
        // full width, which the documents table needs for one column per field.
        <div className="space-y-4">
          <div className="flex gap-1 overflow-x-auto rounded-lg border border-border bg-muted/40 p-1">
            {schemas.map((schema) => (
              <button
                key={schema.schema_id}
                onClick={() => {
                  setSelected(schema.schema_id);
                  setEditing(false);
                }}
                title={schema.name ?? undefined}
                className={`inline-flex shrink-0 items-center gap-1.5 rounded-md px-3 py-1.5 text-sm transition-colors ${
                  schema.schema_id === selected
                    ? "bg-background font-semibold text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                <Tags className="h-3.5 w-3.5" />
                <span className="font-mono">{schema.schema_id}</span>
                <span className="text-[11px] text-muted-foreground tabular-nums">
                  {schema.documents ?? 0}
                </span>
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
                    {!editing && (
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

                <SchemaFillCard schema={current} />

                {/* Its definition first — bounded in size — then what it actually caught. */}
                {editing ? (
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

                {/* Remounted per schema, so a query never carries over to another one. */}
                <div className="mt-8">
                  <SearchPanel
                    key={current.schema_id}
                    kbId={kbId}
                    schemas={schemas}
                    lockedSchemaId={current.schema_id}
                  />
                </div>

                <div className="mt-6">
                  <SchemaDocuments
                    kbId={kbId}
                    schemaId={current.schema_id}
                    fields={Object.keys(current.document_fields)}
                  />
                </div>
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
// One hue per schema, in list order; documents without a schema stay grey.
const SCHEMA_COLORS = ["#009296", "#8b5cf6", "#f59e0b", "#0074d9", "#ec4899", "#6366f1"];

/** How the bank's documents split across its schemas — bank-wide, so it sits above the
 *  list rather than inside one schema. */
function SchemaShareCard({ schemas }: { schemas: KnowledgeSchema[] }) {
  // Every schema carries the same bank total.
  const bankTotal = schemas[0]?.bank_documents ?? 0;
  const assigned = schemas.reduce((sum, s) => sum + (s.documents ?? 0), 0);
  return (
    <div className="rounded-lg border border-border bg-card p-5 h-full">
      <Distribution
        title="Documents by schema"
        items={[
          ...schemas.map((s, index) => ({
            name: s.schema_id,
            value: s.documents ?? 0,
            color: SCHEMA_COLORS[index % SCHEMA_COLORS.length],
          })),
          {
            name: "No schema",
            value: Math.max(0, bankTotal - assigned),
            color: "var(--muted-foreground)",
          },
        ]}
        emptyLabel="No documents in this bank yet."
      />
    </div>
  );
}

/** How many of one schema's documents the extraction has filled. */
function SchemaFillCard({ schema }: { schema: KnowledgeSchema }) {
  const owned = schema.documents ?? 0;
  const missing = owned - schema.documents_with_fields;
  return (
    <div className="mb-4 rounded-lg border border-border bg-card p-5 space-y-3">
      <h4 className="text-[11px] font-semibold text-muted-foreground uppercase tracking-[0.08em]">
        Filled by extraction
      </h4>
      <ProgressRow done={schema.documents_with_fields} total={owned} doneColor="#10b981" />
      <p className="text-xs text-muted-foreground">
        {owned === 0
          ? "No document belongs to this schema yet."
          : missing > 0
            ? `${missing} document${missing === 1 ? " has" : "s have"} no values yet — Actions › Extract missing values fills them.`
            : "Every document of this schema has its values."}
      </p>
    </div>
  );
}

function SchemaDocuments({
  kbId,
  schemaId,
  fields,
}: {
  kbId: string;
  schemaId: string;
  /** The schema's document fields, one column each. */
  fields: string[];
}) {
  const [documents, setDocuments] = useState<KnowledgeDocument[] | null>(null);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);

  useEffect(() => setPage(0), [schemaId]);

  useEffect(() => {
    let live = true;
    kbFetch<{ items: KnowledgeDocument[]; total: number }>(
      `/${encodeURIComponent(kbId)}/documents?limit=${DOCUMENTS_PER_PAGE}&offset=${page * DOCUMENTS_PER_PAGE}&schema_id=${encodeURIComponent(schemaId)}`
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
  }, [kbId, schemaId, page]);

  if (!documents) return <Spinner />;
  if (documents.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">No documents were read with this schema yet.</p>
    );
  }
  return (
    <div className="space-y-2">
      <TableFrame>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Document</TableHead>
              {fields.map((field) => (
                <TableHead key={field} className="font-mono text-xs">
                  {field}
                </TableHead>
              ))}
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
                {fields.map((field) => (
                  <TableCell key={field} className="text-sm">
                    {doc.fields[field] === undefined || doc.fields[field] === null ? (
                      <span className="text-muted-foreground">—</span>
                    ) : (
                      formatFieldValue(doc.fields[field])
                    )}
                  </TableCell>
                ))}
                <TableCell className="text-right text-xs">{doc.passage_count}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableFrame>
      <Pager
        page={page}
        total={total}
        perPage={DOCUMENTS_PER_PAGE}
        noun="document"
        onPage={setPage}
      />
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

type CollectionTab = "records" | "collections";

function CollectionsPanel({ kbId }: { kbId: string }) {
  const [collections, setCollections] = useState<KnowledgeCollection[] | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [tab, setTab] = useState<CollectionTab>("records");

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

  if (!collections) return <Spinner />;

  const tabs: { id: CollectionTab; label: string }[] = [
    { id: "records", label: "Records" },
    { id: "collections", label: "Collections" },
  ];

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2 text-foreground">Collections</h1>
        <p className="text-muted-foreground">
          A collection is a kind of thing the documents talk about; each record folds together what
          every document said about one of them.
        </p>
      </div>
      <div className="border-b border-border mb-5 flex">
        {tabs.map((item) => (
          <button
            key={item.id}
            onClick={() => setTab(item.id)}
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

      {tab === "collections" ? (
        <CollectionDefinitions
          kbId={kbId}
          collections={collections}
          selected={selected}
          onSelect={setSelected}
          onChanged={load}
        />
      ) : (
        <CollectionRecords kbId={kbId} collections={collections} />
      )}
    </div>
  );
}

/** The collections themselves, in the layout the schemas use: the list on the left, the
 *  selected one on the right, and the whole set as a diagram when that reads better. */
function CollectionDefinitions({
  kbId,
  collections,
  selected,
  onSelect,
  onChanged,
}: {
  kbId: string;
  collections: KnowledgeCollection[];
  selected: string | null;
  onSelect: (id: string) => void;
  onChanged: () => Promise<void> | void;
}) {
  const [view, setView] = useState<"list" | "diagram">("list");
  const [editorOpen, setEditorOpen] = useState(false);

  const current = collections.find((c) => c.collection_id === selected) ?? null;

  const derive = async (collectionId: string) => {
    try {
      await kbFetch(
        `/${encodeURIComponent(kbId)}/collections/${encodeURIComponent(collectionId)}/derive`,
        { method: "POST", body: {} }
      );
      toast.success("Deriving in the background — watch it in Settings.");
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
      await onChanged();
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  return (
    <>
      <div className="mb-6 flex items-center justify-between">
        <div className="text-sm text-muted-foreground">
          {collections.length} collection{collections.length === 1 ? "" : "s"}
        </div>
        <div className="flex items-center gap-2">
          <ViewToggle value={view} onChange={setView} />
          <Button size="sm" variant="outline" onClick={() => setEditorOpen(true)}>
            <Pencil className="w-4 h-4 mr-1" /> Edit collections
          </Button>
        </div>
      </div>

      {collections.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          No collections yet. Define one — its fields are what an LLM reads out of each document,
          and the identity field is what makes the same thing found twice one record.
        </p>
      ) : view === "diagram" ? (
        <ErdDiagram
          nodes={collections.map(collectionCard)}
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
          onSelect={onSelect}
        />
      ) : (
        <div className="grid gap-5 md:grid-cols-[minmax(200px,260px)_1fr] items-start">
          <div className="rounded-lg border border-border overflow-hidden divide-y divide-border">
            {collections.map((collection) => (
              <button
                key={collection.collection_id}
                onClick={() => onSelect(collection.collection_id)}
                className={`w-full text-left px-3 py-2.5 transition-colors ${
                  collection.collection_id === selected ? "bg-accent" : "hover:bg-muted/50"
                }`}
              >
                <div className="font-mono text-xs">{collection.collection_id}</div>
                {collection.name && (
                  <div className="text-xs text-muted-foreground truncate">{collection.name}</div>
                )}
                <div className="text-[11px] text-muted-foreground mt-0.5">
                  {Object.keys(collection.fields).length} fields · {collection.records ?? 0} records
                </div>
              </button>
            ))}
          </div>

          <div>
            {!current ? (
              <p className="text-sm text-muted-foreground">Pick a collection to see its fields.</p>
            ) : (
              <>
                <div className="mb-4 flex items-start justify-between gap-3">
                  <div>
                    <div className="font-mono text-sm font-semibold">{current.collection_id}</div>
                    <p className="text-xs text-muted-foreground">
                      {current.description || "Derive it to read its records out of the documents."}
                    </p>
                  </div>
                  <div className="flex gap-2 shrink-0">
                    <Button size="sm" variant="outline" onClick={() => setEditorOpen(true)}>
                      Edit
                    </Button>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="outline" size="sm">
                          Actions
                          <MoreVertical className="w-4 h-4 ml-2" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end" className="w-56">
                        <DropdownMenuItem onClick={() => derive(current.collection_id)}>
                          <RefreshCw className="w-4 h-4 mr-2" />
                          Derive from documents
                        </DropdownMenuItem>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem
                          onClick={() => remove(current.collection_id)}
                          className="text-destructive focus:text-destructive"
                        >
                          <Trash2 className="w-4 h-4 mr-2" />
                          Delete collection
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                </div>

                <CollectionData kbId={kbId} collection={current} />

                <ErdDiagram nodes={[collectionCard(current)]} edges={[]} />
              </>
            )}
          </div>
        </div>
      )}

      <CollectionsEditor
        kbId={kbId}
        open={editorOpen}
        collections={collections}
        onOpenChange={setEditorOpen}
        onApplied={onChanged}
      />
    </>
  );
}

/** What the collection actually holds. The definition is right there in the card below;
 *  what a reader cannot see from it is whether anything filled it — so this counts the
 *  records, the documents behind them, and how much of each field was ever answered. A
 *  field at zero is a field the documents do not carry or the model never finds, and
 *  that is worth knowing before writing a query that filters on it. */
function CollectionData({ kbId, collection }: { kbId: string; collection: KnowledgeCollection }) {
  const [stats, setStats] = useState<CollectionStats | null>(null);

  useEffect(() => {
    let live = true;
    kbFetch<CollectionStats>(
      `/${encodeURIComponent(kbId)}/collections/${encodeURIComponent(collection.collection_id)}/stats`
    )
      .then((s) => live && setStats(s))
      .catch(() => live && setStats(null));
    return () => {
      live = false;
    };
  }, [kbId, collection.collection_id]);

  if (!stats) return <Spinner size="sm" />;

  const filledMost = Math.max(1, ...stats.coverage.map((row) => row.filled));
  return (
    <div className="mb-4 space-y-3">
      <StatStrip>
        <InlineStat icon={Layers} label="Records" value={stats.records} />
        <InlineStat icon={FileText} label="Documents behind them" value={stats.documents} />
        <InlineStat
          icon={Table2}
          label="Fields answered"
          value={stats.coverage.filter((row) => row.filled > 0).length}
        />
      </StatStrip>

      {stats.records > 0 && (
        <div className="rounded-lg border border-border p-4">
          <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-3">
            Field coverage
          </div>
          <div className="space-y-2">
            {stats.coverage.map((row) => {
              const percent = stats.records > 0 ? (row.filled / stats.records) * 100 : 0;
              return (
                <div key={row.field} className="flex items-center gap-3 text-xs">
                  <span className="w-40 shrink-0 truncate font-mono">{row.field}</span>
                  <div className="h-2 flex-1 overflow-hidden rounded-full bg-muted">
                    <div
                      className={
                        row.filled === 0
                          ? "h-full bg-muted-foreground/30"
                          : "h-full bg-primary-gradient"
                      }
                      style={{ width: `${Math.max(percent, row.filled === 0 ? 0 : 2)}%` }}
                    />
                  </div>
                  <span className="w-28 shrink-0 text-right tabular-nums text-muted-foreground">
                    {row.filled} / {stats.records} ({percent.toFixed(0)}%)
                  </span>
                  {/* The widest bar is the field the documents answer best; the empties
                      are the ones to ask about. */}
                  {row.filled === filledMost && row.filled > 0 && (
                    <span className="w-16 shrink-0 text-[10px] text-muted-foreground">fullest</span>
                  )}
                  {row.filled === 0 && (
                    <span className="w-16 shrink-0 text-[10px] text-muted-foreground">
                      never filled
                    </span>
                  )}
                </div>
              );
            })}
          </div>
          {stats.last_updated && (
            <p className="mt-3 text-[11px] text-muted-foreground">
              Last changed {new Date(stats.last_updated).toLocaleString()}.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

/** One collection as a diagram card. */
function collectionCard(collection: KnowledgeCollection): ErdNode {
  return {
    id: collection.collection_id,
    title: collection.collection_id,
    badge: "collection",
    fields: Object.entries(collection.fields).map(([name, spec]) => ({
      name,
      // A relationship field holds another collection's record id, so its "type" is that
      // collection — which is also where its line goes.
      type: spec.collection ?? spec.type ?? "string",
      primary: collection.identity === name,
      relation: Boolean(spec.collection),
      values: spec.values as (string | number | boolean)[] | undefined,
    })),
  };
}

const RECORDS_PER_PAGE = 25;

/** Previous / Next under a server-paginated table: documents, a schema's documents,
 *  a collection's records. */
function Pager({
  page,
  total,
  perPage,
  noun,
  onPage,
}: {
  page: number;
  total: number;
  perPage: number;
  noun: string;
  onPage: (page: number) => void;
}) {
  const pages = Math.max(1, Math.ceil(total / perPage));
  return (
    <div className="flex items-center justify-between text-xs text-muted-foreground">
      <span>
        {total.toLocaleString()} {noun}
        {total === 1 ? "" : "s"}
      </span>
      <div className="flex items-center gap-2">
        <Button
          size="sm"
          variant="outline"
          disabled={page === 0}
          onClick={() => onPage(Math.max(0, page - 1))}
        >
          Previous
        </Button>
        <span className="tabular-nums">
          {page + 1} / {pages}
        </span>
        <Button
          size="sm"
          variant="outline"
          disabled={page + 1 >= pages}
          onClick={() => onPage(page + 1)}
        >
          Next
        </Button>
      </div>
    </div>
  );
}

const DOCUMENTS_PER_PAGE = 50;

/** The records themselves: a page at a time, with the query language above them. A row
 *  opens the record, because a folded record is only trustworthy with its evidence. */
function CollectionRecords({
  kbId,
  collections,
}: {
  kbId: string;
  collections: KnowledgeCollection[];
}) {
  const [collectionId, setCollectionId] = useState(collections[0]?.collection_id ?? "");
  const [page, setPage] = useState(0);
  const [search, setSearch] = useState("");
  const [records, setRecords] = useState<{ items: KnowledgeRecord[]; total: number } | null>(null);
  const [open, setOpen] = useState<string | null>(null);

  const [queryOpen, setQueryOpen] = useState(false);
  const [query, setQuery] = useState<Record<string, unknown>>({ select: [{ count: "*" }] });
  const [result, setResult] = useState<QueryResult | null>(null);
  const [running, setRunning] = useState(false);

  const collection = collections.find((c) => c.collection_id === collectionId) ?? null;

  const load = useCallback(async () => {
    if (!collectionId) return;
    const params = new URLSearchParams({
      limit: String(RECORDS_PER_PAGE),
      offset: String(page * RECORDS_PER_PAGE),
    });
    if (search.trim()) params.set("q", search.trim());
    try {
      setRecords(
        await kbFetch<{ items: KnowledgeRecord[]; total: number }>(
          `/${encodeURIComponent(kbId)}/collections/${encodeURIComponent(collectionId)}/records?${params}`
        )
      );
    } catch (e) {
      toast.error((e as Error).message);
    }
  }, [kbId, collectionId, page, search]);

  useEffect(() => {
    const timer = setTimeout(load, search ? 250 : 0);
    return () => clearTimeout(timer);
  }, [load, search]);

  const run = async () => {
    if (!collectionId) return;
    setRunning(true);
    try {
      setResult(
        await kbFetch<QueryResult>(
          `/${encodeURIComponent(kbId)}/collections/${encodeURIComponent(collectionId)}/query`,
          { method: "POST", body: query }
        )
      );
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setRunning(false);
    }
  };

  if (collections.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        No collections yet — define one on the Collections tab, then derive it.
      </p>
    );
  }

  const columns = collection ? Object.keys(collection.fields) : [];
  const total = records?.total ?? 0;

  return (
    <div className="space-y-4">
      {/* The collections as a row of pills rather than a dropdown: a bank has a handful,
          they are the thing being switched between, and a dropdown hides which ones
          exist behind a click. */}
      <div className="flex gap-1 overflow-x-auto rounded-lg border border-border bg-muted/40 p-1">
        {collections.map((c) => {
          const active = c.collection_id === collectionId;
          return (
            <button
              key={c.collection_id}
              onClick={() => {
                setCollectionId(c.collection_id);
                setPage(0);
                setResult(null);
                setSearch("");
              }}
              className={`inline-flex shrink-0 items-center gap-1.5 rounded-md px-3 py-1.5 text-sm transition-colors ${
                active
                  ? "bg-background font-semibold text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              <Table2 className="h-3.5 w-3.5" />
              <span className="font-mono">{c.collection_id}</span>
              <span className="text-[11px] text-muted-foreground tabular-nums">
                {c.records ?? 0}
              </span>
            </button>
          );
        })}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <div className="relative min-w-[220px] flex-1">
          <SearchIcon className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(0);
            }}
            placeholder="Filter by record id or any value…"
            className="pl-8 h-9"
          />
        </div>
        <DisclosureButton open={queryOpen} onToggle={() => setQueryOpen((v) => !v)} label="Query" />
      </div>

      {queryOpen && (
        <div className="space-y-2 rounded-lg border border-border p-4">
          <QueryBuilder
            key={collectionId}
            fields={[
              // The record's own columns are always there, whatever the collection defines.
              { name: "record_id" },
              { name: "doc_count", type: "number" },
              { name: "updated_at" },
              ...Object.entries(collection?.fields ?? {}).map(([name, spec]) => ({
                name,
                type: spec.type,
                values: spec.values as (string | number | boolean)[] | undefined,
              })),
            ]}
            onChange={setQuery}
          />
          <div className="flex items-center gap-2 pt-1">
            <Button size="sm" onClick={run} disabled={running}>
              {running ? <Spinner size="sm" /> : "Run query"}
            </Button>
            <p className="text-xs text-muted-foreground">
              The same language the documents use, over records — aggregates, grouping, and joins
              across relationship fields.
            </p>
          </div>
          {result && (
            <Tabs defaultValue="data" className="mt-2">
              <TabsList className="grid w-full max-w-xs grid-cols-2">
                <TabsTrigger value="data">Data</TabsTrigger>
                <TabsTrigger value="json">JSON</TabsTrigger>
              </TabsList>
              <TabsContent value="data" className="mt-3">
                <ResultTable
                  result={result}
                  onRowClick={(row) => {
                    // A result that carries record_id is a set of records, so a row is a
                    // record and opens as one.
                    const index = result.columns.indexOf("record_id");
                    if (index >= 0 && row[index]) setOpen(String(row[index]));
                  }}
                />
              </TabsContent>
              <TabsContent value="json" className="mt-3">
                <pre className="max-h-[600px] overflow-auto rounded-lg border border-border bg-muted/30 p-4 text-xs leading-relaxed">
                  {JSON.stringify(result, null, 2)}
                </pre>
              </TabsContent>
            </Tabs>
          )}
        </div>
      )}

      {!records ? (
        <Spinner />
      ) : records.items.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          {search ? "No records match that filter." : "No records yet. Derive the collection."}
        </p>
      ) : (
        <>
          <TableFrame>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-56">Record</TableHead>
                  {columns.map((name) => (
                    <TableHead key={name}>{name}</TableHead>
                  ))}
                  <TableHead className="w-28 text-right">Documents</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {records.items.map((record) => (
                  <TableRow
                    key={record.record_id}
                    className="cursor-pointer"
                    onClick={() => setOpen(record.record_id)}
                  >
                    <TableCell className="font-mono text-xs">{record.record_id}</TableCell>
                    {columns.map((name) => (
                      <TableCell key={name} className="text-xs">
                        {formatFieldValue(record.values?.[name])}
                      </TableCell>
                    ))}
                    <TableCell className="text-right text-xs">
                      {record.doc_ids?.length ?? 0}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableFrame>

          <Pager
            page={page}
            total={total}
            perPage={RECORDS_PER_PAGE}
            noun="record"
            onPage={setPage}
          />
        </>
      )}

      <RecordDetail
        kbId={kbId}
        collectionId={collectionId}
        recordId={open}
        onClose={() => setOpen(null)}
        onChanged={load}
      />
    </div>
  );
}

/** One record: what it says, where each value came from, and the documents behind it. */
function RecordDetail({
  kbId,
  collectionId,
  recordId,
  onClose,
  onChanged,
}: {
  kbId: string;
  collectionId: string;
  recordId: string | null;
  onClose: () => void;
  onChanged: () => void;
}) {
  const [record, setRecord] = useState<KnowledgeRecordDetail | null>(null);

  useEffect(() => {
    if (!recordId) {
      setRecord(null);
      return;
    }
    let live = true;
    kbFetch<KnowledgeRecordDetail>(
      `/${encodeURIComponent(kbId)}/collections/${encodeURIComponent(collectionId)}/records/${encodeURIComponent(recordId)}`
    )
      .then((r) => live && setRecord(r))
      .catch((e) => toast.error((e as Error).message));
    return () => {
      live = false;
    };
  }, [kbId, collectionId, recordId]);

  const remove = async () => {
    if (!recordId) return;
    try {
      await kbFetch(
        `/${encodeURIComponent(kbId)}/collections/${encodeURIComponent(collectionId)}/records/${encodeURIComponent(recordId)}`,
        { method: "DELETE" }
      );
      onClose();
      onChanged();
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  return (
    <Dialog open={recordId !== null} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-3xl max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="font-mono text-base">{recordId}</DialogTitle>
          <DialogDescription>
            Folded from {record?.doc_ids?.length ?? 0} document
            {(record?.doc_ids?.length ?? 0) === 1 ? "" : "s"} in {collectionId}.
          </DialogDescription>
        </DialogHeader>

        {!record ? (
          <Spinner />
        ) : (
          <div className="space-y-4">
            <InfoCard title="Values" icon={<Table2 className="w-3.5 h-3.5" />}>
              {Object.entries(record.values ?? {}).map(([name, value]) => (
                <MetadataRow
                  key={name}
                  label={name}
                  value={
                    <span className="flex items-center gap-2">
                      {formatFieldValue(value)}
                      {record.pinned && name in record.pinned && (
                        <span className="rounded bg-amber-100 dark:bg-amber-500/20 px-1.5 py-0.5 text-[10px]">
                          pinned
                        </span>
                      )}
                    </span>
                  }
                />
              ))}
            </InfoCard>

            {/* Evidence is what makes a folded value auditable: a total nobody can trace
                is a claim, not data. */}
            {record.evidence && Object.keys(record.evidence).length > 0 && (
              <InfoCard title="Evidence" icon={<FileText className="w-3.5 h-3.5" />}>
                {Object.entries(record.evidence).map(([name, quotes]) => (
                  <MetadataRow
                    key={name}
                    label={name}
                    value={
                      <ul className="space-y-1">
                        {(quotes ?? []).map((quote, i) => (
                          <li key={i} className="text-xs text-muted-foreground">
                            <span className="font-mono">{quote.doc_id}</span> —{" "}
                            {/* A quote written straight through the records API is
                                whatever the caller sent, so it is printed as text
                                rather than assumed to be a string. */}
                            “{formatFieldValue(quote.quote)}”
                          </li>
                        ))}
                      </ul>
                    }
                  />
                ))}
              </InfoCard>
            )}

            <InfoCard title="Documents" icon={<FileText className="w-3.5 h-3.5" />}>
              <div className="flex flex-wrap gap-1">
                {(record.doc_ids ?? []).map((docId) => (
                  <span
                    key={docId}
                    className="rounded bg-muted px-1.5 py-0.5 font-mono text-[11px]"
                  >
                    {docId}
                  </span>
                ))}
              </div>
            </InfoCard>

            <div className="flex justify-end">
              <Button variant="outline" size="sm" className="text-destructive" onClick={remove}>
                <Trash2 className="w-4 h-4 mr-1" /> Delete record
              </Button>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

function ResultTable({
  result,
  onRowClick,
}: {
  result: QueryResult;
  onRowClick?: (row: unknown[]) => void;
}) {
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
            <TableRow
              key={i}
              className={onRowClick ? "cursor-pointer" : undefined}
              onClick={() => onRowClick?.(row)}
            >
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

// The task types a knowledge bank queues, for the shared operations list's filter.
const KB_OPERATION_TYPES = [
  "all",
  "knowledge_write_batch",
  "knowledge_file_convert",
  "knowledge_extract_fields",
  "knowledge_derive_records",
] as const;

type SettingsTab = "general" | "configuration" | "llm-requests";

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
    { id: "llm-requests", label: "LLM Requests" },
  ];
  const { features } = useFeatures();
  return (
    <div>
      <div className="flex justify-between items-start mb-6">
        <div>
          <h1 className="text-3xl font-bold mb-2 text-foreground">Settings</h1>
          <p className="text-muted-foreground">
            What is in this bank, what it is doing, and how it is configured.
          </p>
        </div>
        <SettingsActions kbId={kbId} />
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
          <Stats bank={bank} />
          <BankOperationsView bankId={kbId} operationTypes={KB_OPERATION_TYPES} />
        </div>
      ) : tab === "configuration" ? (
        <Configuration kbId={kbId} />
      ) : features?.llm_trace ? (
        <div>
          <p className="text-sm text-muted-foreground mb-4">
            Every LLM call this bank made while extracting fields and records.
          </p>
          <LLMRequestsView bankId={kbId} />
        </div>
      ) : (
        <FeatureNotEnabled
          title="LLM request tracing is not enabled"
          description={
            <>
              Set{" "}
              <code className="px-1 py-0.5 bg-muted rounded text-xs">
                HINDSIGHT_API_LLM_TRACE_ENABLED=true
              </code>{" "}
              on the API to record them.
            </>
          }
        />
      )}
    </div>
  );
}

type KbConfigField = {
  key: string;
  label: string;
  description: string;
  kind: "number" | "boolean";
};

// Grouped the way the memory-bank Configuration groups its settings: one card per stage,
// each with its own Save. Descriptions follow docs/developer/configuration.mdx.
const KB_CONFIG_SECTIONS: { title: string; description: string; fields: KbConfigField[] }[] = [
  {
    title: "Passages",
    description: "How documents are cut into passages on write.",
    fields: [
      {
        key: "kb_passage_size",
        label: "Passage size",
        description: "Tokens per passage, cut on paragraph, then sentence, then word boundaries.",
        kind: "number",
      },
      {
        key: "kb_passage_overlap",
        label: "Passage overlap",
        description: "Tokens of the previous passage repeated at the start of the next.",
        kind: "number",
      },
    ],
  },
  {
    title: "Search",
    description: "How search finds and ranks passages.",
    fields: [
      {
        key: "kb_search_candidates",
        label: "Candidates",
        description:
          "Candidates each arm (vector, keyword) contributes before fusion and reranking.",
        kind: "number",
      },
      {
        key: "kb_search_vector_weight",
        label: "Vector weight",
        description: "Weight of the vector arm in the fusion; keyword gets the rest. 0.5 is even.",
        kind: "number",
      },
      {
        key: "kb_search_rerank",
        label: "Rerank",
        description: "Rerank the fused candidates with the configured reranker.",
        kind: "boolean",
      },
    ],
  },
  {
    title: "Extraction",
    description: "How the LLM fills schema fields and records.",
    fields: [
      {
        key: "kb_field_extraction",
        label: "Field extraction",
        description: "Fill the bank's schema fields with the LLM on write.",
        kind: "boolean",
      },
      {
        key: "kb_schema_classification",
        label: "Schema classification",
        description: "With several schemas and none given on write, ask the LLM which one fits.",
        kind: "boolean",
      },
      {
        key: "kb_field_extraction_max_chars",
        label: "Max characters",
        description: "How much of a document (and of a passage) the extraction LLM reads.",
        kind: "number",
      },
      {
        key: "kb_field_extraction_concurrency",
        label: "Concurrency",
        description: "Passage-level extraction calls run at once per document.",
        kind: "number",
      },
      {
        key: "kb_record_identity_similarity",
        label: "Record name similarity",
        description:
          "How close a misspelled record name must be to match an existing one. 0 is off.",
        kind: "number",
      },
    ],
  },
];

function Configuration({ kbId }: { kbId: string }) {
  const [config, setConfig] = useState<Record<string, unknown> | null>(null);

  const load = useCallback(async () => {
    // Bank config is the memory banks' endpoint: a knowledge bank is a bank row.
    const response = await fetch(withBasePath(`/api/banks/${encodeURIComponent(kbId)}/config`));
    const data = await response.json();
    setConfig(data.config ?? {});
  }, [kbId]);

  useEffect(() => {
    load().catch((e) => toast.error((e as Error).message));
  }, [load]);

  if (!config) return <Spinner />;
  return (
    <div className="space-y-8">
      {KB_CONFIG_SECTIONS.map((section) => (
        <KbConfigSection
          key={section.title}
          kbId={kbId}
          config={config}
          onSaved={load}
          {...section}
        />
      ))}
    </div>
  );
}

function KbConfigSection({
  kbId,
  config,
  onSaved,
  title,
  description,
  fields,
}: {
  kbId: string;
  config: Record<string, unknown>;
  onSaved: () => Promise<void>;
  title: string;
  description: string;
  fields: KbConfigField[];
}) {
  const initial = useCallback(
    () => Object.fromEntries(fields.map((f) => [f.key, String(config[f.key] ?? "")])),
    [config, fields]
  );
  const [draft, setDraft] = useState<Record<string, string>>(initial);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => setDraft(initial()), [initial]);

  const dirty = fields.some((f) => draft[f.key] !== String(config[f.key] ?? ""));

  const save = async () => {
    const updates = Object.fromEntries(
      fields.map((f) => [
        f.key,
        f.kind === "boolean" ? draft[f.key] === "true" : Number(draft[f.key]),
      ])
    );
    setSaving(true);
    setError(null);
    try {
      const response = await fetch(withBasePath(`/api/banks/${encodeURIComponent(kbId)}/config`), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ updates }),
      });
      if (!response.ok) {
        setError(`Could not save: HTTP ${response.status}`);
        return;
      }
      toast.success("Saved. It applies to the next write and the next search.");
      await onSaved();
    } finally {
      setSaving(false);
    }
  };

  return (
    <ConfigSection
      title={title}
      description={description}
      error={error}
      dirty={dirty}
      saving={saving}
      onSave={save}
    >
      {fields.map((field) => (
        <FieldRow key={field.key} label={field.label} description={field.description}>
          {field.kind === "boolean" ? (
            <Select
              value={draft[field.key]}
              onValueChange={(v) => setDraft({ ...draft, [field.key]: v })}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="true">On</SelectItem>
                <SelectItem value="false">Off</SelectItem>
              </SelectContent>
            </Select>
          ) : (
            <Input
              type="number"
              value={draft[field.key]}
              onChange={(e) => setDraft({ ...draft, [field.key]: e.target.value })}
            />
          )}
        </FieldRow>
      ))}
    </ConfigSection>
  );
}

/** The Settings actions menu, as the memory banks have it — minus the memory-only
 *  operations (consolidation, observations, clone). */
function SettingsActions({ kbId }: { kbId: string }) {
  const router = useRouter();
  const { features } = useFeatures();
  const [showHealth, setShowHealth] = useState(false);
  const [confirm, setConfirm] = useState<"delete" | "reset" | null>(null);
  const [busy, setBusy] = useState(false);

  const run = async () => {
    setBusy(true);
    try {
      if (confirm === "delete") {
        await kbFetch(`/${encodeURIComponent(kbId)}`, { method: "DELETE" });
        router.push("/knowledge-banks");
      } else {
        await client.resetBankConfig(kbId);
        toast.success("Configuration reset to the server defaults.");
        // The Configuration tab loads on mount; a reload shows the defaults.
        router.refresh();
      }
      setConfirm(null);
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="outline" size="sm">
            Actions
            <MoreVertical className="w-4 h-4 ml-2" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-48">
          {features?.bank_llm_health && (
            <>
              <DropdownMenuItem onClick={() => setShowHealth(true)}>
                <Activity className="w-4 h-4 mr-2" />
                Health
              </DropdownMenuItem>
              <DropdownMenuSeparator />
            </>
          )}
          <DropdownMenuItem
            onClick={() => setConfirm("reset")}
            disabled={!features?.bank_config_api}
            className="text-amber-600 dark:text-amber-400 focus:text-amber-700 dark:focus:text-amber-300"
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            Reset configuration
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem
            onClick={() => setConfirm("delete")}
            className="text-red-600 dark:text-red-400 focus:text-red-700 dark:focus:text-red-300"
          >
            <Trash2 className="w-4 h-4 mr-2" />
            Delete bank
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      <LlmHealthDialog bankId={kbId} open={showHealth} onOpenChange={setShowHealth} />

      <AlertDialog open={confirm !== null} onOpenChange={(open) => !open && setConfirm(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {confirm === "delete" ? "Delete knowledge bank" : "Reset configuration"}
            </AlertDialogTitle>
            <AlertDialogDescription asChild>
              <div className="space-y-2 text-sm text-muted-foreground">
                <p>
                  {confirm === "delete" ? "Delete " : "Reset the configuration of "}
                  <span className="font-semibold text-foreground">{kbId}</span>?
                </p>
                {confirm === "delete" ? (
                  <p className="text-red-600 dark:text-red-400 font-medium">
                    Every document, passage, schema, collection and record goes with it. This cannot
                    be undone.
                  </p>
                ) : (
                  <p className="text-amber-600 dark:text-amber-400 font-medium">
                    Every per-bank setting goes back to the server default.
                  </p>
                )}
              </div>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={busy}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={run}
              disabled={busy}
              className={
                confirm === "delete"
                  ? "bg-destructive text-destructive-foreground hover:bg-destructive/90"
                  : undefined
              }
            >
              {busy && <Spinner size="sm" className="mr-2" />}
              {confirm === "delete" ? "Delete bank" : "Reset configuration"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
