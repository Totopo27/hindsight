"use client";

// The list of knowledge banks, and creating one.

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { Database, FileText, Plus, Search } from "lucide-react";
import { BankSelector } from "@/components/bank-selector";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { kbFetch, type KnowledgeBank } from "@/components/knowledge-bank-api";

export default function KnowledgeBanksPage() {
  const router = useRouter();
  const [banks, setBanks] = useState<KnowledgeBank[] | null>(null);
  const [query, setQuery] = useState("");
  const [creating, setCreating] = useState(false);
  const [newId, setNewId] = useState("");

  const load = useCallback(async () => {
    try {
      const params = query ? `?q=${encodeURIComponent(query)}` : "";
      setBanks((await kbFetch<{ items: KnowledgeBank[] }>(params)).items);
    } catch (e) {
      toast.error((e as Error).message);
    }
  }, [query]);

  useEffect(() => {
    load();
  }, [load]);

  const create = async () => {
    try {
      await kbFetch("", { method: "POST", body: { id: newId } });
      setCreating(false);
      setNewId("");
      router.push(`/knowledge-banks/${encodeURIComponent(newId)}`);
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <BankSelector />
      <main className="flex-1 p-6">
        <div className="max-w-[1024px] xl:max-w-[1280px] mx-auto w-full space-y-5">
          <div className="flex items-center gap-3">
            <div className="flex-1">
              <h1 className="text-[28px] font-semibold leading-[34px] tracking-[-0.4px]">
                Knowledge banks
              </h1>
              <p className="text-sm text-muted-foreground mt-1">
                Documents, chunked and searchable. No memory extraction.
              </p>
            </div>
            <Input
              placeholder="Filter…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="w-56"
            />
            <Button onClick={() => setCreating(true)}>
              <Plus className="w-4 h-4 mr-1" /> New knowledge bank
            </Button>
          </div>

          {!banks ? (
            <Spinner />
          ) : banks.length === 0 ? (
            <div className="rounded-[16px] border border-border p-[21px] text-sm text-muted-foreground">
              No knowledge banks yet. Create one, then write documents into it through the API or
              the Documents tab.
            </div>
          ) : (
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
              {banks.map((bank) => (
                <button
                  key={bank.bank_id}
                  onClick={() =>
                    router.push(`/knowledge-banks/${encodeURIComponent(bank.bank_id)}`)
                  }
                  className="text-left rounded-[16px] border border-border bg-card p-[21px] hover:border-primary/60 transition"
                >
                  <div className="flex items-center gap-2 font-semibold">
                    <Database className="w-4 h-4 text-primary" />
                    <span className="truncate">{bank.bank_id}</span>
                  </div>
                  {bank.name !== bank.bank_id && (
                    <div className="text-xs text-muted-foreground mt-0.5 truncate">{bank.name}</div>
                  )}
                  <div className="mt-3 flex gap-4 text-sm text-muted-foreground">
                    <span className="flex items-center gap-1">
                      <FileText className="w-3.5 h-3.5" /> {bank.documents} documents
                    </span>
                    <span className="flex items-center gap-1">
                      <Search className="w-3.5 h-3.5" /> {bank.chunks} chunks
                    </span>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      </main>

      <Dialog open={creating} onOpenChange={setCreating}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>New knowledge bank</DialogTitle>
            <DialogDescription>
              The id is how the API addresses it. Knowledge banks and memory banks have separate
              ids.
            </DialogDescription>
          </DialogHeader>
          <Input placeholder="contracts" value={newId} onChange={(e) => setNewId(e.target.value)} />
          <DialogFooter>
            <Button disabled={!newId.trim()} onClick={create}>
              Create
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
