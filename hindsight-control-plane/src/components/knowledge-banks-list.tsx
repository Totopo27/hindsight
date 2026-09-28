"use client";

// The knowledge banks, as a table in the banks overview. It is the same list the header
// selector shows under its Knowledge tab; clicking a row opens that bank's workspace.

import * as React from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { Plus, Trash2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { kbFetch, type KnowledgeBank } from "@/components/knowledge-bank-api";
import { NewKnowledgeBankDialog } from "@/components/new-knowledge-bank-dialog";

export function KnowledgeBanksList({ search }: { search: string }) {
  const router = useRouter();
  const [banks, setBanks] = React.useState<KnowledgeBank[] | null>(null);
  const [creating, setCreating] = React.useState(false);

  const load = React.useCallback(async () => {
    try {
      const params = search ? `?q=${encodeURIComponent(search)}` : "";
      setBanks((await kbFetch<{ items: KnowledgeBank[] }>(params)).items);
    } catch (e) {
      toast.error((e as Error).message);
      setBanks([]);
    }
  }, [search]);

  React.useEffect(() => {
    load();
  }, [load]);

  const remove = async (bankId: string) => {
    try {
      await kbFetch(`/${encodeURIComponent(bankId)}`, { method: "DELETE" });
      await load();
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  return (
    <>
      <div className="mb-4 flex justify-end">
        <Button variant="outline" size="sm" onClick={() => setCreating(true)}>
          <Plus className="mr-1.5 h-4 w-4" />
          New knowledge bank
        </Button>
      </div>

      {!banks ? (
        <div className="flex items-center justify-center py-16">
          <Spinner size="sm" />
        </div>
      ) : banks.length === 0 ? (
        <div className="rounded-xl border border-border bg-card py-16 text-center text-muted-foreground">
          No knowledge banks yet. Create one, then write documents or upload files into it.
        </div>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Bank</TableHead>
              <TableHead className="w-32">Documents</TableHead>
              <TableHead className="w-32">Passages</TableHead>
              <TableHead className="w-40">Created</TableHead>
              <TableHead className="w-12" />
            </TableRow>
          </TableHeader>
          <TableBody>
            {banks.map((bank) => (
              <TableRow
                key={bank.bank_id}
                onClick={() => router.push(`/knowledge-banks/${encodeURIComponent(bank.bank_id)}`)}
                className="cursor-pointer"
              >
                <TableCell>
                  <div className="font-medium text-card-foreground">{bank.bank_id}</div>
                  {bank.name && bank.name !== bank.bank_id && (
                    <div className="truncate text-xs text-muted-foreground">{bank.name}</div>
                  )}
                </TableCell>
                <TableCell className="tabular-nums">{bank.documents}</TableCell>
                <TableCell className="tabular-nums">{bank.passages}</TableCell>
                <TableCell className="text-sm text-muted-foreground">
                  {new Date(bank.created_at).toLocaleDateString()}
                </TableCell>
                <TableCell onClick={(e) => e.stopPropagation()}>
                  <Button
                    variant="ghost"
                    size="sm"
                    title="Delete"
                    className="text-destructive"
                    onClick={() => remove(bank.bank_id)}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}

      <NewKnowledgeBankDialog open={creating} onOpenChange={setCreating} />
    </>
  );
}
