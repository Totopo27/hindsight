"use client";

// Moving a knowledge bank in and out of the control plane.
//
// Both directions are async operations, so both dialogs do the same thing: submit,
// poll the bank's operation until it is not pending, then act on the result. The
// export downloads through the CP's file proxy (the same one the operations table
// uses), because the API's download url is not reachable from the browser.

import * as React from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Spinner } from "@/components/ui/spinner";
import { kbFetch } from "@/components/knowledge-bank-api";
import { withBasePath } from "@/lib/base-path";

interface OperationResult {
  status: string;
  error_message?: string | null;
  result_metadata?: Record<string, unknown> | null;
}

/** Wait for one operation to finish. Transfers are seconds, not minutes. */
async function awaitOperation(kbId: string, operationId: string): Promise<OperationResult> {
  for (let attempt = 0; attempt < 600; attempt++) {
    const operation = await kbFetch<OperationResult>(
      `/${encodeURIComponent(kbId)}/operations/${operationId}`
    );
    if (operation.status !== "pending" && operation.status !== "processing") {
      if (operation.status !== "completed") {
        throw new Error(operation.error_message || `Operation ${operation.status}`);
      }
      return operation;
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error("The operation is still running — watch it under Settings › General.");
}

/** A labelled checkbox, matching the clone dialog's scope rows on a memory bank. */
function ScopeRow({
  checked,
  onChange,
  title,
  description,
}: {
  checked: boolean;
  onChange: (next: boolean) => void;
  title: string;
  description: string;
}) {
  return (
    <label className="flex cursor-pointer items-start gap-3 rounded-lg border border-border p-3">
      <input
        type="checkbox"
        className="mt-0.5"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span>
        <span className="block text-sm font-medium text-foreground">{title}</span>
        <span className="block text-xs text-muted-foreground">{description}</span>
      </span>
    </label>
  );
}

export function KbExportDialog({
  kbId,
  open,
  onOpenChange,
}: {
  kbId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [data, setData] = React.useState(true);
  const [config, setConfig] = React.useState(true);
  const [busy, setBusy] = React.useState(false);

  const run = async () => {
    setBusy(true);
    try {
      const submitted = await kbFetch<{ operation_id: string }>(
        `/${encodeURIComponent(kbId)}/transfer/export?include_data=${data}&include_config=${config}`,
        { method: "POST" }
      );
      const operation = await awaitOperation(kbId, submitted.operation_id);
      const url = operation.result_metadata?.download_url;
      if (typeof url !== "string") throw new Error("The export finished without an archive.");
      const anchor = document.createElement("a");
      anchor.href = withBasePath(`/api/files/download?path=${encodeURIComponent(url)}`);
      anchor.download = String(operation.result_metadata?.filename ?? `${kbId}-knowledge.zip`);
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      onOpenChange(false);
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Export this bank</DialogTitle>
          <DialogDescription>
            A ZIP you can import here or on another instance. Passages and embeddings are not
            carried — the import re-reads the documents with the target bank&apos;s own settings.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-2">
          <ScopeRow
            checked={config}
            onChange={setConfig}
            title="Definitions"
            description="The bank, its schemas and its collections. On their own, this is a template."
          />
          <ScopeRow
            checked={data}
            onChange={setData}
            title="Content"
            description="The documents and the records derived from them, with their evidence."
          />
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={busy}>
            Cancel
          </Button>
          <Button onClick={run} disabled={busy || (!data && !config)}>
            {busy && <Spinner size="sm" className="mr-2" />}
            Export
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export function KbImportDialog({
  kbId,
  open,
  onOpenChange,
}: {
  kbId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const router = useRouter();
  const [file, setFile] = React.useState<File | null>(null);
  // Default to this bank: the common case from here is applying a template to the bank
  // you are looking at. A new bank is one edit away.
  const [target, setTarget] = React.useState(kbId);
  const [busy, setBusy] = React.useState(false);

  React.useEffect(() => {
    if (open) setTarget(kbId);
  }, [open, kbId]);

  const intoThisBank = target === kbId;

  const run = async () => {
    if (!file) return;
    setBusy(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const query = new URLSearchParams({
        mode: intoThisBank ? "merge" : "restore",
        target_bank_id: target,
      });
      const response = await fetch(
        withBasePath(
          `/api/knowledge-banks/${encodeURIComponent(kbId)}/transfer/import?${query.toString()}`
        ),
        { method: "POST", body: form }
      );
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`);
      const operation = await awaitOperation(kbId, body.operation_id);
      const counts = operation.result_metadata ?? {};
      toast.success(
        `Imported into ${target}: ${counts.schemas ?? 0} schemas, ${counts.collections ?? 0} collections, ${counts.documents ?? 0} documents.`
      );
      onOpenChange(false);
      if (intoThisBank) router.refresh();
      else router.push(`/knowledge-banks/${encodeURIComponent(target)}`);
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Import an archive</DialogTitle>
          <DialogDescription>
            A ZIP from the export above. Into this bank it is a merge — schemas, collections and
            documents are written over what is there; into a new id it restores the whole bank.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4">
          <div className="space-y-1.5">
            <Label htmlFor="kb-import-file">Archive</Label>
            <Input
              id="kb-import-file"
              type="file"
              accept=".zip,application/zip"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="kb-import-target">Into</Label>
            <Input
              id="kb-import-target"
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              placeholder={kbId}
            />
            <p className="text-xs text-muted-foreground">
              {intoThisBank
                ? "This bank — the archive is folded into it."
                : `A new bank called ${target || "…"}, which must not exist yet.`}
            </p>
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={busy}>
            Cancel
          </Button>
          <Button onClick={run} disabled={busy || !file || !target.trim()}>
            {busy && <Spinner size="sm" className="mr-2" />}
            Import
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
