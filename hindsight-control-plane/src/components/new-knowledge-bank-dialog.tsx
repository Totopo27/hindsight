"use client";

// Creating a knowledge bank. Shared, because it is reachable from two places — the
// header selector's footer and the knowledge list on the overview — and a second copy
// would be a second set of rules about what an id may be.

import * as React from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { kbFetch } from "@/components/knowledge-bank-api";

export function NewKnowledgeBankDialog({
  open,
  onOpenChange,
  onCreated,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Called after the bank exists, before navigating — for a list that must refresh. */
  onCreated?: (bankId: string) => void;
}) {
  const router = useRouter();
  const [id, setId] = React.useState("");
  const [creating, setCreating] = React.useState(false);

  const create = async () => {
    const bankId = id.trim();
    if (!bankId) return;
    setCreating(true);
    try {
      await kbFetch("", { method: "POST", body: { id: bankId } });
      setId("");
      onOpenChange(false);
      onCreated?.(bankId);
      // Straight into the new bank: there is nothing to see on the list that the
      // workspace does not show, and the next thing anyone does is write to it.
      router.push(`/knowledge-banks/${encodeURIComponent(bankId)}`);
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setCreating(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>New knowledge bank</DialogTitle>
          <DialogDescription>
            The id is how the API addresses it. Knowledge banks and memory banks have separate ids.
          </DialogDescription>
        </DialogHeader>
        <Input
          autoFocus
          placeholder="product-docs"
          value={id}
          onChange={(e) => setId(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") create();
          }}
        />
        <DialogFooter>
          <Button disabled={!id.trim() || creating} onClick={create}>
            {creating ? "Creating…" : "Create"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
