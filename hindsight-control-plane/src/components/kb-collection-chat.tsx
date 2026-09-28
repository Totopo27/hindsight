"use client";

// Talking to a model about a bank's collections.
//
// The agent has one tool and it does not apply anything: a tool call comes back as a
// proposal card in the transcript, which the person approves. Approving is this
// component's PUT or DELETE — the model never holds the pen. The conversation lives
// here rather than on the server, and every turn re-reads the bank, so after an approval
// the next answer is about what the bank now is.

import * as React from "react";
import { toast } from "sonner";
import { Check, Send, Sparkles, Wrench, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { kbFetch, type CollectionProposal } from "@/components/knowledge-bank-api";
import { cn } from "@/lib/utils";

interface Turn {
  role: "user" | "assistant";
  content: string;
  /** Proposals the assistant made on this turn; approved/dismissed in place. */
  proposals?: CollectionProposal[];
  applied?: boolean;
  dismissed?: boolean;
}

/** The order a set of changes can be applied in: a collection nothing points at first,
 *  its referrers after it, deletes last. Same rule the staged editor uses, because the
 *  API validates a relationship against what exists at the time of the call. */
function applyOrder(proposals: CollectionProposal[]): CollectionProposal[] {
  const writes = proposals.filter((p) => p.action !== "delete");
  const deletes = proposals.filter((p) => p.action === "delete");
  const placed = new Set<string>();
  const ordered: CollectionProposal[] = [];
  let remaining = [...writes];
  while (remaining.length > 0) {
    const ready = remaining.filter((p) =>
      Object.values(p.definition?.fields ?? {})
        .map((spec) => spec.collection)
        .filter((target): target is string => Boolean(target) && target !== p.collection_id)
        .every((target) => placed.has(target) || !writes.some((w) => w.collection_id === target))
    );
    const batch = ready.length > 0 ? ready : remaining;
    for (const p of batch) {
      ordered.push(p);
      placed.add(p.collection_id);
    }
    remaining = remaining.filter((p) => !batch.includes(p));
  }
  return [...ordered, ...deletes];
}

function summarise(proposal: CollectionProposal): string {
  if (proposal.action === "delete") return "remove this collection";
  const fields = Object.keys(proposal.definition?.fields ?? {});
  return `${fields.length} field${fields.length === 1 ? "" : "s"}: ${fields.join(", ")}`;
}

export function CollectionChat({
  kbId,
  onApplied,
}: {
  kbId: string;
  /** Called after an approval lands, so the page re-reads what changed. */
  onApplied: () => Promise<void> | void;
}) {
  const [turns, setTurns] = React.useState<Turn[]>([]);
  const [draft, setDraft] = React.useState("");
  const [thinking, setThinking] = React.useState(false);
  const [applying, setApplying] = React.useState(false);
  const bottom = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, thinking]);

  const send = async () => {
    const text = draft.trim();
    if (!text || thinking) return;
    const next: Turn[] = [...turns, { role: "user", content: text }];
    setTurns(next);
    setDraft("");
    setThinking(true);
    try {
      const response = await kbFetch<{
        reply: string;
        proposals: CollectionProposal[];
        warnings: string[];
        documents_read: number;
      }>(`/${encodeURIComponent(kbId)}/collections/chat`, {
        method: "POST",
        // Only the text goes back: a proposal is a fact about this conversation, and the
        // bank's real state is read server-side on every turn anyway.
        body: {
          messages: next.map((turn) => ({ role: turn.role, content: turn.content })),
          sample_documents: 8,
        },
      });
      setTurns([
        ...next,
        {
          role: "assistant",
          // A model that only calls the tool says nothing; the proposal below is the
          // answer, so the bubble introduces it rather than reading as a blank turn.
          content:
            response.reply ||
            (response.proposals.length > 0 ? "Here is what I suggest:" : "I have nothing to add."),
          proposals: response.proposals.length > 0 ? response.proposals : undefined,
        },
      ]);
      for (const warning of response.warnings) toast.warning(warning);
    } catch (e) {
      toast.error((e as Error).message);
      setTurns(next);
    } finally {
      setThinking(false);
    }
  };

  const approve = async (index: number) => {
    const proposals = turns[index]?.proposals;
    if (!proposals) return;
    setApplying(true);
    try {
      for (const proposal of applyOrder(proposals)) {
        const path = `/${encodeURIComponent(kbId)}/collections/${encodeURIComponent(proposal.collection_id)}`;
        if (proposal.action === "delete") await kbFetch(path, { method: "DELETE" });
        else await kbFetch(path, { method: "PUT", body: proposal.definition });
      }
      setTurns((prev) => prev.map((turn, i) => (i === index ? { ...turn, applied: true } : turn)));
      toast.success("Applied. Derive the collections to fill them from the documents.");
      await onApplied();
    } catch (e) {
      toast.error(`${(e as Error).message} — earlier changes were applied.`);
    } finally {
      setApplying(false);
    }
  };

  return (
    <div className="flex h-full flex-col gap-3">
      <div className="flex-1 space-y-3 overflow-y-auto pr-1">
        {turns.length === 0 && (
          <div className="rounded-lg border border-dashed border-border p-4 text-sm text-muted-foreground">
            <p className="mb-2 flex items-center gap-1.5 font-medium text-foreground">
              <Sparkles className="h-4 w-4" /> Ask about this bank&apos;s collections
            </p>
            <p>
              It reads the collections you have and a sample of your documents. It can answer, or
              propose changes — which it cannot apply. You approve them here.
            </p>
            <p className="mt-2 text-xs">
              Try: “what should I be extracting from these documents?” · “split vendors into vendors
              and contacts” · “why is my country field always empty?”
            </p>
          </div>
        )}

        {turns.map((turn, index) => (
          <div key={index} className={cn(turn.role === "user" && "flex justify-end")}>
            <div
              className={cn(
                "max-w-[85%] rounded-lg px-3 py-2 text-sm whitespace-pre-wrap",
                turn.role === "user"
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted text-foreground"
              )}
            >
              {turn.content}
            </div>

            {turn.proposals && (
              <div className="mt-2 space-y-2">
                {turn.proposals.map((proposal) => (
                  <div
                    key={proposal.collection_id}
                    className="rounded-lg border border-primary/40 bg-primary/5 p-3"
                  >
                    <div className="flex items-center gap-2 text-sm">
                      <Wrench className="h-3.5 w-3.5 text-primary" />
                      <span className="font-semibold capitalize">{proposal.action}</span>
                      <span className="font-mono">{proposal.collection_id}</span>
                    </div>
                    <p className="mt-1 text-xs text-muted-foreground">{summarise(proposal)}</p>
                    {proposal.reason && (
                      <p className="mt-1 text-xs text-muted-foreground italic">{proposal.reason}</p>
                    )}
                    {proposal.definition?.identity && (
                      <p className="mt-1 text-xs text-muted-foreground">
                        Identified by{" "}
                        <span className="font-mono">{proposal.definition.identity}</span>
                      </p>
                    )}
                  </div>
                ))}

                {turn.applied ? (
                  <p className="text-xs text-muted-foreground">
                    <Check className="mr-1 inline h-3 w-3" /> Applied.
                  </p>
                ) : turn.dismissed ? (
                  <p className="text-xs text-muted-foreground">Dismissed.</p>
                ) : (
                  <div className="flex gap-2">
                    <Button size="sm" disabled={applying} onClick={() => approve(index)}>
                      {applying ? <Spinner size="sm" /> : <Check className="h-4 w-4 mr-1" />}
                      Approve and apply
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() =>
                        setTurns((prev) =>
                          prev.map((t, i) => (i === index ? { ...t, dismissed: true } : t))
                        )
                      }
                    >
                      <X className="h-4 w-4 mr-1" /> Dismiss
                    </Button>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}

        {thinking && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Spinner size="sm" /> Reading your documents…
          </div>
        )}
        <div ref={bottom} />
      </div>

      <div className="flex gap-2">
        <Input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && send()}
          placeholder="Ask, or describe what you want to track…"
          disabled={thinking}
        />
        <Button onClick={send} disabled={thinking || !draft.trim()}>
          <Send className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}
