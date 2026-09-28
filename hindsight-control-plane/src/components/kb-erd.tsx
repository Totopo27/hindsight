"use client";

// The shape of a knowledge bank as an entity diagram: one card per schema or collection,
// its fields listed with their types, and a curved line from a relationship field to the
// thing it points at. Schemas have no relationships, so their diagram is the cards alone
// — still the fastest way to see what a bank of six schemas actually defines.
//
// Positions are not stored: the cards flow in columns and the edges are measured from
// the laid-out DOM, so nothing has to be dragged and nothing can go stale.

import * as React from "react";
import { KeyRound, Link2, Lock } from "lucide-react";
import { cn } from "@/lib/utils";

export interface ErdField {
  name: string;
  type: string;
  /** A field that identifies the record, shown with a key. */
  primary?: boolean;
  /** A field that points at another card, shown with a link and drawn as an edge. */
  relation?: boolean;
  /** Derived or structural — shown quieter, like a system column. */
  muted?: boolean;
  /** Short flags after the name, e.g. filterable / indexed. */
  badges?: string[];
  /** What the field is, in the document's words; shown under the row. */
  description?: string;
  /** A classification's allowed values. */
  values?: (string | number | boolean)[];
}

export interface ErdNode {
  id: string;
  title: string;
  badge?: string;
  fields: ErdField[];
}

export interface ErdEdge {
  /** Node id the line leaves from. */
  from: string;
  /** Field on that node the line leaves from, so it starts at the right row. */
  fromField: string;
  /** Node id the line arrives at. */
  to: string;
}

interface Point {
  x: number;
  y: number;
}

export function ErdDiagram({
  nodes,
  edges,
  selected,
  onSelect,
  emptyMessage = "Nothing to draw yet.",
}: {
  nodes: ErdNode[];
  edges: ErdEdge[];
  selected?: string | null;
  onSelect?: (id: string) => void;
  emptyMessage?: string;
}) {
  // A card is placed right after the one that points at it, so a relationship is a
  // short line to a neighbour rather than one crossing the whole diagram. Cheap, and
  // enough for the handful of collections a bank has; a real layout is for when it isn't.
  const ordered = React.useMemo(() => {
    const byId = new Map(nodes.map((node) => [node.id, node]));
    const seen = new Set<string>();
    const out: ErdNode[] = [];
    const visit = (id: string) => {
      const node = byId.get(id);
      if (!node || seen.has(id)) return;
      seen.add(id);
      out.push(node);
      for (const edge of edges) if (edge.from === id) visit(edge.to);
    };
    for (const node of nodes) visit(node.id);
    return out;
  }, [nodes, edges]);

  const container = React.useRef<HTMLDivElement>(null);
  const rowRefs = React.useRef(new Map<string, HTMLElement>());
  const cardRefs = React.useRef(new Map<string, HTMLElement>());
  const [paths, setPaths] = React.useState<{ id: string; d: string; end: Point }[]>([]);

  // Edges are measured after layout and again whenever the container resizes, because
  // the cards reflow into one column on a narrow screen and every line moves with them.
  React.useLayoutEffect(() => {
    const measure = () => {
      const box = container.current?.getBoundingClientRect();
      if (!box) return;
      const next: { id: string; d: string; end: Point }[] = [];
      for (const edge of edges) {
        const source = rowRefs.current.get(`${edge.from}.${edge.fromField}`);
        const target = cardRefs.current.get(edge.to);
        if (!source || !target) continue;
        const a = source.getBoundingClientRect();
        const b = target.getBoundingClientRect();
        // Leave from whichever side of the row faces the target, and arrive on the
        // facing side of the card: a line that crosses its own card reads as a mistake.
        const leftToRight = b.left + b.width / 2 >= a.left + a.width / 2;
        const start = {
          x: (leftToRight ? a.right : a.left) - box.left,
          y: a.top + a.height / 2 - box.top,
        };
        const end = {
          x: (leftToRight ? b.left : b.right) - box.left,
          y: b.top + Math.min(b.height / 2, 28) - box.top,
        };
        const bend = Math.max(28, Math.abs(end.x - start.x) / 2);
        const c1 = { x: start.x + (leftToRight ? bend : -bend), y: start.y };
        const c2 = { x: end.x + (leftToRight ? -bend : bend), y: end.y };
        next.push({
          id: `${edge.from}.${edge.fromField}->${edge.to}`,
          d: `M ${start.x} ${start.y} C ${c1.x} ${c1.y}, ${c2.x} ${c2.y}, ${end.x} ${end.y}`,
          end,
        });
      }
      setPaths(next);
    };
    measure();
    const observer = new ResizeObserver(measure);
    if (container.current) observer.observe(container.current);
    window.addEventListener("resize", measure);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [nodes, edges]);

  if (nodes.length === 0) {
    return <p className="text-sm text-muted-foreground">{emptyMessage}</p>;
  }

  return (
    <div
      ref={container}
      className="relative rounded-lg border border-border bg-muted/20 p-6 overflow-x-auto"
      style={{
        backgroundImage:
          "radial-gradient(circle, color-mix(in srgb, currentColor 12%, transparent) 1px, transparent 1px)",
        backgroundSize: "16px 16px",
      }}
    >
      <svg className="pointer-events-none absolute inset-0 h-full w-full overflow-visible">
        {paths.map((path) => (
          <g key={path.id}>
            <path d={path.d} fill="none" className="stroke-primary/60" strokeWidth={1.5} />
            <circle cx={path.end.x} cy={path.end.y} r={3} className="fill-primary/70" />
          </g>
        ))}
      </svg>

      <div className="relative grid gap-6 md:grid-cols-2 xl:grid-cols-3">
        {ordered.map((node) => (
          <div
            key={node.id}
            ref={(el) => {
              if (el) cardRefs.current.set(node.id, el);
              else cardRefs.current.delete(node.id);
            }}
            onClick={() => onSelect?.(node.id)}
            className={cn(
              "h-fit rounded-lg border bg-card shadow-sm overflow-hidden transition-colors",
              onSelect && "cursor-pointer hover:border-primary/60",
              selected === node.id ? "border-primary" : "border-border"
            )}
          >
            <div className="flex items-center justify-between gap-2 bg-muted/60 px-3 py-2">
              <span className="font-mono text-sm font-semibold truncate">{node.title}</span>
              {node.badge && (
                <span className="shrink-0 rounded bg-primary/15 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-primary">
                  {node.badge}
                </span>
              )}
            </div>
            <div className="divide-y divide-border/60">
              {node.fields.length === 0 && (
                <div className="px-3 py-2 text-xs text-muted-foreground">No fields</div>
              )}
              {node.fields.map((field) => (
                <div
                  key={field.name}
                  ref={(el) => {
                    const key = `${node.id}.${field.name}`;
                    if (el) rowRefs.current.set(key, el);
                    else rowRefs.current.delete(key);
                  }}
                  className={cn("px-3 py-1.5 text-xs", field.muted && "text-muted-foreground")}
                >
                  <div className="flex items-center gap-2">
                    {field.primary ? (
                      <KeyRound className="h-3 w-3 shrink-0 text-amber-500" />
                    ) : field.relation ? (
                      <Link2 className="h-3 w-3 shrink-0 text-primary" />
                    ) : field.muted ? (
                      <Lock className="h-3 w-3 shrink-0 opacity-60" />
                    ) : (
                      <span className="w-3 shrink-0" />
                    )}
                    <span className="font-mono truncate">{field.name}</span>
                    {field.badges?.map((badge) => (
                      <span
                        key={badge}
                        className="shrink-0 rounded bg-muted px-1 py-0.5 text-[10px] text-muted-foreground"
                      >
                        {badge}
                      </span>
                    ))}
                    <span className="ml-auto shrink-0 text-muted-foreground">{field.type}</span>
                  </div>
                  {field.values && field.values.length > 0 && (
                    <div className="pl-5 text-[10px] text-muted-foreground">
                      {field.values.map(String).join(" · ")}
                    </div>
                  )}
                  {field.description && (
                    <div className="pl-5 text-[10px] text-muted-foreground">
                      {field.description}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
