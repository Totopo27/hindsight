"use client";

// The stat strip the bank stats page opens with: an icon, a label and one number per
// cell, divided by hairlines. Shared so a second page showing counts doesn't invent a
// second way to show them.

import * as React from "react";
import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

/** 1_234 → "1.2k". Long numbers make a row of stats unreadable; the exact value stays
 *  in the title attribute for anyone who needs it. */
export function formatCompact(n: number): string {
  if (n < 1000) return n.toString();
  if (n < 1_000_000) {
    const k = n / 1000;
    return `${k >= 10 ? k.toFixed(0) : k.toFixed(1).replace(/\.0$/, "")}k`;
  }
  if (n < 1_000_000_000) {
    const m = n / 1_000_000;
    return `${m >= 10 ? m.toFixed(0) : m.toFixed(1).replace(/\.0$/, "")}M`;
  }
  const b = n / 1_000_000_000;
  return `${b >= 10 ? b.toFixed(0) : b.toFixed(1).replace(/\.0$/, "")}B`;
}

export function CompactNumber({ value, className }: { value: number; className?: string }) {
  return (
    <span className={className} title={value.toLocaleString()}>
      {formatCompact(value)}
    </span>
  );
}

export function InlineStat({
  icon: Icon,
  label,
  value,
  suffix,
}: {
  icon: LucideIcon;
  label: string;
  value: number;
  /** Printed after the number, e.g. "%". */
  suffix?: string;
}) {
  return (
    <div className="flex items-center gap-3 p-4">
      <div className="p-2 rounded-md bg-muted">
        <Icon className="w-4 h-4 text-muted-foreground" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-xs text-muted-foreground font-medium">{label}</p>
        <span className="text-2xl font-semibold text-foreground leading-tight tabular-nums block">
          <CompactNumber value={value} />
          {suffix}
        </span>
      </div>
    </div>
  );
}

/** The cells in a row, hairline-divided, as a bordered strip. */
export function StatStrip({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "grid grid-cols-1 rounded-lg border border-border bg-card sm:grid-cols-3 sm:divide-x divide-y sm:divide-y-0 divide-border/60",
        className
      )}
    >
      {children}
    </div>
  );
}
