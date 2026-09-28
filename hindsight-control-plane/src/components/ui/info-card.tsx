"use client";

// The detail-dialog furniture: a titled card, a labelled row inside it, and the label
// itself. Three copies of these existed — documents-view, mental-model-detail-modal and
// the knowledge-bank page — and they had already drifted in type size and casing, which
// is exactly what makes two dialogs in one product look like two products.

import * as React from "react";

export function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground mb-1">
      {children}
    </div>
  );
}

export function InfoCard({
  title,
  icon,
  children,
}: {
  title: string;
  icon?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-border bg-muted/20 overflow-hidden">
      <div className="flex items-center gap-1.5 px-4 py-2 border-b border-border bg-muted/40 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        {icon}
        {title}
      </div>
      <div className="p-4 space-y-4">{children}</div>
    </div>
  );
}

export function MetadataRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <SectionLabel>{label}</SectionLabel>
      <div className="text-sm text-foreground">{value}</div>
    </div>
  );
}
