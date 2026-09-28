"use client";

// Left rail for a knowledge bank. The chrome and the item classes come from BankRail,
// which the memory-bank rail uses too; only the items differ — a knowledge bank has
// documents and search, not memories.

import { FileText, LayoutGrid, ListChecks, Search, Settings, Table2, Tags } from "lucide-react";
import { BankRail, railItemClass } from "@/components/bank-rail";

export type KbSection =
  | "overview"
  | "documents"
  | "search"
  | "schemas"
  | "collections"
  | "operations"
  | "settings";

const ITEMS: { id: KbSection; label: string; icon: typeof LayoutGrid }[] = [
  { id: "overview", label: "Overview", icon: LayoutGrid },
  { id: "documents", label: "Documents", icon: FileText },
  { id: "search", label: "Search", icon: Search },
  { id: "schemas", label: "Schemas", icon: Tags },
  { id: "collections", label: "Collections", icon: Table2 },
  { id: "operations", label: "Operations", icon: ListChecks },
  { id: "settings", label: "Settings", icon: Settings },
];

export function KnowledgeBankSidebar({
  current,
  onChange,
}: {
  current: KbSection;
  onChange: (section: KbSection) => void;
}) {
  return (
    <BankRail
      testId="kb-sidebar"
      labels={{
        expand: "Expand sidebar",
        collapse: "Collapse sidebar",
        collapseAction: "Collapse",
      }}
    >
      {(isCollapsed) =>
        ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <li key={item.id}>
              <button
                onClick={(e) => {
                  // Clicking an item navigates; it does not toggle the rail.
                  e.stopPropagation();
                  onChange(item.id);
                }}
                title={item.label}
                className={railItemClass(current === item.id, isCollapsed)}
              >
                <Icon className="w-5 h-5 shrink-0" />
                {!isCollapsed && <span className="truncate">{item.label}</span>}
              </button>
            </li>
          );
        })
      }
    </BankRail>
  );
}
