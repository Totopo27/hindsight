"use client";

// Left rail for a knowledge bank. Mirrors the memory-bank rail (src/components/sidebar.tsx)
// — same collapse behaviour, same active styling — so the two detail pages read as one
// product. Only the items differ: a knowledge bank has documents and search, not memories.

import { useState } from "react";
import { FileText, LayoutGrid, Search, Settings, ListChecks, Tags } from "lucide-react";
import { cn } from "@/lib/utils";

export type KbSection =
  | "overview"
  | "documents"
  | "search"
  | "metadata"
  | "operations"
  | "configuration";

const ITEMS: { id: KbSection; label: string; icon: typeof LayoutGrid }[] = [
  { id: "overview", label: "Overview", icon: LayoutGrid },
  { id: "documents", label: "Documents", icon: FileText },
  { id: "search", label: "Search", icon: Search },
  { id: "metadata", label: "Metadata", icon: Tags },
  { id: "operations", label: "Operations", icon: ListChecks },
  { id: "configuration", label: "Configuration", icon: Settings },
];

export function KnowledgeBankSidebar({
  current,
  onChange,
}: {
  current: KbSection;
  onChange: (section: KbSection) => void;
}) {
  const [isCollapsed, setIsCollapsed] = useState(true);
  const toggle = () => setIsCollapsed((v) => !v);

  return (
    <aside
      onClick={toggle}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.target !== e.currentTarget) return;
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          toggle();
        }
      }}
      aria-label={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
      aria-expanded={!isCollapsed}
      data-testid="kb-sidebar"
      className={cn(
        "bg-card border-r border-border flex flex-col h-full transition-all duration-300 cursor-pointer select-none",
        isCollapsed ? "w-16" : "w-64"
      )}
    >
      <nav className="flex-1 p-3 pt-4">
        <ul className="space-y-1">
          {ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = current === item.id;
            return (
              <li key={item.id}>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onChange(item.id);
                  }}
                  title={item.label}
                  className={cn(
                    "w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all cursor-pointer",
                    isActive
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:bg-muted hover:text-foreground"
                  )}
                >
                  <Icon className="w-5 h-5 shrink-0" />
                  {!isCollapsed && <span className="truncate">{item.label}</span>}
                </button>
              </li>
            );
          })}
        </ul>
      </nav>
    </aside>
  );
}
