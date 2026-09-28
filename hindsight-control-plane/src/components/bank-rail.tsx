"use client";

// The left rail both bank pages use. It was copied once for knowledge banks and the two
// copies drifted immediately — the collapsed copy kept its horizontal padding, so every
// icon sat off-centre against the memory rail's. The chrome (width, collapse behaviour,
// footer) and the item classes live here now, so a change to one rail is a change to both.
//
// What each page still owns is its items and what an item *is*: the memory rail's are
// Links (so middle-click opens a tab), the knowledge rail's are buttons.

import * as React from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

/** The classes one rail item wears, active or not, collapsed or not. */
export function railItemClass(isActive: boolean, isCollapsed: boolean): string {
  return cn(
    "w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all cursor-pointer",
    isActive
      ? "bg-primary-gradient text-white shadow-sm"
      : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
    isCollapsed && "justify-center px-0"
  );
}

export function BankRail({
  labels,
  footerNote,
  children,
  testId,
}: {
  labels: { expand: string; collapse: string; collapseAction: string };
  /** A line above the collapse button, e.g. the API version. Gets the collapsed state. */
  footerNote?: (isCollapsed: boolean) => React.ReactNode;
  /** The items, rendered by the page; it is handed the collapsed state. */
  children: (isCollapsed: boolean) => React.ReactNode;
  testId?: string;
}) {
  const [isCollapsed, setIsCollapsed] = React.useState(true);
  const toggle = () => setIsCollapsed((v) => !v);

  return (
    <aside
      onClick={toggle}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        // Only when the aside itself has focus. Without this guard, pressing Enter on a
        // focused nav item bubbles up here and collapses the rail on every keyboard
        // navigation.
        if (e.target !== e.currentTarget) return;
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          toggle();
        }
      }}
      aria-label={isCollapsed ? labels.expand : labels.collapse}
      aria-expanded={!isCollapsed}
      data-testid={testId}
      className={cn(
        "bg-card border-r border-border flex flex-col h-full transition-all duration-300 cursor-pointer select-none",
        isCollapsed ? "w-16" : "w-64"
      )}
    >
      <nav className="flex-1 p-3 pt-4">
        <ul className="space-y-1">{children(isCollapsed)}</ul>
      </nav>

      {/* An explicit affordance alongside the ambient click-to-toggle on the chrome. */}
      <div className="p-3 border-t border-border">
        {footerNote?.(isCollapsed)}
        <button
          onClick={(e) => {
            // Don't double-toggle when the aside's onClick also fires.
            e.stopPropagation();
            toggle();
          }}
          className={cn(
            "w-full flex items-center gap-3 px-4 py-2 rounded-lg text-sm text-muted-foreground hover:bg-accent hover:text-accent-foreground transition-colors cursor-pointer",
            isCollapsed && "justify-center px-0"
          )}
          title={isCollapsed ? labels.expand : labels.collapse}
        >
          {isCollapsed ? (
            <ChevronRight className="w-5 h-5" />
          ) : (
            <>
              <ChevronLeft className="w-5 h-5" />
              <span>{labels.collapseAction}</span>
            </>
          )}
        </button>
      </div>
    </aside>
  );
}
