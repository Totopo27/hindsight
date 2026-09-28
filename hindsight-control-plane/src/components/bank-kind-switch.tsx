"use client";

// Memory banks and knowledge banks are both banks, so they are chosen in one place
// rather than living on separate pages: this switch appears in the banks overview and
// in the header selector, and both read the same two values.

import { Brain, Database } from "lucide-react";
import { cn } from "@/lib/utils";

export type BankKind = "memory" | "knowledge";

export function BankKindSwitch({
  value,
  onChange,
  size = "default",
}: {
  value: BankKind;
  onChange: (kind: BankKind) => void;
  size?: "default" | "sm";
}) {
  const options: { kind: BankKind; label: string; Icon: typeof Brain }[] = [
    { kind: "memory", label: "Memory", Icon: Brain },
    { kind: "knowledge", label: "Knowledge", Icon: Database },
  ];
  return (
    <div className="inline-flex rounded-md border border-border bg-muted/40 p-0.5">
      {options.map(({ kind, label, Icon }) => (
        <button
          key={kind}
          type="button"
          onClick={() => onChange(kind)}
          aria-pressed={value === kind}
          className={cn(
            "inline-flex items-center gap-1.5 rounded-[5px] font-medium transition-colors",
            size === "sm" ? "px-2 py-1 text-xs" : "px-3 py-1.5 text-sm",
            value === kind
              ? "bg-background text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground"
          )}
        >
          <Icon className={size === "sm" ? "h-3.5 w-3.5" : "h-4 w-4"} />
          {label}
        </button>
      ))}
    </div>
  );
}
