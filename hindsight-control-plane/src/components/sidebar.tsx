"use client";

import { useEffect, useState } from "react";
import { useTranslations } from "next-intl";
import { useBank } from "@/lib/bank-context";
import { bankRoute } from "@/lib/bank-url";
import { Home, Search, Sparkles, Database, FileText, Users, Network, Settings } from "lucide-react";
import { cn } from "@/lib/utils";
import { BankRail, railItemClass } from "@/components/bank-rail";
import Link from "next/link";
import { client } from "@/lib/api";

type NavItem =
  | "home"
  | "recall"
  | "reflect"
  | "data"
  | "documents"
  | "entities"
  | "knowledge"
  | "profile";

interface SidebarProps {
  currentTab: NavItem;
  onTabChange: (tab: NavItem) => void;
}

export function Sidebar({ currentTab, onTabChange }: SidebarProps) {
  const t = useTranslations("bank.sidebar");
  const tBank = useTranslations("bank");
  const { currentBank } = useBank();
  const [apiVersion, setApiVersion] = useState<string | null>(null);

  useEffect(() => {
    client
      .getVersion()
      .then((v) => setApiVersion(v.api_version))
      .catch(() => setApiVersion(null));
  }, []);

  if (!currentBank) {
    return null;
  }

  const navItems = [
    { id: "home" as NavItem, label: t("home"), icon: Home },
    { id: "data" as NavItem, label: t("memories"), icon: Database },
    { id: "knowledge" as NavItem, label: t("knowledge"), icon: Network },
    { id: "recall" as NavItem, label: t("recall"), icon: Search },
    { id: "reflect" as NavItem, label: t("reflect"), icon: Sparkles },
    { id: "documents" as NavItem, label: t("documents"), icon: FileText },
    { id: "entities" as NavItem, label: t("entities"), icon: Users },
    { id: "profile" as NavItem, label: tBank("bankConfiguration"), icon: Settings },
  ];

  return (
    <BankRail
      labels={{
        expand: t("expandSidebar"),
        collapse: t("collapseSidebar"),
        collapseAction: t("collapse"),
      }}
      footerNote={(isCollapsed) =>
        apiVersion ? (
          <div
            className={cn(
              "mb-2 text-xs text-muted-foreground/60 text-center select-none",
              isCollapsed ? "px-0" : "px-1"
            )}
            title={`Hindsight API v${apiVersion}`}
          >
            {isCollapsed ? `v${apiVersion}` : `Hindsight v${apiVersion}`}
          </div>
        ) : null
      }
    >
      {(isCollapsed) =>
        navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentTab === item.id;
          const href = bankRoute(currentBank, `?view=${item.id}`);

          return (
            <li key={item.id}>
              <Link
                href={href}
                onClick={(e) => {
                  // Don't bubble — clicking an item navigates, it doesn't toggle the rail.
                  e.stopPropagation();
                  // For left-click, handle navigation in the parent so there is no full
                  // page reload; middle-click and Ctrl/Cmd+click open a tab as usual.
                  if (e.button === 0 && !e.ctrlKey && !e.metaKey) {
                    e.preventDefault();
                    onTabChange(item.id);
                    // Give the header logo a playful spin on navigation.
                    window.dispatchEvent(new CustomEvent("hindsight:logo-spin"));
                  }
                }}
                className={railItemClass(isActive, isCollapsed)}
                title={isCollapsed ? item.label : undefined}
              >
                <Icon className="w-5 h-5 flex-shrink-0" />
                {!isCollapsed && <span>{item.label}</span>}
              </Link>
            </li>
          );
        })
      }
    </BankRail>
  );
}
