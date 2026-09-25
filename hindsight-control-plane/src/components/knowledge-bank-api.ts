"use client";

// Thin client for the knowledge-bank endpoints, proxied by
// src/app/api/knowledge-banks/[[...path]]/route.ts.

import { withBasePath } from "@/lib/base-path";

export interface KnowledgeBank {
  bank_id: string;
  name: string;
  documents: number;
  chunks: number;
  last_write_at?: string | null;
  operations_in_flight?: number;
  created_at: string;
  updated_at: string;
}

export interface KnowledgeDocument {
  doc_id: string;
  title: string | null;
  tags: string[];
  metadata: Record<string, unknown>;
  chunk_count: number;
  chars: number;
  created_at: string;
  updated_at: string;
}

export interface SearchResult {
  document_id: string;
  chunk_index: number;
  text: string;
  score: number;
  ranks: { vector?: number; keyword?: number };
}

export interface KnowledgeOperation {
  id: string;
  task_type: string;
  status: string;
  created_at: string;
  updated_at: string | null;
  error_message: string | null;
  details?: string | null;
}

export async function kbFetch<T>(
  path: string,
  init?: { method?: string; body?: unknown }
): Promise<T> {
  const response = await fetch(withBasePath(`/api/knowledge-banks${path}`), {
    method: init?.method ?? "GET",
    headers: { "Content-Type": "application/json" },
    body: init?.body === undefined ? undefined : JSON.stringify(init.body),
  });
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  if (!response.ok) {
    const detail = typeof data.detail === "string" ? data.detail : `HTTP ${response.status}`;
    throw new Error(detail);
  }
  return data as T;
}
