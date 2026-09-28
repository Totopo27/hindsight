"use client";

// Thin client for the knowledge-bank endpoints, proxied by
// src/app/api/knowledge-banks/[[...path]]/route.ts.

import { withBasePath } from "@/lib/base-path";

export interface KnowledgeBank {
  bank_id: string;
  name: string;
  documents: number;
  passages: number;
  last_write_at?: string | null;
  operations_in_flight?: number;
  created_at: string;
  updated_at: string;
}

export interface SchemaField {
  type: "string" | "integer" | "number" | "boolean" | "date" | "datetime" | "array" | "object";
  description?: string;
  values?: (string | number | boolean)[];
  items?: string;
  filterable?: boolean;
  indexed?: boolean;
}

export interface KnowledgeSchema {
  schema_id: string;
  name: string | null;
  description: string | null;
  document_fields: Record<string, SchemaField>;
  passage_fields: Record<string, SchemaField>;
  documents_with_fields: number;
  passages_with_fields: number;
  updated_at: string | null;
}

export interface KnowledgeCollection {
  collection_id: string;
  name: string | null;
  description: string | null;
  fields: Record<string, { type?: string; collection?: string; values?: unknown[] }>;
  identity: string | null;
  derive_on_write?: boolean;
  records?: number;
}

export interface QueryResult {
  columns: string[];
  rows: unknown[][];
  row_count: number;
  grouped: boolean;
}

export interface KnowledgeDocument {
  doc_id: string;
  title: string | null;
  tags: string[];
  metadata: Record<string, unknown>;
  fields: Record<string, unknown>;
  /** Which schema's fields were read out of this document; null when none applied. */
  schema_id: string | null;
  passage_count: number;
  chars: number;
  created_at: string;
  updated_at: string;
}

export interface SearchResult {
  document_id: string;
  passage_index: number;
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

/** Upload files to a knowledge bank; each one is converted and then written. */
export async function kbUpload(
  kbId: string,
  files: File[],
  options: { tags?: string[]; schema_id?: string | null }
): Promise<{ operation_ids: string[] }> {
  const form = new FormData();
  for (const file of files) form.append("files", file);
  form.append(
    "request",
    JSON.stringify({
      tags: options.tags ?? [],
      ...(options.schema_id ? { schema_id: options.schema_id } : {}),
    })
  );
  // No Content-Type header: the browser sets it with the multipart boundary.
  const response = await fetch(
    withBasePath(`/api/knowledge-banks/${encodeURIComponent(kbId)}/files`),
    { method: "POST", body: form }
  );
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  if (!response.ok) {
    throw new Error(typeof data.detail === "string" ? data.detail : `HTTP ${response.status}`);
  }
  return data as { operation_ids: string[] };
}
