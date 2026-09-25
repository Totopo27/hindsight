# Knowledge banks — design

Status: **draft, iterating.**

## Goal

Users who want their documents searchable build a RAG pipeline themselves: chunking, embeddings, a
vector store, a keyword index, metadata extraction, a query layer, and glue to keep it all in sync.
Knowledge banks replace that pipeline with a bank documents go into.

**Source connectors are not in scope.** Users keep whatever they already use to fetch documents
from Drive, S3, a CMS or their own app, and hand us the text.

## Requirements

- Users can drop their own RAG pipeline and get the same results or better, without writing
  retrieval code.
- Good answers out of the box: no tuning of chunking, embeddings or ranking to get started.
- Not only search: answers to questions the documents can support, including counting, filtering
  and comparing across documents.
- Every answer traces back to the documents it came from.
- Users can see and correct what the system concluded, instead of trusting a black box.
- Costs are predictable and visible before they are spent, and nothing expensive happens unasked.
- Changing your mind is cheap: reorganising what is extracted should not mean reprocessing
  everything.
- One API and one UI for the whole thing, in the product users already run.
- Documents stay isolated per bank and per tenant.

## Phases

Each phase ships on its own.

### v1 — Documents and hybrid search

- Send documents, search them; nothing else to run.
- Results as good as a tuned hybrid pipeline (meaning-based and keyword search combined).
- Filter by tags, and inspect a result to see why it matched.
- No LLM cost at ingestion.

**API** — its own resource, `/knowledge-banks`, beside `/banks`. A memory bank's API is retain /
recall / reflect over facts; almost none of it means anything for documents, so sharing `/banks`
would mean a type check on every route and two meanings per path.

- `POST /v1/default/knowledge-banks` — create: `{ "id": "contracts", "name": "Contracts" }`
- `GET /v1/default/knowledge-banks` — list, paginated and searchable
- `GET /v1/default/knowledge-banks/{kb}` — documents, chunks, last write
- `DELETE /v1/default/knowledge-banks/{kb}`
- `PUT …/{kb}/documents/{id}` — `{ "text", "tags": [], "date" }`; idempotent on content, so
  re-sending the same text is a no-op
- `GET …/{kb}/documents` — list; `GET …/{kb}/documents/{id}` — the text and its chunks
- `DELETE …/{kb}/documents/{id}` — takes everything derived from it with it
- `POST …/{kb}/search` — `{ "query", "top_k", "mode": "hybrid|vector|keyword", "tags" }`; each hit
  returns document, chunk, text, score and the rank from each arm
- Everything in later phases hangs off `…/{kb}/…` too
- `default` is the fixed segment every path already uses; tenancy comes from the API key, not the
  URL
- Ids are their own namespace: a memory bank and a knowledge bank can both be called `contracts`
- Name clash to settle: `/banks/{id}/knowledge-base` already exists and means something else (the
  page tree over a memory bank's mental models)
- MCP: one new tool, `search_documents`, so an agent can use a knowledge bank alongside memory

**UI**

- The bank picker lists memory banks and knowledge banks in two groups, with a type chip; "New"
  asks which kind
- A knowledge bank opens on its own page, with two tabs:
  - **Documents** — list with size and chunk count; add, delete, open to read
  - **Search** — query box, arm switch (hybrid / vector / keyword), tag filter; results show
    document, chunk, both ranks and highlighted terms

### v2 — Metadata extraction and classification

Replaces a metadata-extraction pipeline stage (e.g. Vectorize's automatic metadata extraction).

- Users define a schema of the properties they want: at document level (title, author, document
  type…) and at chunk level (clause type, section, party, dates…).
- Properties are typed — text, number, boolean, date — and can be a fixed set of values, which is
  how classification is expressed: the LLM picks from the list or says none applies.
- The LLM reads each document and fills the schema in: document properties once per document,
  chunk properties per chunk, only for the chunks the property applies to.
- Starting points: blank, a template for common document types, or a schema proposed from a sample
  of the user's own documents.
- Search filters on any of it, and facet counts show how the corpus splits per value.
- Optionally the values are prepended to the chunk text before embedding, so classification also
  improves semantic matching.
- Re-extraction is a decision, with the cost shown first.

**API**

- `PUT …/{kb}/metadata-schema` — `{ "document": {...}, "chunks": {...} }`: properties with type,
  optional `values` (the classification list) and a description of what to extract
- `GET …/{kb}/metadata-schema` — the schema, how many documents and chunks are extracted, counts
  per value
- `POST …/{kb}/metadata-schema/suggest` — propose a schema from a sample of the bank's documents
- `POST …/{kb}/metadata-schema/apply` — `dry_run` first: documents and chunks to read, LLM calls,
  sample output; then run as a background operation with progress
- `POST …/{kb}/search` — takes `metadata: {...}` filters (document-level and chunk-level) next to
  `tags`, and returns facet counts
- New documents are extracted on write with the schema in force

**UI**

- A **Metadata** tab: the two schemas (document, chunk) edited in place, "Suggest from my
  documents", a bar per property showing how the corpus splits, and "Apply" showing cost and
  sample output before it runs
- In Search, metadata filters sit next to the tag filter, and each result shows the values that
  matched
- On a document, its properties and each chunk's properties are visible

### v3 — Structured records

- Describe the records wanted (vendors, contracts, invoices) and the system fills them in from the
  documents.
- One record per real-world thing, gathered across every document that mentions it — not one per
  file.
- Questions RAG cannot answer become ordinary queries: totals, counts, filters, sorting, "which of
  these end after 2026".
- Every value shows the document and sentence it came from; disagreements between documents are
  visible; a human correction sticks.
- Records link to each other: from a contract to its vendor and back.

**API**

- `PUT …/{kb}/tables/{t}` — what one row is, its fields, how rows are recognised across documents,
  how values merge; `GET …/{kb}/tables`
- `POST …/{kb}/rows/query` — `{ "from", "where", "order_by", "group_by", "aggregate", "limit" }`
- `GET …/{kb}/tables/{t}/rows/{id}` — the row, with the rows it links to
- `GET …/{kb}/tables/{t}/rows/{id}/evidence` — each value with its document, date and quote
- `PUT …/{kb}/tables/{t}/rows/{id}/pins` — a corrected value that outranks the documents
- `POST …/{kb}/tables/{t}/rows:merge` — two rows that are the same record
- `POST …/{kb}/search` — takes `where: {table, filter}`: only documents behind matching records
- MCP: `query_rows`, next to `search_documents`

**UI**

- A **Records** tab: table picker, rows in a grid with clickable links and disagreements marked,
  and a query box with ready-made examples
- Clicking a row opens a panel: every field with the documents and quotes behind it, the competing
  values when there are any, a pin button, and the rows that reference it

### v4 — Schema editing and assistant

- Describe the change in words, get a concrete proposal to review.
- Nothing changes until it is accepted, and the cost of each change is shown first.
- Cheap changes stay cheap: renaming, removing, or changing how conflicting values are resolved
  does not re-read documents.
- The same review path whether the change came from a person or the assistant.

**API**

- `GET …/{kb}/model` — tables, links, row counts
- `POST …/{kb}/model/preview` — a list of changes → the resulting model, problems, each change
  tagged free or re-reads documents, and the LLM calls it would take
- `POST …/{kb}/model/apply` — the same list, applied together
- `POST …/{kb}/model/assistant` — `{ messages, pending }` → reply plus proposed changes
- One list of change types (add / drop / rename a table or field, change a field, change how it
  merges, change identity) and one validator, whoever sent them

**UI**

- A **Model** tab: tables drawn as cards with their links, an editor panel for the selected table,
  and a chat beside it
- Edits from either side pile up as pending changes; one **Review N changes** screen lists them
  with their cost and applies them together

### v5 — Ask over records and documents

- One chat over the bank that answers from the records, from the documents, or both, as the
  question requires.
- It shows what it looked up, and cites documents and records.
- Available to agents as a tool, not only in the UI.

**API**

- `POST …/{kb}/ask` — `{ messages }` → the answer, the steps it took (tool, arguments, result) and
  citations to documents and rows
- Same shape as `reflect`, over a knowledge bank; MCP: `ask`

**UI**

- An **Ask** tab: the conversation, each answer preceded by the lookups it made (collapsed), with
  citations that open the document or the row

## Out of scope

Source connectors and sync, per-document access control, billing and quotas, cross-bank record
linking.
