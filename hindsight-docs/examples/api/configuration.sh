#!/bin/bash
# Configuration examples for Hindsight (bank config over HTTP)
# Run: bash examples/api/configuration.sh

set -e

HINDSIGHT_API_URL="${HINDSIGHT_API_URL:-http://localhost:8888}"

# =============================================================================
# Doc Examples
# =============================================================================

# [docs:plain-retrieval-bank]
curl -X PUT "$HINDSIGHT_API_URL/v1/default/banks/plain-retrieval-bank" \
  -H "Authorization: Bearer $HINDSIGHT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "retain_extraction_mode": "chunks",
    "enable_observations": false,
    "enable_temporal_retrieval": false,
    "enable_graph_retrieval": false,
    "enable_reranking": false
  }'
# [/docs:plain-retrieval-bank]
echo

curl -sf "$HINDSIGHT_API_URL/v1/default/banks/plain-retrieval-bank/config" \
  | jq -e '.overrides.retain_extraction_mode == "chunks" and .overrides.enable_reranking == false' > /dev/null

# [docs:mcp-tools-per-bank]
# Restrict a specific bank to read-only MCP access
curl -X PATCH http://localhost:8888/v1/default/banks/config-demo-bank/config \
  -H "Content-Type: application/json" \
  -d '{"updates": {"mcp_enabled_tools": ["recall"]}}'
# [/docs:mcp-tools-per-bank]
echo

curl -sf "$HINDSIGHT_API_URL/v1/default/banks/config-demo-bank/config" \
  | jq -e '.overrides.mcp_enabled_tools == ["recall"]' > /dev/null

# [docs:bank-config-examples]
# Update retention settings for a bank
curl -X PATCH http://localhost:8888/v1/default/banks/config-demo-bank/config \
  -H "Content-Type: application/json" \
  -d '{
    "updates": {
      "retain_chunk_size": 4000,
      "retain_extraction_mode": "custom",
      "retain_custom_instructions": "Focus on technical details and implementation specifics"
    }
  }'

# Note: retain_extraction_mode must be "custom" to use retain_custom_instructions

# View resolved config (respects permissions)
curl http://localhost:8888/v1/default/banks/config-demo-bank/config

# Reset to defaults
curl -X DELETE http://localhost:8888/v1/default/banks/config-demo-bank/config
# [/docs:bank-config-examples]
echo

curl -sf "$HINDSIGHT_API_URL/v1/default/banks/config-demo-bank/config" \
  | jq -e '.overrides == {}' > /dev/null

# =============================================================================
# Knowledge-bank metadata extraction
# =============================================================================

curl -s -X POST "$HINDSIGHT_API_URL/v1/default/knowledge-banks" \
  -H "Content-Type: application/json" -d '{"id": "metadata-demo-kb"}' > /dev/null

# [docs:kb-metadata-schema]
# What the LLM extracts from every document, and from every chunk. A property is
# {type, description?, values?, items?}; `values` is a fixed set, which is how
# classification is expressed - the model picks one of them or leaves it out.
curl -X PUT "$HINDSIGHT_API_URL/v1/default/knowledge-banks/metadata-demo-kb/metadata-schema" \
  -H "Content-Type: application/json" \
  -d '{
    "document": {
      "doc_type":  {"type": "string", "values": ["invoice", "contract", "memo"]},
      "vendor":    {"type": "string", "description": "The counterparty"},
      "total":     {"type": "number"},
      "signed_on": {"type": "date"},
      "parties":   {"type": "array", "items": "string"},
      "terms":     {"type": "object"}
    },
    "chunks": {
      "clause": {"type": "string", "values": ["payment", "termination", "liability"]}
    }
  }'
# [/docs:kb-metadata-schema]
echo

curl -sf "$HINDSIGHT_API_URL/v1/default/knowledge-banks/metadata-demo-kb/metadata-schema" \
  | jq -e '.document.doc_type.values | length == 3' > /dev/null

# [docs:kb-metadata-search]
# Filter a search on the extracted values (or on metadata written with the document)
curl -X POST "$HINDSIGHT_API_URL/v1/default/knowledge-banks/metadata-demo-kb/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "late payment penalty",
    "metadata": {"doc_type": "contract", "total": {"$gte": 10000}, "clause": "payment"}
  }'
# [/docs:kb-metadata-search]
echo

# [docs:kb-metadata-extract]
# Re-extract documents already in the bank after changing the schema
curl -X POST "$HINDSIGHT_API_URL/v1/default/knowledge-banks/metadata-demo-kb/metadata/extract" \
  -H "Content-Type: application/json" -d '{"only_missing": true}'
# [/docs:kb-metadata-extract]
echo

# [docs:kb-metadata-supplied]
# A property whose source is "request" is never sent to the LLM: the write supplies it.
# A write may also supply a value for an "extract" property, which skips the call for that
# document - so a corpus whose properties are all known costs no LLM call at all.
curl -X POST "$HINDSIGHT_API_URL/v1/default/knowledge-banks/metadata-demo-kb/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [{
      "id": "inv-1",
      "text": "Acme invoice for 1200 EUR, payable in 30 days.",
      "properties": {"doc_type": "invoice", "vendor": "Acme", "total": 1200},
      "chunk_properties": {"0": {"clause": "payment"}}
    }]
  }'
# [/docs:kb-metadata-supplied]
echo

# [docs:kb-query]
# Aggregate over the corpus: the same filter DSL in `where`, aggregates and arithmetic over
# them in `select`, and `having` over the columns select names.
curl -X POST "$HINDSIGHT_API_URL/v1/default/knowledge-banks/metadata-demo-kb/query" \
  -H "Content-Type: application/json" \
  -d '{
    "from": "chunks",
    "select": [
      {"field": "metadata.doc_type", "as": "doc_type"},
      {"count": "*", "as": "chunks"},
      {"count_distinct": "doc_id", "as": "documents"},
      {"sum": "metadata.total", "as": "total"},
      {"round": [{"divide": [{"sum": "metadata.total"}, {"count_distinct": "doc_id"}]}, 2],
       "as": "per_document"}
    ],
    "where": {"clause": "payment"},
    "group_by": ["metadata.doc_type"],
    "having": {"documents": {"$gte": 1}},
    "order_by": [{"field": "total", "direction": "desc"}],
    "limit": 50
  }'
# [/docs:kb-query]
echo

curl -s -X DELETE "$HINDSIGHT_API_URL/v1/default/knowledge-banks/metadata-demo-kb" > /dev/null

# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
for bank_id in plain-retrieval-bank config-demo-bank; do
  curl -s -X DELETE "${HINDSIGHT_API_URL}/v1/default/banks/${bank_id}" > /dev/null
done

echo "configuration.sh: All examples passed"
