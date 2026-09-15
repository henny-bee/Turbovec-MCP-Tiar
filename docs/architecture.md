# Architecture

## Module map

| Package | Responsibility |
| :--- | :--- |
| `core/` | `Settings` (all tunables), SQLite connection manager and pragmas, schema and migrations, id/text helpers, process bootstrap |
| `storage/` | `VectorIndex` (turbovec wrapper, debounced writes), document metadata file, startup load/rebuild/migrate |
| `graph/` | Row mappers and batched hydration, CRUD repository, traversal and union-find, topology analytics |
| `search/` | Vector and lexical channels, RRF fusion, semantic radar, background discovery daemon |
| `extraction/` | Pre-compiled rule patterns, spaCy-or-regex extractor |
| `memory/` | Ontology, document ingestion, sessions, snapshots, temporal queries, lifecycle, bottles, errors |
| `embeddings/` | Provider protocol, sentence-transformers providers, caching/batching service, factory |
| `librarian/` | DBSCAN clustering and concept synthesis |
| `tools/` | One MCP tool module per domain; `register_tools_and_prompts` wires them |

`VectorDB` (`core/database.py`) is a façade composing those mixins.
`vector_db.py` re-exports it so existing imports keep working.

## Design rules

- **One tunable, one place.** Nothing reads `os.getenv` outside `core/config.py`.
- **Adding a tool touches one file.** Every tool module exposes `register(mcp, db)`.
- **Writes stay consistent.** A node's row, FTS document and vector are written
  in one transaction; the vector index is compensated explicitly because it
  lives outside SQLite.
- **Retrieval fails loudly.** A broken subsystem raises a typed `SearchError`
  rather than returning zero results, so "nothing matched" and "search is
  broken" stay distinguishable.

## Performance decisions

| Change | Why |
| :--- | :--- |
| Batched node and observation hydration | Removed the N+1 query in hybrid search, holograms and snapshots |
| Debounced vector index writes | The index was rewritten to disk on *every* node and observation mutation |
| Batched embeddings + LRU cache | Documents encode in one model call; repeated queries hit the cache |
| Blocked cosine matrix + union-find in the radar | Replaced per-pair encoding and per-pair BFS (`O(n²·E)`) |
| `\|a-b\|² = \|a\|² + \|b\|² - 2ab` in DBSCAN | One BLAS product instead of an `O(n²)` Python loop |
| Indexes on `nodes(name)`, `nodes(status)`, `observations(entity_id, created_at)` | Ingestion and archive filtering were full scans |
| SQLite `cache_size`, `mmap_size`, `busy_timeout` | Read throughput and write contention under concurrent tools |
| `IN (...)` batching via `core.sqlite.chunked` | Queries broke past SQLite's 999-variable limit |
| Per-thread telemetry connection, bounded stats window | Recording sits on the hot path of every search |

## Measured effect

Same workload on both revisions, stubbed embedding model, 1000 entities and
1000 observations, 50 hybrid searches. Seconds, lower is better.

| Operation | Before | After |
| :--- | ---: | ---: |
| Librarian cycle | 22.79 | 0.26 |
| Semantic radar | 2.58 | 0.23 |
| 50 hybrid searches | 1.44 | 0.65 |
| 1000 observation writes | 0.93 | 0.67 |
| 1000 node writes | 0.73 | 0.49 |

The gap widens with graph size: the hot spots removed were quadratic, not constant.

## Testing

`pytest` runs 78 tests. Alongside the end-to-end suites, the units with real
algorithmic content are covered directly — rank fusion, the embedding cache,
index flush thresholds, union-find, DBSCAN, and the radar's similarity and
reachability gating.
