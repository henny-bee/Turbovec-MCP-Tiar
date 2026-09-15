# Tools Reference

35 tools and 2 prompts. Arguments are shown with their defaults; `project_id`
defaults to `"default"` everywhere and scopes results to one workspace.

- [Documents](#documents) · [Graph](#graph) · [Search](#search) · [Sessions](#sessions)
- [Time travel](#time-travel) · [Discovery](#discovery) · [Lifecycle](#lifecycle)
- [Bottles](#bottles) · [Maintenance](#maintenance) · [Prompts](#prompts)

---

## Documents

Unstructured text: chunked, embedded, and searched by meaning. Use these for
files, specs and long notes; use [Graph](#graph) for facts you want to link.

| Tool | Description |
| :--- | :--- |
| `add_knowledge(title, content)` | Chunk, embed and store text. Also extracts entities and relationships into the graph. |
| `add_file_knowledge(file_path)` | Same, reading `content` from a local file. Title is the filename. |
| `search_knowledge(query, top_k=3)` | Vector-only search over chunks. Returns formatted text. |
| `delete_knowledge(title_or_id)` | Delete every chunk matching a title or chunk id. |

---

## Graph

Structured memory. Relationship types are validated against `ontology.json`.

| Tool | Description |
| :--- | :--- |
| `create_entity(id, name, node_type, properties=None, project_id)` | Create an entity node. `node_type` is free-form (`person`, `technology`, `decision`, …). |
| `add_observation(entity_id, content, project_id)` | Attach a timestamped fact to an entity and reindex it. |
| `create_relationship(from_node_id, to_node_id, relationship_type, weight=1.0, properties=None, project_id)` | Create a directed, weighted edge. Rejects unregistered `relationship_type`. |
| `delete_entity(id, project_id)` | Delete a node; its edges, observations and vector cascade. |
| `delete_relationship(from_node_id, to_node_id, relationship_type, project_id)` | Delete one edge. |
| `get_hologram(node_id, depth=1)` | The node, its observations, and its `depth`-hop neighbourhood as nodes + edges. The richest context block for an LLM. |
| `get_neighbors(node_id, depth=1)` | Same traversal, nodes only. |

---

## Search

| Tool | Description |
| :--- | :--- |
| `search_memory(query, limit=10, project_id, channel_weights=None)` | Hybrid vector + FTS5 search fused with RRF, optionally reranked. The main retrieval tool. |

`channel_weights` biases the two channels, e.g. `{"vector": 1.0, "lexical": 0.5}`.
Set one to `0.0` to disable it: `{"vector": 0.0, "lexical": 1.0}` is exact
keyword search. Archived entities are excluded. Each result carries `rrf_score`,
its observations, and `rerank_score` when reranking is on.

---

## Sessions

Groups work into chronological units. Each new session links back to the
previous one with `PRECEDED_BY`, forming a timeline of the project.

| Tool | Description |
| :--- | :--- |
| `start_session(session_id, name, properties=None, project_id)` | Open a session node and chain it to the previous one. |
| `record_breakthrough(session_id, title, content, project_id)` | Record an insight, linked to the session with `BREAKTHROUGH_IN`. |
| `end_session(session_id, summary, project_id)` | Close a session and store its summary. |

---

## Time travel

Every node, edge and observation is timestamped, so past states are
reconstructable — nothing is overwritten in place.

| Tool | Description |
| :--- | :--- |
| `point_in_time_query(as_of_iso_timestamp)` | The whole graph as it existed at an ISO-8601 UTC instant. |
| `diff_knowledge_state(from_timestamp, to_timestamp)` | Deterministic delta between two instants: added, removed and changed nodes and edges. |
| `query_timeline(entity=None, entity_type=None, start=None, end=None, order="ASC", limit=50, offset=0)` | Chronological event feed with filters and pagination. |
| `get_temporal_neighbors(node_id, direction="both", depth=1)` | Neighbours that happened `before`, `after`, or either side of an anchor node. |

---

## Discovery

Finds entities that are semantically close but unconnected in the graph — the
links you would have drawn yourself if you had remembered both notes.

| Tool | Description |
| :--- | :--- |
| `semantic_radar(similarity_threshold=0.7, project_id, auto_create=False)` | Report similar-but-unlinked pairs with a suggested relationship type. |
| `run_relationship_discovery(similarity_threshold=0.7, project_id, auto_create=True)` | Same scan, creating the edges by default. |
| `get_background_discovery_status()` | Whether the daemon is running, plus its interval and threshold. |
| `set_background_discovery(enabled, interval_seconds=300, similarity_threshold=0.7)` | Start, stop or reconfigure the daemon at runtime. |

A lower `similarity_threshold` finds more links and more false positives. Pairs
already connected by any path are skipped.

---

## Lifecycle

Keeps retrieval quality from decaying as the graph grows.

| Tool | Description |
| :--- | :--- |
| `archive_entity(id, reason="Unused/superseded")` | Mark `ARCHIVED` — kept and restorable, but excluded from search. |
| `restore_entity(id)` | Return an archived entity to `ACTIVE`. |
| `list_orphans()` | Entities with no edges at all — usually notes that were never linked. |
| `prune_stale(max_age_days, dry_run=False)` | Permanently delete stale/archived nodes older than N days. **Run with `dry_run=True` first** — deletion cascades to edges and observations. |

---

## Bottles

Messages from one session to the next, independent of the graph. Use them for
hand-offs ("migration half-done, see X") rather than durable facts.

| Tool | Description |
| :--- | :--- |
| `create_bottle(message, priority="medium", expires_at=None, author_session_id=None)` | Leave a note. `expires_at` is an ISO-8601 timestamp. |
| `get_bottles(include_acknowledged=False)` | List unexpired notes. |
| `acknowledge_bottle(id)` | Mark one as read so it stops surfacing. |

---

## Maintenance

| Tool | Description |
| :--- | :--- |
| `search_stats()` | Latency percentiles (p50/p95/p99) per subsystem, error rate, embedding cache hit rate. |
| `analyze_graph()` | Connected components, LPA communities, degree centrality and PageRank; the 20 most central nodes. |
| `run_librarian_cycle()` | One reorganisation pass now: cluster, flag duplicates, discover links, synthesise concepts. |
| `optimize_memory()` | Flush and compact metadata and the vector index. |
| `clear_memory()` | **Deletes everything** — database, index and metadata files. Not reversible. |

---

## Prompts

| Prompt | Description |
| :--- | :--- |
| `qna_with_context(query)` | A Q&A system prompt with the top 5 matching memory chunks pre-injected. |
| `review_codebase()` | A senior-engineer code review system prompt. |
