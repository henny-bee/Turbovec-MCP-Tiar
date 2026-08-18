# Turbovec MCP: Tools Reference Manual

This document provides a comprehensive, categorized reference for all **35 powerful MCP tools and prompts** exposed by the Turbovec MCP Server.

---

## Unstructured Knowledge Tools

These tools manage the semantic indexing of raw text documents or source code snippets using chunking and embedding operations under the vector indexing engine.

- `add_knowledge(title: str, content: str)`:
  - **Description**: Chunk, embed, and store unstructured documents or code snippet texts in the vector memory index.
  - **Usage**: Use this to digest custom standards, local setup guides, API specs, or multi-line files.
- `add_file_knowledge(file_path: str)`:
  - **Description**: Reads a local file, chunks it, and indexes it for semantic search.
  - **Usage**: Automatically reads and indexes files on disk.
- `delete_knowledge(title_or_id: str)`:
  - **Description**: Removes a document chunk from the index and database matching a specific title or ID.
- `search_knowledge(query: str, top_k: int = 3)`:
  - **Description**: Performs vector-similarity search over raw chunked text.

---

## Graph Memory Core CRUD Tools

These tools interact with the structured entity nodes and relationship edges inside SQLite. Validates all entity types and edge types against your defined rules in `ontology.json`.

- `create_entity(id: str, name: str, node_type: str, properties: dict = None, project_id: str = "default")`:
  - **Description**: Creates a structured entity node in the graph.
  - **Validation**: Enforces strict node-type checks (e.g. `person`, `project`, `technology`, `decision`, `event`, `concept`, `file`).
- `add_observation(entity_id: str, content: str, project_id: str = "default")`:
  - **Description**: Attaches a factual observation, take-away, or note to an existing entity and triggers immediate indexing.
- `create_relationship(from_node_id: str, to_node_id: str, relationship_type: str, weight: float = 1.0, properties: dict = None, project_id: str = "default")`:
  - **Description**: Creates a typed, weighted directed edge.
  - **Validation**: Enforces domain-range rules defined in `ontology.json` (e.g. mapping standard phrasing like "lives in" to `LOCATED_IN`, "owns" to `OWNS`).
- `delete_entity(id: str, project_id: str = "default")`:
  - **Description**: Deletes an entity node from the graph, cascading deletions automatically to edges and mappings.
- `delete_relationship(from_node_id: str, to_node_id: str, relationship_type: str, project_id: str = "default")`:
  - **Description**: Removes a relationship edge from the database.

---

## Graph Exploration & Context Tools

Retrieve rich sub-graph context around specific entities. Perfect for reconstructing code architectures or key system relations.

- `get_hologram(node_id: str, depth: int = 1)`:
  - **Description**: Returns a complete context block containing the node's properties, all attached observations, and adjacent neighbors (ideal context formatting for LLMs).
- `get_neighbors(node_id: str, depth: int = 1)`:
  - **Description**: Returns a simple list of connected nodes starting from an anchor node.

---

## Hybrid Search & Reranking

- `search_memory(query: str, limit: int = 10, project_id: str = "default", channel_weights: dict = None)`:
  - **Description**: Executes an enterprise hybrid (Vector + FTS5) search using **Reciprocal Rank Fusion (RRF)**, and optional CrossEncoder Reranking.
  - **Parameters**:
    - `channel_weights`: Dict defining custom weights for vector/lexical score balancing.

---

## Temporal Session & "Time-Travel"

Maintains chronological context, tracks development progress, and allows querying older graph snapshots.

- `start_session(session_id: str, name: str, properties: dict = None, project_id: str = "default")`:
  - **Description**: Starts a temporal session, creating a session node and linking it to any prior closed session.
- `end_session(session_id: str, summary: str, project_id: str = "default")`:
  - **Description**: Closes a session, saving a summary markdown block and changing its status to closed.
- `record_breakthrough(session_id: str, title: str, content: str, project_id: str = "default")`:
  - **Description**: Records a learning breakthrough, linking it directly to the active session.
- `point_in_time_query(as_of_iso_timestamp: str)`:
  - **Description**: Returns the state of the memory graph as it existed at that specific ISO-8601 UTC timestamp.
- `diff_knowledge_state(from_timestamp: str, to_timestamp: str)`:
  - **Description**: Computes a deterministic delta between two points in time (added, removed, changed nodes/edges).
- `query_timeline(entity: str = None, entity_type: str = None, start: str = None, end: str = None, order: str = "ASC", limit: int = 50, offset: int = 0)`:
  - **Description**: Queries timeline events chronologically with filtering and pagination.
- `get_temporal_neighbors(node_id: str, direction: str = "both", depth: int = 1)`:
  - **Description**: Traverses chronological neighbors relative to an anchor node.

---

## Semantic Discovery & Background Workers

Manages autonomous background analysis to find connections between disconnected entities.

- `semantic_radar(similarity_threshold: float = 0.7, project_id: str = "default", auto_create: bool = False)`:
  - **Description**: Scans the graph to detect highly similar disconnected entities, suggesting missing edges.
- `run_relationship_discovery(similarity_threshold: float = 0.7, project_id: str = "default", auto_create: bool = True)`:
  - **Description**: Runs semantic radar discovery with automatic edge creation.
- `get_background_discovery_status()`:
  - **Description**: Returns background worker running health, interval, and settings.
- `set_background_discovery(enabled: bool, interval_seconds: int = 300, similarity_threshold: float = 0.7)`:
  - **Description**: Configures and manages background discovery.

---

## Advanced Parity & Lifecycle Tools

Provides maintenance commands to prevent performance degradation from clutter or stale historical knowledge.

- `archive_entity(id: str, reason: str = "Unused/superseded")`:
  - **Description**: Archives an entity (status = `ARCHIVED`), excluding it from active searches.
- `restore_entity(id: str)`:
  - **Description**: Restores an archived entity node back to active retrieval status.
- `list_orphans()`:
  - **Description**: Identifies and lists orphaned entity nodes (nodes with 0 connected edges).
- `prune_stale(max_age_days: int, dry_run: bool = False)`:
  - **Description**: Identifies stale or archived memories older than `max_age_days` and prunes them.

---

## Inter-Session Provenance & Bottles

Inter-session coordination tools to preserve developer notes or instructions across completely independent chat windows.

- `create_bottle(message: str, priority: str = "medium", expires_at: str = None, author_session_id: str = None)`:
  - **Description**: Leaves a persistent note/bottle for subsequent sessions to discover.
- `get_bottles(include_acknowledged: bool = False)`:
  - **Description**: Lists active and optionally acknowledged bottle notes.
- `acknowledge_bottle(id: str)`:
  - **Description**: Marks a bottle note as acknowledged/read.

---

## Telemetry, Analytics & Autonomous Librarian

Deep system metrics and automated housekeeping.

- `search_stats()`:
  - **Description**: Calculates rolling search statistics, latencies per subsystem, percentiles, error rates, and cache hits.
- `analyze_graph()`:
  - **Description**: Evaluates graph topologies, PageRank, degree centrality, connected components, and LPA communities.
- `run_librarian_cycle()`:
  - **Description**: Triggers one autonomous librarian cycle immediately (clustering, duplicate detection, and concept synthesis).
- `clear_memory()`:
  - **Description**: Wipes the index, metadata, and database files on disk, starting fresh.
- `optimize_memory()`:
  - **Description**: Compacts database storage, removes soft-deleted chunks, and optimizes the vector index.

---

## Registered MCP Prompts

Convenient, ready-to-use LLM system prompts pre-configured for specialized developer operations.

- `qna_with_context(query: str)`:
  - **Description**: Formulates an optimized Q&A prompt pre-injecting the top 5 relevant memory chunks from the database.
- `review_codebase()`:
  - **Description**: Invokes a system context role as a Senior Software Engineer to review codebase snippets.
