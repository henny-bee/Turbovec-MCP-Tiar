# Configuration & Setup

- [Editor integration](#editor-integration)
- [Environment variables](#environment-variables)
- [Docker](#docker)
- [Developer dashboard](#developer-dashboard)
- [Storage schema](#storage-schema)

---

## Editor integration

All clients start the same process; only the config file location differs.

> **Always use the absolute path to your venv's Python.** A bare `python` makes
> the editor use the system interpreter, which produces `ModuleNotFoundError`.

### Claude Desktop

`%APPDATA%/Claude/claude_desktop_config.json` (Windows) or
`~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "turbovec": {
      "command": "C:/path/to/venv/Scripts/python.exe",
      "args": ["C:/path/to/main.py"],
      "env": {
        "PYTHONUNBUFFERED": "1",
        "BACKGROUND_DISCOVERY": "true"
      }
    }
  }
}
```

### Cursor

**Settings → Features → MCP → Add New MCP Server**

- **Name**: `turbovec`
- **Type**: `command`
- **Command**: `C:/path/to/venv/Scripts/python.exe` (Windows) or `/path/to/venv/bin/python`
- **Args**: `["/absolute/path/to/main.py"]`

### Cline / Roo Code

`%APPDATA%/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`:

```json
{
  "mcpServers": {
    "turbovec": {
      "command": "C:/path/to/venv/Scripts/python.exe",
      "args": ["C:/absolute/path/to/main.py"],
      "env": { "PYTHONUNBUFFERED": "1" },
      "disabled": false
    }
  }
}
```

### SSE mode (shared server)

Runs the server as a standalone process instead of one child per editor. Useful
when you want a single memory shared by several clients, or your machine is too
constrained to load a model per session.

1. Create a `.env` file next to `main.py`:
   ```env
   run_mode=local
   ```
2. Start it: `python main.py` → serves `http://localhost:4392/sse`
3. Point the client at the URL:
   ```json
   { "mcpServers": { "turbovec": { "url": "http://localhost:4392/sse" } } }
   ```

---

## Environment variables

Read from the shell, your MCP client's `env` block, or a `.env` file. Every
value is defined once in `core/config.py` (`Settings`) — nothing else reads the
environment directly.

### Runtime

| Variable | Default | Description |
| :--- | :--- | :--- |
| `run_mode` | *(unset)* | Set to `local` to serve SSE on `localhost:4392` instead of stdio. |
| `PYTHONUNBUFFERED` | `1` | Flush output immediately; prevents JSON-RPC deadlocks. |
| `DEFAULT_PROJECT_ID` | `default` | Workspace used when a tool omits `project_id`. |

### Storage

| Variable | Default | Description |
| :--- | :--- | :--- |
| `SQLITE_DB_FILE` | `memory.db` | Graph, FTS and telemetry database. |
| `METADATA_FILE` | `metadata.json` | Document chunk metadata. |
| `INDEX_FILE` | `index.tvim` | Vector index. |
| `ONTOLOGY_FILE` | `ontology.json` | Allowed entity and relationship types. |
| `INDEX_FLUSH_OPS` | `64` | Write the index to disk after N mutations… |
| `INDEX_FLUSH_INTERVAL` | `2.0` | …or after N seconds, whichever comes first. Set `INDEX_FLUSH_OPS=1` to write on every change. |
| `SQLITE_CACHE_KB` | `64000` | SQLite page cache. |
| `SQLITE_MMAP_BYTES` | `268435456` | Memory-mapped I/O budget (256 MiB). |
| `SQLITE_BUSY_TIMEOUT_MS` | `10000` | How long a write waits on a locked database. |
| `SQLITE_TIMEOUT` | `30.0` | Seconds the Python driver waits to acquire a connection. |

### Embeddings

| Variable | Default | Description |
| :--- | :--- | :--- |
| `EMBEDDING_PROVIDER` | `minilm` | `minilm` (384d) or `bge` (1024d). |
| `EMBEDDING_DIMENSION` | `384` | Must match the provider. |
| `EMBEDDING_MODEL` | provider default | Override the sentence-transformers model name. |
| `EMBEDDING_CACHE_SIZE` | `512` | LRU entries for repeated query text. `0` disables. |
| `EMBEDDING_BATCH_SIZE` | `64` | Texts per model call. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Document chunking window, in characters. |

### Retrieval

| Variable | Default | Description |
| :--- | :--- | :--- |
| `RRF_K` | `60.0` | Rank fusion constant. Lower favours top-ranked hits more sharply. |
| `VECTOR_OVERSAMPLE` | `5` | ANN candidates fetched per requested result. |
| `RERANKER_ENABLED` | `false` | Enable cross-encoder reranking. Slower, more accurate. |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranking model. |
| `TELEMETRY_ENABLED` | `true` | Record per-search latency. |
| `TELEMETRY_WINDOW` | `10000` | Rows `search_stats` aggregates over. |

### Background discovery

| Variable | Default | Description |
| :--- | :--- | :--- |
| `BACKGROUND_DISCOVERY` | `false` | Run the relationship-discovery daemon. |
| `BG_DISCOVERY_INTERVAL` | `300` | Seconds between scans. |
| `BG_DISCOVERY_THRESHOLD` | `0.7` | Cosine similarity required to infer an edge. |
| `RADAR_MAX_NODES` | `5000` | Cap on nodes per scan — the comparison is quadratic. |

---

## Docker

```bash
docker-compose up -d
```

Data lives in the `turbovec_data` volume, so `memory.db` and `index.tvim`
survive rebuilds.

---

## Developer dashboard

A Streamlit app for browsing the graph, search latencies and librarian history.

```bash
pip install -r requirements-dashboard.txt
streamlit run dashboard/app.py
```

Opens on `http://localhost:8501`, reading your local `memory.db` directly. Set
`SQLITE_DB_FILE` first if the database is elsewhere.

---

## Storage schema

One SQLite file holds everything.

| Table | Purpose |
| :--- | :--- |
| `nodes` | Entities, sessions and documents, with certainty, salience score and lifecycle status (`ACTIVE` / `ARCHIVED`). |
| `edges` | Directed, typed, weighted relationships between nodes. |
| `observations` | Timestamped facts attached to an entity. Kept separate so a node's text can grow without rewriting the node. |
| `entities_fts` | FTS5 virtual table (Porter stemmer, Unicode) backing the lexical channel. |
| `vector_id_mapping` | Maps string node ids to the 64-bit integer keys `turbovec` requires, resolving hash collisions. |
| `bottles` | Notes left by one session for the next. |
| `search_metrics` | Per-search latency for each subsystem, plus error codes. |
| `librarian_runs` | History of autonomous reorganisation cycles. |

Writes to `nodes`, `entities_fts` and the vector index happen in one
transaction, so a failure cannot leave a node indexed but unsearchable.

### Ontology

`ontology.json` whitelists relationship types; `create_relationship` rejects
anything outside it (case-insensitively). Entity types are **not** validated —
any `node_type` string is accepted.

Add a type by editing the file:

```json
{ "version": 1, "entities": ["person", "..."], "relations": ["depends_on", "..."] }
```

If you extend the built-in text extractor with new relationship verbs, register
the matching type here too — an unregistered type raises and aborts extraction
for that document.
