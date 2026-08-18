# Turbovec MCP: Configuration & Setup Guide

This guide provides deep-dive instructions for configuring, deploying, and integrating **Turbovec MCP** with various AI development environments and clients.

---

## Storage Layer Schema

Turbovec operates fully in-process inside your local workspace. All operations are atomic, local, and private. Below is the internal schema details of the persistent layer:

1. **`nodes` Table**: Stores entities, sessions, breakthroughs, and files with metadata, certainty ratings (`speculative`, `confirmed`, etc.), salience scores, and lifecycle statuses (`ACTIVE`, `ARCHIVED`).
2. **`edges` Table**: Represents directed, typed, and weighted relationships (e.g., `DEPENDS_ON`, `PRECEDED_BY`, `IMPLEMENTS`) connecting nodes.
3. **`observations` Table**: Holds localized facts, takeaways, or content blocks associated with specific entity nodes, keeping search payloads clean and highly searchable.
4. **`entities_fts` Table**: A Porter Stemmer-powered Unicode virtual table inside SQLite for highly efficient lexical keyword matches.
5. **`vector_id_mapping` Table**: Bridges UUID string graph identifiers with deterministic, collision-resistant signed 64-bit integer index keys required by `turbovec`.
6. **`bottles` Table**: Persists inter-session messaging notes left by agents to coordinate instructions across independent sessions.
7. **`search_metrics` Table**: Collects latency metrics per subsystem (FTS, vector search, graph hydration, embedding generation, reranker) and error flags for full search telemetry.

---

## Configuration & Environment Variables

The server and background workers can be configured via environment variables on startup. These can be defined in a `.env` file in the root folder or set directly in your shell or MCP client configuration file:

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `BACKGROUND_DISCOVERY` | Boolean | `false` | Enables/disables the background semantic relationship discovery worker. |
| `BG_DISCOVERY_INTERVAL`| Integer | `300` | Loop interval in seconds for the background relationship worker. |
| `BG_DISCOVERY_THRESHOLD`| Float | `0.7` | Cosine similarity threshold for automatic background edge inferences. |
| `SQLITE_DB_FILE` | String | `memory.db` | Path to the SQLite metadata and graph database. |
| `PYTHONUNBUFFERED` | String | `1` | Ensures python outputs are flushed immediately, avoiding JSON-RPC deadlocks. |
| `run_mode` | String | `None` | Set to `local` (e.g., via `.env`) to run the MCP server over SSE on `localhost:4392` instead of `stdio`. |

---

## Advanced Docker Setup

If you prefer containing the server environment:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/henny-bee/Turbovec-MCP-Server.git
   cd turbovec-mcp-server
   ```

2. **Run with Docker Compose**:
   ```bash
   docker-compose up -d
   ```
   *This starts the server inside a container and mounts the persistent volume `turbovec_data` to protect the database files (`memory.db` and `index.tvim`).*

---

## Developer Dashboard Setup

To run the interactive developer dashboard for visualizing graph schemas, latencies, and telemetry logs:

1. **Install Streamlit & Pandas**:
   ```bash
   pip install streamlit pandas
   ```

2. **Launch the Dashboard**:
   ```bash
   streamlit run dashboard/app.py
   ```
   *This spins up a local web server (usually at `http://localhost:8501`) connected directly to your local `memory.db` file.*

---

## Editor & AI Client Configuration

To integrate Turbovec with your preferred AI environment, add the following to your local MCP server configuration settings (typically located in your global `mcp_settings.json` file or editor preferences).

### 1. Claude Desktop Setup
Modify your `%APPDATA%/Claude/claude_desktop_config.json` (Windows) or `~/Library/Application Support/Claude/claude_desktop_config.json` (Mac):

```json
{
  "mcpServers": {
    "turbovec-mcp": {
      "command": "C:/path/to/turbovec-mcp-server/venv/Scripts/python.exe",
      "args": [
        "C:/path/to/turbovec-mcp-server/main.py"
      ],
      "env": {
        "PYTHONUNBUFFERED": "1",
        "BACKGROUND_DISCOVERY": "true",
        "BG_DISCOVERY_INTERVAL": "300"
      }
    }
  }
}
```

### 2. Cursor Setup
Go to **Settings > Features > MCP**, click **Add New MCP Server**:
- **Name**: `turbovec-mcp`
- **Type**: `command`
- **Command**: `C:/path/to/turbovec-mcp-server/venv/Scripts/python.exe` (on Windows) or `/path/to/venv/bin/python` (on Mac/Linux)
- **Args**: `["C:/path/to/turbovec-mcp-server/main.py"]`

### 3. Cline / Roo Code Setup (Standard stdio)
Modify your system's Cline MCP settings file (`%APPDATA%/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json` on Windows):

```json
{
  "mcpServers": {
    "turbovec-mcp": {
      "command": "python",
      "args": ["C:/absolute/path/to/turbovec-mcp-server/main.py"],
      "env": {
        "PYTHONUNBUFFERED": "1"
      },
      "disabled": false
    }
  }
}
```
> **Troubleshooting Tip**: If you get `ModuleNotFoundError` during setup, your editor is calling the global system Python instead of your virtual environment. Resolve this by specifying the absolute path of your venv's Python executable as the `"command"` property (e.g. `C:/turbovec-mcp-server/venv/Scripts/python.exe`).

### 4. Low Memory / No Docker Setup (Local SSE Mode)
If you cannot run Docker due to low memory, or want to keep the MCP server running independently of the AI editor session, you can run the server in `local` mode via Server-Sent Events (SSE).

1. First, create a `.env` file in the `turbovec-mcp-server` directory and add:
   ```env
   run_mode=local
   ```
2. Manually run the server in your terminal:
   ```bash
   python main.py
   ```
   *(It will start the server on `http://localhost:4392/sse`)*
3. Then, modify your system's Cline MCP settings file to connect via SSE:
   ```json
   {
     "mcpServers": {
       "turbovec-local": {
         "url": "http://localhost:4392/sse"
       }
     }
   }
   ```
