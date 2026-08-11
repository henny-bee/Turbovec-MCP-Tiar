<img width="1984" height="576" alt="vnuw9pe8htgfwu87qgfbw" src="https://github.com/user-attachments/assets/f781a60d-11b9-4198-b80b-fca16a927184" />

# Turbovec MCP Server (Long-Term Memory RAG for AI)

Turbovec MCP Server is a **Model Context Protocol (MCP)** implementation that acts as a persistent, long-term memory (Semantic RAG) for AI coding assistants like **Zoo Code**, **Claude Desktop**, and **Cursor**.

By running this local server, your AI assistant gains the ability to "read", "remember", and "semantically search" through vast amounts of code and documentation across different chat sessions, completely bypassing token limitations.

## The Problem it Solves

1. **Context Window Limits**: When working on large projects, pasting hundreds of files into the AI chat will exceed token limits or cause the AI to hallucinate.
2. **AI Amnesia (Stateless Chats)**: Whenever you start a new chat tab, the AI forgets everything you discussed in the previous session (e.g., project architecture, specific coding guidelines).
3. **Literal Search vs. Semantic Search**: Standard file search (CTRL+F) requires exact keyword matches. This server allows the AI to search by _meaning_ (e.g., searching for "user authentication" will find `login_handler`).

## Key Features & Advantages

- **Persistent Local Memory**: Data is safely saved to your local disk (`metadata.json` and `index.bin`). It never expires and survives across system restarts.
- **Intelligent Text Chunking**: Automatically breaks down large documents into overlapping semantic chunks (1000 chars) before embedding, ensuring context is never lost.
- **Flawless MCP Stdio Communication**: Strictly intercepts and suppresses rogue C-level progress bars (like `tqdm` from `sentence-transformers`) that normally corrupt JSON-RPC streams, ensuring a stable connection.
- **100% Local Privacy**: Runs entirely on your machine using the `all-MiniLM-L6-v2` embedding model. No data is sent to external cloud APIs for indexing.

## Architecture

- **Protocol**: FastMCP (running over `stdio`).
- **Embedding Model**: `sentence-transformers` (`all-MiniLM-L6-v2`) generating 384-dimensional vectors.
- **Vector Database**: `turbovec` (`IdMapIndex`) for ultra-fast, locally persisted similarity search with stable external IDs and O(1) deletions.
- **Storage Layer**: Local JSON mapping for metadata (storing documents in a dictionary alongside a `next_id` counter for safe ID management), allowing automated fallback and index rebuilding if the vector index file is lost.
- **Modular Codebase**: The project is cleanly separated into `main.py` (entry point), `vector_db.py` (database logic), and `tools.py` (MCP tool definitions).

---

## How It Works: AI & MCP Interaction Flow

The Turbovec MCP Server acts as an invisible bridge between your AI client and a persistent local memory database. Here is the step-by-step logic of how they interact:

1. **User Prompt**: The user asks a question or assigns a task in their AI Client (e.g., Zoo Code, Claude Desktop, Cursor).
2. **LLM Tool Call**: The LLM evaluates the prompt and determines it needs past context or codebase knowledge, triggering an MCP tool (like `search_knowledge` or `add_knowledge`).
3. **MCP Execution**: The AI Client forwards this tool request to the Turbovec MCP Server running locally via the standardized JSON-RPC protocol over `stdio`.
4. **Vector Database**: The MCP server interacts with the `turbovec` local vector database to embed the query, search for semantic matches, or store new text chunks.
5. **Context Return & Generation**: The retrieved data is returned to the AI Client and passed back to the LLM. The LLM seamlessly incorporates this retrieved memory into its final context-aware response to the user.

<img width="3600" height="1771" alt="627eriqdfvafasf" src="https://github.com/user-attachments/assets/dd4e4422-7d2f-4f5a-bb6b-a79a00fb9020" />

---

## Installation

You can run Turbovec MCP Server locally via Python or using Docker.

### Option A: Local Python Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/henny-bee/Turbovec-MCP-Server.git
   cd turbovec-mcp-server
   ```

2. **Create a Virtual Environment** (Recommended):

   ```bash
   python -m venv venv

   # Windows
   .\venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify it Runs**:
   ```bash
   # Windows
   .\venv\Scripts\python.exe main.py
   # Mac/Linux
   ./venv/bin/python main.py
   ```
   _You should see a success message: `Turbovec MCP Server is successfully running`_

### Option B: Docker Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/henny-bee/Turbovec-MCP-Server.git
   cd turbovec-mcp-server
   ```

2. **Run with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

Alternatively, you can build and run it directly using the provided `Dockerfile`.

---

## Testing

The project uses `pytest` for testing to ensure the database and tools work correctly.

1. **Install development dependencies**:

   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Run the tests**:
   ```bash
   pytest tests/
   ```

---

## Zoo Code / AI Editor Integration

To use this server in your AI coding assistant (like **Zoo Code**, **Cursor**, or **Claude Desktop**), add it to your MCP configuration settings (usually found in Settings > MCP Servers, or `mcp_settings.json`).

### Configuration

Add the following block to your `mcpServers` configuration:

```json
{
  "mcpServers": {
    "turbovec-mcp": {
      "command": "python",
      "args": ["C:/absolute/path/to/turbovec-mcp-server/main.py"],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

> **Troubleshooting Tip:** If you encounter a `ModuleNotFoundError` (e.g., missing `numpy` or `turbovec`), it means the editor is using the system Python instead of the virtual environment. To fix this, change `"command": "python"` to the absolute path of your virtual environment's Python executable (e.g., `"C:/path/to/turbovec-mcp-server/venv/Scripts/python.exe"` on Windows, or `"/path/to/venv/bin/python"` on Mac/Linux).

### Custom Instructions (Recommended)

To ensure your AI assistant seamlessly and proactively uses the memory server without asking for permission, we highly recommend adding the following to your AI's **Custom Instructions** or **System Prompt**:

/Rules.md

## Available MCP Tools

Once connected, the AI will have access to the following tools:

- `add_knowledge(title, content)`: Embeds and saves raw text into memory.
- `add_file_knowledge(file_path)`: Reads a local file, chunks it, and saves it into memory.
- `search_knowledge(query, top_k)`: Performs a semantic search to retrieve context from the database.
- `delete_knowledge(title_or_id)`: Removes a specific piece of knowledge from the database.
- `optimize_memory()`: Performs hard-deletion and garbage collection of the vector database to optimize memory usage.
- `clear_memory()`: Completely wipes the local database and vector index.

---

**Sponsored by**

<a href="https://www.iseekaigo.com/">
  <img src="logo-iseekaigo-line.png" alt="ISEEKAIGO" height="50">
</a>
