# Turbovec MCP Server (Long-Term Memory RAG for AI)

Turbovec MCP Server is a **Model Context Protocol (MCP)** implementation that acts as a persistent, long-term memory (Semantic RAG) for AI coding assistants like **Zoo Code**, **Claude Desktop**, and **Cursor**. 

By running this local server, your AI assistant gains the ability to "read", "remember", and "semantically search" through vast amounts of code and documentation across different chat sessions, completely bypassing token limitations.

## The Problem it Solves
1. **Context Window Limits**: When working on large projects, pasting hundreds of files into the AI chat will exceed token limits or cause the AI to hallucinate. 
2. **AI Amnesia (Stateless Chats)**: Whenever you start a new chat tab, the AI forgets everything you discussed in the previous session (e.g., project architecture, specific coding guidelines).
3. **Literal Search vs. Semantic Search**: Standard file search (CTRL+F) requires exact keyword matches. This server allows the AI to search by *meaning* (e.g., searching for "user authentication" will find `login_handler`).

## Key Features & Advantages
- **Persistent Local Memory**: Data is safely saved to your local disk (`metadata.json` and `index.bin`). It never expires and survives across system restarts.
- **Intelligent Text Chunking**: Automatically breaks down large documents into overlapping semantic chunks (1000 chars) before embedding, ensuring context is never lost.
- **Flawless MCP Stdio Communication**: Strictly intercepts and suppresses rogue C-level progress bars (like `tqdm` from `sentence-transformers`) that normally corrupt JSON-RPC streams, ensuring a stable connection.
- **100% Local Privacy**: Runs entirely on your machine using the `all-MiniLM-L6-v2` embedding model. No data is sent to external cloud APIs for indexing.

## Architecture
- **Protocol**: FastMCP (running over `stdio`).
- **Embedding Model**: `sentence-transformers` (`all-MiniLM-L6-v2`) generating 384-dimensional vectors.
- **Vector Database**: `turbovec` (TurboQuantIndex) for ultra-fast, locally persisted similarity search.
- **Storage Layer**: Local JSON mapping for metadata, allowing automated fallback and index rebuilding if the `.bin` file is lost.

---

## Installation

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
   .\venv\Scripts\python.exe server.py
   # Mac/Linux
   ./venv/bin/python server.py
   ```
   *You should see a success message: `Turbovec MCP Server is successfully running`*

---

## Integration 

To use this server in your AI assistant, add it to your MCP configuration file (usually found in Settings > MCP Servers, or `mcp.json`).

**Example Configuration:**
```json
{
  "mcpServers": {
    "turbovec-mcp": {
      "command": "C:/path/to/your/turbovec-mcp-server/venv/Scripts/python.exe",
      "args": [
        "C:/path/to/your/turbovec-mcp-server/server.py"
      ],
      "disabled": false,
      "alwaysAllow": []
    }
  }
}
```
*(⚠️ **Important:** Make sure to use the absolute path to the Python executable inside your `venv` to ensure the server has access to the installed dependencies).*

## 🛠️ Available MCP Tools
Once connected, the AI will have access to the following tools:
- `add_knowledge(title, content)`: Embeds and saves raw text into memory.
- `add_file_knowledge(file_path)`: Reads a local file, chunks it, and saves it into memory.
- `search_knowledge(query, top_k)`: Performs a semantic search to retrieve context from the database.
- `delete_knowledge(title_or_id)`: Removes a specific piece of knowledge from the database.
- `clear_memory()`: Completely wipes the local database and vector index.

### Pro-Tip for Zoo Code Users
To make the AI use this memory automatically, add this to your **Custom Instructions** / `.cursorrules`:
> *"Whenever I ask a question about the project context, ALWAYS use the `search_knowledge` tool first before answering."*
