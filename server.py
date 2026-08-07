import os
import sys
import json
import warnings
import logging

# 1. COMPLETELY DISABLE ALL LOGGING, WARNINGS AND PROGRESS BARS
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["DISABLE_TQDM"] = "1"

# 2. TEMPORARILY REDIRECT STDOUT AND STDERR TO DEVNULL TO PREVENT ANY C-LEVEL OR LIBRARY LOGS
_original_stdout = sys.stdout
_original_stderr = sys.stderr
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

try:
    import numpy as np
    from mcp.server.fastmcp import FastMCP
    from sentence_transformers import SentenceTransformer
    import turbovec
    import huggingface_hub.utils as hf_utils
    hf_utils.disable_progress_bars()
except Exception:
    pass

# 3. RESTORE STDERR AFTER IMPORTS
sys.stderr = _original_stderr

# Initialize MCP Server
mcp = FastMCP("TurbovecSemanticSearch")

# Load Local Embedding Model
model = SentenceTransformer('all-MiniLM-L6-v2')
DIMENSION = 384 # Vector dimension for MiniLM model

# Initialize Storage & Turbovec Index
index = turbovec.TurboQuantIndex(DIMENSION) 
document_store = []
METADATA_FILE = "metadata.json"
INDEX_FILE = "index.bin"

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Intelligent text chunking with overlap."""
    chunks = []
    start = 0
    text_len = len(text)
    
    if text_len == 0:
        return []
        
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size
        
    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= text_len:
            break
        start += step
        
    return chunks

def save_storage():
    """Saves metadata to JSON and attempts to save turbovec index."""
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(document_store, f, indent=2)
    try:
        index.write(INDEX_FILE)
    except Exception:
        pass # turbovec library might not support save or threw an error

def load_storage():
    """Loads metadata and index from disk at startup."""
    global document_store, index
    
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                document_store = json.load(f)
        except Exception as e:
            print(f"Failed to load metadata: {e}", file=sys.stderr)
            document_store = []
            
    loaded_index = False
    if os.path.exists(INDEX_FILE):
        try:
            index.load(INDEX_FILE)
            loaded_index = True
        except AttributeError:
            pass
        except Exception as e:
            print(f"Failed to load index: {e}", file=sys.stderr)
            
    # Fallback: rebuild index if we have metadata but failed to load index file
    if not loaded_index and document_store:
        for doc in document_store:
            if doc is None or doc.get("deleted"):
                index.add(np.zeros((1, DIMENSION), dtype=np.float32))
            else:
                vector = model.encode(doc["content"])
                index.add(np.array([vector]))

# Load storage at startup
load_storage()

@mcp.tool()
def add_knowledge(title: str, content: str) -> str:
    """
    Use this tool to insert documentation, code, or information 
    into the Turbovec vector memory using chunking.
    """
    chunks = chunk_text(content, chunk_size=1000, overlap=200)
    
    for i, chunk in enumerate(chunks):
        vector = model.encode(chunk)
        index.add(np.array([vector]))
        
        doc_id = len(document_store)
        document_store.append({
            "id": doc_id,
            "title": title,
            "chunk_index": i,
            "content": chunk,
            "deleted": False
        })
        
    save_storage()
    return f"Successfully added '{title}' ({len(chunks)} chunks) into Turbovec memory."

@mcp.tool()
def add_file_knowledge(file_path: str) -> str:
    """Reads a local file, chunks it, and indexes it for semantic search."""
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' does not exist."
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return f"Error reading file '{file_path}': {e}"
        
    title = os.path.basename(file_path)
    return add_knowledge(title, content)

@mcp.tool()
def delete_knowledge(title_or_id: str) -> str:
    """Removes a document from the index and metadata database."""
    global document_store
    deleted_count = 0
    
    for doc in document_store:
        if doc is None or doc.get("deleted"):
            continue
        if str(doc["id"]) == title_or_id or doc["title"] == title_or_id:
            doc["deleted"] = True
            deleted_count += 1
            
    if deleted_count > 0:
        save_storage()
        return f"Successfully deleted {deleted_count} chunks matching '{title_or_id}'."
    return f"No document found matching '{title_or_id}'."

@mcp.tool()
def clear_memory() -> str:
    """Wipes the index and metadata database completely."""
    global document_store, index
    document_store = []
    # Reinitialize index to clear it
    index = turbovec.TurboQuantIndex(DIMENSION)
    
    # Remove files if they exist
    for f in [METADATA_FILE, INDEX_FILE]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass
                
    return "Memory cleared successfully."

@mcp.tool()
def search_knowledge(query: str, top_k: int = 3) -> str:
    """
    Use this tool to search for specific information from memory based on semantic meaning (Semantic Search).
    """
    valid_docs = [d for d in document_store if d and not d.get("deleted")]
    if not valid_docs:
        return "Database is still empty. Please add knowledge first using the add_knowledge tool."

    query_vector = model.encode(query)
    
    # Retrieve more results just in case some top matches were deleted
    search_k = min(top_k * 3, len(document_store))
    if search_k == 0: 
        search_k = 1
        
    distances, indices = index.search(np.array([query_vector]), k=search_k)
    
    results = []
    for i, doc_idx in enumerate(indices[0]):
        if 0 <= doc_idx < len(document_store):
            doc = document_store[doc_idx]
            if doc and not doc.get("deleted"):
                score = distances[0][i]
                results.append(
                    f"📄 **Source**: {doc['title']} (ID: {doc['id']})\n"
                    f"🎯 **Score**: {score:.4f}\n"
                    f"📝 **Snippet**:\n{doc['content']}"
                )
                if len(results) >= top_k:
                    break
                    
    if not results:
        return "No relevant information found."
        
    return "Semantic Search Results:\n\n" + "\n\n---\n\n".join(results)

# MCP Prompts
@mcp.prompt()
def qna_with_context(query: str) -> str:
    """Creates a system prompt for Q&A using search context."""
    context = search_knowledge(query, top_k=5)
    return f"""You are a helpful assistant. Use the following context retrieved from our knowledge base to answer the user's query.

Context:
{context}

User Query: {query}
Answer:"""

@mcp.prompt()
def review_codebase() -> str:
    """Creates a system prompt for reviewing the codebase."""
    return """You are a senior software engineer. Please review the provided codebase snippets for potential bugs, performance issues, and best practices. Suggest improvements where applicable."""

# Run the server
if __name__ == "__main__":
    # Print a success message to stderr so it doesn't corrupt MCP JSON-RPC on stdout
    print("Turbovec MCP Server is successfully running", file=_original_stderr)
    
    # RESTORE STDOUT FOR MCP COMMUNICATION
    sys.stdout = _original_stdout
    
    mcp.run(transport='stdio')
