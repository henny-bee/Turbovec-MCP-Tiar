import os
import logging
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from vector_db import VectorDB

logger = logging.getLogger(__name__)


def register_tools_and_prompts(mcp: FastMCP, db: VectorDB):
    logger.info("Registering MCP tools and prompts")

    @mcp.tool()
    def add_knowledge(
        title: str = Field(
            ..., description="The title or brief summary of the knowledge to add"
        ),
        content: str = Field(
            ..., description="The actual documentation, code, or information to store"
        ),
    ) -> str:
        """
        Use this tool to insert documentation, code, or information
        into the Turbovec vector memory using chunking.
        """
        logger.info(f"Tool 'add_knowledge' called with title: {title}")
        return db.add_knowledge(title, content)

    @mcp.tool()
    def add_file_knowledge(
        file_path: str = Field(
            ...,
            description="Absolute or relative path to the local file to read and index",
        )
    ) -> str:
        """Reads a local file, chunks it, and indexes it for semantic search."""
        logger.info(f"Tool 'add_file_knowledge' called with file_path: {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return f"Error: File '{file_path}' does not exist."
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading file '{file_path}': {e}", exc_info=True)
            return f"Error reading file '{file_path}': {e}"

        title = os.path.basename(file_path)
        return db.add_knowledge(title, content)

    @mcp.tool()
    def delete_knowledge(
        title_or_id: str = Field(
            ...,
            description="The exact title or ID of the document to remove from memory",
        )
    ) -> str:
        """Removes a document from the index and metadata database."""
        logger.info(f"Tool 'delete_knowledge' called for: {title_or_id}")
        return db.delete_knowledge(title_or_id)

    @mcp.tool()
    def clear_memory() -> str:
        """Wipes the index and metadata database completely."""
        logger.info("Tool 'clear_memory' called")
        return db.clear_memory()

    @mcp.tool()
    def optimize_memory() -> str:
        """Permanently removes soft-deleted chunks (garbage collection) and rebuilds the vector index."""
        logger.info("Tool 'optimize_memory' called")
        return db.optimize_index()

    @mcp.tool()
    def search_knowledge(
        query: str = Field(
            ..., description="The semantic search query to find relevant information"
        ),
        top_k: int = Field(3, description="Maximum number of results to return"),
    ) -> str:
        """
        Use this tool to search for specific information from memory based on semantic meaning (Semantic Search).
        """
        logger.info(
            f"Tool 'search_knowledge' called with query: '{query}', top_k: {top_k}"
        )
        return db.search_knowledge(query, top_k)

    # MCP Prompts
    @mcp.prompt()
    def qna_with_context(
        query: str = Field(
            ...,
            description="The user's question to be answered using the knowledge base",
        )
    ) -> str:
        """Creates a system prompt for Q&A using search context."""
        logger.info(f"Prompt 'qna_with_context' called with query: '{query}'")
        context = db.search_knowledge(query, top_k=5)
        return f"""You are a helpful assistant. Use the following context retrieved from our knowledge base to answer the user's query.

Context:
{context}

User Query: {query}
Answer:"""

    @mcp.prompt()
    def review_codebase() -> str:
        """Creates a system prompt for reviewing the codebase."""
        logger.info("Prompt 'review_codebase' called")
        return """You are a senior software engineer. Please review the provided codebase snippets for potential bugs, performance issues, and best practices. Suggest improvements where applicable."""

    logger.info("Successfully registered all MCP tools and prompts")
