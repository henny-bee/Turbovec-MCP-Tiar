"""MCP prompts backed by the memory engine."""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP
from pydantic import Field

logger = logging.getLogger(__name__)

__all__ = ["register"]

_REVIEW_PROMPT = (
    "You are a senior software engineer. Please review the provided codebase "
    "snippets for potential bugs, performance issues, and best practices. "
    "Suggest improvements where applicable."
)


def register(mcp: FastMCP, db) -> None:
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
        return _REVIEW_PROMPT
