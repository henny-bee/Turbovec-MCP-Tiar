import os
import logging
import uuid
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from vector_db import VectorDB
from pydantic.fields import FieldInfo

logger = logging.getLogger(__name__)


def resolve_field(val, default=None):
    return default if isinstance(val, FieldInfo) else val


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

    @mcp.tool()
    def create_entity(
        id: str,
        name: str,
        node_type: str,
        properties: dict = None,
        project_id: str = "default",
    ) -> str:
        """Creates a structured entity node in the graph memory."""
        logger.info(f"Tool 'create_entity' called for ID: {id}, Name: {name}")
        try:
            db.create_node(
                node_id=id,
                name=name,
                node_type=node_type,
                properties=properties,
                project_id=project_id,
            )
            return f"Successfully created entity '{name}' with ID '{id}'."
        except Exception as e:
            logger.error(f"Error creating entity '{name}': {e}", exc_info=True)
            return f"Error: {e}"

    @mcp.tool()
    def add_observation(
        entity_id: str,
        content: str,
        project_id: str = "default",
    ) -> str:
        """Attaches a new observation/fact to an existing entity and updates indices."""
        logger.info(f"Tool 'add_observation' called for Entity: {entity_id}")
        try:
            obs_id = str(uuid.uuid4())
            db.create_observation(
                obs_id=obs_id,
                entity_id=entity_id,
                content=content,
            )
            return f"Successfully added observation with ID '{obs_id}' to entity '{entity_id}'."
        except Exception as e:
            logger.error(f"Error adding observation to entity '{entity_id}': {e}", exc_info=True)
            return f"Error: {e}"

    @mcp.tool()
    def create_relationship(
        from_node_id: str,
        to_node_id: str,
        relationship_type: str,
        weight: float = 1.0,
        properties: dict = None,
        project_id: str = "default",
    ) -> str:
        """Creates a typed, weighted directed relationship between two entities."""
        logger.info(f"Tool 'create_relationship' called: {from_node_id} -> {to_node_id} ({relationship_type})")
        try:
            edge_id = f"edge-{from_node_id}-{to_node_id}-{relationship_type}"
            db.create_edge(
                edge_id=edge_id,
                from_node_id=from_node_id,
                to_node_id=to_node_id,
                relationship_type=relationship_type,
                weight=weight,
                properties=properties,
            )
            return f"Successfully created relationship '{relationship_type}' between '{from_node_id}' and '{to_node_id}'."
        except Exception as e:
            logger.error(f"Error creating relationship: {e}", exc_info=True)
            return f"Error: {e}"

    @mcp.tool()
    def delete_entity(
        id: str,
        project_id: str = "default",
    ) -> str:
        """Deletes an entity from the graph memory, cascading deletes to edges and vectors."""
        logger.info(f"Tool 'delete_entity' called for ID: {id}")
        try:
            success = db.delete_node(node_id=id)
            if success:
                return f"Successfully deleted entity with ID '{id}'."
            else:
                return f"Entity with ID '{id}' not found."
        except Exception as e:
            logger.error(f"Error deleting entity '{id}': {e}", exc_info=True)
            return f"Error: {e}"

    @mcp.tool()
    def delete_relationship(
        from_node_id: str,
        to_node_id: str,
        relationship_type: str,
        project_id: str = "default",
    ) -> str:
        """Deletes a relationship edge from the graph memory."""
        logger.info(f"Tool 'delete_relationship' called: {from_node_id} -> {to_node_id} ({relationship_type})")
        try:
            edge_id = f"edge-{from_node_id}-{to_node_id}-{relationship_type}"
            success = db.delete_edge(edge_id=edge_id)
            if success:
                return f"Successfully deleted relationship '{relationship_type}' between '{from_node_id}' and '{to_node_id}'."
            else:
                return f"Relationship not found."
        except Exception as e:
            logger.error(f"Error deleting relationship: {e}", exc_info=True)
            return f"Error: {e}"

    @mcp.tool()
    def get_hologram(
        node_id: str,
        depth: int = 1,
    ) -> dict:
        """Retrieves a node, its observations, and its neighbor network up to the specified depth."""
        logger.info(f"Tool 'get_hologram' called for: {node_id}")
        try:
            return db.get_hologram(node_id=node_id, depth=depth)
        except Exception as e:
            logger.error(f"Error getting hologram for '{node_id}': {e}", exc_info=True)
            return {"error": str(e)}

    @mcp.tool()
    def get_neighbors(
        node_id: str,
        depth: int = 1,
    ) -> list[dict]:
        """Explores connected nodes in the graph starting from an anchor node up to a specified depth."""
        logger.info(f"Tool 'get_neighbors' called for: {node_id}")
        try:
            return db.get_neighbors(node_id=node_id, depth=depth)
        except Exception as e:
            logger.error(f"Error getting neighbors for '{node_id}': {e}", exc_info=True)
            return [{"error": str(e)}]

    @mcp.tool()
    def search_memory(
        query: str,
        limit: int = 10,
        project_id: str = "default",
        channel_weights: dict = None,
    ) -> list[dict]:
        """Runs a hybrid vector-lexical search over the memory graph and returns ranked nodes."""
        logger.info(f"Tool 'search_memory' called with query: '{query}'")
        try:
            return db.search_hybrid(
                query=query,
                project_id=project_id,
                limit=limit,
                channel_weights=channel_weights,
            )
        except Exception as e:
            logger.error(f"Error searching memory for '{query}': {e}", exc_info=True)
            return [{"error": str(e)}]

    @mcp.tool()
    def start_session(
        session_id: str,
        name: str,
        properties: dict = None,
        project_id: str = "default",
    ) -> str:
        """Starts a session, creating a session node and linking to any previous session."""
        logger.info(f"Tool 'start_session' called: {session_id} - {name}")
        try:
            return db.start_session(
                session_id=session_id,
                name=name,
                properties=properties,
            )
        except Exception as e:
            logger.error(f"Error starting session '{session_id}': {e}", exc_info=True)
            return f"Error: {e}"

    @mcp.tool()
    def end_session(
        session_id: str,
        summary: str,
        project_id: str = "default",
    ) -> str:
        """Ends an active session, saving its summary and updating status to closed."""
        logger.info(f"Tool 'end_session' called for: {session_id}")
        try:
            db.end_session(session_id=session_id, summary=summary)
            return f"Successfully ended session '{session_id}'."
        except Exception as e:
            logger.error(f"Error ending session '{session_id}': {e}", exc_info=True)
            return f"Error: {e}"

    @mcp.tool()
    def record_breakthrough(
        session_id: str,
        title: str,
        content: str,
        project_id: str = "default",
    ) -> str:
        """Records a learning breakthrough and links it to the active session."""
        logger.info(f"Tool 'record_breakthrough' called for Session: {session_id}")
        try:
            bt_id = db.record_breakthrough(
                session_id=session_id,
                title=title,
                content=content,
            )
            return f"Successfully recorded breakthrough with ID '{bt_id}' linked to session '{session_id}'."
        except Exception as e:
            logger.error(f"Error recording breakthrough: {e}", exc_info=True)
            return f"Error: {e}"

    @mcp.tool()
    def point_in_time_query(
        as_of_iso_timestamp: str,
    ) -> dict:
        """Queries the state of the memory graph as of a specific date and time."""
        logger.info(f"Tool 'point_in_time_query' called for: {as_of_iso_timestamp}")
        try:
            return db.get_graph_state_as_of(as_of_iso_timestamp=as_of_iso_timestamp)
        except Exception as e:
            logger.error(f"Error in point_in_time_query: {e}", exc_info=True)
            return {"error": str(e)}

    @mcp.tool()
    def semantic_radar(
        similarity_threshold: float = 0.7,
        project_id: str = "default",
        auto_create: bool = False,
    ) -> list[dict]:
        """Scans the graph to detect pairs of entities with high semantic similarity but no path between them, with optional automatic creation of inferred edges."""
        logger.info(f"Tool 'semantic_radar' called with threshold: {similarity_threshold}, auto_create: {auto_create}")
        try:
            return db.semantic_radar(
                similarity_threshold=similarity_threshold,
                project_id=project_id,
                auto_create=auto_create,
            )
        except Exception as e:
            logger.error(f"Error in semantic_radar: {e}", exc_info=True)
            return [{"error": str(e)}]

    @mcp.tool()
    def run_relationship_discovery(
        similarity_threshold: float = 0.7,
        project_id: str = "default",
        auto_create: bool = True,
    ) -> list[dict]:
        """Runs the semantic radar discovery process to find disconnected highly similar nodes and optionally create inferred relationship edges."""
        logger.info(f"Tool 'run_relationship_discovery' called with similarity_threshold: {similarity_threshold}, auto_create: {auto_create}")
        try:
            return db.semantic_radar(
                similarity_threshold=similarity_threshold,
                project_id=project_id,
                auto_create=auto_create,
            )
        except Exception as e:
            logger.error(f"Error in run_relationship_discovery: {e}", exc_info=True)
            return [{"error": str(e)}]

    @mcp.tool()
    def get_background_discovery_status() -> dict:
        """Returns the current status, interval, and similarity threshold of the background relationship discovery worker."""
        logger.info("Tool 'get_background_discovery_status' called")
        is_running = db.bg_thread is not None and db.bg_thread.is_alive()
        return {
            "background_discovery_enabled_at_startup": db.bg_enabled,
            "background_worker_running": is_running,
            "interval_seconds": db.bg_interval,
            "similarity_threshold": db.bg_similarity_threshold,
        }

    @mcp.tool()
    def set_background_discovery(
        enabled: bool,
        interval_seconds: int = 300,
        similarity_threshold: float = 0.7,
    ) -> str:
        """Dynamically starts, stops, or configures the background relationship discovery daemon worker."""
        logger.info(f"Tool 'set_background_discovery' called with enabled: {enabled}, interval_seconds: {interval_seconds}, similarity_threshold: {similarity_threshold}")
        try:
            db.bg_interval = interval_seconds
            db.bg_similarity_threshold = similarity_threshold
            if enabled:
                db.start_background_discovery()
                return f"Successfully started/updated background relationship discovery with interval of {interval_seconds}s and similarity threshold {similarity_threshold}."
            else:
                db.stop_background_discovery()
                return "Successfully stopped background relationship discovery."
        except Exception as e:
            logger.error(f"Error in set_background_discovery: {e}", exc_info=True)
            return f"Error: {e}"

    @mcp.tool()
    def diff_knowledge_state(
        from_timestamp: str = Field(..., description="Starting ISO 8601 timestamp for the diff window"),
        to_timestamp: str = Field(..., description="Ending ISO 8601 timestamp for the diff window"),
    ) -> dict:
        """Computes a deterministic delta between two points in time (added, removed, changed nodes/edges)."""
        logger.info(f"Tool 'diff_knowledge_state' called from {from_timestamp} to {to_timestamp}")
        return db.diff_knowledge_state(
            from_timestamp=resolve_field(from_timestamp, None),
            to_timestamp=resolve_field(to_timestamp, None)
        )

    @mcp.tool()
    def query_timeline(
        entity: str = Field(None, description="Optional entity name/attribute keyword filter"),
        entity_type: str = Field(None, description="Optional node type filter"),
        start: str = Field(None, description="Optional starting ISO 8601 timestamp filter"),
        end: str = Field(None, description="Optional ending ISO 8601 timestamp filter"),
        order: str = Field("ASC", description="Chronological sorting order: ASC or DESC"),
        limit: int = Field(50, description="Maximum number of timeline events to return"),
        offset: int = Field(0, description="Offset for pagination"),
    ) -> list[dict]:
        """Queries the timeline of events chronologically with strict filtering and pagination."""
        logger.info("Tool 'query_timeline' called")
        return db.query_timeline(
            entity=resolve_field(entity, None),
            entity_type=resolve_field(entity_type, None),
            start=resolve_field(start, None),
            end=resolve_field(end, None),
            order=resolve_field(order, "ASC"),
            limit=resolve_field(limit, 50),
            offset=resolve_field(offset, 0)
        )

    @mcp.tool()
    def get_temporal_neighbors(
        node_id: str = Field(..., description="The anchor entity ID to start traversal from"),
        direction: str = Field("both", description="Temporal relationship direction: before, after, or both"),
        depth: int = Field(1, description="Graph traversal depth to explore"),
    ) -> list[dict]:
        """Explores connected nodes along temporal edges or chronologically related nodes relative to an anchor node."""
        logger.info(f"Tool 'get_temporal_neighbors' called for: {node_id}")
        return db.get_temporal_neighbors(
            node_id=resolve_field(node_id, None),
            direction=resolve_field(direction, "both"),
            depth=resolve_field(depth, 1)
        )

    @mcp.tool()
    def archive_entity(
        id: str = Field(..., description="The ID of the entity node to archive"),
        reason: str = Field("Unused/superseded", description="Reason for archiving the entity"),
    ) -> dict:
        """Archives an entity and updates its status to ARCHIVED, excluding it from active retrieval."""
        logger.info(f"Tool 'archive_entity' called for ID: {id}")
        return db.archive_entity(id, reason=resolve_field(reason, "Unused/superseded"))

    @mcp.tool()
    def restore_entity(
        id: str = Field(..., description="The ID of the entity node to restore to ACTIVE status"),
    ) -> dict:
        """Restores an archived entity node back to ACTIVE status."""
        logger.info(f"Tool 'restore_entity' called for ID: {id}")
        return db.restore_entity(id)

    @mcp.tool()
    def list_orphans() -> list[dict]:
        """Identifies and lists orphaned entities (nodes with zero edges)."""
        logger.info("Tool 'list_orphans' called")
        return db.list_orphans()

    @mcp.tool()
    def prune_stale(
        max_age_days: int = Field(..., description="Maximum age in days for STALE or ARCHIVED nodes to be pruned"),
        dry_run: bool = Field(False, description="If True, only lists the candidates without deleting them"),
    ) -> list[dict]:
        """Identifies stale or archived memories older than a specified duration and prunes them from the database."""
        logger.info(f"Tool 'prune_stale' called with max_age_days: {max_age_days}, dry_run: {dry_run}")
        return db.prune_stale(
            max_age_days=resolve_field(max_age_days, None),
            dry_run=resolve_field(dry_run, False)
        )

    @mcp.tool()
    def create_bottle(
        message: str = Field(..., description="The content of the note/message left in the bottle"),
        priority: str = Field("medium", description="Priority tier: low, medium, or high"),
        expires_at: str = Field(None, description="Optional ISO 8601 expiration timestamp"),
        author_session_id: str = Field(None, description="Optional session ID of the authoring agent"),
    ) -> dict:
        """Creates a standardized bottle note for subsequent sessions or agents to discover."""
        logger.info("Tool 'create_bottle' called")
        return db.create_bottle(
            message=resolve_field(message, None),
            priority=resolve_field(priority, "medium"),
            expires_at=resolve_field(expires_at, None),
            author_session_id=resolve_field(author_session_id, None)
        )

    @mcp.tool()
    def get_bottles(
        include_acknowledged: bool = Field(False, description="If True, includes acknowledged bottle notes in list"),
    ) -> list[dict]:
        """Lists active and optionally acknowledged bottle notes left by agents."""
        logger.info("Tool 'get_bottles' called")
        return db.get_bottles(include_acknowledged=resolve_field(include_acknowledged, False))

    @mcp.tool()
    def acknowledge_bottle(
        id: str = Field(..., description="The ID of the bottle note to acknowledge"),
    ) -> bool:
        """Marks a bottle note as acknowledged/read."""
        logger.info(f"Tool 'acknowledge_bottle' called for ID: {id}")
        return db.acknowledge_bottle(id)

    @mcp.tool()
    def search_stats() -> dict:
        """Calculates rolling statistics, latencies per subsystem, percentiles, error rates, and cache performance."""
        logger.info("Tool 'search_stats' called")
        return db.search_stats()

    @mcp.tool()
    def analyze_graph() -> dict:
        """Runs connected components, LPA community detection, degree centrality, and PageRank analytics over the graph topology."""
        logger.info("Tool 'analyze_graph' called")
        return db.analyze_graph()

    @mcp.tool()
    def run_librarian_cycle() -> dict:
        """Executes one background librarian cycle to organize, cluster, detect duplicates, and synthesize concepts autonomously."""
        logger.info("Tool 'run_librarian_cycle' called")
        return db.run_librarian_cycle()

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
