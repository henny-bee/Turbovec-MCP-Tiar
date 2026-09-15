"""Graph CRUD and neighbourhood MCP tools."""

from __future__ import annotations

import logging
import uuid

from mcp.server.fastmcp import FastMCP

from tools._shared import error_dict, error_list, failure

logger = logging.getLogger(__name__)

__all__ = ["register"]


def register(mcp: FastMCP, db) -> None:
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
        except Exception as exc:
            return failure(logger, f"Error creating entity '{name}'", exc)

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
            db.create_observation(obs_id=obs_id, entity_id=entity_id, content=content)
            return (
                f"Successfully added observation with ID '{obs_id}' "
                f"to entity '{entity_id}'."
            )
        except Exception as exc:
            return failure(
                logger, f"Error adding observation to entity '{entity_id}'", exc
            )

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
        logger.info(
            f"Tool 'create_relationship' called: {from_node_id} -> {to_node_id} "
            f"({relationship_type})"
        )
        try:
            db.create_edge(
                edge_id=f"edge-{from_node_id}-{to_node_id}-{relationship_type}",
                from_node_id=from_node_id,
                to_node_id=to_node_id,
                relationship_type=relationship_type,
                weight=weight,
                properties=properties,
            )
            return (
                f"Successfully created relationship '{relationship_type}' "
                f"between '{from_node_id}' and '{to_node_id}'."
            )
        except Exception as exc:
            return failure(logger, "Error creating relationship", exc)

    @mcp.tool()
    def delete_entity(
        id: str,
        project_id: str = "default",
    ) -> str:
        """Deletes an entity from the graph memory, cascading deletes to edges and vectors."""
        logger.info(f"Tool 'delete_entity' called for ID: {id}")
        try:
            if db.delete_node(node_id=id):
                return f"Successfully deleted entity with ID '{id}'."
            return f"Entity with ID '{id}' not found."
        except Exception as exc:
            return failure(logger, f"Error deleting entity '{id}'", exc)

    @mcp.tool()
    def delete_relationship(
        from_node_id: str,
        to_node_id: str,
        relationship_type: str,
        project_id: str = "default",
    ) -> str:
        """Deletes a relationship edge from the graph memory."""
        logger.info(
            f"Tool 'delete_relationship' called: {from_node_id} -> {to_node_id} "
            f"({relationship_type})"
        )
        try:
            deleted = db.delete_edge(
                edge_id=f"edge-{from_node_id}-{to_node_id}-{relationship_type}"
            )
            if deleted:
                return (
                    f"Successfully deleted relationship '{relationship_type}' "
                    f"between '{from_node_id}' and '{to_node_id}'."
                )
            return "Relationship not found."
        except Exception as exc:
            return failure(logger, "Error deleting relationship", exc)

    @mcp.tool()
    def get_hologram(
        node_id: str,
        depth: int = 1,
    ) -> dict:
        """Retrieves a node, its observations, and its neighbor network up to the specified depth."""
        logger.info(f"Tool 'get_hologram' called for: {node_id}")
        try:
            return db.get_hologram(node_id=node_id, depth=depth)
        except Exception as exc:
            return error_dict(logger, f"Error getting hologram for '{node_id}'", exc)

    @mcp.tool()
    def get_neighbors(
        node_id: str,
        depth: int = 1,
    ) -> list[dict]:
        """Explores connected nodes in the graph starting from an anchor node up to a specified depth."""
        logger.info(f"Tool 'get_neighbors' called for: {node_id}")
        try:
            return db.get_neighbors(node_id=node_id, depth=depth)
        except Exception as exc:
            return error_list(logger, f"Error getting neighbors for '{node_id}'", exc)
