import os
import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from mcp.server.fastmcp import FastMCP
from vector_db import VectorDB
from tools import register_tools_and_prompts


@pytest.fixture
def temp_files(tmp_path):
    metadata_file = tmp_path / "metadata.json"
    index_file = tmp_path / "index.tvim"
    sqlite_db_file = tmp_path / "memory.db"
    return str(metadata_file), str(index_file), str(sqlite_db_file)


@pytest.fixture
def mock_sentence_transformer():
    with patch("embeddings.minilm.SentenceTransformer") as MockST:
        instance = MockST.return_value

        def mock_encode(text, **kwargs):
            np.random.seed(abs(hash(text)) % (2**32))
            return np.random.rand(384).astype(np.float32)

        instance.encode.side_effect = mock_encode
        yield instance


@pytest.fixture
def db(temp_files, mock_sentence_transformer):
    metadata_file, index_file, sqlite_db_file = temp_files
    db = VectorDB(
        dimension=384,
        metadata_file=metadata_file,
        index_file=index_file,
        sqlite_db_file=sqlite_db_file,
    )
    return db


@pytest.fixture
def mcp(db):
    mcp = FastMCP("test-server")
    register_tools_and_prompts(mcp, db)
    return mcp


@pytest.mark.asyncio
async def test_tool_registration(mcp):
    """Verify all 13 new hybrid tools and existing tools are registered on the FastMCP server."""
    tools = mcp._tool_manager._tools
    assert "add_knowledge" in tools
    assert "add_file_knowledge" in tools
    assert "delete_knowledge" in tools
    assert "clear_memory" in tools
    assert "optimize_memory" in tools
    assert "search_knowledge" in tools

    # Verify new tools
    assert "create_entity" in tools
    assert "add_observation" in tools
    assert "create_relationship" in tools
    assert "delete_entity" in tools
    assert "delete_relationship" in tools
    assert "get_hologram" in tools
    assert "get_neighbors" in tools
    assert "search_memory" in tools
    assert "start_session" in tools
    assert "end_session" in tools
    assert "record_breakthrough" in tools
    assert "point_in_time_query" in tools
    assert "semantic_radar" in tools

    # Verify brand new parity tools
    assert "diff_knowledge_state" in tools
    assert "query_timeline" in tools
    assert "get_temporal_neighbors" in tools
    assert "archive_entity" in tools
    assert "restore_entity" in tools
    assert "list_orphans" in tools
    assert "prune_stale" in tools
    assert "create_bottle" in tools
    assert "get_bottles" in tools
    assert "acknowledge_bottle" in tools
    assert "search_stats" in tools
    assert "analyze_graph" in tools
    assert "run_librarian_cycle" in tools


@pytest.mark.asyncio
async def test_create_and_delete_entity_tools(mcp, db):
    """Verify create_entity and delete_entity tools execute successfully."""
    # Retrieve the tool function directly to test execution synchronously or asynchronously
    create_entity_fn = mcp._tool_manager._tools["create_entity"].fn
    delete_entity_fn = mcp._tool_manager._tools["delete_entity"].fn

    # Create entity
    res_create = create_entity_fn(
        id="ent-1",
        name="Entity 1",
        node_type="concept",
        properties={"desc": "Test entity"},
    )
    assert "Successfully created" in res_create

    # Verify node exists in DB
    cursor = db.conn.cursor()
    cursor.execute("SELECT name FROM nodes WHERE id = 'ent-1'")
    assert cursor.fetchone()[0] == "Entity 1"

    # Delete entity
    res_delete = delete_entity_fn(id="ent-1")
    assert "Successfully deleted" in res_delete

    # Verify node is gone
    cursor.execute("SELECT COUNT(*) FROM nodes WHERE id = 'ent-1'")
    assert cursor.fetchone()[0] == 0


@pytest.mark.asyncio
async def test_add_observation_tool(mcp, db):
    """Verify add_observation tool executes successfully and updates index."""
    create_entity_fn = mcp._tool_manager._tools["create_entity"].fn
    add_observation_fn = mcp._tool_manager._tools["add_observation"].fn

    create_entity_fn(id="ent-obs", name="Observant Node", node_type="entity")
    res_obs = add_observation_fn(
        entity_id="ent-obs", content="This is an interesting fact."
    )
    assert "Successfully added observation" in res_obs

    # Check SQLite
    cursor = db.conn.cursor()
    cursor.execute("SELECT content FROM observations WHERE entity_id = 'ent-obs'")
    assert cursor.fetchone()[0] == "This is an interesting fact."


@pytest.mark.asyncio
async def test_create_and_delete_relationship_tools(mcp, db):
    """Verify create_relationship and delete_relationship tools execute successfully."""
    create_entity_fn = mcp._tool_manager._tools["create_entity"].fn
    create_relationship_fn = mcp._tool_manager._tools["create_relationship"].fn
    delete_relationship_fn = mcp._tool_manager._tools["delete_relationship"].fn

    create_entity_fn(id="n1", name="Node 1", node_type="entity")
    create_entity_fn(id="n2", name="Node 2", node_type="entity")

    res_rel = create_relationship_fn(
        from_node_id="n1",
        to_node_id="n2",
        relationship_type="CONNECTS_TO",
        weight=0.5,
    )
    assert "Successfully created relationship" in res_rel

    # Check SQLite
    cursor = db.conn.cursor()
    cursor.execute(
        "SELECT relationship_type, weight FROM edges WHERE from_node_id = 'n1'"
    )
    row = cursor.fetchone()
    assert row[0] == "CONNECTS_TO"
    assert row[1] == 0.5

    # Delete relationship
    res_del_rel = delete_relationship_fn(
        from_node_id="n1", to_node_id="n2", relationship_type="CONNECTS_TO"
    )
    assert "Successfully deleted" in res_del_rel

    # Check SQLite
    cursor.execute("SELECT COUNT(*) FROM edges WHERE from_node_id = 'n1'")
    assert cursor.fetchone()[0] == 0


@pytest.mark.asyncio
async def test_get_hologram_and_neighbors_tools(mcp, db):
    """Verify get_hologram and get_neighbors tools retrieve expected graph context."""
    create_entity_fn = mcp._tool_manager._tools["create_entity"].fn
    create_relationship_fn = mcp._tool_manager._tools["create_relationship"].fn
    get_hologram_fn = mcp._tool_manager._tools["get_hologram"].fn
    get_neighbors_fn = mcp._tool_manager._tools["get_neighbors"].fn

    create_entity_fn(id="anchor", name="Anchor", node_type="entity")
    create_entity_fn(id="friend", name="Friend", node_type="entity")
    create_relationship_fn(
        from_node_id="anchor", to_node_id="friend", relationship_type="KNOWS"
    )

    # Test hologram
    holo = get_hologram_fn(node_id="anchor", depth=1)
    assert "nodes" in holo
    assert "edges" in holo
    assert len(holo["nodes"]) == 2
    assert len(holo["edges"]) == 1

    # Test neighbors
    neighbors = get_neighbors_fn(node_id="anchor", depth=1)
    assert isinstance(neighbors, list)
    assert len(neighbors) == 2


@pytest.mark.asyncio
async def test_search_memory_tool(mcp, db):
    """Verify search_memory tool executes hybrid search."""
    create_entity_fn = mcp._tool_manager._tools["create_entity"].fn
    search_memory_fn = mcp._tool_manager._tools["search_memory"].fn

    create_entity_fn(id="n-search", name="Search Target Node", node_type="entity")

    results = search_memory_fn(query="Search Target", limit=5)
    assert len(results) > 0
    assert results[0]["id"] == "n-search"


@pytest.mark.asyncio
async def test_session_and_breakthrough_tools(mcp, db):
    """Verify session management and breakthrough recording tools."""
    start_session_fn = mcp._tool_manager._tools["start_session"].fn
    end_session_fn = mcp._tool_manager._tools["end_session"].fn
    record_breakthrough_fn = mcp._tool_manager._tools["record_breakthrough"].fn

    # Start session
    sid = start_session_fn(
        session_id="sess-01", name="Session 01", properties={"focus": "test"}
    )
    assert sid == "sess-01"

    # Record breakthrough
    res_bt = record_breakthrough_fn(
        session_id="sess-01", title="Aha Moment", content="I figured it out!"
    )
    assert "Successfully recorded breakthrough" in res_bt

    # End session
    res_end = end_session_fn(session_id="sess-01", summary="All done.")
    assert "Successfully ended session" in res_end


@pytest.mark.asyncio
async def test_point_in_time_query_tool(mcp, db):
    """Verify point_in_time_query tool returns historical state."""
    create_entity_fn = mcp._tool_manager._tools["create_entity"].fn
    point_in_time_query_fn = mcp._tool_manager._tools["point_in_time_query"].fn

    t1 = "2026-08-16T08:00:00Z"
    t2 = "2026-08-16T08:30:00Z"

    create_entity_fn(id="n-pit", name="Historical Node", node_type="entity")

    # Update created_at to t2
    with db.conn:
        db.conn.execute("UPDATE nodes SET created_at = ? WHERE id = 'n-pit'", (t2,))

    # Query as of t1 (should be empty)
    state_t1 = point_in_time_query_fn(as_of_iso_timestamp=t1)
    assert len(state_t1["nodes"]) == 0

    # Query as of t2 (should have our node)
    state_t2 = point_in_time_query_fn(as_of_iso_timestamp=t2)
    assert len(state_t2["nodes"]) == 1
    assert state_t2["nodes"][0]["id"] == "n-pit"


@pytest.mark.asyncio
async def test_semantic_radar_tool(mcp, db):
    """Verify semantic_radar tool detects missing relationships."""
    create_entity_fn = mcp._tool_manager._tools["create_entity"].fn
    semantic_radar_fn = mcp._tool_manager._tools["semantic_radar"].fn

    # Create two nodes with highly similar vectors but no edge between them
    # Note: Our mock SentenceTransformer is deterministic, so we can mock or just insert two nodes.
    # Since our mock is based on string hash, let's patch SentenceTransformer to return identical vectors
    # for these two nodes.
    vec = np.zeros(384, dtype=np.float32)
    vec[0] = 1.0

    with patch.object(db.model, "encode", return_value=vec):
        create_entity_fn(id="node-1", name="First", node_type="entity")
        create_entity_fn(id="node-2", name="Second", node_type="entity")

        gaps = semantic_radar_fn(similarity_threshold=0.9, project_id="default")
        assert len(gaps) == 1
        assert gaps[0]["similarity"] == pytest.approx(1.0)
        assert {gaps[0]["node_1"]["id"], gaps[0]["node_2"]["id"]} == {
            "node-1",
            "node-2",
        }


@pytest.mark.asyncio
async def test_add_knowledge_pipeline_integration(mcp, db):
    """Verify that add_knowledge automatically runs extraction pipeline."""
    add_knowledge_fn = mcp._tool_manager._tools["add_knowledge"].fn

    text = (
        "John Doe works at Google. Python is a programming language. I prefer VS Code."
    )
    res = add_knowledge_fn(title="Unstructured Doc", content=text)
    assert "Successfully added" in res

    # Verify that entities and relationships were extracted and added to SQLite Graph
    cursor = db.conn.cursor()

    # Check that "John Doe" and "Google" exist as nodes
    cursor.execute("SELECT id FROM nodes WHERE name = 'John Doe'")
    john_row = cursor.fetchone()
    assert john_row is not None

    cursor.execute("SELECT id FROM nodes WHERE name = 'Google'")
    google_row = cursor.fetchone()
    assert google_row is not None

    # Check that WORKS_AT relationship exists
    cursor.execute(
        "SELECT relationship_type FROM edges WHERE from_node_id = ? AND to_node_id = ?",
        (john_row[0], google_row[0]),
    )
    assert cursor.fetchone()[0] == "WORKS_AT"


@pytest.mark.asyncio
async def test_relationship_inference_and_auto_creation(mcp, db):
    """Verify that semantic_radar automatically infers relationships and can create discovered edges."""
    create_entity_fn = mcp._tool_manager._tools["create_entity"].fn
    run_relationship_discovery_fn = mcp._tool_manager._tools[
        "run_relationship_discovery"
    ].fn

    # Prepare two highly similar nodes with different types (e.g. Person and Session)
    vec = np.zeros(384, dtype=np.float32)
    vec[0] = 1.0

    with patch.object(db.model, "encode", return_value=vec):
        create_entity_fn(id="p-1", name="Alice", node_type="Person")
        create_entity_fn(id="s-1", name="Project Kickoff", node_type="Session")

        # Run discovery without auto_create
        results = run_relationship_discovery_fn(
            similarity_threshold=0.9, project_id="default", auto_create=False
        )
        assert len(results) == 1
        assert results[0]["suggested_relationship"] == "MENTIONED_IN"
        assert results[0]["created_edge"] is False

        # Verify no edge actually created in SQLite yet
        cursor = db.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM edges")
        assert cursor.fetchone()[0] == 0

        # Run discovery with auto_create
        results_with_create = run_relationship_discovery_fn(
            similarity_threshold=0.9, project_id="default", auto_create=True
        )
        assert len(results_with_create) == 1
        assert results_with_create[0]["created_edge"] is True
        assert results_with_create[0]["suggested_relationship"] == "MENTIONED_IN"

        # Verify edge exists in SQLite now
        cursor.execute(
            "SELECT relationship_type FROM edges WHERE from_node_id = 'p-1' AND to_node_id = 's-1'"
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "MENTIONED_IN"


@pytest.mark.asyncio
async def test_background_discovery_management_tool(mcp, db):
    """Verify background discovery status query and start/stop controls."""
    status_fn = mcp._tool_manager._tools["get_background_discovery_status"].fn
    set_bg_fn = mcp._tool_manager._tools["set_background_discovery"].fn

    # Start with stopped worker
    assert db.bg_thread is None
    status = status_fn()
    assert status["background_worker_running"] is False

    # Turn background discovery on
    msg = set_bg_fn(enabled=True, interval_seconds=10, similarity_threshold=0.85)
    assert "Successfully started" in msg
    assert db.bg_thread is not None
    assert db.bg_thread.is_alive()

    # Verify status reflects it running
    status = status_fn()
    assert status["background_worker_running"] is True
    assert status["interval_seconds"] == 10
    assert status["similarity_threshold"] == 0.85

    # Stop background discovery
    msg_stop = set_bg_fn(enabled=False)
    assert "Successfully stopped" in msg_stop
    assert db.bg_thread is None

    # Status reflects stopped
    status = status_fn()
    assert status["background_worker_running"] is False


@pytest.mark.asyncio
async def test_new_parity_tools_and_wrappers(mcp, db):
    """Test all the brand new parity tools exposed on the MCP server."""
    create_entity_fn = mcp._tool_manager._tools["create_entity"].fn
    add_observation_fn = mcp._tool_manager._tools["add_observation"].fn

    # 1. Setup entities for temporal & lifecycle testing
    # Use exact timestamps to test diff_knowledge_state and query_timeline
    t1 = "2026-08-16T10:00:00Z"
    t2 = "2026-08-16T10:30:00Z"
    t3 = "2026-08-16T11:00:00Z"

    create_entity_fn(id="n-time1", name="Temporal Entity 1", node_type="entity")
    with db.conn:
        db.conn.execute(
            "UPDATE nodes SET created_at = ?, occurred_at = ? WHERE id = 'n-time1'",
            (t1, t1),
        )

    create_entity_fn(id="n-time2", name="Temporal Entity 2", node_type="entity")
    with db.conn:
        db.conn.execute(
            "UPDATE nodes SET created_at = ?, occurred_at = ? WHERE id = 'n-time2'",
            (t2, t2),
        )

    add_observation_fn(entity_id="n-time1", content="Fact registered at t2.")
    with db.conn:
        db.conn.execute(
            "UPDATE observations SET created_at = ?, occurred_at = ? WHERE entity_id = 'n-time1'",
            (t2, t2),
        )

    # 2. Test diff_knowledge_state tool
    diff_fn = mcp._tool_manager._tools["diff_knowledge_state"].fn
    diff_res = diff_fn(from_timestamp=t1, to_timestamp=t3)
    assert diff_res["from"] == t1
    assert diff_res["to"] == t3
    assert len(diff_res["added"]) >= 1

    # 3. Test query_timeline tool
    timeline_fn = mcp._tool_manager._tools["query_timeline"].fn
    timeline_res = timeline_fn(start=t1, end=t3, order="ASC", limit=10)
    assert len(timeline_res) >= 1

    # 4. Test get_temporal_neighbors tool
    create_relationship_fn = mcp._tool_manager._tools["create_relationship"].fn
    create_relationship_fn(
        from_node_id="n-time1", to_node_id="n-time2", relationship_type="RELATED_TO"
    )

    temp_neigh_fn = mcp._tool_manager._tools["get_temporal_neighbors"].fn
    temp_neigh_res = temp_neigh_fn(node_id="n-time1", direction="both", depth=1)
    assert len(temp_neigh_res) >= 1

    # 5. Test archive_entity / restore_entity tools
    archive_fn = mcp._tool_manager._tools["archive_entity"].fn
    restore_fn = mcp._tool_manager._tools["restore_entity"].fn

    arch_res = archive_fn(id="n-time1", reason="Testing archive")
    assert arch_res["status"] == "ARCHIVED"

    # Verify node status is updated in SQLite
    cursor = db.conn.cursor()
    cursor.execute("SELECT status FROM nodes WHERE id = 'n-time1'")
    assert cursor.fetchone()[0] == "ARCHIVED"

    rest_res = restore_fn(id="n-time1")
    assert rest_res["status"] == "ACTIVE"

    # 6. Test list_orphans tool
    list_orphans_fn = mcp._tool_manager._tools["list_orphans"].fn
    # Since we have edges, let's create a completely isolated node to test orphan detection
    create_entity_fn(id="isolated-node", name="Isolated Node", node_type="entity")
    orphans = list_orphans_fn()
    assert any(o["id"] == "isolated-node" for o in orphans)

    # 7. Test prune_stale tool
    prune_stale_fn = mcp._tool_manager._tools["prune_stale"].fn
    # Mark n-time2 as STALE and with older created_at to be pruned
    with db.conn:
        db.conn.execute(
            "UPDATE nodes SET status = 'STALE', created_at = '2020-01-01T00:00:00Z' WHERE id = 'n-time2'"
        )

    # Dry run prune
    pruned_dry = prune_stale_fn(max_age_days=30, dry_run=True)
    assert any(p["id"] == "n-time2" for p in pruned_dry)

    # Actual prune
    pruned_real = prune_stale_fn(max_age_days=30, dry_run=False)
    assert any(p["id"] == "n-time2" for p in pruned_real)

    # Verify node is deleted in DB
    cursor.execute("SELECT COUNT(*) FROM nodes WHERE id = 'n-time2'")
    assert cursor.fetchone()[0] == 0

    # 8. Test Bottle note management tools (create_bottle, get_bottles, acknowledge_bottle)
    create_bottle_fn = mcp._tool_manager._tools["create_bottle"].fn
    get_bottles_fn = mcp._tool_manager._tools["get_bottles"].fn
    acknowledge_bottle_fn = mcp._tool_manager._tools["acknowledge_bottle"].fn

    bottle = create_bottle_fn(message="Important system note.", priority="high")
    assert bottle["message"] == "Important system note."
    assert bottle["priority"] == "high"

    bottles_list = get_bottles_fn(include_acknowledged=False)
    assert any(b["id"] == bottle["id"] for b in bottles_list)

    ack_res = acknowledge_bottle_fn(id=bottle["id"])
    assert ack_res is True

    # 9. Test search_stats tool
    stats_fn = mcp._tool_manager._tools["search_stats"].fn
    stats = stats_fn()
    assert "total_searches" in stats

    # 10. Test analyze_graph tool
    analyze_fn = mcp._tool_manager._tools["analyze_graph"].fn
    analysis = analyze_fn()
    assert "nodes" in analysis
    assert "edges" in analysis

    # 11. Test run_librarian_cycle tool
    run_lib_fn = mcp._tool_manager._tools["run_librarian_cycle"].fn
    lib_res = run_lib_fn()
    assert "status" in lib_res
