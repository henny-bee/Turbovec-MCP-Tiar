import os
import json
import sqlite3
import datetime
import pandas as pd

try:
    import streamlit as st
except ImportError:
    # Streamlit is optional, print a mock/help message if run as CLI
    print("Streamlit is not installed. To run the developer dashboard, please run:")
    print("  pip install streamlit")
    import sys

    sys.exit(0)

# Page configuration
st.set_page_config(
    page_title="Turbovec MCP Developer Dashboard", page_icon="🧬", layout="wide"
)

st.title("Turbovec-MCP-Server Developer Dashboard")
st.markdown("---")

DB_FILE = os.getenv("SQLITE_DB_FILE", "memory.db")

if not os.path.exists(DB_FILE):
    st.error(
        f"SQLite Database not found at '{DB_FILE}'. Please start the server and run some queries first."
    )
else:
    # Connect to db
    conn = sqlite3.connect(DB_FILE)

    # 1. Sidebar statistics
    st.sidebar.header("Database Info")
    st.sidebar.info(f"Database File: `{DB_FILE}`")

    try:
        nodes_df = pd.read_sql_query("SELECT count(*) as count FROM nodes", conn)
        edges_df = pd.read_sql_query("SELECT count(*) as count FROM edges", conn)
        obs_df = pd.read_sql_query("SELECT count(*) as count FROM observations", conn)

        st.sidebar.metric("Total Entities (Nodes)", int(nodes_df["count"][0]))
        st.sidebar.metric("Total Relationships (Edges)", int(edges_df["count"][0]))
        st.sidebar.metric("Total Contextual Observations", int(obs_df["count"][0]))
    except Exception as e:
        st.sidebar.error(f"Error loading stats: {e}")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Telemetry & Latency",
            "Graph Topology & Communities",
            "Librarian Cycles",
            "Temporal Changes & Timeline",
            "Bottles & Notes",
        ]
    )

    # Tab 1: Telemetry & Latency
    with tab1:
        st.subheader("Search Performance Telemetry")
        try:
            metrics_df = pd.read_sql_query(
                "SELECT * FROM search_metrics ORDER BY id DESC", conn
            )
            if metrics_df.empty:
                st.warning(
                    "No search metrics recorded yet. Try running some semantic searches via MCP!"
                )
            else:
                st.write(f"Showing last {len(metrics_df)} searches")

                # Metrics Summary Cards
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Searches Run", len(metrics_df))
                col2.metric(
                    "P50 Latency (s)", f"{metrics_df['total_latency'].median():.4f}"
                )
                col3.metric(
                    "Max Latency (s)", f"{metrics_df['total_latency'].max():.4f}"
                )

                error_rate = (metrics_df["is_error"].sum() / len(metrics_df)) * 100
                col4.metric("Error Rate (%)", f"{error_rate:.2f}%")

                # Latency chart
                st.subheader("Total Latency Trend over Time")
                st.line_chart(metrics_df, x="timestamp", y="total_latency")

                # Subsystem latencies comparison
                st.subheader("Subsystem Latency Analysis (Averages)")
                avg_latencies = metrics_df[
                    [
                        "fts_latency",
                        "vector_latency",
                        "graph_latency",
                        "embedding_latency",
                        "reranker_latency",
                    ]
                ].mean()
                st.bar_chart(avg_latencies)

                st.subheader("Raw Telemetry Log")
                st.dataframe(metrics_df)
        except Exception as e:
            st.error(f"Search metrics table not available: {e}")

    # Tab 2: Graph Topology
    with tab2:
        st.subheader("Registered Graph Entities")
        try:
            nodes_data = pd.read_sql_query(
                "SELECT id, name, node_type, status, created_at, certainty, salience_score, retrieval_count FROM nodes",
                conn,
            )
            st.dataframe(nodes_data)

            # Node types distribution
            st.subheader("Entity Types Distribution")
            type_counts = nodes_data["node_type"].value_counts()
            st.bar_chart(type_counts)

            st.subheader("Relationships (Edges)")
            edges_data = pd.read_sql_query(
                "SELECT id, from_node_id, to_node_id, relationship_type, confidence, weight, created_at FROM edges",
                conn,
            )
            st.dataframe(edges_data)
        except Exception as e:
            st.error(f"Error loading topology data: {e}")

    # Tab 3: Librarian Cycles
    with tab3:
        st.subheader("Librarian Autonomous Organization History")
        try:
            runs_df = pd.read_sql_query(
                "SELECT * FROM librarian_runs ORDER BY id DESC", conn
            )
            if runs_df.empty:
                st.warning("No autonomous Librarian runs recorded yet.")
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Librarian Runs", len(runs_df))
                col2.metric(
                    "Total Synthesized Concepts",
                    int(runs_df["synthesized_concepts"].sum()),
                )
                col3.metric(
                    "Total Inferred Relationships", int(runs_df["edges_created"].sum())
                )

                st.subheader("Librarian Runs Log")
                st.dataframe(runs_df)
        except Exception as e:
            st.error(f"Librarian runs log table not available: {e}")

    # Tab 4: Temporal Changes & Timeline
    with tab4:
        st.subheader("Chronological Event Log")
        try:
            nodes_timeline = pd.read_sql_query(
                "SELECT created_at as timestamp, 'Entity Created' as event_type, name || ' (' || node_type || ')' as detail FROM nodes",
                conn,
            )
            obs_timeline = pd.read_sql_query(
                "SELECT o.created_at as timestamp, 'Observation Recorded' as event_type, n.name || ': ' || o.content as detail FROM observations o JOIN nodes n ON o.entity_id = n.id",
                conn,
            )

            timeline_all = pd.concat([nodes_timeline, obs_timeline]).sort_values(
                by="timestamp", ascending=False
            )
            if timeline_all.empty:
                st.warning("Timeline is empty.")
            else:
                st.dataframe(timeline_all)
        except Exception as e:
            st.error(f"Error loading timeline: {e}")

    # Tab 5: Bottles & Notes
    with tab5:
        st.subheader("Inter-session Message Bottles")
        try:
            bottles_df = pd.read_sql_query(
                "SELECT * FROM bottles ORDER BY created_at DESC", conn
            )
            if bottles_df.empty:
                st.info("No message bottles in the database.")
            else:
                col1, col2 = st.columns(2)
                active_b = len(bottles_df[bottles_df["acknowledged"] == 0])
                col1.metric("Active Bottles", active_b)
                col2.metric("Acknowledged Bottles", len(bottles_df) - active_b)

                st.dataframe(bottles_df)
        except Exception as e:
            st.error(f"Bottles table not available yet: {e}")

    conn.close()
st.sidebar.markdown("---")
st.sidebar.markdown("© 2026 Turbovec-MCP-Server")
