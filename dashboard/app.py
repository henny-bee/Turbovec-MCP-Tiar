import os
import json
import sqlite3
import datetime
import pandas as pd

try:
    import streamlit as st
    import streamlit.components.v1 as components
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

# HTML and JS template for Vis.js interactive graph
HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Memory Graph Overview</title>
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style type="text/css">
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin: 0;
      padding: 0;
      background-color: #0e1117;
      color: #fafafa;
    }
    .container {
      display: flex;
      height: 800px;
      border: 1px solid #31333f;
      border-radius: 8px;
      overflow: hidden;
      background-color: #0e1117;
    }
    #mynetwork {
      flex: 7;
      height: 100%;
      background-color: #0e1117;
    }
    #details-panel {
      flex: 3;
      height: 100%;
      border-left: 1px solid #31333f;
      padding: 20px;
      overflow-y: auto;
      background-color: #0e1117;
    }
    h3 {
      margin-top: 0;
      color: #fafafa;
      border-bottom: 2px solid #31333f;
      padding-bottom: 8px;
    }
    h4 {
      margin-top: 20px;
      margin-bottom: 10px;
      color: #fafafa;
      border-bottom: 1px solid #31333f;
      padding-bottom: 4px;
    }
    .badge {
      display: inline-block;
      padding: 0.35em 0.65em;
      font-size: 75%;
      font-weight: 700;
      line-height: 1;
      text-align: center;
      white-space: nowrap;
      vertical-align: baseline;
      border-radius: 0.25rem;
      margin-right: 5px;
    }
    .badge-type {
      background-color: #17a2b8;
      color: white;
    }
    .badge-status {
      background-color: #28a745;
      color: white;
    }
    .badge-certainty {
      background-color: #ffc107;
      color: #212529;
    }
    .observation-item {
      padding: 10px 12px;
      margin-bottom: 8px;
      background-color: #262730;
      border-left: 4px solid #007bff;
      border-radius: 0 6px 6px 0;
      font-size: 0.9em;
      box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }
    .properties-table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      font-size: 0.85em;
    }
    .properties-table th, .properties-table td {
      border: 1px solid #31333f;
      padding: 8px;
      text-align: left;
    }
    .properties-table th {
      background-color: #262730;
      color: #fafafa;
    }
    .legend-container {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      padding: 12px;
      background-color: #0e1117;
      border-bottom: 1px solid #31333f;
      font-size: 0.82em;
      align-items: center;
    }
    .legend-title {
      font-weight: bold;
      color: #fafafa;
      margin-right: 5px;
    }
    .legend-item {
      display: flex;
      align-items: center;
      gap: 5px;
      color: #fafafa;
    }
    .legend-color {
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 3px;
      border: 1.5px solid;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-all;
      font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
    }
    .vis-network .vis-navigation .vis-button {
      background-color: rgba(255, 255, 255, 0.7);
      border-radius: 50%;
    }
    .vis-network .vis-navigation .vis-button:hover {
      background-color: rgba(255, 255, 255, 1);
      box-shadow: 0 0 5px rgba(255,255,255,0.8);
    }
  </style>
</head>
<body>
  <div class="legend-container">
    <div class="legend-title">Entity Type Colors:</div>
    <div class="legend-item"><span class="legend-color" style="background:#ffebee; border-color:#d32f2f;"></span>breakthrough</div>
    <div class="legend-item"><span class="legend-color" style="background:#e1f5fe; border-color:#0288d1;"></span>concept</div>
    <div class="legend-item"><span class="legend-color" style="background:#fff3e0; border-color:#f57c00;"></span>decision</div>
    <div class="legend-item"><span class="legend-color" style="background:#e0f2f1; border-color:#00796b;"></span>document</div>
    <div class="legend-item"><span class="legend-color" style="background:#e8f5e9; border-color:#388e3c;"></span>entity</div>
    <div class="legend-item"><span class="legend-color" style="background:#fffde7; border-color:#fbc02d;"></span>event</div>
    <div class="legend-item"><span class="legend-color" style="background:#eceff1; border-color:#455a64;"></span>file</div>
    <div class="legend-item"><span class="legend-color" style="background:#fce4ec; border-color:#c2185b;"></span>person</div>
    <div class="legend-item"><span class="legend-color" style="background:#f3e5f5; border-color:#7b1fa2;"></span>preference</div>
    <div class="legend-item"><span class="legend-color" style="background:#efebe9; border-color:#5d4037;"></span>project</div>
    <div class="legend-item"><span class="legend-color" style="background:#e0f7fa; border-color:#00838f;"></span>session</div>
    <div class="legend-item"><span class="legend-color" style="background:#e8eaf6; border-color:#3f51b5;"></span>technology</div>
  </div>

  <div class="container">
    <div id="mynetwork"></div>
    <div id="details-panel">
      <div id="details-content">
        <h3 style="color: #7f8c8d; border-bottom: none; text-align: center; margin-top: 50px;">No Entity Selected</h3>
        <p style="color: #7f8c8d; text-align: center;">Click on any node in the graph to view its saved memories, relationships, and observations.</p>
      </div>
    </div>
  </div>

  <script type="text/javascript">
    // Parse the JSON data injected from Python
    var nodesData = /*NODES_JSON*/;
    var edgesData = /*EDGES_JSON*/;
    var observationsData = /*OBS_JSON*/;

    // Prepare vis nodes with custom styling based on node type
    var colors = {
      "breakthrough": { background: "#ffebee", border: "#d32f2f" },
      "concept": { background: "#e1f5fe", border: "#0288d1" },
      "decision": { background: "#fff3e0", border: "#f57c00" },
      "document": { background: "#e0f2f1", border: "#00796b" },
      "entity": { background: "#e8f5e9", border: "#388e3c" },
      "event": { background: "#fffde7", border: "#fbc02d" },
      "file": { background: "#eceff1", border: "#455a64" },
      "person": { background: "#fce4ec", border: "#c2185b" },
      "preference": { background: "#f3e5f5", border: "#7b1fa2" },
      "project": { background: "#efebe9", border: "#5d4037" },
      "session": { background: "#e0f7fa", border: "#00838f" },
      "technology": { background: "#e8eaf6", border: "#3f51b5" }
    };

    var defaultColor = { background: "#f8f9fa", border: "#adb5bd" };

    var formattedNodes = nodesData.map(function(node) {
      var typeColor = colors[node.node_type] || defaultColor;
      return {
        id: node.id,
        label: node.name,
        color: {
          background: typeColor.background,
          border: typeColor.border,
          highlight: {
            background: typeColor.background,
            border: "#111111"
          }
        },
        font: { color: "#212529", size: 13, face: "-apple-system, BlinkMacSystemFont, Segoe UI" },
        shape: "box",
        margin: 10,
        borderWidth: 2,
        title: "Type: " + node.node_type + " | Status: " + node.status
      };
    });

    var formattedEdges = edgesData.map(function(edge) {
      return {
        from: edge.from_node_id,
        to: edge.to_node_id,
        label: edge.relationship_type,
        font: { align: "middle", size: 10, color: "#555555", face: "-apple-system, BlinkMacSystemFont, Segoe UI" },
        arrows: "to",
        color: { color: "#ced4da", highlight: "#16a085" },
        width: 1.5 + (edge.weight || 1.0) * 0.5
      };
    });

    // create a network
    var container = document.getElementById("mynetwork");
    var data = {
      nodes: new vis.DataSet(formattedNodes),
      edges: new vis.DataSet(formattedEdges)
    };
    var options = {
      physics: {
        forceAtlas2Based: {
          gravitationalConstant: -70,
          centralGravity: 0.015,
          springLength: 120,
          springConstant: 0.05
        },
        solver: "forceAtlas2Based",
        stabilization: {
          iterations: 100,
          fit: true
        }
      },
      interaction: {
        hover: true,
        tooltipDelay: 100,
        navigationButtons: true,
        keyboard: true
      }
    };
    var network = new vis.Network(container, data, options);

    // Event listener for clicking a node
    network.on("click", function (params) {
      if (params.nodes.length > 0) {
        var nodeId = params.nodes[0];
        showNodeDetails(nodeId);
      } else {
        resetDetailsPanel();
      }
    });

    function showNodeDetails(nodeId) {
      var node = nodesData.find(function(n) { return n.id === nodeId; });
      if (!node) return;

      var nodeObs = observationsData[nodeId] || [];
      
      var html = "<h3>" + escapeHtml(node.name) + "</h3>";
      html += "<div style='margin-bottom: 15px;'>";
      html += "<span class='badge badge-type'>" + escapeHtml(node.node_type) + "</span>";
      html += "<span class='badge badge-status' style='background-color: " + (node.status === "ACTIVE" ? "#28a745" : "#6c757d") + ";'>" + escapeHtml(node.status) + "</span>";
      if (node.certainty) {
        html += "<span class='badge badge-certainty'>" + escapeHtml(node.certainty) + "</span>";
      }
      html += "</div>";

      html += "<p style='margin: 6px 0;'><strong>ID:</strong> <code style='font-size:0.85em; background:#000000; padding:2px 4px; border-radius:3px;'>" + escapeHtml(node.id) + "</code></p>";
      html += "<p style='margin: 6px 0;'><strong>Certainty:</strong> " + escapeHtml(node.certainty || "confirmed") + "</p>";
      html += "<p style='margin: 6px 0;'><strong>Salience Score:</strong> " + (node.salience_score !== undefined ? node.salience_score : "1.0") + "</p>";
      if (node.retrieval_count !== undefined) {
        html += "<p style='margin: 6px 0;'><strong>Retrieval Count:</strong> " + node.retrieval_count + "</p>";
      }

      // Properties
      var props = {};
      try {
        if (node.properties) {
          props = typeof node.properties === 'string' ? JSON.parse(node.properties) : node.properties;
        }
      } catch(e) {
        console.error("Error parsing node properties", e);
      }

      if (Object.keys(props).length > 0) {
        html += "<h4>Node Properties</h4>";
        html += "<table class='properties-table'>";
        html += "<thead><tr><th>Property</th><th>Value</th></tr></thead><tbody>";
        for (var key in props) {
          if (props.hasOwnProperty(key)) {
            var val = props[key];
            var displayVal = typeof val === 'object' ? JSON.stringify(val, null, 2) : String(val);
            html += "<tr><td style='width:30%; word-break:break-all;'><strong>" + escapeHtml(key) + "</strong></td><td><pre>" + escapeHtml(displayVal) + "</pre></td></tr>";
          }
        }
        html += "</tbody></table>";
      }

      // Observations
      html += "<h4>Saved Observations & Memories</h4>";
      if (nodeObs.length === 0) {
        html += "<p style='color:#7f8c8d; font-style:italic;'>No contextual observations saved for this entity.</p>";
      } else {
        nodeObs.forEach(function(obs) {
          html += "<div class='observation-item'>";
          html += "<div>" + escapeHtml(obs.content) + "</div>";
          html += "<div style='font-size: 0.75em; color: #7f8c8d; margin-top: 6px; text-align: right;'>Certainty: " + escapeHtml(obs.certainty || "confirmed") + " | " + escapeHtml(obs.created_at || "") + "</div>";
          html += "</div>";
        });
      }

      document.getElementById("details-content").innerHTML = html;
    }

    function resetDetailsPanel() {
      document.getElementById("details-content").innerHTML = 
        "<h3 style='color: #7f8c8d; border-bottom: none; text-align: center; margin-top: 50px;'>No Entity Selected</h3>" +
        "<p style='color: #7f8c8d; text-align: center;'>Click on any node in the graph to view its saved memories, relationships, and observations.</p>";
    }

    function escapeHtml(str) {
      if (typeof str !== 'string') return String(str);
      return str
        .split('&').join('&' + 'amp;')
        .split('<').join('&' + 'lt;')
        .split('>').join('&' + 'gt;')
        .split('"').join('&' + 'quot;')
        .split("'").join('&' + '#039;');
    }
  </script>
</body>
</html>
"""

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
            "Interactive Graph & Topology",
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

    # Tab 2: Graph Topology (Interactive overview)
    with tab2:
        st.subheader("🧬 Interactive Memory Graph & Relationships")
        st.markdown(
            "Explore nodes representing saved entities/concepts, and directed edges representing their relationships. "
            "**Click on any node** to inspect its full properties, certainty scores, and specific saved observations/memories."
        )

        try:
            # Load basic tables
            nodes_data = pd.read_sql_query(
                "SELECT id, name, node_type, status, created_at, certainty, salience_score, retrieval_count, properties FROM nodes",
                conn,
            )
            edges_data = pd.read_sql_query(
                "SELECT id, from_node_id, to_node_id, relationship_type, confidence, weight, created_at, properties FROM edges",
                conn,
            )
            obs_data = pd.read_sql_query(
                "SELECT id, entity_id, content, certainty, created_at FROM observations",
                conn,
            )

            # --- FILTERS PANEL ---
            st.markdown("---")
            with st.expander("🔍 Filter & Render Controls", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    search_query = st.text_input("Search Node Name:", value="")
                    hide_orphans = st.checkbox(
                        "Hide Orphan Nodes (show only nodes with relationships)",
                        value=False,
                    )
                with col2:
                    unique_types = (
                        sorted(list(nodes_data["node_type"].unique()))
                        if not nodes_data.empty
                        else []
                    )
                    selected_types = st.multiselect(
                        "Filter by Entity Type:",
                        options=unique_types,
                        default=unique_types,
                    )
                with col3:
                    unique_statuses = (
                        sorted(list(nodes_data["status"].unique()))
                        if not nodes_data.empty
                        else ["ACTIVE"]
                    )
                    selected_statuses = st.multiselect(
                        "Filter by Status:",
                        options=unique_statuses,
                        default=["ACTIVE"],  # default to ACTIVE to reduce noise
                    )
                    max_nodes = st.slider(
                        "Max Nodes to Render in Graph:",
                        min_value=10,
                        max_value=500,
                        value=150,
                    )

            # Apply filters to nodes
            filtered_nodes = nodes_data.copy()
            if search_query:
                filtered_nodes = filtered_nodes[
                    filtered_nodes["name"].str.contains(
                        search_query, case=False, na=False
                    )
                ]
            if selected_types:
                filtered_nodes = filtered_nodes[
                    filtered_nodes["node_type"].isin(selected_types)
                ]
            if selected_statuses:
                filtered_nodes = filtered_nodes[
                    filtered_nodes["status"].isin(selected_statuses)
                ]

            # Filter edges to connect only filtered nodes
            filtered_edges = edges_data[
                edges_data["from_node_id"].isin(filtered_nodes["id"])
                & edges_data["to_node_id"].isin(filtered_nodes["id"])
            ]

            # Hide orphans if requested
            if hide_orphans and not filtered_edges.empty:
                connected_ids = set(filtered_edges["from_node_id"]).union(
                    set(filtered_edges["to_node_id"])
                )
                filtered_nodes = filtered_nodes[
                    filtered_nodes["id"].isin(connected_ids)
                ]
            elif hide_orphans and filtered_edges.empty:
                # If there are no edges, show empty nodes
                filtered_nodes = filtered_nodes.head(0)

            # Apply head limit
            filtered_nodes = filtered_nodes.head(max_nodes)

            # Final check of edges to align with limited nodes
            filtered_edges = filtered_edges[
                filtered_edges["from_node_id"].isin(filtered_nodes["id"])
                & filtered_edges["to_node_id"].isin(filtered_nodes["id"])
            ]

            # Group observations by entity_id
            obs_dict = {}
            if not obs_data.empty:
                for _, row in obs_data.iterrows():
                    e_id = row["entity_id"]
                    if e_id not in obs_dict:
                        obs_dict[e_id] = []
                    obs_dict[e_id].append(
                        {
                            "content": row["content"],
                            "certainty": row["certainty"],
                            "created_at": row["created_at"],
                        }
                    )

            # Convert filtered sets to JSON
            nodes_json = filtered_nodes.to_json(orient="records")
            edges_json = filtered_edges.to_json(orient="records")
            obs_json = json.dumps(obs_dict)

            # Inline visualization component
            st.markdown("### 🕸️ Interactive Visualization")

            # Construct HTML from template
            html_content = (
                HTML_TEMPLATE.replace("/*NODES_JSON*/", nodes_json)
                .replace("/*EDGES_JSON*/", edges_json)
                .replace("/*OBS_JSON*/", obs_json)
            )

            # Render in Streamlit
            components.html(html_content, height=880, scrolling=False)

            st.markdown("---")
            st.subheader("📊 Raw Database Data & Tables")

            # Nodes Table
            with st.expander("📁 Raw Entities Table (Nodes)", expanded=False):
                st.dataframe(nodes_data)
                # Node types distribution chart
                type_counts = nodes_data["node_type"].value_counts()
                st.bar_chart(type_counts)

            # Edges Table
            with st.expander("🔗 Raw Relationships Table (Edges)", expanded=False):
                st.dataframe(edges_data)

            # Observations Table
            with st.expander("📝 Raw Contextual Observations Table", expanded=False):
                st.dataframe(obs_data)

        except Exception as e:
            st.error(f"Error loading topology data: {e}")
            import traceback

            st.exception(e)

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
