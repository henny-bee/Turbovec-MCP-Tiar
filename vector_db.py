import os
import json
import logging
import sqlite3
import threading
import hashlib
import datetime
import uuid
import time
import numpy as np

# We still need to configure the environment before importing SentenceTransformers
# to prevent it from outputting to stdout/stderr.
# This might be handled in main.py, but it's safe to do the import inside the class
# or ensure main.py imports vector_db after setting up the environment.

import turbovec
from sentence_transformers import SentenceTransformer

# Import auxiliary modules for enhanced reliability, temporal features, ontology, and librarian
from memory.errors import (
    SearchError,
    MEMORY_LAYER_DEGRADED,
    SQLITE_UNAVAILABLE,
    SQLITE_TRANSACTION_FAILED,
    VECTOR_INDEX_UNAVAILABLE,
    VECTOR_INDEX_CORRUPTED,
    EMBEDDING_FAILED,
    RERANKER_FAILED,
)
from memory.ontology import OntologyManager
from embeddings.minilm import MiniLMProvider
from reranking.cross_encoder import CrossEncoderReranker
from telemetry.metrics import SearchTelemetry
from memory.provenance import BottleManager
from graph.analytics import GraphAnalytics
from librarian.librarian import LibrarianService
from memory.temporal import (
    diff_knowledge_state,
    query_timeline,
    get_temporal_neighbors,
)
from memory.lifecycle import (
    archive_node,
    restore_node,
    list_orphans,
    prune_stale,
)

logger = logging.getLogger(__name__)


class VectorDB:
    def __init__(
        self,
        dimension: int = 384,
        metadata_file: str = "metadata.json",
        index_file: str = "index.tvim",
        sqlite_db_file: str = "memory.db",
    ):
        self.dimension = dimension
        self.metadata_file = metadata_file
        self.index_file = index_file

        # If sqlite_db_file is the default 'memory.db', place it in the same directory as metadata_file
        if sqlite_db_file == "memory.db" and os.path.dirname(metadata_file):
            self.sqlite_db_file = os.path.join(
                os.path.dirname(metadata_file), "memory.db"
            )
        else:
            self.sqlite_db_file = sqlite_db_file

        logger.info("Initializing SentenceTransformer model: all-MiniLM-L6-v2")
        self.ontology_manager = OntologyManager()
        self.embedding_provider = MiniLMProvider()
        self.model = self.embedding_provider._model
        self.index = turbovec.IdMapIndex(self.dimension)
        self.document_store = {}
        self.next_id = 0

        self._local = threading.local()
        self._init_sqlite_db()
        self.load_storage()

        # Initialize telemetry, bottles, analytics, and librarian services
        self.telemetry = SearchTelemetry(self.sqlite_db_file)
        self.bottle_manager = BottleManager(self)
        self.analytics = GraphAnalytics(self)
        self.librarian = LibrarianService(self)
        self.reranker = CrossEncoderReranker(enabled=False)

        # Background relationship discovery configuration
        self.bg_thread = None
        self.bg_stop_event = threading.Event()
        self.bg_interval = int(os.getenv("BG_DISCOVERY_INTERVAL", "300"))
        self.bg_enabled = os.getenv("BACKGROUND_DISCOVERY", "false").lower() == "true"
        self.bg_similarity_threshold = float(os.getenv("BG_DISCOVERY_THRESHOLD", "0.7"))

        if self.bg_enabled:
            self.start_background_discovery()

    @property
    def conn(self) -> sqlite3.Connection:
        """Returns the thread-local SQLite connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = self._init_connection()
        return self._local.conn

    def _init_connection(self) -> sqlite3.Connection:
        """Initializes a new SQLite connection with WAL, synchronous normal, and foreign keys ON."""
        conn = sqlite3.connect(self.sqlite_db_file, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_sqlite_db(self):
        """Initializes SQLite database and tables/indexes."""
        conn = self.conn
        with conn:
            # Create nodes table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    node_type TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    occurred_at TEXT,
                    certainty TEXT NOT NULL DEFAULT 'confirmed',
                    salience_score REAL NOT NULL DEFAULT 1.0,
                    retrieval_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    properties TEXT
                );
                """)
            # Schema Migration Check for status column
            try:
                conn.execute(
                    "ALTER TABLE nodes ADD COLUMN status TEXT NOT NULL DEFAULT 'ACTIVE';"
                )
                logger.info(
                    "Migrated SQLite schema: added 'status' column to 'nodes' table."
                )
            except sqlite3.OperationalError:
                pass  # status column already exists
            # Create edges table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    id TEXT PRIMARY KEY,
                    from_node_id TEXT NOT NULL,
                    to_node_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    weight REAL NOT NULL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    properties TEXT,
                    FOREIGN KEY (from_node_id) REFERENCES nodes(id) ON DELETE CASCADE,
                    FOREIGN KEY (to_node_id) REFERENCES nodes(id) ON DELETE CASCADE
                );
                """)
            # Create observations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS observations (
                    id TEXT PRIMARY KEY,
                    entity_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    certainty TEXT NOT NULL DEFAULT 'confirmed',
                    created_at TEXT NOT NULL,
                    occurred_at TEXT,
                    FOREIGN KEY (entity_id) REFERENCES nodes(id) ON DELETE CASCADE
                );
                """)
            # Create entities_fts virtual table (FTS5)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
                    entity_id UNINDEXED,
                    project_id UNINDEXED,
                    name,
                    node_type,
                    description,
                    observations,
                    tokenize='porter unicode61'
                );
                """)
            # Create vector_id_mapping table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vector_id_mapping (
                    vector_id INTEGER PRIMARY KEY,
                    node_id TEXT UNIQUE NOT NULL,
                    FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
                );
                """)
            # Create indexes
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_nodes_project_type ON nodes(project_id, node_type);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_nodes_created_at ON nodes(created_at);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_node_id);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_node_id);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(relationship_type);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_observations_entity ON observations(entity_id);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_observations_created_at ON observations(created_at);"
            )

    def _to_signed_64(self, val: int) -> int:
        """Converts an unsigned 64-bit integer to a signed 64-bit integer for SQLite."""
        val = val & 0xFFFFFFFFFFFFFFFF
        if val >= 0x8000000000000000:
            return val - 0x10000000000000000
        return val

    def _to_unsigned_64(self, val: int) -> int:
        """Converts a signed 64-bit integer from SQLite back to an unsigned 64-bit integer."""
        return val & 0xFFFFFFFFFFFFFFFF

    def generate_vector_id(self, node_id: str, conn: sqlite3.Connection) -> int:
        """
        Computes a candidate uint64 ID by taking the first 8 bytes of the SHA-256 hash of node_id.
        If a collision occurs in vector_id_mapping, increments the candidate ID until an unused slot is found.
        """
        try:
            # Backward compatibility: if node_id is a simple integer, use it directly as the candidate vector ID
            candidate_id = int(node_id)
            if not (0 <= candidate_id <= 0xFFFFFFFFFFFFFFFF):
                raise ValueError()
        except ValueError:
            # Compute SHA-256 hash of node_id
            hash_bytes = hashlib.sha256(node_id.encode("utf-8")).digest()
            candidate_id = (
                int.from_bytes(hash_bytes[:8], byteorder="big") & 0xFFFFFFFFFFFFFFFF
            )

        cursor = conn.cursor()
        while True:
            signed_id = self._to_signed_64(candidate_id)
            cursor.execute(
                "SELECT node_id FROM vector_id_mapping WHERE vector_id = ?",
                (signed_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return candidate_id
            elif row[0] == node_id:
                return candidate_id
            else:
                candidate_id = (candidate_id + 1) & 0xFFFFFFFFFFFFFFFF

    def build_embedding_text(
        self, name: str, node_type: str, description: str, observations: list[str]
    ) -> str:
        """Single source of truth for embedding generation text."""
        max_observations = 20
        max_chars_per_obs = 500

        parts = [name, node_type, description]
        for obs in observations[:max_observations]:
            parts.append(obs[:max_chars_per_obs])

        return " ".join(p for p in parts if p)

    def _get_description(self, properties) -> str:
        """Extracts description from properties or serializes properties to a string."""
        if not properties:
            return ""
        if isinstance(properties, str):
            try:
                properties = json.loads(properties)
            except Exception:
                return properties
        if isinstance(properties, dict):
            return properties.get("description", json.dumps(properties))
        return str(properties)

    def chunk_text(
        self, text: str, chunk_size: int = 1000, overlap: int = 200
    ) -> list[str]:
        """Intelligent text chunking with overlap."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")

        chunks = []
        start = 0
        text_len = len(text)

        if text_len == 0:
            return []

        step = chunk_size - overlap

        while start < text_len:
            end = start + chunk_size
            chunks.append(text[start:end])
            if end >= text_len:
                break
            start += step

        return chunks

    def save_storage(self):
        """Saves metadata to JSON and attempts to save turbovec index."""
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(
                    {"next_id": self.next_id, "documents": self.document_store},
                    f,
                    indent=2,
                )
            logger.info(f"Metadata saved to {self.metadata_file}")
        except Exception as e:
            logger.error(
                f"Failed to save metadata to {self.metadata_file}: {e}", exc_info=True
            )

        try:
            self.index.write(self.index_file)
            logger.info(f"Index saved to {self.index_file}")
        except Exception as e:
            logger.error(
                f"Failed to save index to {self.index_file}: {e}", exc_info=True
            )

    def load_storage(self):
        """Loads metadata and index from disk at startup."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "documents" in data:
                        self.document_store = {
                            int(k): v for k, v in data["documents"].items()
                        }
                        self.next_id = data.get(
                            "next_id", max(self.document_store.keys(), default=-1) + 1
                        )
                    elif isinstance(data, list):
                        self.document_store = {
                            doc["id"]: doc
                            for doc in data
                            if doc and not doc.get("deleted")
                        }
                        self.next_id = max(self.document_store.keys(), default=-1) + 1
                logger.info(
                    f"Loaded {len(self.document_store)} documents from {self.metadata_file}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to load metadata from {self.metadata_file}: {e}",
                    exc_info=True,
                )
                self.document_store = {}
                self.next_id = 0

        loaded_index = False
        if os.path.exists(self.index_file):
            try:
                self.index = turbovec.IdMapIndex.load(self.index_file)

                if len(self.index) != len(self.document_store):
                    logger.warning(
                        "Index and metadata contain different numbers of entries"
                    )
                loaded_index = True
                logger.info(f"Loaded index from {self.index_file}")
            except Exception as e:
                logger.error(
                    f"Failed to load index from {self.index_file}: {e}", exc_info=True
                )
                self.index = turbovec.IdMapIndex(self.dimension)

        # Fallback: rebuild index if we have metadata but failed to load index file
        if not loaded_index and self.document_store:
            logger.info("Rebuilding index from document store...")
            try:
                ids = []
                vectors = []
                for doc_id, doc in self.document_store.items():
                    ids.append(doc_id)
                    vectors.append(self.model.encode(doc["content"]))

                if ids:
                    self.index.add_with_ids(
                        np.asarray(vectors, dtype=np.float32),
                        np.asarray(ids, dtype=np.uint64),
                    )
                self.save_storage()
                logger.info("Finished rebuilding index.")
            except Exception:
                self.index = turbovec.IdMapIndex(self.dimension)
                logger.error("Failed to rebuild index from metadata", exc_info=True)
                raise

        # Migration / Import check:
        # If we loaded documents from metadata.json but our nodes table is empty, import them!
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM nodes")
        if cursor.fetchone()[0] == 0 and self.document_store:
            logger.info("Migrating existing document store to SQLite...")
            try:
                with self.conn as conn:
                    for doc_id, doc in self.document_store.items():
                        node_id = str(doc_id)
                        created_at = datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat()
                        conn.execute(
                            """
                            INSERT INTO nodes (
                                id, name, node_type, project_id, created_at, updated_at, certainty, properties
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                node_id,
                                doc.get("title", "Untitled"),
                                "document",
                                "default",
                                created_at,
                                created_at,
                                "confirmed",
                                json.dumps(
                                    {
                                        "content": doc.get("content", ""),
                                        "chunk_index": doc.get("chunk_index", 0),
                                    }
                                ),
                            ),
                        )
                        desc = doc.get("content", "")
                        conn.execute(
                            """
                            INSERT INTO entities_fts (
                                entity_id, project_id, name, node_type, description, observations
                            ) VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                node_id,
                                "default",
                                doc.get("title", "Untitled"),
                                "document",
                                desc,
                                "",
                            ),
                        )
                        conn.execute(
                            "INSERT OR IGNORE INTO vector_id_mapping (vector_id, node_id) VALUES (?, ?)",
                            (self._to_signed_64(int(doc_id)), node_id),
                        )
                logger.info("Migration complete.")
            except Exception as e:
                logger.error(f"Migration failed: {e}", exc_info=True)

    def add_knowledge(self, title: str, content: str) -> str:
        if not title.strip():
            return "Error: Title must not be empty."
        if not content.strip():
            return "Error: Content must not be empty."

        chunks = self.chunk_text(content, chunk_size=1000, overlap=200)
        if not chunks:
            return "Error: No chunks generated from content."

        try:
            vectors = [
                np.asarray(self.model.encode(chunk), dtype=np.float32)
                for chunk in chunks
            ]
        except Exception as e:
            logger.error(f"Failed to encode '{title}': {e}", exc_info=True)
            return f"Error: Failed to process '{title}'."

        ids = []
        chunk_docs = []

        # Write to SQLite atomically
        try:
            with self.conn as conn:
                for i, chunk in enumerate(chunks):
                    node_id = str(uuid.uuid4())
                    doc_id = self.generate_vector_id(node_id, conn)
                    ids.append(doc_id)

                    created_at = datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat()
                    conn.execute(
                        """
                        INSERT INTO nodes (
                            id, name, node_type, project_id, created_at, updated_at, certainty, properties
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            node_id,
                            title,
                            "document",
                            "default",
                            created_at,
                            created_at,
                            "confirmed",
                            json.dumps({"content": chunk, "chunk_index": i}),
                        ),
                    )
                    conn.execute(
                        """
                        INSERT INTO entities_fts (
                            entity_id, project_id, name, node_type, description, observations
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (node_id, "default", title, "document", chunk, ""),
                    )
                    conn.execute(
                        "INSERT INTO vector_id_mapping (vector_id, node_id) VALUES (?, ?)",
                        (self._to_signed_64(doc_id), node_id),
                    )

                    chunk_docs.append(
                        {
                            "id": doc_id,
                            "title": title,
                            "chunk_index": i,
                            "content": chunk,
                        }
                    )

                # Add to FAISS index inside the transaction so SQLite rolls back if FAISS fails
                self.index.add_with_ids(
                    np.asarray(vectors, dtype=np.float32),
                    np.asarray(ids, dtype=np.uint64),
                )

        except Exception as e:
            logger.error(f"Insertion failed during add_knowledge: {e}", exc_info=True)
            return f"Error: Failed to save knowledge: {e}"

        for doc in chunk_docs:
            self.document_store[doc["id"]] = doc

        self.save_storage()

        # Run extraction pipeline on incoming text
        try:
            extracted = self.extract_entities_and_relations(content)
            cursor = self.conn.cursor()

            # Create/update extracted entities
            for ent in extracted.get("entities", []):
                cursor.execute("SELECT id FROM nodes WHERE name = ?", (ent["name"],))
                row = cursor.fetchone()
                if not row:
                    self.create_node(
                        node_id=str(uuid.uuid4()),
                        name=ent["name"],
                        node_type=ent["type"],
                        properties=ent["properties"],
                    )

            # Create relationships
            for rel in extracted.get("relationships", []):
                cursor.execute("SELECT id FROM nodes WHERE name = ?", (rel["from"],))
                from_row = cursor.fetchone()
                cursor.execute("SELECT id FROM nodes WHERE name = ?", (rel["to"],))
                to_row = cursor.fetchone()
                if from_row and to_row:
                    from_id = from_row[0]
                    to_id = to_row[0]
                    edge_id = f"edge-{from_id}-{to_id}-{rel['type']}"
                    cursor.execute(
                        "SELECT COUNT(*) FROM edges WHERE id = ?", (edge_id,)
                    )
                    if cursor.fetchone()[0] == 0:
                        self.create_edge(
                            edge_id=edge_id,
                            from_node_id=from_id,
                            to_node_id=to_id,
                            relationship_type=rel["type"],
                            properties=rel.get("properties"),
                        )

            # Create observations
            for obs in extracted.get("observations", []):
                cursor.execute(
                    "SELECT id FROM nodes WHERE name = ?", (obs["entity_name"],)
                )
                ent_row = cursor.fetchone()
                if ent_row:
                    ent_id = ent_row[0]
                    self.create_observation(
                        obs_id=str(uuid.uuid4()),
                        entity_id=ent_id,
                        content=obs["content"],
                    )
        except Exception as e:
            logger.error(
                f"Extraction pipeline failed during add_knowledge: {e}", exc_info=True
            )

        return (
            f"Successfully added '{title}' ({len(chunks)} chunks) into Turbovec memory."
        )

    def delete_knowledge(self, title_or_id: str) -> str:
        deleted_count = 0
        to_delete = []

        for doc_id, doc in self.document_store.items():
            if str(doc["id"]) == title_or_id or doc["title"] == title_or_id:
                to_delete.append(doc_id)

        for doc_id in to_delete:
            if self.index.remove(doc_id):
                del self.document_store[doc_id]
                deleted_count += 1
                # Also delete from SQLite!
                try:
                    with self.conn as conn:
                        node_id = str(doc_id)
                        conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
                        conn.execute(
                            "DELETE FROM entities_fts WHERE entity_id = ?", (node_id,)
                        )
                except Exception as e:
                    logger.error(
                        f"Failed to delete {doc_id} from SQLite: {e}", exc_info=True
                    )

        if deleted_count > 0:
            self.save_storage()
            logger.info(f"Deleted {deleted_count} chunks matching '{title_or_id}'")
            return (
                f"Successfully deleted {deleted_count} chunks matching '{title_or_id}'."
            )

        logger.info(f"No document found matching '{title_or_id}' for deletion")
        return f"No document found matching '{title_or_id}'."

    def clear_memory(self) -> str:
        self.document_store = {}
        self.next_id = 0
        self.index = turbovec.IdMapIndex(self.dimension)

        for f in [self.metadata_file, self.index_file, self.sqlite_db_file]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                    logger.info(f"Deleted file: {f}")
                except Exception as e:
                    logger.error(f"Failed to delete file {f}: {e}", exc_info=True)

        # Re-initialize local connection and tables
        if hasattr(self._local, "conn"):
            if self._local.conn is not None:
                try:
                    self._local.conn.close()
                except Exception:
                    pass
            self._local.conn = None
        self._init_sqlite_db()

        return "Memory cleared successfully."

    def optimize_index(self) -> str:
        """Permanently removes soft-deleted chunks and rebuilds the index."""
        logger.info("Starting index optimization (garbage collection)...")
        self.save_storage()
        return (
            f"Optimization complete. Current size: {len(self.document_store)} chunks."
        )

    def search_knowledge(self, query: str, top_k: int = 3) -> str:
        if not query.strip():
            return "Error: Search query must not be empty."
        if top_k <= 0:
            return "Error: top_k must be greater than zero."

        valid_count = len(self.document_store)
        if valid_count == 0:
            return "Database is still empty. Please add knowledge first using the add_knowledge tool."

        try:
            query_vector = self.model.encode(query)
            search_k = min(len(self.index), top_k + 100)
            distances, ids = self.index.search(
                np.asarray([query_vector], dtype=np.float32),
                k=search_k,
            )

            results = []
            for i, doc_id in enumerate(ids[0]):
                doc_id = int(doc_id)
                if doc_id in self.document_store:
                    doc = self.document_store[doc_id]
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
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}", exc_info=True)
            return f"Error during search: {e}"

    # ==========================================
    # Graph Memory Core CRUD Implementation
    # ==========================================

    def create_node(
        self,
        node_id: str,
        name: str,
        node_type: str,
        project_id: str = "default",
        properties: dict = None,
        certainty: str = "confirmed",
        salience_score: float = 1.0,
        occurred_at: str = None,
    ) -> dict:
        """Creates a structured node in the graph, registers its FTS and Vector representations."""
        created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        updated_at = created_at
        properties_json = json.dumps(properties or {})

        try:
            with self.conn as conn:
                # 1. Insert into nodes table
                conn.execute(
                    """
                    INSERT INTO nodes (
                        id, name, node_type, project_id, created_at, updated_at, occurred_at, certainty, salience_score, retrieval_count, properties
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        node_id,
                        name,
                        node_type,
                        project_id,
                        created_at,
                        updated_at,
                        occurred_at,
                        certainty,
                        salience_score,
                        0,
                        properties_json,
                    ),
                )

                # 2. Insert into entities_fts
                desc = self._get_description(properties)
                conn.execute(
                    """
                    INSERT INTO entities_fts (
                        entity_id, project_id, name, node_type, description, observations
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (node_id, project_id, name, node_type, desc, ""),
                )

                # 3. Generate and insert mapping
                vector_id = self.generate_vector_id(node_id, conn)
                signed_vector_id = self._to_signed_64(vector_id)
                conn.execute(
                    "INSERT INTO vector_id_mapping (vector_id, node_id) VALUES (?, ?)",
                    (signed_vector_id, node_id),
                )

                # 4. Generate embedding and add to turbovec index
                embedding_text = self.build_embedding_text(name, node_type, desc, [])
                vector = self.model.encode(embedding_text)
                self.index.add_with_ids(
                    np.asarray([vector], dtype=np.float32),
                    np.asarray([vector_id], dtype=np.uint64),
                )
                self.index.write(self.index_file)
        except Exception as e:
            # Compensating rollback for turbovec in-memory
            try:
                self.index.remove(vector_id)
            except Exception:
                pass
            logger.error(f"Failed to create node {node_id}: {e}", exc_info=True)
            raise e

        return {
            "id": node_id,
            "name": name,
            "node_type": node_type,
            "project_id": project_id,
            "created_at": created_at,
            "updated_at": updated_at,
            "occurred_at": occurred_at,
            "certainty": certainty,
            "salience_score": salience_score,
            "properties": properties or {},
        }

    def update_node(
        self,
        node_id: str,
        name: str = None,
        node_type: str = None,
        properties: dict = None,
        certainty: str = None,
        salience_score: float = None,
        occurred_at: str = None,
    ) -> dict:
        """Updates a node's attributes, synchronizes with FTS and Vector representations."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM nodes WHERE id = ?", (node_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Node {node_id} not found")

        current_name = row[1]
        current_node_type = row[2]
        project_id = row[3]
        created_at = row[4]
        current_occurred_at = row[6]
        current_certainty = row[7]
        current_salience_score = row[8]
        current_properties = json.loads(row[11]) if row[11] else {}

        new_name = name if name is not None else current_name
        new_node_type = node_type if node_type is not None else current_node_type
        new_certainty = certainty if certainty is not None else current_certainty
        new_salience_score = (
            salience_score if salience_score is not None else current_salience_score
        )
        new_occurred_at = (
            occurred_at if occurred_at is not None else current_occurred_at
        )

        new_properties = dict(current_properties)
        if properties is not None:
            new_properties.update(properties)
        properties_json = json.dumps(new_properties)

        updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

        try:
            with self.conn as conn:
                # Update nodes
                conn.execute(
                    """
                    UPDATE nodes SET
                        name = ?,
                        node_type = ?,
                        updated_at = ?,
                        occurred_at = ?,
                        certainty = ?,
                        salience_score = ?,
                        properties = ?
                    WHERE id = ?
                    """,
                    (
                        new_name,
                        new_node_type,
                        updated_at,
                        new_occurred_at,
                        new_certainty,
                        new_salience_score,
                        properties_json,
                        node_id,
                    ),
                )

                # Sync FTS and embedding
                self._sync_node_fts_and_embedding(node_id, conn)
        except Exception as e:
            logger.error(f"Failed to update node {node_id}: {e}", exc_info=True)
            raise e

        return {
            "id": node_id,
            "name": new_name,
            "node_type": new_node_type,
            "project_id": project_id,
            "created_at": created_at,
            "updated_at": updated_at,
            "occurred_at": new_occurred_at,
            "certainty": new_certainty,
            "salience_score": new_salience_score,
            "properties": new_properties,
        }

    def delete_node(self, node_id: str) -> bool:
        """Deletes a node from the graph, cascading deletes to edges, observations, and vectors."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT vector_id FROM vector_id_mapping WHERE node_id = ?", (node_id,)
        )
        v_row = cursor.fetchone()
        if not v_row:
            return False
        vector_id = self._to_unsigned_64(v_row[0])

        try:
            with self.conn as conn:
                # Delete from nodes (cascades to edges, observations, vector_id_mapping)
                conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
                # Delete from entities_fts
                conn.execute("DELETE FROM entities_fts WHERE entity_id = ?", (node_id,))

                # Remove from turbovec index
                self.index.remove(vector_id)
                self.index.write(self.index_file)
        except Exception as e:
            logger.error(f"Failed to delete node {node_id}: {e}", exc_info=True)
            raise e
        return True

    def create_edge(
        self,
        edge_id: str,
        from_node_id: str,
        to_node_id: str,
        relationship_type: str,
        confidence: float = 1.0,
        weight: float = 1.0,
        properties: dict = None,
    ) -> dict:
        """Creates a directed relationship edge between two nodes."""
        self.ontology_manager.validate_relation_type(relationship_type)
        created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        properties_json = json.dumps(properties or {})

        try:
            with self.conn as conn:
                conn.execute(
                    """
                    INSERT INTO edges (
                        id, from_node_id, to_node_id, relationship_type, confidence, weight, created_at, properties
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        edge_id,
                        from_node_id,
                        to_node_id,
                        relationship_type,
                        confidence,
                        weight,
                        created_at,
                        properties_json,
                    ),
                )
        except Exception as e:
            logger.error(f"Failed to create edge {edge_id}: {e}", exc_info=True)
            raise e

        return {
            "id": edge_id,
            "from_node_id": from_node_id,
            "to_node_id": to_node_id,
            "relationship_type": relationship_type,
            "confidence": confidence,
            "weight": weight,
            "created_at": created_at,
            "properties": properties or {},
        }

    def update_edge(
        self,
        edge_id: str,
        confidence: float = None,
        weight: float = None,
        properties: dict = None,
    ) -> dict:
        """Updates an edge's weights and properties."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM edges WHERE id = ?", (edge_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Edge {edge_id} not found")

        from_node_id = row[1]
        to_node_id = row[2]
        relationship_type = row[3]
        current_confidence = row[4]
        current_weight = row[5]
        created_at = row[6]
        current_properties = json.loads(row[7]) if row[7] else {}

        new_confidence = confidence if confidence is not None else current_confidence
        new_weight = weight if weight is not None else current_weight
        new_properties = dict(current_properties)
        if properties is not None:
            new_properties.update(properties)
        properties_json = json.dumps(new_properties)

        try:
            with self.conn as conn:
                conn.execute(
                    """
                    UPDATE edges SET
                        confidence = ?,
                        weight = ?,
                        properties = ?
                    WHERE id = ?
                    """,
                    (new_confidence, new_weight, properties_json, edge_id),
                )
        except Exception as e:
            logger.error(f"Failed to update edge {edge_id}: {e}", exc_info=True)
            raise e

        return {
            "id": edge_id,
            "from_node_id": from_node_id,
            "to_node_id": to_node_id,
            "relationship_type": relationship_type,
            "confidence": new_confidence,
            "weight": new_weight,
            "created_at": created_at,
            "properties": new_properties,
        }

    def delete_edge(self, edge_id: str) -> bool:
        """Deletes an edge from the graph."""
        try:
            with self.conn as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
                if cursor.rowcount == 0:
                    return False
        except Exception as e:
            logger.error(f"Failed to delete edge {edge_id}: {e}", exc_info=True)
            raise e
        return True

    def create_observation(
        self,
        obs_id: str,
        entity_id: str,
        content: str,
        certainty: str = "confirmed",
        occurred_at: str = None,
    ) -> dict:
        """Creates a contextual observation linked to an entity, and updates FTS and Vector indices."""
        created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

        try:
            with self.conn as conn:
                # 1. Insert observation
                conn.execute(
                    """
                    INSERT INTO observations (
                        id, entity_id, content, certainty, created_at, occurred_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (obs_id, entity_id, content, certainty, created_at, occurred_at),
                )

                # 2. Update the node's embedding and FTS
                self._sync_node_fts_and_embedding(entity_id, conn)
        except Exception as e:
            logger.error(f"Failed to create observation {obs_id}: {e}", exc_info=True)
            raise e

        return {
            "id": obs_id,
            "entity_id": entity_id,
            "content": content,
            "certainty": certainty,
            "created_at": created_at,
            "occurred_at": occurred_at,
        }

    def update_observation(
        self,
        obs_id: str,
        content: str = None,
        certainty: str = None,
        occurred_at: str = None,
    ) -> dict:
        """Updates an observation and synchronizes the parent entity's FTS and Vector indices."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM observations WHERE id = ?", (obs_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Observation {obs_id} not found")

        entity_id = row[1]
        current_content = row[2]
        current_certainty = row[3]
        created_at = row[4]
        current_occurred_at = row[5]

        new_content = content if content is not None else current_content
        new_certainty = certainty if certainty is not None else current_certainty
        new_occurred_at = (
            occurred_at if occurred_at is not None else current_occurred_at
        )

        try:
            with self.conn as conn:
                conn.execute(
                    """
                    UPDATE observations SET
                        content = ?,
                        certainty = ?,
                        occurred_at = ?
                    WHERE id = ?
                    """,
                    (new_content, new_certainty, new_occurred_at, obs_id),
                )

                # Sync FTS and embedding
                self._sync_node_fts_and_embedding(entity_id, conn)
        except Exception as e:
            logger.error(f"Failed to update observation {obs_id}: {e}", exc_info=True)
            raise e

        return {
            "id": obs_id,
            "entity_id": entity_id,
            "content": new_content,
            "certainty": new_certainty,
            "created_at": created_at,
            "occurred_at": new_occurred_at,
        }

    def delete_observation(self, obs_id: str) -> bool:
        """Deletes an observation and synchronizes the parent entity's FTS and Vector indices."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT entity_id FROM observations WHERE id = ?", (obs_id,))
        row = cursor.fetchone()
        if not row:
            return False
        entity_id = row[0]

        try:
            with self.conn as conn:
                conn.execute("DELETE FROM observations WHERE id = ?", (obs_id,))
                # Sync FTS and embedding
                self._sync_node_fts_and_embedding(entity_id, conn)
        except Exception as e:
            logger.error(f"Failed to delete observation {obs_id}: {e}", exc_info=True)
            raise e
        return True

    def _sync_node_fts_and_embedding(self, node_id: str, conn: sqlite3.Connection):
        """
        Fetches the node and all of its observations, updates its entities_fts row,
        and updates its turbovec vector representation.
        """
        # 1. Fetch node
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name, node_type, properties FROM nodes WHERE id = ?", (node_id,)
        )
        node_row = cursor.fetchone()
        if not node_row:
            raise ValueError(f"Node {node_id} not found during sync")
        name, node_type, properties_json = node_row
        properties = json.loads(properties_json) if properties_json else {}

        # 2. Fetch all observations
        cursor.execute(
            "SELECT content FROM observations WHERE entity_id = ? ORDER BY created_at DESC",
            (node_id,),
        )
        obs_rows = cursor.fetchall()
        observations = [r[0] for r in obs_rows]

        # 3. Update entities_fts
        desc = self._get_description(properties)
        obs_concat = " ".join(observations)
        conn.execute(
            """
            UPDATE entities_fts SET
                name = ?,
                node_type = ?,
                description = ?,
                observations = ?
            WHERE entity_id = ?
            """,
            (name, node_type, desc, obs_concat, node_id),
        )

        # 4. Fetch vector_id
        cursor.execute(
            "SELECT vector_id FROM vector_id_mapping WHERE node_id = ?", (node_id,)
        )
        v_row = cursor.fetchone()
        if not v_row:
            raise ValueError(f"Vector mapping not found for node {node_id}")
        vector_id = self._to_unsigned_64(v_row[0])

        # 5. Generate new embedding
        embedding_text = self.build_embedding_text(name, node_type, desc, observations)
        vector = self.model.encode(embedding_text)

        # 6. Update turbovec index
        self.index.remove(vector_id)
        self.index.add_with_ids(
            np.asarray([vector], dtype=np.float32),
            np.asarray([vector_id], dtype=np.uint64),
        )
        self.index.write(self.index_file)

    def extract_entities_and_relations(self, text: str) -> dict:
        """
        Extracts entities, relationships, and observations from unstructured text.
        Attempts to use spaCy (en_core_web_sm) if installed.
        Falls back to a robust, deterministic Python-based rule/regex extractor.
        """
        import re

        entities = {}  # name -> {"name": str, "type": str, "properties": dict}
        relationships = []
        observations = []

        # Try to load spaCy
        nlp = None
        try:
            import spacy

            # Lazy-load to avoid slow imports on start
            nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.info(
                f"spaCy not available or failed to load: {e}. Using regex fallback."
            )

        # Common stop words to ignore as standalone fallback named entities
        STOP_WORDS = {
            "I",
            "The",
            "A",
            "An",
            "They",
            "He",
            "She",
            "It",
            "We",
            "You",
            "But",
            "And",
            "Or",
            "If",
            "Then",
            "Else",
            "When",
            "Where",
            "Why",
            "How",
            "This",
            "That",
            "These",
            "Those",
            "My",
            "Your",
            "His",
            "Her",
            "Its",
            "Our",
            "Their",
        }

        # 1. Base Entity Extraction
        if nlp is not None:
            try:
                doc = nlp(text)
                for ent in doc.ents:
                    name = ent.text.strip()
                    if not name or len(name) < 2:
                        continue
                    ent_type = ent.label_
                    if name in STOP_WORDS:
                        continue
                    if name not in entities:
                        entities[name] = {
                            "name": name,
                            "type": ent_type,
                            "properties": {"description": f"Extracted as {ent_type}"},
                        }
            except Exception as e:
                logger.error(f"Error during spaCy extraction: {e}")
                nlp = None  # Force fallback if spaCy fails during processing

        if nlp is None:
            # Fallback named entity extraction using regex (capitalized phrases/nouns)
            ne_pattern = re.compile(r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b")
            for match in ne_pattern.finditer(text):
                name = match.group(0).strip()
                if name in STOP_WORDS or len(name) < 2:
                    continue
                if name not in entities:
                    entities[name] = {
                        "name": name,
                        "type": "entity",
                        "properties": {"description": "Extracted named entity"},
                    }

        # 2. Key-Value, Definition, Preference, and Relationship Pattern extraction
        sentences = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            for s in re.split(r"(?<=[.!?])\s+", line):
                s = s.strip()
                if s:
                    sentences.append(s)

        for sentence in sentences:
            # Definition: "X is (a/an) Y"
            def_match = re.search(
                r"\b(?P<entity>[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)\s+is\s+(?:a|an)?\s*(?P<definition>[^.!?]+)",
                sentence,
            )
            if def_match:
                ent_name = def_match.group("entity").strip()
                definition = def_match.group("definition").strip()
                if ent_name not in STOP_WORDS and len(ent_name) >= 2:
                    if ent_name in entities:
                        entities[ent_name]["properties"]["description"] = definition
                    else:
                        entities[ent_name] = {
                            "name": ent_name,
                            "type": "entity",
                            "properties": {"description": definition},
                        }
                    observations.append(
                        {
                            "entity_name": ent_name,
                            "content": f"{ent_name} is {definition}",
                        }
                    )

            # User preference: "I prefer X", "I like X", "User likes X", etc.
            pref_match = re.search(
                r"\b(?:I|User)\s+(?:prefer|like|enjoy|love)s?\s+(?P<preference>[^.!?]+)",
                sentence,
                re.IGNORECASE,
            )
            if pref_match:
                pref_item = pref_match.group("preference").strip()
                pref_item = re.sub(
                    r"^(?:a|an|the)\s+", "", pref_item, flags=re.IGNORECASE
                )
                if pref_item and len(pref_item) >= 2:
                    if pref_item not in entities:
                        entities[pref_item] = {
                            "name": pref_item,
                            "type": "PREFERENCE",
                            "properties": {"description": "Preferred by user"},
                        }
                    else:
                        entities[pref_item]["type"] = "PREFERENCE"
                        entities[pref_item]["properties"][
                            "description"
                        ] = "Preferred by user"
                    observations.append(
                        {"entity_name": "User", "content": f"Prefers {pref_item}"}
                    )
                    if "User" not in entities:
                        entities["User"] = {
                            "name": "User",
                            "type": "person",
                            "properties": {"description": "The active user"},
                        }

            # Relationship: "X works at Y", "X is part of Y", etc.
            rel_patterns = [
                (
                    "WORKS_AT",
                    r"\b(?P<from_node>[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)\s+works?\s+at\s+(?P<to_node>[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)\b",
                ),
                (
                    "PART_OF",
                    r"\b(?P<from_node>[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)\s+is\s+part\s+of\s+(?P<to_node>[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)\b",
                ),
                (
                    "DEPENDS_ON",
                    r"\b(?P<from_node>[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)\s+depends?\s+on\s+(?P<to_node>[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)\b",
                ),
                (
                    "MEMBER_OF",
                    r"\b(?P<from_node>[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)\s+(?:is\s+)?members?\s+of\s+(?P<to_node>[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)\b",
                ),
                (
                    "LOCATED_IN",
                    r"\b(?P<from_node>[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)\s+lives?\s+in\s+(?P<to_node>[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)\b",
                ),
                (
                    "CREATED_BY",
                    r"\b(?P<from_node>[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)\s+created\s+(?P<to_node>[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)\b",
                ),
                (
                    "OWNS",
                    r"\b(?P<from_node>[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)\s+owns?\s+(?P<to_node>[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)\b",
                ),
                (
                    "SUPPORTS",
                    r"\b(?P<from_node>[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)\s+supports?\s+(?P<to_node>[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)\b",
                ),
                (
                    "PARTNER_WITH",
                    r"\b(?P<from_node>[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)\s+is\s+partners?\s+of\s+(?P<to_node>[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)\b",
                ),
                (
                    "CONNECTS_TO",
                    r"\b(?P<from_node>[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)\s+is\s+connected\s+to\s+(?P<to_node>[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)\b",
                ),
            ]

            for verb, pattern in rel_patterns:
                rel_match = re.search(pattern, sentence, re.IGNORECASE)
                if rel_match:
                    from_node = rel_match.group("from_node").strip()
                    to_node = rel_match.group("to_node").strip()
                    if from_node not in STOP_WORDS and to_node not in STOP_WORDS:
                        if from_node not in entities:
                            entities[from_node] = {
                                "name": from_node,
                                "type": "entity",
                                "properties": {"description": "Extracted named entity"},
                            }
                        if to_node not in entities:
                            entities[to_node] = {
                                "name": to_node,
                                "type": "entity",
                                "properties": {"description": "Extracted named entity"},
                            }
                        rel_type = verb
                        relationships.append(
                            {
                                "from": from_node,
                                "to": to_node,
                                "type": rel_type,
                                "properties": {},
                            }
                        )

        return {
            "entities": list(entities.values()),
            "relationships": relationships,
            "observations": observations,
        }

    def start_session(self, session_id: str, name: str, properties: dict = None) -> str:
        """
        Starts a session by creating a node of type 'session' in the nodes table.
        If a previous session exists, links the new session to the previous session with a 'PRECEDED_BY' edge.
        """
        props = dict(properties) if properties else {}
        props.setdefault("status", "active")

        # Find previous session
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id FROM nodes WHERE node_type = 'session' ORDER BY created_at DESC LIMIT 1"
        )
        prev_row = cursor.fetchone()

        # Create the session node
        self.create_node(
            node_id=session_id,
            name=name,
            node_type="session",
            properties=props,
        )

        # If previous session exists, link them
        if prev_row:
            prev_session_id = prev_row[0]
            edge_id = f"edge-{session_id}-{prev_session_id}"
            self.create_edge(
                edge_id=edge_id,
                from_node_id=session_id,
                to_node_id=prev_session_id,
                relationship_type="PRECEDED_BY",
            )

        return session_id

    def end_session(self, session_id: str, summary: str) -> None:
        """
        Ends a session by updating its status to 'closed' and saving the summary.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT properties FROM nodes WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Session {session_id} not found")

        properties = json.loads(row[0]) if row[0] else {}
        properties["status"] = "closed"
        properties["summary"] = summary
        properties["ended_at"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()

        self.update_node(
            node_id=session_id,
            properties=properties,
        )

    def record_breakthrough(self, session_id: str, title: str, content: str) -> str:
        """
        Records a learning breakthrough and links it to the specified session.
        Returns the breakthrough node ID.
        """
        breakthrough_id = str(uuid.uuid4())

        self.create_node(
            node_id=breakthrough_id,
            name=title,
            node_type="breakthrough",
            properties={"content": content},
        )

        edge_id = f"edge-{breakthrough_id}-{session_id}"
        self.create_edge(
            edge_id=edge_id,
            from_node_id=breakthrough_id,
            to_node_id=session_id,
            relationship_type="BREAKTHROUGH_IN",
        )

        return breakthrough_id

    def get_graph_state_as_of(self, as_of_iso_timestamp: str) -> dict:
        """
        Returns the state of the memory graph as of the specified ISO 8601 UTC timestamp.
        Excludes any nodes, edges, or observations created after the timestamp.
        """
        cursor = self.conn.cursor()

        # Fetch nodes created on or before the timestamp
        cursor.execute(
            """
            SELECT id, name, node_type, project_id, created_at, updated_at, occurred_at, certainty, salience_score, retrieval_count, properties
            FROM nodes
            WHERE created_at <= ?
            """,
            (as_of_iso_timestamp,),
        )
        node_rows = cursor.fetchall()

        nodes = []
        node_ids = set()
        for row in node_rows:
            node_id = row[0]
            node_ids.add(node_id)

            # Fetch observations for this node created on or before the timestamp
            cursor.execute(
                """
                SELECT id, content, certainty, created_at, occurred_at
                FROM observations
                WHERE entity_id = ? AND created_at <= ?
                ORDER BY created_at ASC
                """,
                (node_id, as_of_iso_timestamp),
            )
            obs_rows = cursor.fetchall()
            observations = []
            for obs_row in obs_rows:
                observations.append(
                    {
                        "id": obs_row[0],
                        "content": obs_row[1],
                        "certainty": obs_row[2],
                        "created_at": obs_row[3],
                        "occurred_at": obs_row[4],
                    }
                )

            nodes.append(
                {
                    "id": node_id,
                    "name": row[1],
                    "node_type": row[2],
                    "project_id": row[3],
                    "created_at": row[4],
                    "updated_at": row[5],
                    "occurred_at": row[6],
                    "certainty": row[7],
                    "salience_score": row[8],
                    "retrieval_count": row[9],
                    "properties": json.loads(row[10]) if row[10] else {},
                    "observations": observations,
                }
            )

        # Fetch edges created on or before the timestamp
        cursor.execute(
            """
            SELECT id, from_node_id, to_node_id, relationship_type, confidence, weight, created_at, properties
            FROM edges
            WHERE created_at <= ?
            """,
            (as_of_iso_timestamp,),
        )
        edge_rows = cursor.fetchall()

        edges = []
        for row in edge_rows:
            from_node = row[1]
            to_node = row[2]
            if from_node in node_ids and to_node in node_ids:
                edges.append(
                    {
                        "id": row[0],
                        "from_node_id": from_node,
                        "to_node_id": to_node,
                        "relationship_type": row[3],
                        "confidence": row[4],
                        "weight": row[5],
                        "created_at": row[6],
                        "properties": json.loads(row[7]) if row[7] else {},
                    }
                )

        return {
            "nodes": nodes,
            "edges": edges,
        }

    def _sanitize_query(self, query: str) -> str:
        """Sanitize a user query for FTS5 MATCH syntax."""
        cleaned = query.replace("*", "").replace("(", "").replace(")", "")
        cleaned = cleaned.replace(":", " ").replace("^", " ")
        terms = [t.strip() for t in cleaned.split() if t.strip()]
        if not terms:
            return ""
        return " ".join(terms)

    def search_hybrid(
        self,
        query: str,
        project_id: str = "default",
        limit: int = 10,
        channel_weights: dict = None,
    ) -> list[dict]:
        """
        Runs parallel Vector and Lexical searches, combines with RRF,
        optionally reranks with CrossEncoder, and logs fail-loud telemetry.
        """
        start_total = time.perf_counter()

        t_fts = 0.0
        t_vec = 0.0
        t_embed = 0.0
        t_rerank = 0.0
        t_graph = 0.0

        is_err = False
        err_code = None
        candidate_count = 0
        result_count = 0

        if channel_weights is None:
            channel_weights = {}
        w_vector = channel_weights.get("vector", 1.0)
        w_lexical = channel_weights.get("lexical", 1.0)

        # Fail-Loud check: Ensure database is not empty
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM nodes")
            if cursor.fetchone()[0] == 0:
                # Return empty or handle gracefully
                return []
        except sqlite3.Error as e:
            self.telemetry.record_search(
                total_latency=time.perf_counter() - start_total,
                fts_latency=0,
                vector_latency=0,
                graph_latency=0,
                embedding_latency=0,
                reranker_latency=0,
                candidate_count=0,
                result_count=0,
                is_error=True,
                error_code=SQLITE_UNAVAILABLE,
            )
            raise SearchError(
                code=SQLITE_UNAVAILABLE,
                message=f"SQLite Database Unavailable: {e}",
                subsystem="SQLITE",
                retry_safe=True,
            )

        # 1. Vector Channel
        vector_results = []
        if w_vector > 0.0 and query.strip():
            try:
                # Embedding step
                s_embed = time.perf_counter()
                try:
                    query_vector = self.model.encode(query)
                except Exception as e:
                    raise SearchError(
                        code=EMBEDDING_FAILED,
                        message=f"Embedding model failure: {e}",
                        subsystem="EMBEDDING",
                        retry_safe=True,
                    )
                t_embed = time.perf_counter() - s_embed

                # Vector Search step
                s_vec = time.perf_counter()
                cursor = self.conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM vector_id_mapping")
                total_nodes = cursor.fetchone()[0]

                if total_nodes > 0:
                    try:
                        search_k = min(total_nodes, max(100, limit * 5))
                        distances, ids = self.index.search(
                            np.asarray([query_vector], dtype=np.float32),
                            k=search_k,
                        )
                    except Exception as e:
                        raise SearchError(
                            code=VECTOR_INDEX_UNAVAILABLE,
                            message=f"Turbovec index search failed: {e}",
                            subsystem="VECTOR_INDEX",
                            retry_safe=True,
                        )

                    # Map unsigned vector IDs back to node IDs
                    signed_ids = [self._to_signed_64(int(vid)) for vid in ids[0]]
                    if signed_ids:
                        placeholders = ",".join("?" for _ in signed_ids)
                        cursor.execute(
                            f"""
                            SELECT n.id, m.vector_id 
                            FROM nodes n 
                            JOIN vector_id_mapping m ON n.id = m.node_id 
                            WHERE n.project_id = ? AND m.vector_id IN ({placeholders})
                              AND n.status != 'ARCHIVED'
                            """,
                            (project_id, *signed_ids),
                        )
                        node_map = {
                            self._to_unsigned_64(row[1]): row[0]
                            for row in cursor.fetchall()
                        }

                        # Maintain rank order from vector search
                        for vid in ids[0]:
                            vid_val = int(vid)
                            if vid_val in node_map:
                                vector_results.append(node_map[vid_val])
                t_vec = time.perf_counter() - s_vec
            except SearchError as se:
                self.telemetry.record_search(
                    total_latency=time.perf_counter() - start_total,
                    fts_latency=0,
                    vector_latency=t_vec,
                    graph_latency=0,
                    embedding_latency=t_embed,
                    reranker_latency=0,
                    candidate_count=0,
                    result_count=0,
                    is_error=True,
                    error_code=se.code,
                )
                raise
            except Exception as e:
                logger.error(f"Vector search failed unexpectedly: {e}")
                self.telemetry.record_search(
                    total_latency=time.perf_counter() - start_total,
                    fts_latency=0,
                    vector_latency=t_vec,
                    graph_latency=0,
                    embedding_latency=t_embed,
                    reranker_latency=0,
                    candidate_count=0,
                    result_count=0,
                    is_error=True,
                    error_code=MEMORY_LAYER_DEGRADED,
                )
                raise SearchError(
                    code=MEMORY_LAYER_DEGRADED,
                    message=f"Vector index search degraded: {e}",
                    subsystem="VECTOR_INDEX",
                    retry_safe=True,
                )

        # 2. Lexical Channel
        lexical_results = []
        if w_lexical > 0.0 and query.strip():
            s_fts = time.perf_counter()
            safe_query = self._sanitize_query(query)
            if safe_query:
                try:
                    cursor = self.conn.cursor()
                    cursor.execute(
                        """
                        SELECT entity_id 
                        FROM entities_fts 
                        WHERE project_id = ? AND entities_fts MATCH ? 
                        ORDER BY rank
                        """,
                        (project_id, safe_query),
                    )
                    lexical_results = [row[0] for row in cursor.fetchall()]
                    # Filter out archived nodes
                    if lexical_results:
                        placeholders = ",".join("?" for _ in lexical_results)
                        cursor.execute(
                            f"SELECT id FROM nodes WHERE status != 'ARCHIVED' AND id IN ({placeholders})",
                            tuple(lexical_results),
                        )
                        active_lex = {r[0] for r in cursor.fetchall()}
                        lexical_results = [
                            r for r in lexical_results if r in active_lex
                        ]
                except sqlite3.OperationalError as e:
                    logger.warning(
                        f"FTS search channel failed for query {query!r}: {e}"
                    )
            t_fts = time.perf_counter() - s_fts

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}

        # Process Vector rank scoring
        for rank_idx, node_id in enumerate(vector_results):
            rank = rank_idx + 1
            rrf_scores[node_id] = rrf_scores.get(node_id, 0.0) + w_vector / (
                60.0 + rank
            )

        # Process Lexical rank scoring
        for rank_idx, node_id in enumerate(lexical_results):
            rank = rank_idx + 1
            rrf_scores[node_id] = rrf_scores.get(node_id, 0.0) + w_lexical / (
                60.0 + rank
            )

        if not rrf_scores:
            self.telemetry.record_search(
                total_latency=time.perf_counter() - start_total,
                fts_latency=t_fts,
                vector_latency=t_vec,
                graph_latency=0,
                embedding_latency=t_embed,
                reranker_latency=0,
                candidate_count=0,
                result_count=0,
                is_error=False,
            )
            return []

        # Sort by RRF score descending and apply larger limit for reranking if enabled
        candidate_limit = limit * 3 if self.reranker.enabled else limit
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[
            :candidate_limit
        ]
        matched_node_ids = [node_id for node_id, _ in sorted_rrf]
        candidate_count = len(matched_node_ids)

        # 4. Hydrate Node Details
        s_graph = time.perf_counter()
        try:
            with self.conn as conn:
                placeholders = ",".join("?" for _ in matched_node_ids)
                conn.execute(
                    f"UPDATE nodes SET retrieval_count = retrieval_count + 1 WHERE id IN ({placeholders})",
                    tuple(matched_node_ids),
                )
        except Exception as e:
            logger.error(f"Failed to increment retrieval count: {e}", exc_info=True)

        nodes_by_id = {}
        try:
            cursor = self.conn.cursor()
            placeholders = ",".join("?" for _ in matched_node_ids)
            cursor.execute(
                f"""
                SELECT id, name, node_type, project_id, created_at, updated_at, occurred_at, certainty, salience_score, retrieval_count, status, properties 
                FROM nodes 
                WHERE id IN ({placeholders})
                """,
                tuple(matched_node_ids),
            )
            for row in cursor.fetchall():
                node_id = row[0]

                # Fetch observations for this node
                cursor.execute(
                    """
                    SELECT id, content, certainty, created_at, occurred_at 
                    FROM observations 
                    WHERE entity_id = ? 
                    ORDER BY created_at ASC
                    """,
                    (node_id,),
                )
                obs_rows = cursor.fetchall()
                observations = []
                for obs_row in obs_rows:
                    observations.append(
                        {
                            "id": obs_row[0],
                            "content": obs_row[1],
                            "certainty": obs_row[2],
                            "created_at": obs_row[3],
                            "occurred_at": obs_row[4],
                        }
                    )

                nodes_by_id[node_id] = {
                    "id": node_id,
                    "name": row[1],
                    "node_type": row[2],
                    "project_id": row[3],
                    "created_at": row[4],
                    "updated_at": row[5],
                    "occurred_at": row[6],
                    "certainty": row[7],
                    "salience_score": row[8],
                    "retrieval_count": row[9],
                    "status": row[10],
                    "properties": json.loads(row[11]) if row[11] else {},
                    "observations": observations,
                }
        except Exception as e:
            logger.error(f"Failed to fetch node details: {e}", exc_info=True)
            self.telemetry.record_search(
                total_latency=time.perf_counter() - start_total,
                fts_latency=t_fts,
                vector_latency=t_vec,
                graph_latency=0,
                embedding_latency=t_embed,
                reranker_latency=0,
                candidate_count=candidate_count,
                result_count=0,
                is_error=True,
                error_code=SQLITE_TRANSACTION_FAILED,
            )
            raise SearchError(
                code=SQLITE_TRANSACTION_FAILED,
                message=f"Database retrieval failure during hydrate: {e}",
                subsystem="SQLITE",
                retry_safe=True,
            )
        t_graph = time.perf_counter() - s_graph

        # Build initial candidate list
        results = []
        for node_id, score in sorted_rrf:
            if node_id in nodes_by_id:
                node_data = dict(nodes_by_id[node_id])
                node_data["rrf_score"] = score
                results.append(node_data)

        # 5. Optional CrossEncoder Reranking
        if self.reranker.enabled:
            s_rerank = time.perf_counter()
            try:
                results = self.reranker.rerank(query, results, limit)
            except Exception as e:
                self.telemetry.record_search(
                    total_latency=time.perf_counter() - start_total,
                    fts_latency=t_fts,
                    vector_latency=t_vec,
                    graph_latency=t_graph,
                    embedding_latency=t_embed,
                    reranker_latency=0,
                    candidate_count=candidate_count,
                    result_count=0,
                    is_error=True,
                    error_code=RERANKER_FAILED,
                )
                raise SearchError(
                    code=RERANKER_FAILED,
                    message=f"Reranking stage failed: {e}",
                    subsystem="RERANKER",
                    retry_safe=True,
                )
            t_rerank = time.perf_counter() - s_rerank
        else:
            results = results[:limit]

        result_count = len(results)

        # Record final successful search telemetry
        total_lat = time.perf_counter() - start_total
        self.telemetry.record_search(
            total_latency=total_lat,
            fts_latency=t_fts,
            vector_latency=t_vec,
            graph_latency=t_graph,
            embedding_latency=t_embed,
            reranker_latency=t_rerank,
            candidate_count=candidate_count,
            result_count=result_count,
            is_error=False,
        )

        return results

    def get_hologram(self, node_id: str, depth: int = 1) -> dict:
        """
        Retrieves a node, its properties, its associated observations,
        and its neighboring nodes and edges up to the specified depth.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM nodes WHERE id = ?", (node_id,))
        if cursor.fetchone()[0] == 0:
            raise ValueError(f"Node {node_id} not found")

        visited_nodes = {node_id}
        current_level = {node_id}

        for d in range(depth):
            if not current_level:
                break
            placeholders = ",".join("?" for _ in current_level)
            cursor.execute(
                f"""
                SELECT from_node_id, to_node_id 
                FROM edges 
                WHERE from_node_id IN ({placeholders}) OR to_node_id IN ({placeholders})
                """,
                tuple(current_level) + tuple(current_level),
            )
            next_level = set()
            for from_id, to_id in cursor.fetchall():
                for nid in (from_id, to_id):
                    if nid not in visited_nodes:
                        visited_nodes.add(nid)
                        next_level.add(nid)
            current_level = next_level

        nodes_list = []
        if visited_nodes:
            placeholders = ",".join("?" for _ in visited_nodes)
            cursor.execute(
                f"""
                SELECT id, name, node_type, project_id, created_at, updated_at, occurred_at, certainty, salience_score, retrieval_count, properties 
                FROM nodes 
                WHERE id IN ({placeholders})
                """,
                tuple(visited_nodes),
            )
            for row in cursor.fetchall():
                nid = row[0]
                # Fetch observations for this node
                cursor.execute(
                    """
                    SELECT id, content, certainty, created_at, occurred_at 
                    FROM observations 
                    WHERE entity_id = ? 
                    ORDER BY created_at ASC
                    """,
                    (nid,),
                )
                obs_rows = cursor.fetchall()
                observations = []
                for obs_row in obs_rows:
                    observations.append(
                        {
                            "id": obs_row[0],
                            "content": obs_row[1],
                            "certainty": obs_row[2],
                            "created_at": obs_row[3],
                            "occurred_at": obs_row[4],
                        }
                    )

                nodes_list.append(
                    {
                        "id": nid,
                        "name": row[1],
                        "node_type": row[2],
                        "project_id": row[3],
                        "created_at": row[4],
                        "updated_at": row[5],
                        "occurred_at": row[6],
                        "certainty": row[7],
                        "salience_score": row[8],
                        "retrieval_count": row[9],
                        "properties": json.loads(row[10]) if row[10] else {},
                        "observations": observations,
                    }
                )

        edges_list = []
        if visited_nodes:
            placeholders = ",".join("?" for _ in visited_nodes)
            cursor.execute(
                f"""
                SELECT id, from_node_id, to_node_id, relationship_type, confidence, weight, created_at, properties 
                FROM edges 
                WHERE from_node_id IN ({placeholders}) AND to_node_id IN ({placeholders})
                """,
                tuple(visited_nodes) + tuple(visited_nodes),
            )
            for row in cursor.fetchall():
                edges_list.append(
                    {
                        "id": row[0],
                        "from_node_id": row[1],
                        "to_node_id": row[2],
                        "relationship_type": row[3],
                        "confidence": row[4],
                        "weight": row[5],
                        "created_at": row[6],
                        "properties": json.loads(row[7]) if row[7] else {},
                    }
                )

        return {
            "nodes": nodes_list,
            "edges": edges_list,
        }

    def get_neighbors(self, node_id: str, depth: int = 1) -> list[dict]:
        """
        Explores connected nodes in the graph starting from an anchor node up to a specified depth.
        Returns a list of node dictionaries.
        """
        hologram = self.get_hologram(node_id, depth=depth)
        return hologram["nodes"]

    def semantic_radar(
        self,
        similarity_threshold: float = 0.7,
        project_id: str = "default",
        auto_create: bool = False,
    ) -> list[dict]:
        """
        Scans the graph to detect pairs of entities with high semantic similarity but no path between them,
        indicating knowledge gaps.
        If auto_create is True, automatically infers and creates relationship edges between these nodes.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, name, node_type, properties FROM nodes WHERE project_id = ?",
            (project_id,),
        )
        rows = cursor.fetchall()
        if len(rows) < 2:
            return []

        nodes = []
        vectors = {}
        for row in rows:
            nid, name, ntype, props_json = row
            props = json.loads(props_json) if props_json else {}
            desc = self._get_description(props)

            cursor.execute(
                "SELECT content FROM observations WHERE entity_id = ? ORDER BY created_at DESC",
                (nid,),
            )
            obs_rows = cursor.fetchall()
            observations = [r[0] for r in obs_rows]

            emb_text = self.build_embedding_text(name, ntype, desc, observations)
            vector = self.model.encode(emb_text)

            nodes.append(
                {
                    "id": nid,
                    "name": name,
                    "node_type": ntype,
                    "project_id": project_id,
                }
            )
            vectors[nid] = vector

        suggestions = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_a = nodes[i]
                node_b = nodes[j]

                v_a = vectors[node_a["id"]]
                v_b = vectors[node_b["id"]]

                dot = np.dot(v_a, v_b)
                norm_a = np.linalg.norm(v_a)
                norm_b = np.linalg.norm(v_b)
                if norm_a == 0 or norm_b == 0:
                    similarity = 0.0
                else:
                    similarity = float(dot / (norm_a * norm_b))

                if similarity >= similarity_threshold:
                    if not self._has_path(node_a["id"], node_b["id"]):
                        # 1. Automatic relationship inference
                        rel_type = self._infer_relationship_type(
                            node_a["node_type"],
                            node_b["node_type"],
                            node_a["project_id"],
                            node_b["project_id"],
                        )

                        reasoning = (
                            f"High semantic similarity ({similarity:.2f}) but no graph path "
                            f"— potential bridge"
                        )

                        created_edge = False
                        edge_details = None

                        # 2. Automatic creation of discovered edges
                        if auto_create:
                            try:
                                edge_id = (
                                    f"edge-{node_a['id']}-{node_b['id']}-{rel_type}"
                                )
                                # Verify if it already exists to be safe
                                cursor.execute(
                                    "SELECT COUNT(*) FROM edges WHERE id = ?",
                                    (edge_id,),
                                )
                                if cursor.fetchone()[0] == 0:
                                    edge_details = self.create_edge(
                                        edge_id=edge_id,
                                        from_node_id=node_a["id"],
                                        to_node_id=node_b["id"],
                                        relationship_type=rel_type,
                                        properties={
                                            "reason": "Automatic relationship inference from semantic_radar"
                                        },
                                    )
                                    created_edge = True
                            except Exception as e:
                                logger.error(
                                    f"Failed to auto-create edge in semantic_radar: {e}"
                                )

                        suggestions.append(
                            {
                                "node_1": node_a,
                                "node_2": node_b,
                                "similarity": similarity,
                                "suggested_relationship": rel_type,
                                "reasoning": reasoning,
                                "created_edge": created_edge,
                                "edge_details": edge_details,
                            }
                        )

        suggestions.sort(key=lambda x: x["similarity"], reverse=True)
        return suggestions

    def _has_path(self, start_id: str, end_id: str) -> bool:
        if start_id == end_id:
            return True
        visited = {start_id}
        queue = [start_id]
        cursor = self.conn.cursor()
        while queue:
            curr = queue.pop(0)
            cursor.execute(
                "SELECT from_node_id, to_node_id FROM edges WHERE from_node_id = ? OR to_node_id = ?",
                (curr, curr),
            )
            for from_id, to_id in cursor.fetchall():
                neighbor = to_id if from_id == curr else from_id
                if neighbor == end_id:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False

    def _infer_relationship_type(
        self,
        source_type: str,
        candidate_type: str,
        source_project: str,
        candidate_project: str,
    ) -> str:
        """Heuristic relationship type inference from node types."""
        if source_project and candidate_project and source_project != candidate_project:
            return "BRIDGES_TO"

        types = frozenset({source_type, candidate_type})

        if types in ({"Concept"}, {"Breakthrough"}) or (
            "Concept" in types and "Analogy" in types
        ):
            return "ANALOGOUS_TO"

        if "Tool" in types and "Procedure" in types:
            return "ENABLES"

        for node_type, rel in (
            ("Decision", "DECIDED_IN"),
            ("Session", "MENTIONED_IN"),
            ("Person", "CREATED_BY"),
        ):
            if node_type in types:
                return rel

        return "RELATED_TO"

    def start_background_discovery(self) -> None:
        """Starts the background relationship discovery daemon thread."""
        if self.bg_thread is not None and self.bg_thread.is_alive():
            logger.info("Background relationship discovery thread is already running.")
            return
        self.bg_stop_event.clear()
        self.bg_thread = threading.Thread(
            target=self._background_discovery_loop, daemon=True
        )
        self.bg_thread.start()
        logger.info("Background relationship discovery thread started.")

    def stop_background_discovery(self) -> None:
        """Stops the background relationship discovery daemon thread."""
        if self.bg_thread is None:
            return
        self.bg_stop_event.set()
        self.bg_thread.join(timeout=2.0)
        self.bg_thread = None
        logger.info("Background relationship discovery thread stopped.")

    def _background_discovery_loop(self) -> None:
        """Periodic background relationship discovery loop."""
        while not self.bg_stop_event.is_set():
            # Sleep in small chunks to remain responsive to stop event
            for _ in range(self.bg_interval):
                if self.bg_stop_event.is_set():
                    return
                time.sleep(1)

            logger.info("Triggering background relationship discovery scan...")
            try:
                projects = []
                try:
                    cursor = self.conn.cursor()
                    cursor.execute("SELECT DISTINCT project_id FROM nodes")
                    projects = [row[0] for row in cursor.fetchall()]
                except Exception as e:
                    logger.error(
                        f"Failed to query projects for background discovery: {e}"
                    )

                if not projects:
                    projects = ["default"]

                for proj in projects:
                    if self.bg_stop_event.is_set():
                        return
                    logger.info(
                        f"Scanning project '{proj}' for relationship discovery..."
                    )
                    created_results = self.semantic_radar(
                        similarity_threshold=self.bg_similarity_threshold,
                        project_id=proj,
                        auto_create=True,
                    )
                    actual_created = sum(
                        1 for item in created_results if item.get("created_edge")
                    )
                    if actual_created > 0:
                        logger.info(
                            f"Background discovery successfully created {actual_created} edges in project '{proj}'."
                        )
            except Exception as e:
                logger.error(
                    f"Error in background relationship discovery loop: {e}",
                    exc_info=True,
                )

    # ==========================================
    # Parity & Auxiliary Wrapper Methods
    # ==========================================

    def diff_knowledge_state(self, from_timestamp: str, to_timestamp: str) -> dict:
        """Computes a deterministic delta between two points in time."""
        return diff_knowledge_state(self, from_timestamp, to_timestamp)

    def query_timeline(
        self,
        entity: str = None,
        entity_type: str = None,
        start: str = None,
        end: str = None,
        order: str = "ASC",
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """Queries timeline of events (nodes, observations) chronologically with strict filtering and pagination."""
        return query_timeline(
            self,
            entity=entity,
            entity_type=entity_type,
            start=start,
            end=end,
            order=order,
            limit=limit,
            offset=offset,
        )

    def get_temporal_neighbors(
        self, node_id: str, direction: str = "both", depth: int = 1
    ) -> list[dict]:
        """Explores connected nodes along temporal edges or nodes that are chronologically related."""
        return get_temporal_neighbors(self, node_id, direction=direction, depth=depth)

    def archive_entity(self, node_id: str, reason: str = "Unused/superseded") -> dict:
        """Archives an entity and its historical records."""
        return archive_node(self, node_id, reason=reason)

    def restore_entity(self, node_id: str) -> dict:
        """Restores an archived entity to ACTIVE status."""
        return restore_node(self, node_id)

    def list_orphans(self) -> list[dict]:
        """Identifies and lists orphaned entities (nodes with zero edges)."""
        return list_orphans(self)

    def prune_stale(self, max_age_days: int, dry_run: bool = False) -> list[dict]:
        """Identifies stale/archived memories older than max_age_days and prunes them."""
        return prune_stale(self, max_age_days, dry_run=dry_run)

    def create_bottle(
        self,
        message: str,
        priority: str = "medium",
        expires_at: str = None,
        author_session_id: str = None,
    ) -> dict:
        """Creates a standardized bottle note in the database."""
        return self.bottle_manager.create_bottle(
            message, priority, expires_at, author_session_id
        )

    def get_bottles(self, include_acknowledged: bool = False) -> list[dict]:
        """Lists active and optionally acknowledged bottles."""
        return self.bottle_manager.list_bottles(include_acknowledged)

    def acknowledge_bottle(self, bottle_id: str) -> bool:
        """Marks a bottle note as acknowledged."""
        return self.bottle_manager.acknowledge_bottle(bottle_id)

    def search_stats(self) -> dict:
        """Calculates rolling statistics, percentiles, and sub-system latency analysis."""
        return self.telemetry.get_stats()

    def analyze_graph(self) -> dict:
        """Runs connected components, LPA community detection, centrality, and PageRank."""
        return self.analytics.analyze_graph()

    def run_librarian_cycle(self) -> dict:
        """Executes one background/autonomous librarian reorganization and synthesis cycle."""
        return self.librarian.run_cycle()
