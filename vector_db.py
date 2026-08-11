import os
import json
import logging
import numpy as np

# We still need to configure the environment before importing SentenceTransformers
# to prevent it from outputting to stdout/stderr.
# This might be handled in main.py, but it's safe to do the import inside the class
# or ensure main.py imports vector_db after setting up the environment.

import turbovec
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class VectorDB:
    def __init__(
        self,
        dimension: int = 384,
        metadata_file: str = "metadata.json",
        index_file: str = "index.tvim",
    ):
        self.dimension = dimension
        self.metadata_file = metadata_file
        self.index_file = index_file

        logger.info("Initializing SentenceTransformer model: all-MiniLM-L6-v2")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = turbovec.IdMapIndex(self.dimension)
        self.document_store = {}
        self.next_id = 0

        self.load_storage()

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

        start_id = self.next_id
        ids = np.arange(start_id, start_id + len(chunks), dtype=np.uint64)

        try:
            self.index.add_with_ids(np.asarray(vectors, dtype=np.float32), ids)
        except Exception as e:
            logger.error(f"Failed to add '{title}' to the index: {e}", exc_info=True)
            return f"Error: Failed to add '{title}' to the index."

        for i, chunk in enumerate(chunks):
            doc_id = int(ids[i])
            self.document_store[doc_id] = {
                "id": doc_id,
                "title": title,
                "chunk_index": i,
                "content": chunk,
            }

        self.next_id += len(chunks)
        self.save_storage()
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

        for f in [self.metadata_file, self.index_file]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                    logger.info(f"Deleted file: {f}")
                except Exception as e:
                    logger.error(f"Failed to delete file {f}: {e}", exc_info=True)

        return "Memory cleared successfully."

    def optimize_index(self) -> str:
        """Permanently removes soft-deleted chunks and rebuilds the index."""
        # With IdMapIndex, we actually delete from the index immediately in delete_knowledge.
        # So optimize_index doesn't strictly need to rebuild to remove deleted items.
        # We just sync it to disk.
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
            search_k = min(top_k, valid_count)
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
