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
    def __init__(self, dimension: int = 384, metadata_file: str = "metadata.json", index_file: str = "index.bin"):
        self.dimension = dimension
        self.metadata_file = metadata_file
        self.index_file = index_file
        
        logger.info("Initializing SentenceTransformer model: all-MiniLM-L6-v2")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = turbovec.TurboQuantIndex(self.dimension)
        self.document_store = []
        
        self.load_storage()

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
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

    def save_storage(self):
        """Saves metadata to JSON and attempts to save turbovec index."""
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.document_store, f, indent=2)
            logger.info(f"Metadata saved to {self.metadata_file}")
        except Exception as e:
            logger.error(f"Failed to save metadata to {self.metadata_file}: {e}", exc_info=True)

        try:
            self.index.write(self.index_file)
            logger.info(f"Index saved to {self.index_file}")
        except Exception as e:
            logger.error(f"Failed to save index to {self.index_file}: {e}", exc_info=True)

    def load_storage(self):
        """Loads metadata and index from disk at startup."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    self.document_store = json.load(f)
                logger.info(f"Loaded {len(self.document_store)} documents from {self.metadata_file}")
            except Exception as e:
                logger.error(f"Failed to load metadata from {self.metadata_file}: {e}", exc_info=True)
                self.document_store = []
                
        loaded_index = False
        if os.path.exists(self.index_file):
            try:
                self.index.load(self.index_file)
                loaded_index = True
                logger.info(f"Loaded index from {self.index_file}")
            except Exception as e:
                logger.error(f"Failed to load index from {self.index_file}: {e}", exc_info=True)
                
        # Fallback: rebuild index if we have metadata but failed to load index file
        if not loaded_index and self.document_store:
            logger.info("Rebuilding index from document store...")
            for doc in self.document_store:
                if doc is None or doc.get("deleted"):
                    self.index.add(np.zeros((1, self.dimension), dtype=np.float32))
                else:
                    try:
                        vector = self.model.encode(doc["content"])
                        self.index.add(np.array([vector]))
                    except Exception as e:
                        logger.error(f"Failed to encode document {doc.get('id')}: {e}", exc_info=True)
            logger.info("Finished rebuilding index.")

    def add_knowledge(self, title: str, content: str) -> str:
        chunks = self.chunk_text(content, chunk_size=1000, overlap=200)
        
        for i, chunk in enumerate(chunks):
            try:
                vector = self.model.encode(chunk)
                self.index.add(np.array([vector]))
                
                doc_id = len(self.document_store)
                self.document_store.append({
                    "id": doc_id,
                    "title": title,
                    "chunk_index": i,
                    "content": chunk,
                    "deleted": False
                })
            except Exception as e:
                logger.error(f"Failed to add chunk {i} of '{title}': {e}", exc_info=True)
                return f"Error: Failed to process chunk {i} of '{title}'."
            
        self.save_storage()
        return f"Successfully added '{title}' ({len(chunks)} chunks) into Turbovec memory."

    def delete_knowledge(self, title_or_id: str) -> str:
        deleted_count = 0
        
        for doc in self.document_store:
            if doc is None or doc.get("deleted"):
                continue
            if str(doc["id"]) == title_or_id or doc["title"] == title_or_id:
                doc["deleted"] = True
                deleted_count += 1
                
        if deleted_count > 0:
            self.save_storage()
            logger.info(f"Deleted {deleted_count} chunks matching '{title_or_id}'")
            return f"Successfully deleted {deleted_count} chunks matching '{title_or_id}'."
            
        logger.info(f"No document found matching '{title_or_id}' for deletion")
        return f"No document found matching '{title_or_id}'."

    def clear_memory(self) -> str:
        self.document_store = []
        self.index = turbovec.TurboQuantIndex(self.dimension)
        
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
        logger.info("Starting index optimization (garbage collection)...")
        
        valid_docs = [doc for doc in self.document_store if doc is not None and not doc.get("deleted")]
        old_count = len(self.document_store)
        
        if len(valid_docs) == old_count:
            logger.info("No deleted documents found to optimize.")
            return "No deleted documents found. Index is already optimal."
            
        new_index = turbovec.TurboQuantIndex(self.dimension)
        new_store = []
        
        try:
            for doc in valid_docs:
                vector = self.model.encode(doc["content"])
                new_index.add(np.array([vector]))
                
                # Update document ID to match new index position
                new_doc = doc.copy()
                new_doc["id"] = len(new_store)
                new_store.append(new_doc)
                
            self.index = new_index
            self.document_store = new_store
            
            self.save_storage()
            
            removed_count = old_count - len(self.document_store)
            msg = f"Optimization complete. Removed {removed_count} deleted chunks. Current size: {len(self.document_store)} chunks."
            logger.info(msg)
            return msg
        except Exception as e:
            logger.error(f"Optimization failed: {e}", exc_info=True)
            return f"Optimization failed: {e}"

    def search_knowledge(self, query: str, top_k: int = 3) -> str:
        valid_docs = [d for d in self.document_store if d and not d.get("deleted")]
        if not valid_docs:
            return "Database is still empty. Please add knowledge first using the add_knowledge tool."

        try:
            query_vector = self.model.encode(query)
            
            search_k = min(top_k * 3, len(self.document_store))
            if search_k == 0: 
                search_k = 1
                
            distances, indices = self.index.search(np.array([query_vector]), k=search_k)
            
            results = []
            for i, doc_idx in enumerate(indices[0]):
                if 0 <= doc_idx < len(self.document_store):
                    doc = self.document_store[doc_idx]
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
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}", exc_info=True)
            return f"Error during search: {e}"
