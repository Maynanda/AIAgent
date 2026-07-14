"""
ARIA / Hermes — ChromaDB Vector Store Client
Manages local persistent vector storage using ChromaDB.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Lazy load chromadb only when needed
_chroma_client = None
_entities_collection = None
_chunks_collection = None

def get_chroma_client():
    global _chroma_client, _entities_collection, _chunks_collection
    if _chroma_client is not None:
        return _chroma_client, _entities_collection, _chunks_collection

    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        
        # Path to local persistent chroma db
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
        os.makedirs(db_path, exist_ok=True)
        
        _chroma_client = chromadb.PersistentClient(
            path=db_path,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Get or create collections
        # nomic-embed-text-v1.5 has 768 dimensions
        _entities_collection = _chroma_client.get_or_create_collection(
            name="entities",
            metadata={"hnsw:space": "cosine"}
        )
        _chunks_collection = _chroma_client.get_or_create_collection(
            name="doc_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"✅ Local ChromaDB initialized successfully at: {db_path}")
        return _chroma_client, _entities_collection, _chunks_collection
    except Exception as e:
        logger.error(f"❌ Failed to initialize ChromaDB (check if chromadb package is installed): {e}")
        return None, None, None


def upsert_entity(entity_id: str, embedding: list[float], metadata: dict[str, Any] | None = None) -> bool:
    """Insert or update an entity vector in ChromaDB."""
    _, col, _ = get_chroma_client()
    if col is None:
        return False
    try:
        # Convert all metadata values to string/int/float/bool (Chroma requirement)
        flat_meta = {}
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    flat_meta[k] = v
                else:
                    flat_meta[k] = str(v)
        
        col.upsert(
            ids=[entity_id],
            embeddings=[embedding],
            metadatas=[flat_meta] if flat_meta else None
        )
        return True
    except Exception as e:
        logger.error(f"Failed to upsert entity {entity_id} to ChromaDB: {e}")
        return False


def delete_entity(entity_id: str) -> bool:
    """Delete an entity from ChromaDB."""
    _, col, _ = get_chroma_client()
    if col is None:
        return False
    try:
        col.delete(ids=[entity_id])
        return True
    except Exception as e:
        logger.error(f"Failed to delete entity {entity_id} from ChromaDB: {e}")
        return False


def query_entities(query_vector: list[float], limit: int = 10) -> list[dict[str, Any]]:
    """Retrieve top N matching entities using cosine similarity."""
    _, col, _ = get_chroma_client()
    if col is None:
        return []
    try:
        res = col.query(
            query_embeddings=[query_vector],
            n_results=limit
        )
        results = []
        if res and res["ids"] and len(res["ids"][0]) > 0:
            for idx in range(len(res["ids"][0])):
                # Cosine distance to cosine similarity: score = 1.0 - distance
                dist = res["distances"][0][idx] if "distances" in res and res["distances"] else 0.0
                score = 1.0 - float(dist)
                results.append({
                    "id": res["ids"][0][idx],
                    "score": score,
                    "metadata": res["metadatas"][0][idx] if "metadatas" in res and res["metadatas"] else {},
                })
        return results
    except Exception as e:
        logger.error(f"Failed to query entities from ChromaDB: {e}")
        return []


def upsert_chunk(chunk_id: str, embedding: list[float], content: str, metadata: dict[str, Any] | None = None) -> bool:
    """Insert or update a document chunk vector in ChromaDB."""
    _, _, col = get_chroma_client()
    if col is None:
        return False
    try:
        flat_meta = {"content": content}
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    flat_meta[k] = v
                else:
                    flat_meta[k] = str(v)
        
        col.upsert(
            ids=[chunk_id],
            embeddings=[embedding],
            metadatas=[flat_meta]
        )
        return True
    except Exception as e:
        logger.error(f"Failed to upsert chunk {chunk_id} to ChromaDB: {e}")
        return False


def delete_chunk(chunk_id: str) -> bool:
    """Delete a chunk from ChromaDB."""
    _, _, col = get_chroma_client()
    if col is None:
        return False
    try:
        col.delete(ids=[chunk_id])
        return True
    except Exception as e:
        logger.error(f"Failed to delete chunk {chunk_id} from ChromaDB: {e}")
        return False


def query_chunks(query_vector: list[float], limit: int = 10) -> list[dict[str, Any]]:
    """Retrieve top N matching document chunks using cosine similarity."""
    _, _, col = get_chroma_client()
    if col is None:
        return []
    try:
        res = col.query(
            query_embeddings=[query_vector],
            n_results=limit
        )
        results = []
        if res and res["ids"] and len(res["ids"][0]) > 0:
            for idx in range(len(res["ids"][0])):
                dist = res["distances"][0][idx] if "distances" in res and res["distances"] else 0.0
                score = 1.0 - float(dist)
                results.append({
                    "id": res["ids"][0][idx],
                    "score": score,
                    "metadata": res["metadatas"][0][idx] if "metadatas" in res and res["metadatas"] else {},
                })
        return results
    except Exception as e:
        logger.error(f"Failed to query chunks from ChromaDB: {e}")
        return []
