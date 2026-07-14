"""
ARIA / Hermes — Hybrid RAG Retriever
Combines Vector Search (pgvector), Keyword Search (trgm), and Graph Traversal.
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select, text, and_
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import Entity, Relation, DocChunk
from rag.embedder import embedder

logger = logging.getLogger(__name__)


async def retrieve_context(
    query: str,
    db: AsyncSession,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Hybrid Search Retrieval:
    1. Vector Search on doc_chunks (nomic-embed + pgvector distance)
    2. Vector Search on entities (nomic-embed + pgvector distance)
    3. Keyword (Trigram) Search on doc_chunks and entities
    4. Graph Traversal: walk 1-2 hops out from matching nodes
    5. Fusion & Reranking: combine results into formatted list
    """
    if not query.strip():
        return []

    # Get query embedding vector
    try:
        query_vector = embedder.embed_query(query)
    except Exception as e:
        logger.error(f"Failed to embed query: {e}")
        query_vector = None

    results: list[dict[str, Any]] = []
    seen_entity_ids: set[str] = set()

    # ── 1. Vector Search on DocChunks ─────────────────────────────────────────
    if query_vector:
        if "sqlite" in str(db.bind.url):
            # SQLite fallback: fetch doc chunks and compute similarity in Python
            try:
                stmt = select(DocChunk).where(DocChunk.embedding != None)
                res = await db.execute(stmt)
                chunks = res.scalars().all()
                for chunk in chunks:
                    if chunk.embedding:
                        emb = chunk.embedding
                        if isinstance(emb, str):
                            emb = json.loads(emb)
                        score = embedder.similarity(query_vector, emb)
                        if score < 0.3:
                            continue
                        results.append({
                            "type": "doc_chunk",
                            "entity_id": chunk.entity_id,
                            "content": chunk.content,
                            "score": score,
                            "metadata": chunk.metadata_,
                        })
            except Exception as e:
                logger.warning(f"SQLite doc_chunks fallback search failed: {e}")
        else:
            try:
                # pgvector distance operator: <=> (cosine distance)
                stmt = (
                    select(DocChunk, DocChunk.embedding.cosine_distance(query_vector).label("distance"))
                    .order_by(text("distance"))
                    .limit(limit)
                )
                res = await db.execute(stmt)
                for chunk, dist in res.all():
                    score = 1.0 - float(dist)
                    if score < 0.3:  # cutoff threshold
                        continue
                    results.append({
                        "type": "doc_chunk",
                        "entity_id": chunk.entity_id,
                        "content": chunk.content,
                        "score": score,
                        "metadata": chunk.metadata_,
                    })
            except Exception as e:
                logger.warning(f"pgvector doc_chunks search failed: {e}")

    # ── 2. Vector Search on Entities ──────────────────────────────────────────
    if query_vector:
        if "sqlite" in str(db.bind.url):
            # SQLite fallback: fetch entities and compute similarity in Python
            try:
                stmt = select(Entity).where(Entity.embedding != None)
                res = await db.execute(stmt)
                entities = res.scalars().all()
                for entity in entities:
                    if entity.embedding:
                        emb = entity.embedding
                        if isinstance(emb, str):
                            emb = json.loads(emb)
                        score = embedder.similarity(query_vector, emb)
                        if score < 0.35:
                            continue
                        seen_entity_ids.add(entity.id)
                        results.append({
                            "type": "entity",
                            "entity_id": entity.id,
                            "content": f"Entity: {entity.name} (type: {entity.type}). Description: {entity.description or 'None'}",
                            "score": score + 0.1,
                            "metadata": entity.metadata_,
                        })
            except Exception as e:
                logger.warning(f"SQLite entities fallback search failed: {e}")
        else:
            try:
                stmt = (
                    select(Entity, Entity.embedding.cosine_distance(query_vector).label("distance"))
                    .order_by(text("distance"))
                    .limit(limit)
                )
                res = await db.execute(stmt)
                for entity, dist in res.all():
                    score = 1.0 - float(dist)
                    if score < 0.35:
                        continue
                    seen_entity_ids.add(entity.id)
                    results.append({
                        "type": "entity",
                        "entity_id": entity.id,
                        "content": f"Entity: {entity.name} (type: {entity.type}). Description: {entity.description or 'None'}",
                        "score": score + 0.1,  # boost entity match slightly
                        "metadata": entity.metadata_,
                    })
            except Exception as e:
                logger.warning(f"pgvector entities search failed: {e}")

    # ── 3. Trigram (Keyword) Similarity Search ────────────────────────────────
    try:
        if "sqlite" in str(db.bind.url):
            # SQLite keyword fallback
            trigm_stmt = (
                select(Entity)
                .where((Entity.name.like(f"%{query}%")) | (Entity.description.like(f"%{query}%")))
                .limit(limit)
            )
        else:
            # Check trigram match on entities (name and description)
            trigm_stmt = (
                select(Entity)
                .where(text("name % :q OR description % :q"))
                .params(q=query)
                .limit(limit)
            )
        trigm_res = await db.execute(trigm_stmt)
        for entity in trigm_res.scalars().all():
            if entity.id in seen_entity_ids:
                continue
            seen_entity_ids.add(entity.id)
            results.append({
                "type": "entity_keyword",
                "entity_id": entity.id,
                "content": f"Entity: {entity.name} (type: {entity.type}). Description: {entity.description or 'None'}",
                "score": 0.5,
                "metadata": entity.metadata_,
            })
    except Exception as e:
        logger.warning(f"Trigram search failed: {e}")

    # ── 4. Graph Traversal ────────────────────────────────────────────────────
    # For every highly-relevant entity, fetch its neighbors up to 1 hop out.
    if seen_entity_ids:
        try:
            hops_stmt = (
                select(Relation, Entity)
                .join(Entity, Relation.to_entity_id == Entity.id)
                .where(Relation.from_entity_id.in_(list(seen_entity_ids)))
                .limit(limit * 2)
            )
            hops_res = await db.execute(hops_stmt)
            for rel, target_entity in hops_res.all():
                if target_entity.id in seen_entity_ids:
                    continue
                results.append({
                    "type": "graph_hop",
                    "entity_id": target_entity.id,
                    "content": f"Graph relation: Entity '{target_entity.name}' (type: {target_entity.type}) is {rel.relation_type} related node.",
                    "score": 0.4,
                    "metadata": target_entity.metadata_,
                })
        except Exception as e:
            logger.warning(f"Graph traversal context search failed: {e}")

    # Sort all results by score desc and deduplicate content
    results.sort(key=lambda x: x["score"], reverse=True)

    unique_results: list[dict[str, Any]] = []
    seen_contents: set[str] = set()

    for item in results:
        cleaned = item["content"].strip().lower()
        if cleaned not in seen_contents:
            seen_contents.add(cleaned)
            unique_results.append(item)

    return unique_results[:limit]


async def assemble_context_block(
    query: str,
    db: AsyncSession,
    limit: int = 8,
) -> str:
    """Convenience helper to retrieve and format search context block for prompting."""
    results = await retrieve_context(query, db, limit)
    if not results:
        return "No relevant background information found."

    lines = ["Here is background context retrieved from your database:"]
    for i, item in enumerate(results):
        lines.append(f"[{i+1}] ({item['type']}): {item['content']}")
    return "\n".join(lines)
