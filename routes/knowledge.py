"""
ARIA / Hermes — Knowledge Graph & Search Routes
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import get_db
from database.models import Entity, Relation

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.get("/graph")
async def get_graph_data(
    min_weight: float = 0.0,
    db: AsyncSession = Depends(get_db),
) -> dict[str, list[dict[str, Any]]]:
    """
    Format knowledge graph entities and relations for D3.js visualization.
    Returns: {"nodes": [...], "links": [...]}
    """
    # 1. Fetch entities
    entity_stmt = select(Entity)
    entities_res = await db.execute(entity_stmt)
    entities = entities_res.scalars().all()

    # 2. Fetch relations
    relation_stmt = select(Relation).where(Relation.weight >= min_weight)
    relations_res = await db.execute(relation_stmt)
    relations = relations_res.scalars().all()

    # Filter out links pointing to non-existent nodes
    node_ids = {e.id for e in entities}
    nodes = [
        {
            "id": e.id,
            "name": e.name,
            "type": e.type,
            "importance": e.importance,
            "description": e.description,
        }
        for e in entities
    ]

    links = [
        {
            "id": r.id,
            "source": r.from_entity_id,
            "target": r.to_entity_id,
            "type": r.relation_type,
            "weight": r.weight,
        }
        for r in relations
        if r.from_entity_id in node_ids and r.to_entity_id in node_ids
    ]

    return {"nodes": nodes, "links": links}


@router.get("/search")
async def search_knowledge(
    q: str,
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """
    Query the knowledge base using advanced hybrid search (semantic + keyword).
    """
    if not q.strip():
        return []

    from rag.retriever import retrieve_context
    try:
        results = await retrieve_context(query=q, db=db, limit=limit)
        return results
    except Exception as e:
        logger.exception("Hybrid search failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/entities")
async def create_custom_entity(
    body: dict[str, Any],
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Manually add an Entity node to the graph."""
    name = body.get("name", "").strip()
    entity_type = body.get("type", "").strip()
    if not name or not entity_type:
        raise HTTPException(status_code=400, detail="name and type are required")

    # Optionally calculate embedding
    embedding_val = None
    if "description" in body and body["description"]:
        from rag.embedder import embedder
        try:
            embedding_val = embedder.embed_document(body["description"])
        except Exception:
            pass

    entity = Entity(
        type=entity_type,
        name=name,
        description=body.get("description"),
        importance=body.get("importance", 0.5),
        embedding=embedding_val,
        metadata_=body.get("metadata", {}),
    )
    db.add(entity)
    await db.commit()
    return entity.to_dict()


@router.post("/relations")
async def create_custom_relation(
    body: dict[str, Any],
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Manually link two Entity nodes in the graph."""
    from_id = body.get("from_entity_id")
    to_id = body.get("to_entity_id")
    rel_type = body.get("relation_type")

    if not from_id or not to_id or not rel_type:
        raise HTTPException(status_code=400, detail="from_entity_id, to_entity_id, and relation_type are required")

    # Verify both exist
    from_entity = await db.get(Entity, from_id)
    to_entity = await db.get(Entity, to_id)
    if not from_entity or not to_entity:
        raise HTTPException(status_code=404, detail="One or both entities not found")

    rel = Relation(
        from_entity_id=from_id,
        to_entity_id=to_id,
        relation_type=rel_type,
        weight=body.get("weight", 1.0),
        metadata_=body.get("metadata", {}),
    )
    db.add(rel)
    await db.commit()

    return {
        "id": rel.id,
        "from_entity_id": rel.from_entity_id,
        "to_entity_id": rel.to_entity_id,
        "relation_type": rel.relation_type,
        "weight": rel.weight,
    }
