"""
ARIA / Hermes — Static Database and Knowledge Graph Tools
Available directly to the orchestrator and knowledge agents.
"""
from __future__ import annotations

import logging
from typing import Any

from database.connection import AsyncSessionLocal
from database.models import Entity, Relation, Activity
from rag.retriever import retrieve_context
from tools.registry import tool

logger = logging.getLogger(__name__)


@tool
async def search_knowledge_base(query: str) -> str:
    """
    Search your entire database (emails, projects, activities, documents, people)
    using advanced hybrid search retrieval.
    Args:
        query: The search term or semantic question to ask.
    """
    async with AsyncSessionLocal() as db:
        results = await retrieve_context(query, db, limit=5)
        if not results:
            return f"No context matches found for search query: '{query}'."

        lines = [f"Background context retrieved for query '{query}':"]
        for i, item in enumerate(results):
            lines.append(f"[{i+1}] ({item['type']}): {item['content']}")
        return "\n".join(lines)


@tool
async def add_entity_relation(from_entity_name: str, to_entity_name: str, relation_type: str) -> str:
    """
    Link two existing entities in the knowledge graph.
    Args:
        from_entity_name: Name of the source entity node
        to_entity_name: Name of the destination entity node
        relation_type: 'works_on', 'assigned_to', 'mentioned_in', 'owns', 'reports_to', 'related_to'
    """
    from sqlalchemy import select

    async with AsyncSessionLocal() as db:
        # Search from entity
        from_stmt = select(Entity).where(Entity.name == from_entity_name)
        from_res = await db.execute(from_stmt)
        from_ent = from_res.scalars().first()

        # Search to entity
        to_stmt = select(Entity).where(Entity.name == to_entity_name)
        to_res = await db.execute(to_stmt)
        to_ent = to_res.scalars().first()

        if not from_ent or not to_ent:
            missing = []
            if not from_ent:
                missing.append(from_entity_name)
            if not to_ent:
                missing.append(to_entity_name)
            return f"Error: Cannot create relation because these entities do not exist: {', '.join(missing)}"

        # Create relation
        rel = Relation(
            from_entity_id=from_ent.id,
            to_entity_id=to_ent.id,
            relation_type=relation_type,
        )
        db.add(rel)
        await db.commit()
        return f"Created relation: '{from_entity_name}' --[{relation_type}]--> '{to_entity_name}'."


@tool
async def record_activity_note(content: str, activity_type: str = "note") -> str:
    """
    Record an activity, meeting note, or personal reminder.
    Args:
        content: The text content of the note
        activity_type: 'note', 'meeting', 'call', 'milestone', 'decision'
    """
    import datetime

    async with AsyncSessionLocal() as db:
        # 1. Entity
        entity = Entity(
            type="activity",
            name=f"Note on {datetime.datetime.utcnow().strftime('%Y-%m-%d')}",
            description=content[:200],
        )
        db.add(entity)
        await db.flush()

        # 2. Activity
        activity = Activity(
            entity_id=entity.id,
            type=activity_type,
            content=content,
            source="agent",
        )
        db.add(activity)
        await db.commit()
        return f"Successfully recorded activity note (ID: {activity.id})."
