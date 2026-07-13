"""
ARIA / Hermes — Database Seeding Script
Populates initial prompt templates, registers static tools, and seeds default demo data.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from sqlalchemy import select

from database.connection import engine, AsyncSessionLocal
from database.models import Base, PromptVersion, ToolRegistry, Entity, Project, Person

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def seed_database() -> None:
    logger.info("Starting database seeding...")

    async with AsyncSessionLocal() as db:
        # ── 1. Initial Prompt Templates ───────────────────────────────────────
        prompts = [
            {
                "key": "orchestrator_system",
                "content": (
                    "You are Hermes — a highly intelligent personal AI assistant and second brain.\n"
                    "You coordinate tasks, projects, emails, and activities using tools."
                ),
            },
            {
                "key": "extract_entities",
                "content": (
                    "Extract all named entities (people, projects, companies, topics, dates) "
                    "and relationships from the text. Output valid JSON only."
                ),
            },
        ]

        for p in prompts:
            # Check if key already seeded
            stmt = select(PromptVersion).where(PromptVersion.prompt_key == p["key"])
            res = await db.execute(stmt)
            if not res.scalars().first():
                db.add(PromptVersion(
                    prompt_key=p["key"],
                    version=1,
                    content=p["content"],
                    is_active=True,
                    created_by="system",
                ))
                logger.info(f"Seeded prompt: {p['key']}")

        # ── 2. Static Tool Registrations ──────────────────────────────────────
        tools = [
            {"name": "create_project", "desc": "Create a new project."},
            {"name": "create_project_block", "desc": "Create a task or block within a project."},
            {"name": "update_block_status", "desc": "Change status of a project block."},
            {"name": "list_active_projects", "desc": "List all active projects in the workspace."},
            {"name": "send_email_message", "desc": "Send an email message to a recipient."},
            {"name": "list_unread_emails", "desc": "See unread/unprocessed emails."},
            {"name": "get_email_content", "desc": "Read full details of an email."},
            {"name": "search_knowledge_base", "desc": "Query database using hybrid search."},
            {"name": "add_entity_relation", "desc": "Link two nodes in the knowledge graph."},
            {"name": "record_activity_note", "desc": "Log a note, decision, or reminder."},
        ]

        for t in tools:
            stmt = select(ToolRegistry).where(ToolRegistry.name == t["name"])
            res = await db.execute(stmt)
            if not res.scalars().first():
                db.add(ToolRegistry(
                    name=t["name"],
                    description=t["desc"],
                    category="static",
                    is_active=True,
                    created_by="system",
                ))
                logger.info(f"Registered static tool: {t['name']}")

        # ── 3. Initial Demo Data (Only if no entities exist) ──────────────────
        ent_stmt = select(Entity)
        ent_res = await db.execute(ent_stmt)
        if not ent_res.scalars().first():
            logger.info("Seeding demo knowledge graph data...")

            # 1. Project Entity
            proj_entity = Entity(type="project", name="Hermes Platform", description="Developing personal AI agent.")
            db.add(proj_entity)
            await db.flush()

            # Link project
            project = Project(
                entity_id=proj_entity.id,
                title="Hermes Platform",
                description="Developing personal AI agent.",
                status="active",
                priority=2,
                progress=25,
            )
            db.add(project)

            # 2. Owner Person Entity
            user_entity = Entity(type="person", name="Alpha Dev", description="Primary workspace owner.")
            db.add(user_entity)
            await db.flush()

            user = Person(entity_id=user_entity.id, name="Alpha Dev", email="alpha@hermes-dev.io")
            db.add(user)

            # Link relation: Person -> owns -> Project
            from database.models import Relation
            rel = Relation(
                from_entity_id=user_entity.id,
                to_entity_id=proj_entity.id,
                relation_type="owns",
            )
            db.add(rel)

            logger.info("Demo data seeded successfully")

        await db.commit()
        logger.info("Database seeding complete!")


if __name__ == "__main__":
    asyncio.run(seed_database())
