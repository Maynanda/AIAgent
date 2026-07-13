"""
ARIA / Hermes — Static Project Tools
Available directly to the orchestrator and project agents.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any

from sqlalchemy import select

from database.connection import AsyncSessionLocal
from database.models import Project, ProjectBlock, Entity, Relation
from tools.registry import tool

logger = logging.getLogger(__name__)


@tool
async def create_project(title: str, description: str | None = None, target_date: str | None = None) -> str:
    """
    Create a new project.
    Args:
        title: The project title (required)
        description: Brief description of the project
        target_date: Target deadline date (YYYY-MM-DD)
    """
    async with AsyncSessionLocal() as db:
        # Create Entity
        entity = Entity(type="project", name=title, description=description)
        db.add(entity)
        await db.flush()

        # Create Project
        proj = Project(
            entity_id=entity.id,
            title=title,
            description=description,
            target_date=date.fromisoformat(target_date) if target_date else None,
            status="active",
        )
        db.add(proj)
        await db.commit()
        return f"Successfully created project '{title}' with ID {proj.id} and entity ID {entity.id}."


@tool
async def create_project_block(project_id: str, title: str, block_type: str = "task", content: str | None = None) -> str:
    """
    Create a card/block (task, milestone, decision) inside a project.
    Args:
        project_id: The project UUID
        title: Title of the block
        block_type: 'task', 'milestone', 'decision', 'note'
        content: Additional content details
    """
    async with AsyncSessionLocal() as db:
        proj = await db.get(Project, project_id)
        if not proj:
            return f"Error: Project with ID {project_id} not found."

        # Create Entity
        entity = Entity(type="task", name=title, description=content)
        db.add(entity)
        await db.flush()

        block = ProjectBlock(
            project_id=project_id,
            entity_id=entity.id,
            title=title,
            block_type=block_type,
            status="todo",
            content=content,
        )
        db.add(block)

        # Link project -> has_task -> task entity
        if proj.entity_id:
            rel = Relation(from_entity_id=proj.entity_id, to_entity_id=entity.id, relation_type="has_task")
            db.add(rel)

        await db.commit()
        return f"Successfully created block '{title}' inside project '{proj.title}'."


@tool
async def update_block_status(block_id: str, status: str) -> str:
    """
    Update status of a project block (Kanban status update).
    Args:
        block_id: The block UUID
        status: 'todo', 'in_progress', 'done', 'blocked'
    """
    async with AsyncSessionLocal() as db:
        block = await db.get(ProjectBlock, block_id)
        if not block:
            return f"Error: Block with ID {block_id} not found."

        old_status = block.status
        block.status = status
        if status == "done":
            import datetime
            block.completed_at = datetime.datetime.utcnow()
        else:
            block.completed_at = None

        await db.commit()
        return f"Successfully updated block '{block.title}' status from '{old_status}' to '{status}'."


@tool
async def list_active_projects() -> str:
    """List names and IDs of all active projects in the workspace."""
    async with AsyncSessionLocal() as db:
        stmt = select(Project).where(Project.status == "active")
        res = await db.execute(stmt)
        projects = res.scalars().all()
        if not projects:
            return "No active projects found."

        lines = ["Active Projects:"]
        for p in projects:
            lines.append(f"- '{p.title}' (ID: {p.id}) - progress: {p.progress}%")
        return "\n".join(lines)
