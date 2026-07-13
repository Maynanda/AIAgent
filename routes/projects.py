"""
ARIA / Hermes — Projects and Blocks Routes
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import get_db
from database.models import Project, ProjectBlock, Entity, Relation

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("")
async def list_projects(
    status: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """List all projects, optionally filtered by status."""
    stmt = select(Project).order_by(Project.priority.asc(), Project.created_at.desc())
    if status:
        stmt = stmt.where(Project.status == status)

    result = await db.execute(stmt)
    projects = result.scalars().all()
    return [p.to_dict() for p in projects]


@router.post("")
async def create_project(
    body: dict[str, Any],
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Create a new project (and its corresponding knowledge graph entity)."""
    title = body.get("title", "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title is required")

    # 1. Create Entity in Knowledge Graph
    entity = Entity(
        type="project",
        name=title,
        description=body.get("description"),
    )
    db.add(entity)
    await db.flush()

    # 2. Create Project linked to Entity
    start_date = body.get("start_date")
    target_date = body.get("target_date")

    project = Project(
        entity_id=entity.id,
        title=title,
        description=body.get("description"),
        status=body.get("status", "active"),
        priority=body.get("priority", 3),
        progress=body.get("progress", 0),
        start_date=date.fromisoformat(start_date) if start_date else None,
        target_date=date.fromisoformat(target_date) if target_date else None,
        auto_created=body.get("auto_created", False),
        confidence_score=body.get("confidence_score"),
        metadata_=body.get("metadata", {}),
    )
    db.add(project)
    await db.commit()

    return project.to_dict()


@router.get("/{project_id}")
async def get_project(
    project_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Fetch details of a single project, including its blocks."""
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    blocks_stmt = select(ProjectBlock).where(ProjectBlock.project_id == project_id).order_by(ProjectBlock.order_index.asc())
    blocks_result = await db.execute(blocks_stmt)
    blocks = blocks_result.scalars().all()

    data = project.to_dict()
    data["blocks"] = [
        {
            "id": b.id,
            "title": b.title,
            "block_type": b.block_type,
            "status": b.status,
            "content": b.content,
            "assignee_id": b.assignee_id,
            "due_date": b.due_date.isoformat() if b.due_date else None,
            "completed_at": b.completed_at.isoformat() if b.completed_at else None,
            "order_index": b.order_index,
            "metadata": b.metadata_,
        }
        for b in blocks
    ]
    return data


@router.patch("/{project_id}")
async def update_project(
    project_id: str,
    body: dict[str, Any],
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Update a project's parameters and synchronize it with the knowledge graph."""
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if "title" in body:
        project.title = body["title"]
        # Sync title back to Entity
        if project.entity_id:
            entity = await db.get(Entity, project.entity_id)
            if entity:
                entity.name = body["title"]

    if "description" in body:
        project.description = body["description"]
        if project.entity_id:
            entity = await db.get(Entity, project.entity_id)
            if entity:
                entity.description = body["description"]

    if "status" in body:
        project.status = body["status"]
        if body["status"] == "completed" and not project.completed_at:
            project.completed_at = datetime.utcnow()
        elif body["status"] != "completed":
            project.completed_at = None

    if "priority" in body:
        project.priority = body["priority"]
    if "progress" in body:
        project.progress = body["progress"]
    if "start_date" in body:
        project.start_date = date.fromisoformat(body["start_date"]) if body["start_date"] else None
    if "target_date" in body:
        project.target_date = date.fromisoformat(body["target_date"]) if body["target_date"] else None
    if "metadata" in body:
        project.metadata_ = body["metadata"]

    await db.commit()
    return project.to_dict()


@router.delete("/{project_id}")
async def delete_project(
    project_id: str,
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    """Delete a project and cleanup its related blocks."""
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # If linked to an entity, we can optionally delete the entity or keep it
    if project.entity_id:
        # Delete entity from Knowledge Graph (cascade deletes relations)
        entity = await db.get(Entity, project.entity_id)
        if entity:
            await db.delete(entity)

    await db.delete(project)
    await db.commit()
    return JSONResponse({"status": "ok", "message": "Project deleted successfully"})


# ── Project Blocks (Kanban items inside a project) ──────────────────────────


@router.post("/{project_id}/blocks")
async def create_block(
    project_id: str,
    body: dict[str, Any],
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Create a new Kanban project block."""
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    title = body.get("title", "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title is required")

    # Create associated task entity in graph if block is a task/milestone
    entity_id = None
    block_type = body.get("block_type", "task")
    if block_type in ["task", "milestone"]:
        entity = Entity(type="task", name=title, description=body.get("content"))
        db.add(entity)
        await db.flush()
        entity_id = entity.id

        # Link relation: project -> has_task -> task_entity
        if project.entity_id:
            rel = Relation(
                from_entity_id=project.entity_id,
                to_entity_id=entity_id,
                relation_type="has_task",
            )
            db.add(rel)

    due_date = body.get("due_date")
    block = ProjectBlock(
        project_id=project_id,
        entity_id=entity_id,
        title=title,
        block_type=block_type,
        status=body.get("status", "todo"),
        content=body.get("content"),
        assignee_id=body.get("assignee_id"),
        due_date=date.fromisoformat(due_date) if due_date else None,
        order_index=body.get("order_index", 0),
        metadata_=body.get("metadata", {}),
    )
    db.add(block)
    await db.commit()

    return {
        "id": block.id,
        "project_id": block.project_id,
        "title": block.title,
        "block_type": block.block_type,
        "status": block.status,
        "content": block.content,
        "due_date": block.due_date.isoformat() if block.due_date else None,
        "order_index": block.order_index,
    }


@router.patch("/{project_id}/blocks/{block_id}")
async def update_block(
    project_id: str,
    block_id: str,
    body: dict[str, Any],
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Update a specific project block's status, assignee, or parameters."""
    block = await db.get(ProjectBlock, block_id)
    if not block or block.project_id != project_id:
        raise HTTPException(status_code=404, detail="Block not found")

    if "title" in body:
        block.title = body["title"]
        if block.entity_id:
            entity = await db.get(Entity, block.entity_id)
            if entity:
                entity.name = body["title"]

    if "content" in body:
        block.content = body["content"]
        if block.entity_id:
            entity = await db.get(Entity, block.entity_id)
            if entity:
                entity.description = body["content"]

    if "status" in body:
        block.status = body["status"]
        if body["status"] == "done" and not block.completed_at:
            block.completed_at = datetime.utcnow()
        elif body["status"] != "done":
            block.completed_at = None

    if "due_date" in body:
        block.due_date = date.fromisoformat(body["due_date"]) if body["due_date"] else None
    if "assignee_id" in body:
        block.assignee_id = body["assignee_id"]
    if "order_index" in body:
        block.order_index = body["order_index"]
    if "metadata" in body:
        block.metadata_ = body["metadata"]

    await db.commit()

    return {
        "id": block.id,
        "project_id": block.project_id,
        "title": block.title,
        "block_type": block.block_type,
        "status": block.status,
        "content": block.content,
        "due_date": block.due_date.isoformat() if block.due_date else None,
        "order_index": block.order_index,
    }


@router.delete("/{project_id}/blocks/{block_id}")
async def delete_block(
    project_id: str,
    block_id: str,
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    """Delete a project block and cleanup associated entities."""
    block = await db.get(ProjectBlock, block_id)
    if not block or block.project_id != project_id:
        raise HTTPException(status_code=404, detail="Block not found")

    if block.entity_id:
        entity = await db.get(Entity, block.entity_id)
        if entity:
            await db.delete(entity)

    await db.delete(block)
    await db.commit()
    return JSONResponse({"status": "ok", "message": "Block deleted successfully"})
