"""
ARIA / Hermes — Tasks API Routes
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import get_db
from database.models import Task, Entity, Relation

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("")
async def list_tasks(
    status: str | None = None,
    project_id: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """List all tasks with optional filters."""
    stmt = select(Task).order_by(Task.priority.asc(), Task.due_date.asc())
    if status:
        stmt = stmt.where(Task.status == status)
    if project_id:
        stmt = stmt.where(Task.project_id == project_id)

    result = await db.execute(stmt)
    tasks = result.scalars().all()
    return [
        {
            "id": t.id,
            "entity_id": t.entity_id,
            "title": t.title,
            "description": t.description,
            "status": t.status,
            "priority": t.priority,
            "project_id": t.project_id,
            "assignee_id": t.assignee_id,
            "due_date": t.due_date.isoformat() if t.due_date else None,
            "completed_at": t.completed_at.isoformat() if t.completed_at else None,
            "source": t.source,
            "metadata": t.metadata_,
            "created_at": t.created_at.isoformat() if t.created_at else None,
        }
        for t in tasks
    ]


@router.post("")
async def create_task(
    body: dict[str, Any],
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Create a task and register it as an entity in the knowledge graph."""
    title = body.get("title", "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title is required")

    # 1. Create Entity node
    entity = Entity(type="task", name=title, description=body.get("description"))
    db.add(entity)
    await db.flush()

    # Link relations if project_id is provided
    project_id = body.get("project_id")
    assignee_id = body.get("assignee_id")

    due_date = body.get("due_date")

    task = Task(
        entity_id=entity.id,
        title=title,
        description=body.get("description"),
        status=body.get("status", "todo"),
        priority=body.get("priority", 3),
        project_id=project_id,
        assignee_id=assignee_id,
        due_date=date.fromisoformat(due_date) if due_date else None,
        source=body.get("source", "manual"),
        metadata_=body.get("metadata", {}),
    )
    db.add(task)
    await db.flush()

    # Create Graph edges
    if project_id:
        # We need the project entity ID
        from database.models import Project
        proj = await db.get(Project, project_id)
        if proj and proj.entity_id:
            rel = Relation(
                from_entity_id=proj.entity_id,
                to_entity_id=entity.id,
                relation_type="has_task",
            )
            db.add(rel)

    if assignee_id:
        rel = Relation(
            from_entity_id=entity.id,
            to_entity_id=assignee_id,
            relation_type="assigned_to",
        )
        db.add(rel)

    await db.commit()

    return {
        "id": task.id,
        "entity_id": task.entity_id,
        "title": task.title,
        "status": task.status,
        "priority": task.priority,
        "project_id": task.project_id,
        "due_date": task.due_date.isoformat() if task.due_date else None,
    }


@router.patch("/{task_id}")
async def update_task(
    task_id: str,
    body: dict[str, Any],
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Update task details and sync status changes to the graph."""
    task = await db.get(Task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if "title" in body:
        task.title = body["title"]
        if task.entity_id:
            entity = await db.get(Entity, task.entity_id)
            if entity:
                entity.name = body["title"]

    if "description" in body:
        task.description = body["description"]
        if task.entity_id:
            entity = await db.get(Entity, task.entity_id)
            if entity:
                entity.description = body["description"]

    if "status" in body:
        task.status = body["status"]
        if body["status"] == "done" and not task.completed_at:
            task.completed_at = datetime.utcnow()
        elif body["status"] != "done":
            task.completed_at = None

    if "priority" in body:
        task.priority = body["priority"]
    if "due_date" in body:
        task.due_date = date.fromisoformat(body["due_date"]) if body["due_date"] else None
    if "assignee_id" in body:
        # Cleanup old assigned relations and add new one
        if task.assignee_id and task.entity_id:
            # Delete old relation
            from sqlalchemy import delete
            stmt = delete(Relation).where(
                Relation.from_entity_id == task.entity_id,
                Relation.to_entity_id == task.assignee_id,
                Relation.relation_type == "assigned_to",
            )
            await db.execute(stmt)

        task.assignee_id = body["assignee_id"]
        if body["assignee_id"] and task.entity_id:
            rel = Relation(
                from_entity_id=task.entity_id,
                to_entity_id=body["assignee_id"],
                relation_type="assigned_to",
            )
            db.add(rel)

    if "metadata" in body:
        task.metadata_ = body["metadata"]

    await db.commit()

    return {
        "id": task.id,
        "title": task.title,
        "status": task.status,
        "priority": task.priority,
        "due_date": task.due_date.isoformat() if task.due_date else None,
    }


@router.delete("/{task_id}")
async def delete_task(
    task_id: str,
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    """Delete a task and its knowledge graph entity."""
    task = await db.get(Task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.entity_id:
        entity = await db.get(Entity, task.entity_id)
        if entity:
            await db.delete(entity)

    await db.delete(task)
    await db.commit()
    return JSONResponse({"status": "ok", "message": "Task deleted successfully"})
