"""
ARIA / Hermes — Dashboard API Route
Provides summary metrics and consolidated views for the UI homepage.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import get_db
from database.models import Project, Task, Entity, Email, Activity

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/summary")
async def get_dashboard_summary(
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Retrieve statistical counters and lists for the dashboard home screen."""
    # 1. Projects Count
    proj_stmt = select(func.count(Project.id)).where(Project.status == "active")
    proj_res = await db.execute(proj_stmt)
    active_projects = proj_res.scalar() or 0

    # 2. Tasks Count
    task_stmt = select(func.count(Task.id)).where(Task.status.in_(["todo", "in_progress"]))
    task_res = await db.execute(task_stmt)
    open_tasks = task_res.scalar() or 0

    # 3. Entities Count
    entity_stmt = select(func.count(Entity.id))
    entity_res = await db.execute(entity_stmt)
    graph_entities = entity_res.scalar() or 0

    # 4. Emails Count
    email_stmt = select(func.count(Email.id))
    email_res = await db.execute(email_stmt)
    emails_processed = email_res.scalar() or 0

    # 5. Top 3 Recent Projects
    recent_projects_stmt = select(Project).order_by(Project.created_at.desc()).limit(3)
    recent_projects_res = await db.execute(recent_projects_stmt)
    recent_projects = recent_projects_res.scalars().all()

    # 6. Top 5 Recent Activities
    recent_activities_stmt = select(Activity).order_by(Activity.occurred_at.desc()).limit(5)
    recent_activities_res = await db.execute(recent_activities_stmt)
    recent_activities = recent_activities_res.scalars().all()

    return {
        "counters": {
            "active_projects": active_projects,
            "open_tasks": open_tasks,
            "graph_entities": graph_entities,
            "emails_processed": emails_processed,
        },
        "projects": [p.to_dict() for p in recent_projects],
        "activities": [
            {
                "id": a.id,
                "type": a.type,
                "content": a.content,
                "source": a.source,
                "occurred_at": a.occurred_at.isoformat() if a.occurred_at else None,
            }
            for a in recent_activities
        ],
    }
