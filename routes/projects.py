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


# ── Project Intelligence Endpoints ───────────────────────────────────────────


@router.get("/{project_id}/leader-blocks")
async def get_leader_blocks(
    project_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Return the live 4 leader blocks + KPI stats for a project.
    Pulls the latest snapshot blocks, plus real-time computed KPIs.
    """
    from services.project_intelligence import compute_project_progress
    from database.models import ProjectWeeklySnapshot
    from sqlalchemy import select as sa_select

    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    kpis = await compute_project_progress(project_id, db)

    # Pull latest snapshot for AI-written blocks
    stmt = (
        sa_select(ProjectWeeklySnapshot)
        .where(ProjectWeeklySnapshot.project_id == project_id)
        .order_by(ProjectWeeklySnapshot.week_start.desc())
        .limit(1)
    )
    snap = (await db.execute(stmt)).scalar_one_or_none()

    # Task items per status for the block lists
    blocks_stmt = sa_select(ProjectBlock).where(ProjectBlock.project_id == project_id)
    all_blocks = (await db.execute(blocks_stmt)).scalars().all()

    def fmt_block(b: ProjectBlock) -> dict:
        return {
            "id": b.id,
            "title": b.title,
            "status": b.status,
            "block_type": b.block_type,
            "due_date": b.due_date.isoformat() if b.due_date else None,
        }

    return {
        "project_id": project_id,
        "project_title": project.title,
        "kpis": kpis,
        # 4 leader blocks (AI text)
        "leader_blocks": {
            "progress": snap.block_progress if snap else f"{kpis['progress_pct']}% complete — {kpis['tasks_done']} of {kpis['tasks_total']} items done.",
            "focus": snap.block_highlights if snap else "No recent highlights yet. Add activities to generate this.",
            "blockers": snap.block_blockers if snap else ("No blockers." if kpis["tasks_blocked"] == 0 else f"{kpis['tasks_blocked']} items blocked."),
            "need_support": snap.block_next_steps if snap else "Add activities or tasks to generate next steps.",
        },
        # Task items grouped by status
        "items": {
            "in_progress": [fmt_block(b) for b in all_blocks if b.status == "in_progress"],
            "todo": [fmt_block(b) for b in all_blocks if b.status == "todo"],
            "blocked": [fmt_block(b) for b in all_blocks if b.status == "blocked"],
            "done": [fmt_block(b) for b in all_blocks if b.status == "done"],
        },
    }


@router.post("/{project_id}/leader-blocks/refresh")
async def refresh_leader_blocks_endpoint(
    project_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Manually trigger AI to re-analyse recent activities and rewrite the 4 leader blocks.
    Call this after adding new activities or tasks.
    """
    from services.project_intelligence import refresh_leader_blocks, compute_project_progress

    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    kpis = await compute_project_progress(project_id, db)
    blocks = await refresh_leader_blocks(project_id, db)

    return {
        "status": "refreshed",
        "kpis": kpis,
        "leader_blocks": {
            "progress": blocks.get("block_progress"),
            "focus": blocks.get("block_highlights"),
            "blockers": blocks.get("block_blockers"),
            "need_support": blocks.get("block_next_steps"),
        },
    }


@router.post("/{project_id}/draft-update")
async def generate_update_draft(
    project_id: str,
    body: dict[str, Any] | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """
    Generate a short communication update draft for the project.
    Optionally pass recipient and tone in the body.
    Body: {recipient?: str, tone?: "formal"|"casual"|"executive", format?: "email"|"slack"|"bullet"}
    """
    from services.project_intelligence import compute_project_progress
    from database.models import ProjectWeeklySnapshot
    from sqlalchemy import select as sa_select

    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    body = body or {}
    recipient = body.get("recipient", "the team")
    tone = body.get("tone", "professional")
    fmt = body.get("format", "email")

    kpis = await compute_project_progress(project_id, db)

    stmt = (
        sa_select(ProjectWeeklySnapshot)
        .where(ProjectWeeklySnapshot.project_id == project_id)
        .order_by(ProjectWeeklySnapshot.week_start.desc())
        .limit(1)
    )
    snap = (await db.execute(stmt)).scalar_one_or_none()

    from llm.client import llm
    system = "You are a professional project communicator. Write clear, concise project updates."
    user = f"""
Project: {project.title}
Description: {project.description or "N/A"}
Progress: {kpis["progress_pct"]}% ({kpis["tasks_done"]}/{kpis["tasks_total"]} items done)
Deadline: {kpis["deadline_label"]}
Blocked items: {kpis["tasks_blocked"]}

Recent highlights: {snap.block_highlights if snap else "N/A"}
Blockers: {snap.block_blockers if snap else "N/A"}
Next steps: {snap.block_next_steps if snap else "N/A"}

Write a short {tone} {fmt} update for {recipient}.
- Format: {fmt} (if email: include subject line; if slack: use emoji; if bullet: bullet points only)
- Tone: {tone}
- Keep it under 150 words
- End with a clear call to action if there are blockers needing support
"""
    try:
        draft = await llm.generate(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_new_tokens=300,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Draft generation failed: {e}")

    return {
        "project_id": project_id,
        "project_title": project.title,
        "format": fmt,
        "tone": tone,
        "recipient": recipient,
        "draft": draft,
    }


@router.get("/{project_id}/insights")
async def list_insights(
    project_id: str,
    status: str = "pending",
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """List AI-detected insights (auto-tasks, risks, blockers) for a project."""
    from database.models import ProjectInsight
    from sqlalchemy import select as sa_select

    stmt = (
        sa_select(ProjectInsight)
        .where(
            ProjectInsight.project_id == project_id,
            ProjectInsight.status == status,
        )
        .order_by(ProjectInsight.created_at.desc())
    )
    result = await db.execute(stmt)
    insights = result.scalars().all()
    return [
        {
            "id": ins.id,
            "insight_type": ins.insight_type,
            "title": ins.title,
            "content": ins.content,
            "confidence": ins.confidence,
            "source_type": ins.source_type,
            "suggested_due_date": ins.suggested_due_date.isoformat() if ins.suggested_due_date else None,
            "status": ins.status,
            "created_at": ins.created_at.isoformat() if ins.created_at else None,
        }
        for ins in insights
    ]


@router.post("/{project_id}/insights/{insight_id}/accept")
async def accept_insight_endpoint(
    project_id: str,
    insight_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Accept an AI insight — promotes it to a real task and kanban block."""
    from services.project_intelligence import accept_insight
    task = await accept_insight(insight_id, db)
    if not task:
        raise HTTPException(status_code=404, detail="Insight not found or already processed")
    return {"status": "accepted", "task_id": task.id, "task_title": task.title}


@router.post("/{project_id}/insights/{insight_id}/reject")
async def reject_insight_endpoint(
    project_id: str,
    insight_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Dismiss an AI insight."""
    from database.models import ProjectInsight
    insight = await db.get(ProjectInsight, insight_id)
    if not insight:
        raise HTTPException(status_code=404, detail="Insight not found")
    insight.status = "rejected"
    await db.commit()
    return {"status": "rejected", "insight_id": insight_id}


@router.get("/{project_id}/weekly-history")
async def get_weekly_history(
    project_id: str,
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """Return week-by-week snapshot history for charting (last 12 weeks)."""
    from database.models import ProjectWeeklySnapshot
    from sqlalchemy import select as sa_select

    stmt = (
        sa_select(ProjectWeeklySnapshot)
        .where(ProjectWeeklySnapshot.project_id == project_id)
        .order_by(ProjectWeeklySnapshot.week_start.asc())
        .limit(12)
    )
    result = await db.execute(stmt)
    snaps = result.scalars().all()
    return [
        {
            "week_start": s.week_start.isoformat(),
            "progress_pct": s.progress_pct,
            "tasks_total": s.tasks_total,
            "tasks_done": s.tasks_done,
            "tasks_blocked": s.tasks_blocked,
            "new_tasks_this_week": s.new_tasks_this_week,
            "completed_tasks_this_week": s.completed_tasks_this_week,
        }
        for s in snaps
    ]


@router.post("/digest/send")
async def send_digest_email(
    body: dict[str, Any],
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Send a project status digest email to a given recipient."""
    from services.project_intelligence import send_projects_digest_email

    recipient = body.get("recipient_email", "").strip()
    if not recipient:
        raise HTTPException(status_code=400, detail="recipient_email is required")

    result = await send_projects_digest_email(recipient, db)
    return {"status": "ok", "message": result}


# ── Per-project Activity Feed ─────────────────────────────────────────────────

@router.get("/{project_id}/activities")
async def list_project_activities(
    project_id: str,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """List activities linked to this project (by entity relation or project_id tag)."""
    from database.models import Activity
    from sqlalchemy import select as sa_select, or_

    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Find activities tagged with this project id in metadata, or whose content
    # mentions the project title — simple match via JSON field
    stmt = (
        sa_select(Activity)
        .where(
            or_(
                Activity.metadata_["project_id"].astext == project_id,
                Activity.related_entities.contains([project.entity_id]) if project.entity_id else False,
            )
        )
        .order_by(Activity.occurred_at.desc())
        .limit(limit)
    )
    try:
        result = await db.execute(stmt)
        activities = result.scalars().all()
    except Exception:
        activities = []

    return [
        {
            "id": str(a.id),
            "type": a.type,
            "content": a.content,
            "source": a.source,
            "occurred_at": a.occurred_at.isoformat() if a.occurred_at else None,
        }
        for a in activities
    ]


@router.post("/{project_id}/activities")
async def log_project_activity(
    project_id: str,
    body: dict[str, Any],
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Log a manual activity (note, update, meeting) directly against a project.
    This automatically triggers an AI insight extraction pass and optionally
    refreshes the 4 leader blocks.
    """
    from database.models import Activity, Entity, Relation

    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    content = body.get("content", "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="content is required")

    activity_type = body.get("type", "note")

    # Create activity entity node
    entity = Entity(
        type="activity",
        name=f"{activity_type.capitalize()} — {project.title} — {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
        description=content[:300],
    )
    db.add(entity)
    await db.flush()

    # Link activity entity → project entity
    if project.entity_id:
        rel = Relation(
            from_entity_id=entity.id,
            to_entity_id=project.entity_id,
            relation_type="related_to",
        )
        db.add(rel)

    activity = Activity(
        entity_id=entity.id,
        type=activity_type,
        content=content,
        source="manual_project",
        related_entities=[project.entity_id] if project.entity_id else [],
        occurred_at=datetime.utcnow(),
        metadata_={"project_id": project_id, "project_title": project.title},
    )
    db.add(activity)
    await db.commit()

    # Fire-and-forget: extract insights + refresh blocks
    import asyncio

    async def _bg() -> None:
        try:
            from database.connection import AsyncSessionLocal
            from services.project_intelligence import extract_insights_from_text, refresh_leader_blocks
            async with AsyncSessionLocal() as bg_db:
                await extract_insights_from_text(
                    text=content,
                    source_type="activity",
                    source_id=str(activity.id),
                    db=bg_db,
                )
                await refresh_leader_blocks(project_id, bg_db)
        except Exception as exc:
            logger.warning(f"Background activity processing failed: {exc}")

    asyncio.create_task(_bg())

    return {
        "id": str(activity.id),
        "type": activity.type,
        "content": activity.content,
        "occurred_at": activity.occurred_at.isoformat(),
    }


@router.post("/{project_id}/auto-update")
async def auto_update_project_from_comms(
    project_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    AI-driven update: scans the last 30 days of emails and activities that
    mention this project, then:
    1. Extracts new insights (risks, blockers, tasks)
    2. Refreshes the 4 leader blocks
    3. Updates the project progress %
    Returns the updated leader blocks + KPIs.
    """
    from services.project_intelligence import (
        extract_insights_from_text,
        refresh_leader_blocks,
        compute_project_progress,
    )
    from database.models import Activity, Email as EmailModel
    from sqlalchemy import select as sa_select
    from datetime import timedelta

    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    cutoff = datetime.utcnow() - timedelta(days=30)

    # Gather activities that reference this project
    acts_stmt = (
        sa_select(Activity)
        .where(Activity.occurred_at >= cutoff)
        .order_by(Activity.occurred_at.desc())
        .limit(40)
    )
    acts = (await db.execute(acts_stmt)).scalars().all()
    relevant_acts = [
        a for a in acts
        if project.title.lower() in (a.content or "").lower()
        or (a.metadata_ or {}).get("project_id") == project_id
    ]

    # Gather emails that mention the project title
    emails_stmt = (
        sa_select(EmailModel)
        .where(EmailModel.received_at >= cutoff)
        .order_by(EmailModel.received_at.desc())
        .limit(30)
    )
    emails = (await db.execute(emails_stmt)).scalars().all()
    relevant_emails = [
        e for e in emails
        if project.title.lower() in f"{e.subject or ''} {e.body or ''}".lower()
    ]

    # Extract insights from each relevant item
    extracted = 0
    for act in relevant_acts:
        try:
            await extract_insights_from_text(
                text=act.content,
                source_type="activity",
                source_id=str(act.id),
                db=db,
            )
            extracted += 1
        except Exception as exc:
            logger.warning(f"Insight extraction failed for activity {act.id}: {exc}")

    for em in relevant_emails:
        try:
            text = f"{em.subject or ''}\n{em.body or ''}"
            await extract_insights_from_text(
                text=text,
                source_type="email",
                source_id=str(em.id),
                db=db,
            )
            extracted += 1
        except Exception as exc:
            logger.warning(f"Insight extraction failed for email {em.id}: {exc}")

    # Refresh leader blocks with AI
    blocks = await refresh_leader_blocks(project_id, db)
    kpis = await compute_project_progress(project_id, db)

    return {
        "status": "updated",
        "items_scanned": len(relevant_acts) + len(relevant_emails),
        "insights_extracted": extracted,
        "kpis": kpis,
        "leader_blocks": {
            "progress": blocks.get("block_progress"),
            "focus": blocks.get("block_highlights"),
            "blockers": blocks.get("block_blockers"),
            "need_support": blocks.get("block_next_steps"),
        },
    }


