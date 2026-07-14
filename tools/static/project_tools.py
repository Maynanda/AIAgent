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


@tool
async def get_project_progress(project_id: str) -> str:
    """
    Get the live 4-block KPI status for a project.
    Returns progress %, task counts, blocked count, and deadline.
    Args:
        project_id: The project UUID
    """
    from services.project_intelligence import compute_project_progress
    async with AsyncSessionLocal() as db:
        proj = await db.get(Project, project_id)
        if not proj:
            return f"Project {project_id} not found."
        kpis = await compute_project_progress(project_id, db)
        return (
            f"📊 {proj.title}\n"
            f"  ① Progress:   {kpis['progress_pct']}% ({kpis['tasks_done']} / {kpis['tasks_total']} done)\n"
            f"  ② In Progress: {kpis['tasks_in_progress']} tasks\n"
            f"  ③ Blockers:    {kpis['tasks_blocked']} blocked\n"
            f"  ④ Deadline:    {kpis['deadline_label']}"
        )


@tool
async def refresh_project_blocks(project_id: str) -> str:
    """
    Trigger AI to re-analyse recent activities and refresh the 4 leader blocks for a project.
    Call this after adding new activities or when progress may have changed.
    Args:
        project_id: The project UUID
    """
    from services.project_intelligence import refresh_leader_blocks
    async with AsyncSessionLocal() as db:
        proj = await db.get(Project, project_id)
        if not proj:
            return f"Project {project_id} not found."
        blocks = await refresh_leader_blocks(project_id, db)
        return (
            f"✅ Leader blocks refreshed for '{proj.title}':\n"
            f"  ① Progress:    {blocks.get('block_progress', '—')}\n"
            f"  ② Highlights:  {blocks.get('block_highlights', '—')}\n"
            f"  ③ Blockers:    {blocks.get('block_blockers', '—')}\n"
            f"  ④ Next Steps:  {blocks.get('block_next_steps', '—')}"
        )


@tool
async def list_project_insights(project_id: str | None = None) -> str:
    """
    List pending AI-detected action items, risks, and blockers.
    If project_id is None, shows unmatched insights from all sources.
    Args:
        project_id: The project UUID, or omit to see all pending insights
    """
    from sqlalchemy import select as sa_select
    from database.models import ProjectInsight
    async with AsyncSessionLocal() as db:
        stmt = sa_select(ProjectInsight).where(ProjectInsight.status == "pending")
        if project_id:
            stmt = stmt.where(ProjectInsight.project_id == project_id)
        res = await db.execute(stmt)
        insights = res.scalars().all()
        if not insights:
            return "No pending AI insights found."
        lines = [f"🔍 {len(insights)} pending insight(s):"]
        for ins in insights:
            due = f" (due {ins.suggested_due_date})" if ins.suggested_due_date else ""
            lines.append(
                f"  [{ins.insight_type.upper()}] {ins.title}{due}\n"
                f"    Source: {ins.source_type} | Confidence: {int(ins.confidence * 100)}% | ID: {ins.id}"
            )
        return "\n".join(lines)


@tool
async def accept_project_insight(insight_id: str) -> str:
    """
    Accept a pending AI insight — creates a real task and kanban block from it.
    Args:
        insight_id: The ProjectInsight UUID
    """
    from services.project_intelligence import accept_insight
    async with AsyncSessionLocal() as db:
        task = await accept_insight(insight_id, db)
        if task:
            return f"✅ Insight accepted. Task created: '{task.title}' (ID: {task.id})"
        return f"❌ Could not accept insight {insight_id} — already processed or not found."


@tool
async def reject_project_insight(insight_id: str) -> str:
    """
    Reject and dismiss a pending AI insight.
    Args:
        insight_id: The ProjectInsight UUID
    """
    from database.models import ProjectInsight
    async with AsyncSessionLocal() as db:
        insight = await db.get(ProjectInsight, insight_id)
        if not insight:
            return f"Insight {insight_id} not found."
        insight.status = "rejected"
        await db.commit()
        return f"✅ Insight '{insight.title}' dismissed."


@tool
async def get_project_weekly_history(project_id: str) -> str:
    """
    Get the week-by-week progress history for a project (last 8 weeks).
    Useful for understanding trend and momentum.
    Args:
        project_id: The project UUID
    """
    from sqlalchemy import select as sa_select
    from database.models import ProjectWeeklySnapshot
    async with AsyncSessionLocal() as db:
        proj = await db.get(Project, project_id)
        if not proj:
            return f"Project {project_id} not found."
        stmt = (
            sa_select(ProjectWeeklySnapshot)
            .where(ProjectWeeklySnapshot.project_id == project_id)
            .order_by(ProjectWeeklySnapshot.week_start.desc())
            .limit(8)
        )
        res = await db.execute(stmt)
        snaps = res.scalars().all()
        if not snaps:
            return f"No weekly history yet for '{proj.title}'."
        lines = [f"📈 Weekly history for '{proj.title}':"]
        for s in reversed(snaps):
            bar = "█" * (s.progress_pct // 10) + "░" * (10 - s.progress_pct // 10)
            lines.append(f"  {s.week_start}  [{bar}] {s.progress_pct}%  ({s.tasks_done}/{s.tasks_total} done, {s.tasks_blocked} blocked)")
        return "\n".join(lines)


@tool
async def send_projects_digest_email(recipient_email: str) -> str:
    """
    Send a project status summary email to a recipient (or yourself).
    The email includes progress, highlights, blockers, and next steps for every active project.
    Args:
        recipient_email: Email address to send the digest to
    """
    from services.project_intelligence import send_projects_digest_email as _send
    from database.connection import AsyncSessionLocal as _db
    async with _db() as db:
        return await _send(recipient_email, db)

