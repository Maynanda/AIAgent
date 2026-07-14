"""
ARIA / Hermes — Project Intelligence Service

Drives the 4 live leader blocks on each project by:
  1. Computing task/block KPIs in real-time
  2. Using LLM to summarise recent activities into 4 human-readable blocks
  3. Scanning new activities/emails for hidden action items (auto-tasks)
  4. Snapshotting weekly state every Monday
  5. Building email digest across all active projects

The 4 leader blocks per project are:
  ① Progress   — "67% complete — 12 of 18 tasks done"
  ② Highlights — "This week: API shipped, auth bug fixed"
  ③ Blockers   — "2 blocked: payment gateway, legal review"
  ④ Next Steps — "Next: finalize UI, run user tests, send proposal"
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import select, func as sqlfunc, and_
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import (
    Activity, Email, Project, ProjectBlock, ProjectInsight,
    ProjectWeeklySnapshot, Task,
)
from llm.client import llm

logger = logging.getLogger(__name__)


# ── Progress computation ──────────────────────────────────────────────────────

async def compute_project_progress(project_id: str, db: AsyncSession) -> dict[str, Any]:
    """
    Calculate the 4 live KPI values for a project and return them.
    Also updates Project.progress in the DB.
    """
    # Count blocks by status
    stmt = select(ProjectBlock).where(ProjectBlock.project_id == project_id)
    result = await db.execute(stmt)
    blocks = result.scalars().all()

    # Count tasks by status
    stmt2 = select(Task).where(Task.project_id == project_id)
    result2 = await db.execute(stmt2)
    tasks = result2.scalars().all()

    all_items = list(blocks) + list(tasks)
    total = len(all_items)
    done = sum(1 for i in all_items if i.status == "done")
    in_progress = sum(1 for i in all_items if i.status == "in_progress")
    blocked = sum(1 for i in all_items if i.status == "blocked")

    progress_pct = int((done / total * 100) if total > 0 else 0)

    # Deadline
    proj = await db.get(Project, project_id)
    days_left = None
    deadline_label = "No deadline set"
    if proj and proj.target_date:
        days_left = (proj.target_date - date.today()).days
        if days_left < 0:
            deadline_label = f"⚠️ Overdue by {abs(days_left)} days"
        elif days_left == 0:
            deadline_label = "⚠️ Due today"
        else:
            deadline_label = f"{days_left} days left"

    # Persist updated progress on the project
    if proj:
        proj.progress = progress_pct
        await db.commit()

    return {
        "progress_pct": progress_pct,
        "tasks_total": total,
        "tasks_done": done,
        "tasks_in_progress": in_progress,
        "tasks_blocked": blocked,
        "days_left": days_left,
        "deadline_label": deadline_label,
    }


# ── 4 Leader Blocks (LLM-generated) ──────────────────────────────────────────

async def refresh_leader_blocks(project_id: str, db: AsyncSession) -> dict[str, str]:
    """
    Generate (or refresh) the 4 AI-written leader block texts for a project.
    Triggered whenever an activity is added that mentions the project,
    or on the daily 6PM job.

    Returns: {block_progress, block_highlights, block_blockers, block_next_steps}
    """
    proj = await db.get(Project, project_id)
    if not proj:
        return {}

    kpis = await compute_project_progress(project_id, db)

    # Fetch recent activities (last 14 days) that mention this project
    cutoff = datetime.utcnow() - timedelta(days=14)
    stmt = (
        select(Activity)
        .where(Activity.occurred_at >= cutoff)
        .order_by(Activity.occurred_at.desc())
        .limit(20)
    )
    result = await db.execute(stmt)
    recent_activities = result.scalars().all()

    # Fetch blocked block titles
    stmt2 = select(ProjectBlock).where(
        and_(ProjectBlock.project_id == project_id, ProjectBlock.status == "blocked")
    )
    result2 = await db.execute(stmt2)
    blocked_blocks = result2.scalars().all()
    blocked_titles = [b.title for b in blocked_blocks]

    # Fetch next open tasks
    stmt3 = (
        select(ProjectBlock)
        .where(and_(
            ProjectBlock.project_id == project_id,
            ProjectBlock.status.in_(["todo", "in_progress"]),
        ))
        .order_by(ProjectBlock.order_index)
        .limit(5)
    )
    result3 = await db.execute(stmt3)
    next_tasks = result3.scalars().all()
    next_titles = [t.title for t in next_tasks]

    activity_snippets = "\n".join(
        f"- [{a.occurred_at.strftime('%Y-%m-%d')}] {a.content[:150]}" for a in recent_activities
    ) or "No recent activities."

    prompt_system = "You are a concise project status assistant. Write short, factual bullet-point updates."
    prompt_user = f"""
Project: {proj.title}
Description: {proj.description or 'N/A'}
Progress: {kpis['progress_pct']}% ({kpis['tasks_done']} of {kpis['tasks_total']} items done)
Blocked items: {', '.join(blocked_titles) if blocked_titles else 'None'}
Next open items: {', '.join(next_titles) if next_titles else 'None'}
Deadline: {kpis['deadline_label']}

Recent activities:
{activity_snippets}

Generate exactly 4 short text blocks (1-2 sentences each). Respond in JSON with keys:
  block_progress   — brief progress status sentence
  block_highlights — what was accomplished recently (from activities)
  block_blockers   — current blockers or risks (say "None" if clear)
  block_next_steps — the most important upcoming actions
"""
    try:
        raw = await llm.generate(
            [{"role": "system", "content": prompt_system}, {"role": "user", "content": prompt_user}],
            json_mode=True,
            max_new_tokens=400,
        )
        blocks = json.loads(raw)
    except Exception as e:
        logger.warning(f"LLM block generation failed for project {project_id}: {e}")
        blocks = {
            "block_progress": f"{kpis['progress_pct']}% complete — {kpis['tasks_done']} of {kpis['tasks_total']} items done.",
            "block_highlights": "Unable to generate highlights — add more activities.",
            "block_blockers": f"{kpis['tasks_blocked']} blocked items." if kpis['tasks_blocked'] else "No blockers.",
            "block_next_steps": ", ".join(next_titles) if next_titles else "All tasks complete.",
        }

    logger.info(f"Leader blocks refreshed for project {proj.title}")
    return blocks


# ── Auto-task extraction ──────────────────────────────────────────────────────

async def extract_insights_from_text(
    text: str,
    source_type: str,
    source_id: str,
    db: AsyncSession,
) -> list[ProjectInsight]:
    """
    Run the LLM over a piece of text (email body, activity note, etc.)
    and extract action items, risks, blockers, or updates.
    Try to match each to an active project.
    """
    # Get active project names for matching
    stmt = select(Project).where(Project.status == "active")
    result = await db.execute(stmt)
    active_projects = result.scalars().all()

    if not active_projects:
        return []

    project_list = "\n".join(f"- {p.title} (id: {p.id})" for p in active_projects)

    system = "You are an intelligent project assistant. Extract actionable items from text."
    user = f"""
Active projects:
{project_list}

Text to analyse:
---
{text[:2000]}
---

Extract any action items, deadlines, risks, or blockers mentioned.
For each one, identify which project it belongs to (or null if unclear).

Respond as JSON array. Each item has:
  title           — short action title (max 80 chars)
  content         — full context sentence
  insight_type    — "auto_task" | "risk" | "blocker" | "update" | "milestone"
  project_id      — matching project UUID or null
  confidence      — 0.0 to 1.0
  suggested_due_date — "YYYY-MM-DD" or null

Return [] if nothing actionable is found.
"""
    try:
        raw = await llm.generate(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            json_mode=True,
            max_new_tokens=600,
        )
        items = json.loads(raw)
        if not isinstance(items, list):
            items = items.get("items", [])
    except Exception as e:
        logger.warning(f"Insight extraction failed: {e}")
        return []

    saved: list[ProjectInsight] = []
    for item in items:
        if not item.get("title"):
            continue
        due = None
        if item.get("suggested_due_date"):
            try:
                due = date.fromisoformat(item["suggested_due_date"])
            except ValueError:
                pass

        insight = ProjectInsight(
            project_id=item.get("project_id"),
            insight_type=item.get("insight_type", "auto_task"),
            title=item["title"][:200],
            content=item.get("content", ""),
            suggested_due_date=due,
            source_type=source_type,
            source_id=source_id,
            confidence=float(item.get("confidence", 0.8)),
            status="pending",
        )
        db.add(insight)
        saved.append(insight)

    if saved:
        await db.commit()
        logger.info(f"Extracted {len(saved)} insights from {source_type}:{source_id}")

    return saved


async def accept_insight(insight_id: str, db: AsyncSession) -> Task | None:
    """
    Promote a ProjectInsight to a real Task + ProjectBlock.
    Updates Project.progress after creation.
    """
    insight = await db.get(ProjectInsight, insight_id)
    if not insight or insight.status != "pending":
        return None

    # Create Task
    task = Task(
        title=insight.title,
        description=insight.content,
        project_id=insight.project_id,
        due_date=insight.suggested_due_date,
        source="agent",
        status="todo",
    )
    db.add(task)
    await db.flush()

    # Create ProjectBlock too (Kanban card)
    if insight.project_id:
        block = ProjectBlock(
            project_id=insight.project_id,
            title=insight.title,
            block_type=_insight_type_to_block_type(insight.insight_type),
            status="todo",
            content=insight.content,
        )
        db.add(block)

    # Mark insight as accepted
    insight.status = "accepted"
    insight.accepted_task_id = task.id
    await db.commit()

    # Refresh progress
    if insight.project_id:
        await compute_project_progress(insight.project_id, db)

    return task


def _insight_type_to_block_type(insight_type: str) -> str:
    return {
        "auto_task": "task",
        "risk": "risk",
        "blocker": "task",
        "update": "note",
        "milestone": "milestone",
    }.get(insight_type, "task")


# ── Weekly snapshot ───────────────────────────────────────────────────────────

async def take_weekly_snapshot(db: AsyncSession) -> str:
    """
    Called every Monday at 7 AM.
    Saves a ProjectWeeklySnapshot for every active project,
    including refreshed 4 leader blocks.
    """
    today = date.today()
    week_start = today - timedelta(days=today.weekday())  # Monday

    stmt = select(Project).where(Project.status == "active")
    result = await db.execute(stmt)
    projects = result.scalars().all()

    count = 0
    for proj in projects:
        try:
            kpis = await compute_project_progress(str(proj.id), db)
            blocks = await refresh_leader_blocks(str(proj.id), db)

            # Count new/completed tasks this week
            week_cutoff = datetime.combine(week_start, datetime.min.time())
            stmt2 = select(sqlfunc.count()).select_from(Task).where(
                and_(Task.project_id == str(proj.id), Task.created_at >= week_cutoff)
            )
            new_this_week = (await db.execute(stmt2)).scalar() or 0

            stmt3 = select(sqlfunc.count()).select_from(Task).where(
                and_(Task.project_id == str(proj.id), Task.completed_at >= week_cutoff)
            )
            done_this_week = (await db.execute(stmt3)).scalar() or 0

            snapshot = ProjectWeeklySnapshot(
                project_id=str(proj.id),
                week_start=week_start,
                progress_pct=kpis["progress_pct"],
                tasks_total=kpis["tasks_total"],
                tasks_done=kpis["tasks_done"],
                tasks_in_progress=kpis["tasks_in_progress"],
                tasks_blocked=kpis["tasks_blocked"],
                new_tasks_this_week=new_this_week,
                completed_tasks_this_week=done_this_week,
                block_progress=blocks.get("block_progress"),
                block_highlights=blocks.get("block_highlights"),
                block_blockers=blocks.get("block_blockers"),
                block_next_steps=blocks.get("block_next_steps"),
            )
            db.add(snapshot)
            count += 1
        except Exception as e:
            logger.error(f"Snapshot failed for project {proj.title}: {e}")

    await db.commit()
    return f"Snapshots saved for {count} projects (week of {week_start})"


# ── Email digest ──────────────────────────────────────────────────────────────

async def generate_projects_email_digest(db: AsyncSession) -> str:
    """
    Build a plain-text + HTML email body summarising all active projects.
    """
    stmt = select(Project).where(Project.status == "active")
    result = await db.execute(stmt)
    projects = result.scalars().all()

    if not projects:
        return "No active projects to report."

    lines = [f"📊 Project Status Digest — {date.today().strftime('%B %d, %Y')}\n"]

    for proj in projects:
        kpis = await compute_project_progress(str(proj.id), db)
        # Get latest snapshot blocks if available
        stmt2 = (
            select(ProjectWeeklySnapshot)
            .where(ProjectWeeklySnapshot.project_id == str(proj.id))
            .order_by(ProjectWeeklySnapshot.week_start.desc())
            .limit(1)
        )
        snap_result = await db.execute(stmt2)
        snap = snap_result.scalar_one_or_none()

        lines.append(f"── {proj.title} ──────────────────────")
        lines.append(f"  Progress:    {kpis['progress_pct']}%  ({kpis['tasks_done']}/{kpis['tasks_total']} tasks done)")
        lines.append(f"  Deadline:    {kpis['deadline_label']}")
        if snap:
            if snap.block_highlights:
                lines.append(f"  Highlights:  {snap.block_highlights}")
            if snap.block_blockers and snap.block_blockers.lower() != "none":
                lines.append(f"  Blockers:    {snap.block_blockers}")
            if snap.block_next_steps:
                lines.append(f"  Next Steps:  {snap.block_next_steps}")
        lines.append("")

    return "\n".join(lines)


async def send_projects_digest_email(recipient_email: str, db: AsyncSession) -> str:
    """Draft and send the project digest to a given recipient."""
    body = await generate_projects_email_digest(db)

    try:
        from services.email_service import send_email
        subject = f"📊 Project Status Digest — {date.today().strftime('%B %d, %Y')}"
        await send_email(
            to=recipient_email,
            subject=subject,
            body=body,
        )
        return f"✅ Project digest sent to {recipient_email}"
    except Exception as e:
        logger.error(f"Failed to send digest email: {e}")
        return f"❌ Failed to send digest: {e}"
