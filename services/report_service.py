"""
ARIA / Hermes — Weekly Report Service
Synthesizes the past 7 days of activities, emails, project progress, and
tasks into a rich weekly HTML report using the LLM.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import AsyncSessionLocal
from database.models import Activity, Email, Project, Task, AgentRun

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


async def generate_weekly_report() -> str:
    """
    Gathers data from the past 7 days and uses the LLM to synthesize
    a structured Markdown weekly report, then saves it as HTML.
    Returns the path to the generated report file.
    """
    async with AsyncSessionLocal() as db:
        week_ago = datetime.utcnow() - timedelta(days=7)

        # 1. Gather activities
        activities_stmt = select(Activity).where(Activity.occurred_at >= week_ago).limit(30)
        acts_res = await db.execute(activities_stmt)
        activities = acts_res.scalars().all()

        # 2. Gather emails
        emails_stmt = select(Email).where(Email.received_at >= week_ago).limit(20)
        emails_res = await db.execute(emails_stmt)
        emails = emails_res.scalars().all()

        # 3. Gather project progress
        projects_stmt = select(Project).where(Project.status == "active")
        projs_res = await db.execute(projects_stmt)
        projects = projs_res.scalars().all()

        # 4. Gather completed tasks
        tasks_stmt = select(Task).where(Task.completed_at >= week_ago).limit(20)
        tasks_res = await db.execute(tasks_stmt)
        completed_tasks = tasks_res.scalars().all()

        # 5. Gather agent runs
        runs_stmt = select(AgentRun).where(AgentRun.created_at >= week_ago)
        runs_res = await db.execute(runs_stmt)
        runs = runs_res.scalars().all()

    # Build data summary for LLM
    summary_text = _build_summary_text(
        activities=activities,
        emails=emails,
        projects=projects,
        completed_tasks=completed_tasks,
        runs=runs,
    )

    # Use LLM to generate rich report
    report_md = await _synthesize_report(summary_text)

    # Convert to HTML
    report_html = _markdown_to_html(report_md)

    # Save file
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    report_path = REPORTS_DIR / f"weekly_report_{date_str}.html"
    report_path.write_text(report_html, encoding="utf-8")
    logger.info(f"Weekly report saved to {report_path}")

    return str(report_path)


def _build_summary_text(
    activities: list, emails: list, projects: list,
    completed_tasks: list, runs: list
) -> str:
    """Build a structured text dump for the LLM to synthesize from."""
    parts = []

    parts.append(f"=== WEEKLY SUMMARY (Past 7 Days) ===\nGenerated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC\n")

    # Activities
    parts.append(f"\n--- ACTIVITIES ({len(activities)}) ---")
    for a in activities[:15]:
        parts.append(f"[{a.type}] {a.content[:150]} ({a.occurred_at.strftime('%b %d')})")

    # Emails
    parts.append(f"\n--- EMAILS RECEIVED ({len(emails)}) ---")
    for e in emails[:10]:
        parts.append(f"From: {e.sender} | Subject: {e.subject} | Sentiment: {e.sentiment or 'N/A'}")

    # Project Status
    parts.append(f"\n--- PROJECT STATUS ({len(projects)} active) ---")
    for p in projects:
        parts.append(f"'{p.title}': {p.progress}% complete | Status: {p.status}")

    # Completed Tasks
    parts.append(f"\n--- TASKS COMPLETED ({len(completed_tasks)}) ---")
    for t in completed_tasks:
        parts.append(f"✓ {t.title} (priority: {t.priority})")

    # Agent Activity
    parts.append(f"\n--- HERMES AI ACTIVITY ({len(runs)} agent runs) ---")
    parts.append(f"Total AI interactions this week: {len(runs)}")

    return "\n".join(parts)


async def _synthesize_report(summary_text: str) -> str:
    """Call LLM to synthesize the weekly report in Markdown."""
    prompt = f"""Based on the following data from the past week, write a comprehensive weekly progress report in Markdown.

Include sections:
1. Executive Summary (2-3 sentences, most important things that happened)
2. Project Progress (how each project moved forward)
3. Key Activities & Decisions (notable meetings, notes)
4. Email Highlights (important communications)
5. Tasks Completed
6. Looking Ahead (what needs attention next week based on patterns)
7. AI Usage Summary (how Hermes was used)

Keep it structured, professional, and actionable. Use bullet points for lists.

DATA:
{summary_text}

Write the report now:"""

    try:
        from llm.client import llm
        messages = [
            {"role": "system", "content": "You are an executive assistant writing a structured weekly progress report."},
            {"role": "user", "content": prompt},
        ]
        report = await llm.generate(messages, max_new_tokens=2048)
        return report
    except Exception as e:
        logger.error(f"LLM report synthesis failed: {e}")
        # Fallback: return the raw summary
        return f"# Weekly Report\n\nFailed to synthesize with LLM.\n\n```\n{summary_text}\n```"


def _markdown_to_html(markdown_text: str) -> str:
    """Convert Markdown to a styled HTML report page."""
    # Simple conversion — production would use markdown2 or mistune
    lines = markdown_text.split("\n")
    html_lines = []

    for line in lines:
        if line.startswith("# "):
            html_lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("- ") or line.startswith("* "):
            html_lines.append(f"<li>{line[2:]}</li>")
        elif line.strip() == "":
            html_lines.append("<br/>")
        else:
            html_lines.append(f"<p>{line}</p>")

    date_str = datetime.utcnow().strftime("%B %d, %Y")
    content = "\n".join(html_lines)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Weekly Report — {date_str}</title>
  <style>
    body {{ font-family: 'Inter', sans-serif; background: #070711; color: #f0f0ff; max-width: 860px; margin: 0 auto; padding: 40px 32px; line-height: 1.7; }}
    h1 {{ font-size: 28px; color: #a78bfa; border-bottom: 1px solid #1a1a3e; padding-bottom: 12px; margin-bottom: 24px; }}
    h2 {{ font-size: 20px; color: #e2e2ff; margin-top: 32px; margin-bottom: 12px; }}
    h3 {{ font-size: 16px; color: #9394b4; margin-top: 20px; }}
    p {{ color: #9394b4; margin: 6px 0; }}
    li {{ color: #c4c4e8; margin: 4px 0; list-style: disc; margin-left: 20px; }}
    .meta {{ font-size: 12px; color: #565675; margin-bottom: 32px; font-family: monospace; }}
  </style>
</head>
<body>
  <div class="meta">Generated by Hermes · {date_str}</div>
  {content}
</body>
</html>"""
