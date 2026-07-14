"""
ARIA / Hermes — Background Worker Scheduler
Manages recurring tasks: graph refinement, email sync, weekly reports.
Uses APScheduler with AsyncIO support.
"""
from __future__ import annotations

import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config import settings

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()


async def _run_nightly_graph_refinement() -> None:
    """Nightly job: decay, prune, and re-score the knowledge graph."""
    logger.info("⚙ Running nightly graph refinement...")
    try:
        from database.connection import AsyncSessionLocal
        from knowledge_graph.updater import KnowledgeGraphUpdater
        async with AsyncSessionLocal() as db:
            updater = KnowledgeGraphUpdater(db)
            result = await updater.run_nightly_refinement()
            logger.info(f"Graph refinement complete: {result}")
    except Exception as e:
        logger.error(f"Nightly graph refinement failed: {e}")


async def _run_email_sync() -> None:
    """Periodic job: fetch new IMAP emails and index them."""
    if not settings.email_address:
        return
    logger.info("📧 Running scheduled email sync...")
    try:
        from database.connection import AsyncSessionLocal
        from services.email_service import sync_imap_emails
        async with AsyncSessionLocal() as db:
            count = await sync_imap_emails(db)
            logger.info(f"Email sync: {count} new emails ingested")
    except Exception as e:
        logger.error(f"Scheduled email sync failed: {e}")


async def _run_weekly_report() -> None:
    """Weekly job: synthesize activities, emails, and project progress into a report."""
    logger.info("📊 Generating weekly report...")
    try:
        from services.report_service import generate_weekly_report
        path = await generate_weekly_report()
        logger.info(f"Weekly report generated successfully: {path}")

        # Send the report by email if credentials are configured
        if settings.email_address:
            try:
                from pathlib import Path
                from services.email_service import smtp_send_email
                from datetime import datetime
                html_content = Path(path).read_text(encoding="utf-8")
                subject = f"📊 Hermes Weekly Report — {datetime.utcnow().strftime('%B %d, %Y')}"
                # Send as plain-text fallback (HTML inline) to configured address
                await smtp_send_email(
                    to_email=settings.email_address,
                    subject=subject,
                    content=html_content,
                )
                logger.info(f"Weekly report emailed to {settings.email_address}")
            except Exception as mail_err:
                logger.warning(f"Failed to email weekly report (report still saved): {mail_err}")
    except Exception as e:
        logger.error(f"Weekly report generation failed: {e}")


async def _run_prompt_evolution() -> None:
    """Weekly job: analyze feedback and propose prompt improvements."""
    logger.info("🧠 Running prompt evolution analysis...")
    try:
        from agents.prompt_evolution import prompt_evolution
        result = await prompt_evolution.run_weekly_analysis()
        logger.info(f"Prompt evolution: {result}")
    except Exception as e:
        logger.error(f"Prompt evolution run failed: {e}")


async def _run_project_task_extraction() -> None:
    """Daily 6 PM job: scan recent emails and activities for action items and risks."""
    logger.info("🔍 Running project insight extraction from recent inputs...")
    try:
        from database.connection import AsyncSessionLocal
        from services.project_intelligence import extract_insights_from_text
        from sqlalchemy import select
        from database.models import Activity, Email
        from datetime import datetime, timedelta

        cutoff = datetime.utcnow() - timedelta(hours=24)
        async with AsyncSessionLocal() as db:
            # Process recent unprocessed activities
            stmt = select(Activity).where(Activity.created_at >= cutoff)
            result = await db.execute(stmt)
            activities = result.scalars().all()
            for act in activities:
                await extract_insights_from_text(
                    text=act.content,
                    source_type="activity",
                    source_id=str(act.id),
                    db=db,
                )

            # Process recent unprocessed emails
            stmt2 = select(Email).where(
                Email.received_at >= cutoff,
                Email.is_processed == False,  # noqa: E712
            )
            result2 = await db.execute(stmt2)
            emails = result2.scalars().all()
            for email in emails:
                text = f"{email.subject or ''}\n{email.body or ''}"
                await extract_insights_from_text(
                    text=text,
                    source_type="email",
                    source_id=str(email.id),
                    db=db,
                )

        logger.info(f"Project extraction: processed {len(activities)} activities, {len(emails)} emails")
    except Exception as e:
        logger.error(f"Project task extraction failed: {e}")


async def _run_weekly_project_snapshots() -> None:
    """Monday 7 AM job: snapshot all active projects + refresh 4 leader blocks."""
    logger.info("📸 Taking weekly project snapshots...")
    try:
        from database.connection import AsyncSessionLocal
        from services.project_intelligence import take_weekly_snapshot
        async with AsyncSessionLocal() as db:
            result = await take_weekly_snapshot(db)
            logger.info(f"Weekly snapshots: {result}")
    except Exception as e:
        logger.error(f"Weekly project snapshots failed: {e}")


def start_scheduler() -> None:
    """Register and start all background cron jobs."""
    # Nightly at 2 AM — graph refinement
    scheduler.add_job(
        _run_nightly_graph_refinement,
        CronTrigger(hour=2, minute=0),
        id="nightly_graph_refinement",
        replace_existing=True,
    )

    # Every 30 minutes — email sync
    scheduler.add_job(
        _run_email_sync,
        CronTrigger(minute="*/30"),
        id="email_sync",
        replace_existing=True,
    )

    # Every Sunday at 8 AM — weekly report
    scheduler.add_job(
        _run_weekly_report,
        CronTrigger(day_of_week="sun", hour=8, minute=0),
        id="weekly_report",
        replace_existing=True,
    )

    # Every Sunday at 9 AM — prompt evolution analysis
    scheduler.add_job(
        _run_prompt_evolution,
        CronTrigger(day_of_week="sun", hour=9, minute=0),
        id="prompt_evolution",
        replace_existing=True,
    )

    # Every day at 6 PM — scan new activities + emails for project insights
    scheduler.add_job(
        _run_project_task_extraction,
        CronTrigger(hour=18, minute=0),
        id="project_task_extraction",
        replace_existing=True,
    )

    # Every Monday at 7 AM — snapshot all active projects
    scheduler.add_job(
        _run_weekly_project_snapshots,
        CronTrigger(day_of_week="mon", hour=7, minute=0),
        id="weekly_project_snapshots",
        replace_existing=True,
    )

    scheduler.start()
    logger.info("✅ Background scheduler started (graph refinement, email sync, weekly reports, prompt evolution, project insights, weekly snapshots)")


def stop_scheduler() -> None:
    if scheduler.running:
        scheduler.shutdown()
        logger.info("Background scheduler stopped")
