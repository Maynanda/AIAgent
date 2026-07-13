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
        await generate_weekly_report()
        logger.info("Weekly report generated successfully")
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

    scheduler.start()
    logger.info("✅ Background scheduler started (graph refinement, email sync, weekly reports, prompt evolution)")


def stop_scheduler() -> None:
    if scheduler.running:
        scheduler.shutdown()
        logger.info("Background scheduler stopped")
