"""
ARIA / Hermes — Static Email Tools
Available to the orchestrator and email agents.
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select

from database.connection import AsyncSessionLocal
from database.models import Email
from services.email_service import smtp_send_email
from tools.registry import tool

logger = logging.getLogger(__name__)


@tool
async def send_email_message(to_email: str, subject: str, content: str) -> str:
    """
    Send an email message to a specified recipient.
    Args:
        to_email: Recipient email address
        subject: Subject line of the email
        content: The text content/body of the email
    """
    try:
        await smtp_send_email(to_email, subject, content)
        return f"Successfully sent email to '{to_email}' with subject '{subject}'."
    except Exception as e:
        logger.error(f"Failed sending email: {e}")
        return f"Error sending email: {e}"


@tool
async def list_unread_emails() -> str:
    """List subjects and sender details of unread/unprocessed emails."""
    async with AsyncSessionLocal() as db:
        stmt = select(Email).where(Email.is_processed == False).order_by(Email.received_at.desc()).limit(10)
        res = await db.execute(stmt)
        emails = res.scalars().all()
        if not emails:
            return "No unread or unprocessed emails in the inbox."

        lines = ["Unprocessed Emails:"]
        for e in emails:
            lines.append(f"- ID: {e.id} | From: {e.sender} | Subject: '{e.subject}' | Date: {e.received_at}")
        return "\n".join(lines)


@tool
async def get_email_content(email_id: str) -> str:
    """
    Fetch the full body of an email to read its details.
    Args:
        email_id: The UUID of the email
    """
    async with AsyncSessionLocal() as db:
        email_item = await db.get(Email, email_id)
        if not email_item:
            return f"Error: Email with ID {email_id} not found."

        # Mark as read/processed when read by the agent
        email_item.is_read = True
        email_item.is_processed = True
        await db.commit()

        return f"From: {email_item.sender}\nSubject: {email_item.subject}\nReceived: {email_item.received_at}\n\n{email_item.body}"
