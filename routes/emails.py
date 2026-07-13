"""
ARIA / Hermes — Emails API Routes
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import get_db
from database.models import Email, Entity

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/emails", tags=["emails"])


@router.get("")
async def list_emails(
    is_processed: bool | None = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """List ingested emails with option to filter by processed status."""
    stmt = select(Email).order_by(Email.received_at.desc()).limit(limit)
    if is_processed is not None:
        stmt = stmt.where(Email.is_processed == is_processed)

    result = await db.execute(stmt)
    emails = result.scalars().all()
    return [
        {
            "id": e.id,
            "entity_id": e.entity_id,
            "subject": e.subject,
            "sender": e.sender,
            "recipients": e.recipients,
            "summary": e.summary,
            "sentiment": e.sentiment,
            "is_read": e.is_read,
            "is_processed": e.is_processed,
            "received_at": e.received_at.isoformat() if e.received_at else None,
            "metadata": e.metadata_,
        }
        for e in emails
    ]


@router.get("/{email_id}")
async def get_email(
    email_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Fetch detail of a single email."""
    email = await db.get(Email, email_id)
    if not email:
        raise HTTPException(status_code=404, detail="Email not found")

    return {
        "id": email.id,
        "entity_id": email.entity_id,
        "subject": email.subject,
        "sender": email.sender,
        "recipients": email.recipients,
        "body": email.body,
        "summary": email.summary,
        "sentiment": email.sentiment,
        "thread_id": email.thread_id,
        "is_read": email.is_read,
        "received_at": email.received_at.isoformat() if email.received_at else None,
        "metadata": email.metadata_,
    }


@router.post("")
async def receive_email(
    body: dict[str, Any],
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Ingest a new email (for webhook triggers or manual testing).
    Auto-creates the corresponding knowledge graph entity.
    """
    message_id = body.get("message_id")
    if not message_id:
        raise HTTPException(status_code=400, detail="message_id is required")

    # Check duplicate
    dup_stmt = select(Email).where(Email.message_id == message_id)
    dup_res = await db.execute(dup_stmt)
    if dup_res.scalars().first():
        raise HTTPException(status_code=409, detail="Email message_id already exists")

    subject = body.get("subject", "(No Subject)")

    # 1. Create Entity
    entity = Entity(type="email", name=subject, description=body.get("body", "")[:300])
    db.add(entity)
    await db.flush()

    received_at_str = body.get("received_at")
    received_at = datetime.fromisoformat(received_at_str) if received_at_str else datetime.utcnow()

    # 2. Save Email
    email = Email(
        entity_id=entity.id,
        subject=subject,
        sender=body.get("sender"),
        recipients=body.get("recipients", []),
        body=body.get("body"),
        summary=body.get("summary"),
        sentiment=body.get("sentiment"),
        thread_id=body.get("thread_id"),
        message_id=message_id,
        is_read=body.get("is_read", False),
        is_processed=False,
        received_at=received_at,
        metadata_=body.get("metadata", {}),
    )
    db.add(email)
    await db.commit()

    return {"id": email.id, "subject": email.subject, "entity_id": email.entity_id}


@router.post("/send")
async def send_email(
    body: dict[str, Any],
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    """
    Send an email through configured SMTP details.
    Drafts are tracked in the database.
    """
    to_email = body.get("to")
    subject = body.get("subject")
    content = body.get("body")

    if not to_email or not subject or not content:
        raise HTTPException(status_code=400, detail="to, subject, and body are required fields")

    # In a full system we call the SMTP client. For now we record the activity of sending.
    from services.email_service import smtp_send_email
    try:
        await smtp_send_email(to_email, subject, content)
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        raise HTTPException(status_code=500, detail=f"SMTP Send Error: {e}")

    # Record sending activity
    entity = Entity(type="email", name=f"Sent: {subject}", description=content[:300])
    db.add(entity)
    await db.flush()

    email = Email(
        entity_id=entity.id,
        subject=subject,
        sender="me",
        recipients=[to_email],
        body=content,
        is_read=True,
        is_processed=True,
        received_at=datetime.utcnow(),
    )
    db.add(email)
    await db.commit()

    return JSONResponse({"status": "ok", "message": "Email sent successfully"})


@router.post("/sync")
async def trigger_sync(db: AsyncSession = Depends(get_db)) -> JSONResponse:
    """Trigger background IMAP synchronization loop."""
    from services.email_service import sync_imap_emails
    try:
        count = await sync_imap_emails(db)
        return JSONResponse({"status": "ok", "synced_count": count})
    except Exception as e:
        logger.exception("IMAP email sync failed")
        raise HTTPException(status_code=500, detail=str(e))
