"""
ARIA / Hermes — Email Service
Handles IMAP mail pulling, email parsing, SMTP sending,
and auto-updating the Knowledge Graph with extracted entities.
"""
from __future__ import annotations

import logging
from datetime import datetime
import email
from email.header import decode_header
from typing import Any

import aiosmtplib
from imapclient import IMAPClient
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from config import settings
from database.models import Email, Entity, Relation

logger = logging.getLogger(__name__)


async def smtp_send_email(to_email: str, subject: str, content: str) -> None:
    """Send an email asynchronously via SMTP client."""
    if not settings.email_address or not settings.email_password:
        logger.warning("SMTP credentials not configured. Skipping actual send.")
        return

    message = email.message.EmailMessage()
    message["From"] = settings.email_address
    message["To"] = to_email
    message["Subject"] = subject
    message.set_content(content)

    await aiosmtplib.send(
        message,
        hostname=settings.email_smtp_host,
        port=settings.email_smtp_port,
        username=settings.email_address,
        password=settings.email_password,
        use_tls=settings.email_smtp_use_tls,
    )
    logger.info(f"Successfully sent SMTP email to {to_email}")


async def sync_imap_emails(db: AsyncSession) -> int:
    """
    Connects to IMAP server, pulls new emails, saves them to DB,
    and runs NER extraction to update the Knowledge Graph.
    """
    if not settings.email_address or not settings.email_password:
        logger.warning("IMAP credentials not configured. Email sync skipped.")
        return 0

    count = 0
    try:
        # Use sync context manager for IMAPClient (since it's a sync library)
        # We run it synchronously as it's safe to block the worker thread if needed
        # or we could use run_in_executor. For simplicity and robustness, we construct the client.
        with IMAPClient(settings.email_imap_host, port=settings.email_imap_port, ssl=settings.email_imap_use_ssl) as client:
            client.login(settings.email_address, settings.email_password)
            client.select_folder("INBOX")

            # Search for unprocessed messages (we search UNSEEN or keep a state in DB)
            messages = client.search(["UNSEEN"])
            logger.info(f"Found {len(messages)} unread emails on IMAP server")

            for msg_id in messages[:10]:  # batch process max 10 emails at a time
                raw_data = client.fetch([msg_id], ["RFC822"])
                email_bytes = raw_data[msg_id][b"RFC822"]

                msg = email.message_from_bytes(email_bytes)

                subject = _decode_header_str(msg.get("Subject", "(No Subject)"))
                sender = _decode_header_str(msg.get("From", ""))
                date_str = msg.get("Date")

                body = _extract_body(msg)
                message_id = msg.get("Message-ID", f"generated-{msg_id}-{datetime.utcnow().timestamp()}")

                # Check duplicate in database
                dup_stmt = select(Email).where(Email.message_id == message_id)
                dup_res = await db.execute(dup_stmt)
                if dup_res.scalars().first():
                    continue

                # 1. Create Entity node
                entity = Entity(type="email", name=subject, description=body[:300])
                db.add(entity)
                await db.flush()

                # Parse date
                received_at = datetime.utcnow()
                if date_str:
                    try:
                        received_at = email.utils.parsedate_to_datetime(date_str)
                    except Exception:
                        pass

                # 2. Save Email
                new_email = Email(
                    entity_id=entity.id,
                    subject=subject,
                    sender=sender,
                    recipients=[_decode_header_str(r) for r in msg.get_all("To", [])],
                    body=body,
                    message_id=message_id,
                    is_read=False,
                    is_processed=False,
                    received_at=received_at,
                )
                db.add(new_email)
                await db.flush()

                # 3. Knowledge Graph linkage (Sender entity, project linkages)
                # Parse sender name/email
                sender_name, sender_addr = email.utils.parseaddr(sender)
                if sender_addr:
                    # Look up if person exists
                    from database.models import Person
                    person_stmt = select(Person).where(Person.email == sender_addr)
                    person_res = await db.execute(person_stmt)
                    person = person_res.scalars().first()

                    if not person:
                        # Create Person and Person Entity
                        person_entity = Entity(type="person", name=sender_name or sender_addr, description=f"Email contact: {sender_addr}")
                        db.add(person_entity)
                        await db.flush()

                        person = Person(entity_id=person_entity.id, name=sender_name or sender_addr, email=sender_addr)
                        db.add(person)
                        await db.flush()

                    # Link sender person -> sent -> email entity
                    rel = Relation(
                        from_entity_id=person.entity_id,
                        to_entity_id=entity.id,
                        relation_type="sent_by",
                    )
                    db.add(rel)

                count += 1
                logger.info(f"Synced email: {subject}")

            await db.commit()
    except Exception as e:
        logger.exception("Error running IMAP email sync")
        raise e

    return count


def _decode_header_str(header_val: Any) -> str:
    """Utility to decode RFC headers safely."""
    if not header_val:
        return ""
    try:
        decoded = decode_header(header_val)
        parts = []
        for text, charset in decoded:
            if isinstance(text, bytes):
                parts.append(text.decode(charset or "utf-8", errors="ignore"))
            else:
                parts.append(str(text))
        return "".join(parts)
    except Exception:
        return str(header_val)


def _extract_body(msg: email.message.Message) -> str:
    """Recursively extract plain text body from multipart email."""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disp = str(part.get("Content-Disposition"))
            if content_type == "text/plain" and "attachment" not in content_disp:
                try:
                    return part.get_payload(decode=True).decode("utf-8", errors="ignore")
                except Exception:
                    pass
    else:
        try:
            return msg.get_payload(decode=True).decode("utf-8", errors="ignore")
        except Exception:
            pass
    return ""
