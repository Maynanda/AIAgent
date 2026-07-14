"""
ARIA / Hermes — Email Service
Handles IMAP mail pulling, local Microsoft Outlook pulling (macOS),
SMTP sending, Outlook sending, and auto-updating the Knowledge Graph.
"""
from __future__ import annotations

import logging
from datetime import datetime
import email
from email.header import decode_header
import subprocess
import asyncio
from typing import Any

import aiosmtplib
import dateutil.parser
from imapclient import IMAPClient
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from config import settings
from database.models import Email, Entity, Relation

logger = logging.getLogger(__name__)


async def smtp_send_email(to_email: str, subject: str, content: str) -> None:
    """Send an email asynchronously via SMTP or Outlook AppleScript."""
    if settings.email_client == "outlook":
        await send_outlook_email(to_email, subject, content)
        return

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


async def send_outlook_email(to_email: str, subject: str, content: str) -> None:
    """Send email using local Microsoft Outlook app on macOS via AppleScript."""
    # Escape quotes for AppleScript string
    escaped_subject = subject.replace('"', '\\"')
    escaped_content = content.replace('"', '\\"').replace('\n', '\\n')

    script = f"""
    tell application "Microsoft Outlook"
        try
            set newMsg to make new outgoing message with properties {{subject: "{escaped_subject}", plain text content: "{escaped_content}"}}
            make new recipient at newMsg with properties {{email address: {{address: "{to_email}"}}}}
            send newMsg
            return "SUCCESS"
        on error errText number errNum
            return "ERROR: " & errText & " (" & (errNum as string) & ")"
        end try
    end tell
    """
    
    proc = await asyncio.create_subprocess_exec(
        'osascript', '-e', script,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    output = stdout.decode('utf-8').strip()
    if proc.returncode != 0 or output.startswith("ERROR"):
        raise Exception(f"Failed to send email via local Outlook: {output or stderr.decode('utf-8')}")
    logger.info(f"Successfully sent Outlook email via AppleScript to {to_email}")


async def get_local_outlook_emails() -> str:
    """Pull unread emails from local Microsoft Outlook on macOS via AppleScript."""
    script = """
    tell application "Microsoft Outlook"
        set results to ""
        try
            set unreadMessages to (every message of inbox whose is read is false)
            repeat with msg in unreadMessages
                set msgId to id of msg
                set msgSubject to subject of msg
                set msgSender to sender of msg
                set senderName to name of msgSender
                set senderAddress to address of msgSender
                set msgBody to plain text content of msg
                set msgDate to (time received of msg) as string
                
                set results to results & "ID: " & msgId & "\n"
                set results to results & "SenderName: " & senderName & "\n"
                set results to results & "SenderAddress: " & senderAddress & "\n"
                set results to results & "Subject: " & msgSubject & "\n"
                set results to results & "Date: " & msgDate & "\n"
                set results to results & "Body: " & msgBody & "\n"
                set results to results & "---ENDMSG---\n"
                
                set is read of msg to true
            end repeat
        on error errText number errNum
            set results to "ERROR: " & errText & " (" & (errNum as string) & ")"
        end try
        return results
    end tell
    """
    
    proc = await asyncio.create_subprocess_exec(
        'osascript', '-e', script,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.error(f"AppleScript execution failed: {stderr.decode('utf-8')}")
        return ""
    return stdout.decode('utf-8')


async def sync_imap_emails(db: AsyncSession) -> int:
    """
    Syncs email messages from the configured provider.
    Routes to local macOS Outlook or IMAP depending on Settings.
    """
    if settings.email_client == "outlook":
        return await sync_outlook_emails_mac(db)

    if not settings.email_address or not settings.email_password:
        logger.warning("IMAP credentials not configured. Email sync skipped.")
        return 0

    count = 0
    try:
        with IMAPClient(settings.email_imap_host, port=settings.email_imap_port, ssl=settings.email_imap_use_ssl) as client:
            client.login(settings.email_address, settings.email_password)
            client.select_folder("INBOX")

            messages = client.search(["UNSEEN"])
            logger.info(f"Found {len(messages)} unread emails on IMAP server")

            for msg_id in messages[:10]:
                raw_data = client.fetch([msg_id], ["RFC822"])
                email_bytes = raw_data[msg_id][b"RFC822"]

                msg = email.message_from_bytes(email_bytes)

                subject = _decode_header_str(msg.get("Subject", "(No Subject)"))
                sender = _decode_header_str(msg.get("From", ""))
                date_str = msg.get("Date")

                body = _extract_body(msg)
                message_id = f"generated-{msg_id}-{datetime.utcnow().timestamp()}"

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
                sender_name, sender_addr = email.utils.parseaddr(sender)
                if sender_addr:
                    from database.models import Person
                    person_stmt = select(Person).where(Person.email == sender_addr)
                    person_res = await db.execute(person_stmt)
                    person = person_res.scalars().first()

                    if not person:
                        person_entity = Entity(type="person", name=sender_name or sender_addr, description=f"Email contact: {sender_addr}")
                        db.add(person_entity)
                        await db.flush()

                        person = Person(entity_id=person_entity.id, name=sender_name or sender_addr, email=sender_addr)
                        db.add(person)
                        await db.flush()

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


async def sync_outlook_emails_mac(db: AsyncSession) -> int:
    """Sync emails from local Outlook desktop app on macOS via AppleScript."""
    output = await get_local_outlook_emails()
    if not output:
        return 0
    if output.startswith("ERROR:"):
        logger.warning(f"Failed to fetch local Outlook emails: {output}")
        return 0

    messages = output.split("---ENDMSG---\n")
    count = 0

    for raw_msg in messages:
        if not raw_msg.strip():
            continue
        
        msg_id = None
        sender_name = ""
        sender_address = ""
        subject = "(No Subject)"
        date_str = ""
        body_lines = []
        in_body = False

        for line in raw_msg.splitlines():
            if line.startswith("ID: "):
                msg_id = line[4:].strip()
            elif line.startswith("SenderName: "):
                sender_name = line[12:].strip()
            elif line.startswith("SenderAddress: "):
                sender_address = line[15:].strip()
            elif line.startswith("Subject: "):
                subject = line[9:].strip()
            elif line.startswith("Date: "):
                date_str = line[6:].strip()
            elif line.startswith("Body: "):
                in_body = True
                body_lines.append(line[6:])
            elif in_body:
                body_lines.append(line)

        body = "\n".join(body_lines).strip()
        if not msg_id:
            continue

        message_id = f"outlook-{msg_id}"

        # Check duplicate
        dup_stmt = select(Email).where(Email.message_id == message_id)
        dup_res = await db.execute(dup_stmt)
        if dup_res.scalars().first():
            continue

        # Create Entity
        entity = Entity(type="email", name=subject, description=body[:300])
        db.add(entity)
        await db.flush()

        # Parse date
        received_at = datetime.utcnow()
        if date_str:
            try:
                received_at = dateutil.parser.parse(date_str)
            except Exception:
                pass

        # Create Email
        sender_full = f"{sender_name} <{sender_address}>" if sender_name else sender_address
        new_email = Email(
            entity_id=entity.id,
            subject=subject,
            sender=sender_full,
            recipients=["me"],
            body=body,
            message_id=message_id,
            is_read=True,
            is_processed=False,
            received_at=received_at,
        )
        db.add(new_email)
        await db.flush()

        # Build Graph relation for Sender
        if sender_address:
            from database.models import Person
            person_stmt = select(Person).where(Person.email == sender_address)
            person_res = await db.execute(person_stmt)
            person = person_res.scalars().first()

            if not person:
                person_entity = Entity(type="person", name=sender_name or sender_address, description=f"Outlook contact: {sender_address}")
                db.add(person_entity)
                await db.flush()

                person = Person(entity_id=person_entity.id, name=sender_name or sender_address, email=sender_address)
                db.add(person)
                await db.flush()

            rel = Relation(
                from_entity_id=person.entity_id,
                to_entity_id=entity.id,
                relation_type="sent_by",
            )
            db.add(rel)

        count += 1
        logger.info(f"Ingested Outlook email: {subject}")

    await db.commit()
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
