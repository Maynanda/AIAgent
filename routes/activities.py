"""
ARIA / Hermes — Activities API Routes
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import get_db
from database.models import Activity, Entity, Relation

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/activities", tags=["activities"])


@router.get("")
async def list_activities(
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """List all activities in reverse chronological order."""
    stmt = select(Activity).order_by(Activity.occurred_at.desc()).limit(limit)
    result = await db.execute(stmt)
    activities = result.scalars().all()
    return [
        {
            "id": a.id,
            "entity_id": a.entity_id,
            "type": a.type,
            "content": a.content,
            "source": a.source,
            "related_entities": [str(uid) for uid in a.related_entities] if a.related_entities else [],
            "occurred_at": a.occurred_at.isoformat() if a.occurred_at else None,
            "metadata": a.metadata_,
        }
        for a in activities
    ]


@router.post("")
async def record_activity(
    body: dict[str, Any],
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Record an activity manually and index it into the knowledge graph."""
    content = body.get("content", "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="content is required")

    activity_type = body.get("type", "note")

    # 1. Create Entity node
    entity = Entity(
        type="activity",
        name=f"{activity_type.capitalize()} on {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
        description=content[:200],
    )
    db.add(entity)
    await db.flush()

    occurred_at_str = body.get("occurred_at")
    occurred_at = datetime.fromisoformat(occurred_at_str) if occurred_at_str else datetime.utcnow()

    # 2. Save Activity
    activity = Activity(
        entity_id=entity.id,
        type=activity_type,
        content=content,
        source=body.get("source", "manual"),
        related_entities=body.get("related_entities", []),
        occurred_at=occurred_at,
        metadata_=body.get("metadata", {}),
    )
    db.add(activity)

    # Link relations
    for rel_id in body.get("related_entities", []):
        rel = Relation(
            from_entity_id=entity.id,
            to_entity_id=rel_id,
            relation_type="related_to",
        )
        db.add(rel)

    await db.commit()

    # ── Fire-and-forget: extract project insights from this activity ──────────
    import asyncio
    async def _extract_in_background() -> None:
        try:
            from services.project_intelligence import extract_insights_from_text
            from database.connection import AsyncSessionLocal
            async with AsyncSessionLocal() as bg_db:
                await extract_insights_from_text(
                    text=content,
                    source_type="activity",
                    source_id=str(activity.id),
                    db=bg_db,
                )
                # Also refresh leader blocks for any project mentioned by name in the content
                from sqlalchemy import select as _sel
                from database.models import Project as _Proj
                stmt = _sel(_Proj).where(_Proj.status == "active")
                res = await bg_db.execute(stmt)
                projects = res.scalars().all()
                from services.project_intelligence import refresh_leader_blocks
                for proj in projects:
                    if proj.title.lower() in content.lower():
                        await refresh_leader_blocks(str(proj.id), bg_db)
        except Exception as e:
            logger.warning(f"Background insight extraction failed: {e}")

    asyncio.create_task(_extract_in_background())

    return {
        "id": activity.id,
        "entity_id": activity.entity_id,
        "type": activity.type,
        "content": activity.content,
    }



@router.post("/with-image")
async def record_activity_with_image(
    content: str = Form(...),
    activity_type: str = Form(default="note"),
    image: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Record an activity with an attached image.
    Hermes uses Qwen's vision to describe the image and merge it with the text note.
    """
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image file is empty")

    # Ask the vision model to describe the image in context of the note
    try:
        from llm.client import llm
        messages = [
            {"role": "system", "content": "You are an assistant that describes images concisely in the context of a user's personal notes."},
            {"role": "user", "content": f"The user wrote this note: '{content}'. Describe what is in the attached image and how it relates to the note. Keep it under 100 words."},
        ]
        vision_description = await llm.generate(messages, images=[image_bytes], max_new_tokens=200)
        enriched_content = f"{content}\n\n[Image description: {vision_description}]"
    except Exception as e:
        logger.warning(f"Vision description failed, saving without it: {e}")
        enriched_content = content

    entity = Entity(
        type="activity",
        name=f"{activity_type.capitalize()} (with image) on {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
        description=enriched_content[:300],
    )
    db.add(entity)
    await db.flush()

    activity = Activity(
        entity_id=entity.id,
        type=activity_type,
        content=enriched_content,
        source="manual_with_image",
        related_entities=[],
        occurred_at=datetime.utcnow(),
        metadata_={"has_image": True, "original_filename": image.filename},
    )
    db.add(activity)
    await db.commit()

    return {
        "id": activity.id,
        "entity_id": activity.entity_id,
        "type": activity.type,
        "content": activity.content,
        "vision_enriched": True,
    }


@router.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
) -> dict[str, str]:
    """Transcribe uploaded audio file using local Whisper model."""
    import tempfile
    import os

    try:
        # Save temporary audio file
        suffix = os.path.splitext(audio.filename or "recording.webm")[1] or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(await audio.read())
            temp_path = temp.name

        # Call Whisper STT
        from services.whisper_service import transcribe_local_audio
        transcript = await transcribe_local_audio(temp_path)

        # Cleanup
        try:
            os.remove(temp_path)
        except OSError:
            pass

        return {"transcript": transcript}
    except Exception as e:
        logger.exception("Audio transcription failed")
        raise HTTPException(status_code=500, detail=str(e))
