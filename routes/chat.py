"""
ARIA / Hermes — Chat WebSocket Route
Handles real-time streaming chat with the Hermes orchestrator.
Supports multimodal vision inputs: images can be sent as base64 or raw bytes.
"""
from __future__ import annotations

import base64
import json
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from agents.orchestrator import OrchestratorAgent
from database.connection import get_db
from database.models import AgentRun
from llm.client import get_llm, HermesLLM
from memory.manager import MemoryManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.websocket("/ws")
async def chat_websocket(
    websocket: WebSocket,
    db: AsyncSession = Depends(get_db),
    llm: HermesLLM = Depends(get_llm),
) -> None:
    """
    WebSocket endpoint for streaming chat with Hermes.

    Protocol:
      Client sends JSON: {"message": "...", "session_id": "optional-uuid"}
      OR sends binary: first receive the image bytes, then a JSON with {"message": ..., "has_image": true}
      Server streams: {"type": "token", "content": "..."}
                      {"type": "done", "metadata": {...}}
                      {"type": "error", "content": "..."}
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    session_id = str(uuid.uuid4())

    try:
        pending_image_bytes: bytes | None = None
        while True:
            # Accept either text (JSON) or binary (image bytes)
            msg = await websocket.receive()

            if msg["type"] == "websocket.receive":
                if "bytes" in msg and msg["bytes"]:
                    # Raw image bytes pre-buffered for next text message
                    pending_image_bytes = msg["bytes"]
                    continue
                raw = msg.get("text", "")
            else:
                continue

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = {"message": raw}

            user_message = data.get("message", "").strip()
            if not user_message:
                continue

            # Collect images: from pending binary frame or base64 field
            images: list[bytes] = []
            if pending_image_bytes:
                images.append(pending_image_bytes)
                pending_image_bytes = None
            if data.get("image_b64"):
                try:
                    images.append(base64.b64decode(data["image_b64"]))
                except Exception:
                    pass

            # Use existing session or create new
            if data.get("session_id"):
                session_id = data["session_id"]

            # Send acknowledgement
            await websocket.send_json({
                "type": "ack",
                "session_id": session_id,
            })

            start_time = time.time()

            try:
                # Build memory context
                memory = MemoryManager(session_id=session_id, db=db)
                memory.add_short_term(f"User: {user_message}")
                
                from rag.context_builder import build_agent_context
                context = await build_agent_context(query=user_message, session_id=session_id, db=db)

                # Initialize agent
                agent = OrchestratorAgent(llm=llm)
                agent.session_id = session_id

                # Stream response token by token
                full_response = ""
                steps_count = 0

                async for chunk in agent.stream_run(
                    user_input=user_message,
                    context=context,
                    session_id=session_id,
                    images=images or None,
                ):
                    # Check for metadata marker
                    if chunk.startswith("\n\n__DONE__:"):
                        try:
                            meta = json.loads(chunk.replace("\n\n__DONE__:", ""))
                            steps_count = meta.get("steps", 0)
                        except Exception:
                            pass
                        break

                    full_response += chunk
                    await websocket.send_json({
                        "type": "token",
                        "content": chunk,
                    })

                # Store episodic memory
                memory.add_short_term(f"Hermes: {full_response[:200]}")
                await memory.add_episodic(
                    content=f"Q: {user_message[:100]} → A: {full_response[:200]}",
                    importance=0.6,
                )

                # Save agent run to DB
                duration_ms = int((time.time() - start_time) * 1000)
                run = AgentRun(
                    session_id=session_id,
                    user_input=user_message,
                    agent_type="orchestrator",
                    steps=[s.__dict__ for s in agent._steps],
                    result=full_response,
                    duration_ms=duration_ms,
                )
                db.add(run)
                await db.commit()

                # Send completion signal
                await websocket.send_json({
                    "type": "done",
                    "metadata": {
                        "session_id": session_id,
                        "duration_ms": duration_ms,
                        "steps": steps_count,
                        "run_id": str(run.id),
                    },
                })

            except Exception as e:
                logger.exception(f"Agent error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "content": f"I encountered an error: {str(e)}",
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: session {session_id}")


@router.post("/message")
async def chat_http(
    message: str = Form(...),
    session_id: str = Form(default=""),
    image: UploadFile | None = File(default=None),
    db: AsyncSession = Depends(get_db),
    llm: HermesLLM = Depends(get_llm),
) -> JSONResponse:
    """
    HTTP fallback for chat (non-streaming), supports optional image attachment.
    Send as multipart/form-data: message + optional image file.
    """
    user_message = message.strip()
    sid = session_id or str(uuid.uuid4())

    if not user_message:
        return JSONResponse({"error": "message is required"}, status_code=400)

    # Read attached image (if any)
    images: list[bytes] = []
    if image:
        image_bytes = await image.read()
        if image_bytes:
            images.append(image_bytes)

    memory = MemoryManager(session_id=sid, db=db)
    memory.add_short_term(f"User: {user_message}")

    from rag.context_builder import build_agent_context
    context = await build_agent_context(query=user_message, session_id=sid, db=db)

    agent = OrchestratorAgent(llm=llm)
    result = await agent.run(
        user_input=user_message,
        context=context,
        session_id=sid,
        images=images or None,
    )

    return JSONResponse({
        "session_id": sid,
        "response": result.result,
        "steps": len(result.steps),
        "duration_ms": result.duration_ms,
        "success": result.success,
    })


@router.post("/{run_id}/feedback")
async def submit_feedback(
    run_id: str,
    body: dict[str, Any],
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    """Submit user feedback (👍/👎) on an agent response."""
    from database.models import ResponseFeedback

    rating = body.get("rating")  # 1-5
    if rating not in range(1, 6):
        return JSONResponse({"error": "rating must be 1-5"}, status_code=400)

    feedback = ResponseFeedback(
        run_id=run_id,
        user_rating=rating,
        notes=body.get("notes"),
    )
    db.add(feedback)
    await db.commit()

    return JSONResponse({"status": "ok", "message": "Feedback recorded. Hermes is learning."})
