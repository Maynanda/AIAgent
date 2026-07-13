"""
ARIA / Hermes — Chat WebSocket Route
Handles real-time streaming chat with the Hermes orchestrator.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
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
      Client sends: {"message": "...", "session_id": "optional-uuid"}
      Server streams: {"type": "token", "content": "..."}
                      {"type": "done", "metadata": {...}}
                      {"type": "error", "content": "..."}
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    session_id = str(uuid.uuid4())

    try:
        while True:
            # Receive message
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = {"message": raw}

            user_message = data.get("message", "").strip()
            if not user_message:
                continue

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
                context = await memory.build_context()

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
    body: dict[str, Any],
    db: AsyncSession = Depends(get_db),
    llm: HermesLLM = Depends(get_llm),
) -> JSONResponse:
    """
    HTTP fallback for chat (non-streaming).
    Body: {"message": "...", "session_id": "optional"}
    """
    user_message = body.get("message", "").strip()
    session_id = body.get("session_id", str(uuid.uuid4()))

    if not user_message:
        return JSONResponse({"error": "message is required"}, status_code=400)

    memory = MemoryManager(session_id=session_id, db=db)
    context = await memory.build_context()

    agent = OrchestratorAgent(llm=llm)
    result = await agent.run(
        user_input=user_message,
        context=context,
        session_id=session_id,
    )

    return JSONResponse({
        "session_id": session_id,
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
