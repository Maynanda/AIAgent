"""
ARIA / Hermes — Tools Management API Routes
Lists static and dynamic tools, enables creating dynamic tools via the agent.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import get_db
from database.models import ToolRegistry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tools", tags=["tools"])


@router.get("")
async def list_tools(db: AsyncSession = Depends(get_db)) -> list[dict[str, Any]]:
    """List all registered tools (static + dynamic)."""
    stmt = select(ToolRegistry).order_by(ToolRegistry.category, ToolRegistry.name)
    res = await db.execute(stmt)
    tools = res.scalars().all()
    return [
        {
            "id": t.id,
            "name": t.name,
            "description": t.description,
            "category": t.category,
            "is_active": t.is_active,
            "created_by": t.created_by,
        }
        for t in tools
    ]


@router.post("/dynamic")
async def create_tool(body: dict[str, Any]) -> dict[str, Any]:
    """
    Create a new dynamic tool (sandbox code validation + hot-load).
    Body: { name, description, code }
    """
    name = body.get("name", "").strip()
    description = body.get("description", "").strip()
    code = body.get("code", "").strip()

    if not name or not code:
        raise HTTPException(status_code=400, detail="name and code are required")

    from agents.tool_builder import create_dynamic_tool
    result = await create_dynamic_tool(name=name, description=description, code=code)
    if not result["success"]:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@router.patch("/{tool_id}/toggle")
async def toggle_tool(tool_id: str, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    """Enable or disable a tool."""
    tool = await db.get(ToolRegistry, tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    tool.is_active = not tool.is_active
    await db.commit()
    return JSONResponse({"name": tool.name, "is_active": tool.is_active})
