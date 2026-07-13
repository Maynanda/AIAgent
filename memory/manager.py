"""
ARIA / Hermes — Memory Manager
Unified interface for short-term, episodic, and semantic memory.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from database.models import AgentMemory

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Manages all memory layers for an agent session:

    - short_term: Current session context (cleared after session)
    - episodic: Past interactions and events (persisted, time-indexed)
    - semantic: Long-term knowledge facts (persisted, importance-weighted)
    - procedural: How-to knowledge, workflows (persisted)
    """

    def __init__(self, session_id: str, db: AsyncSession) -> None:
        self.session_id = session_id
        self.db = db
        self._short_term_cache: list[dict[str, Any]] = []

    # ── Short-term memory (in-session, cached) ───────────────────────────────

    def add_short_term(self, content: str, agent_type: str = "orchestrator") -> None:
        """Add to in-memory short-term store."""
        self._short_term_cache.append({
            "content": content,
            "agent_type": agent_type,
            "timestamp": datetime.utcnow().isoformat(),
        })
        # Keep last 20 items in short-term
        if len(self._short_term_cache) > 20:
            self._short_term_cache = self._short_term_cache[-20:]

    def get_short_term(self) -> list[dict[str, Any]]:
        return self._short_term_cache

    def format_short_term(self) -> str:
        if not self._short_term_cache:
            return ""
        lines = ["[Recent conversation context]"]
        for item in self._short_term_cache[-5:]:  # last 5 items
            lines.append(f"- {item['content']}")
        return "\n".join(lines)

    # ── Episodic memory (DB-backed) ──────────────────────────────────────────

    async def add_episodic(
        self,
        content: str,
        agent_type: str = "orchestrator",
        importance: float = 0.5,
        ttl_days: int = 90,
    ) -> None:
        """Store an episodic memory (past interaction/event)."""
        memory = AgentMemory(
            session_id=self.session_id,
            agent_type=agent_type,
            memory_type="episodic",
            content=content,
            importance=importance,
            expires_at=datetime.utcnow() + timedelta(days=ttl_days),
        )
        self.db.add(memory)
        await self.db.flush()

    async def get_episodic(self, limit: int = 10) -> list[str]:
        """Retrieve recent episodic memories for context."""
        result = await self.db.execute(
            select(AgentMemory)
            .where(
                and_(
                    AgentMemory.memory_type == "episodic",
                    AgentMemory.session_id == self.session_id,
                )
            )
            .order_by(AgentMemory.created_at.desc())
            .limit(limit)
        )
        memories = result.scalars().all()
        return [m.content for m in reversed(memories)]

    # ── Semantic memory (DB-backed, persistent) ───────────────────────────────

    async def add_semantic(
        self,
        content: str,
        importance: float = 0.7,
        agent_type: str = "knowledge_agent",
    ) -> None:
        """Store a long-term semantic fact."""
        memory = AgentMemory(
            session_id="global",  # semantic memory is session-independent
            agent_type=agent_type,
            memory_type="semantic",
            content=content,
            importance=importance,
        )
        self.db.add(memory)
        await self.db.flush()

    async def get_semantic(self, limit: int = 20) -> list[str]:
        """Retrieve high-importance semantic memories."""
        result = await self.db.execute(
            select(AgentMemory)
            .where(AgentMemory.memory_type == "semantic")
            .order_by(AgentMemory.importance.desc())
            .limit(limit)
        )
        return [m.content for m in result.scalars().all()]

    # ── Context assembly ──────────────────────────────────────────────────────

    async def build_context(self) -> str:
        """Assemble full memory context for the agent."""
        parts = []

        short = self.format_short_term()
        if short:
            parts.append(short)

        episodic = await self.get_episodic(limit=5)
        if episodic:
            parts.append("[Recent interactions]\n" + "\n".join(f"- {e}" for e in episodic))

        semantic = await self.get_semantic(limit=10)
        if semantic:
            parts.append("[Important knowledge]\n" + "\n".join(f"- {s}" for s in semantic))

        return "\n\n".join(parts) if parts else "No prior context available."
