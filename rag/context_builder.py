"""
ARIA / Hermes — Context Assembly & Prompt Packing
Combines short-term memory, episodic events, and hybrid RAG searches.
"""
from __future__ import annotations

import logging
from sqlalchemy.ext.asyncio import AsyncSession

from memory.manager import MemoryManager
from rag.retriever import assemble_context_block

logger = logging.getLogger(__name__)


async def build_agent_context(
    query: str,
    session_id: str,
    db: AsyncSession,
) -> str:
    """
    Assembles the complete reasoning context pack:
    1. Short-term session memory (recent lines of chat)
    2. Episodic memory (past relevant interactions)
    3. Hybrid RAG results (vector chunks + graph matches + BM25 keywords)
    """
    # 1. Fetch memory layers
    memory = MemoryManager(session_id=session_id, db=db)
    memory_context = await memory.build_context()

    # 2. Fetch search context
    search_context = await assemble_context_block(query=query, db=db, limit=8)

    # 3. Assemble and pack
    context_pack = f"""
{memory_context}

=== KNOWLEDGE BASE SEARCH RESULTS ===
{search_context}
"""
    return context_pack.strip()
