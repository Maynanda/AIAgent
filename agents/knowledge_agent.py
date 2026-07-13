"""
ARIA / Hermes — Knowledge Specialist Agent
Handles graph queries, entity extraction, relation building, and semantic search.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agents.base import BaseAgent
from tools.static.db_tools import (
    search_knowledge_base,
    add_entity_relation,
    record_activity_note,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are Hermes Knowledge Agent — a specialist in the knowledge graph, semantic search, and information retrieval.

Your responsibilities:
- Search the knowledge base to answer questions accurately
- Add entities and relations to grow the knowledge graph
- Resolve ambiguous references by searching context first
- Record important facts, decisions, or notes to long-term memory
- Synthesize information from multiple sources to answer complex questions

Always search the knowledge base before stating you don't know something.
When you discover new facts from conversations, proactively record them.
"""


class KnowledgeAgent(BaseAgent):
    """Specialist agent for knowledge graph operations and semantic retrieval."""

    @property
    def agent_type(self) -> str:
        return "knowledge_agent"

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    def get_tools(self) -> dict[str, Any]:
        return {
            "search_knowledge_base": search_knowledge_base,
            "add_entity_relation": add_entity_relation,
            "record_activity_note": record_activity_note,
        }
