"""
ARIA / Hermes — Orchestrator Agent
The top-level agent that receives user input, plans, and delegates
to specialist agents or calls tools directly.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from agents.base import BaseAgent
from llm.client import HermesLLM

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "llm" / "prompts" / "system" / "orchestrator.txt"


class OrchestratorAgent(BaseAgent):
    """
    Top-level agent that routes requests to specialist agents.
    Has access to all tools across the system.
    """

    def __init__(self, llm: HermesLLM) -> None:
        super().__init__(llm)
        self._system_prompt: str | None = None

    @property
    def agent_type(self) -> str:
        return "orchestrator"

    @property
    def system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")
        return self._system_prompt

    def get_tools(self) -> dict[str, Any]:
        """Orchestrator has access to the full tool registry."""
        from tools.registry import tool_registry
        return tool_registry.get_all_active()

    async def plan_and_route(self, user_input: str, context: str) -> dict[str, Any]:
        """
        High-level planning step — decide which agents/tools to involve.
        Returns a plan dict.
        """
        plan_prompt = f"""
Given this user request, create a brief execution plan.
Output JSON only:
{{
  "intent": "brief description of what user wants",
  "agents_needed": ["orchestrator", "email_agent", "project_agent", etc.],
  "tools_needed": ["tool_name1", "tool_name2"],
  "steps": ["step 1", "step 2"],
  "is_simple": true/false  // true = handle directly, false = delegate
}}

User request: {user_input}
Context: {context[:500]}
"""
        response = await self.llm.chat(self.system_prompt, plan_prompt, json_mode=True)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"intent": user_input, "is_simple": True, "steps": []}
