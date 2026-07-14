"""
ARIA / Hermes — Orchestrator Agent
Routes user intent to the appropriate specialist agent.
Serves as the top-level reasoning brain with full tool access and delegation.
"""
from __future__ import annotations

import json
import logging
import time
import re
from pathlib import Path
from typing import Any, AsyncGenerator

from agents.base import BaseAgent, AgentResult, AgentStep
from agents.email_agent import EmailAgent
from agents.project_agent import ProjectAgent
from agents.knowledge_agent import KnowledgeAgent
from tools.static.project_tools import create_project, create_project_block, update_block_status, list_active_projects
from tools.static.email_tools import send_email_message, list_unread_emails, get_email_content
from tools.static.db_tools import search_knowledge_base, add_entity_relation, record_activity_note
from config import settings

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "llm" / "prompts" / "system" / "orchestrator.txt"

_DELEGATE_MAP = {
    "email_agent": ["email", "inbox", "send", "reply", "message", "mail", "imap"],
    "project_agent": ["project", "task", "kanban", "milestone", "deadline", "progress", "board"],
    "knowledge_agent": ["who", "what", "search", "find", "know", "remember", "recall", "graph"],
}


class OrchestratorAgent(BaseAgent):
    """
    Top-level orchestrator. Routes intent to specialists or handles directly.
    Uses all tools from all agents plus delegation capability.
    """

    def __init__(self, llm) -> None:
        super().__init__(llm)
        self._sub_agents = {
            "email_agent": EmailAgent(llm),
            "project_agent": ProjectAgent(llm),
            "knowledge_agent": KnowledgeAgent(llm),
        }

    @property
    def agent_type(self) -> str:
        return "orchestrator"

    @property
    def system_prompt(self) -> str:
        if _PROMPT_PATH.exists():
            return _PROMPT_PATH.read_text(encoding="utf-8")
        return (
            "You are Hermes — an advanced autonomous AI assistant and personal second brain. "
            "You have full access to the user's projects, tasks, emails, activities, and knowledge base. "
            "You reason step by step and use the right tools to complete any request.\n\n"
            "Sub-agents you can delegate to:\n"
            "  - email_agent: email reading, writing, inbox triage\n"
            "  - project_agent: project and task management\n"
            "  - knowledge_agent: knowledge graph queries and memory\n\n"
            "To delegate: Action: delegate_to_agent({\"agent\": \"email_agent\", \"task\": \"...\"})\n"
            "Or use any tool directly."
        )

    def get_tools(self) -> dict[str, Any]:
        """Full tool set: all static tools + delegation."""
        return {
            # Project tools
            "create_project": create_project,
            "create_project_block": create_project_block,
            "update_block_status": update_block_status,
            "list_active_projects": list_active_projects,
            # Email tools
            "send_email_message": send_email_message,
            "list_unread_emails": list_unread_emails,
            "get_email_content": get_email_content,
            # Knowledge tools
            "search_knowledge_base": search_knowledge_base,
            "add_entity_relation": add_entity_relation,
            "record_activity_note": record_activity_note,
            # Meta: delegation
            "delegate_to_agent": self._delegate_to_agent,
        }

    def _classify_intent(self, user_input: str) -> str | None:
        """Quick keyword classification to suggest a sub-agent."""
        text = user_input.lower()
        for agent_key, keywords in _DELEGATE_MAP.items():
            if any(kw in text for kw in keywords):
                return agent_key
        return None

    async def _delegate_to_agent(self, agent: str, task: str) -> str:
        """Delegate a sub-task to a specialist agent and return its result."""
        if agent not in self._sub_agents:
            return f"Unknown agent: {agent}. Available: {list(self._sub_agents.keys())}"

        sub_agent = self._sub_agents[agent]
        sub_agent.session_id = self.session_id
        logger.info(f"[Orchestrator] Delegating to {agent}: {task[:80]}")

        result = await sub_agent.run(user_input=task, context="", session_id=self.session_id)
        return f"[{agent} result]: {result.result}"

    async def stream_run(
        self,
        user_input: str,
        context: str = "",
        session_id: str | None = None,
        images: list[Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Full streaming ReAct loop — yields tokens live.
        Intermediate thoughts/actions are streamed so the UI can show reasoning.
        Images are sent on the first iteration only for visual context grounding.
        """
        if session_id:
            self.session_id = session_id

        start_time = time.time()
        self._steps: list[AgentStep] = []
        tools = self.get_tools()
        steps_count = 0

        for iteration in range(settings.agent_max_iterations):
            steps_count += 1

            # Build prompt
            prompt = self._build_react_prompt(user_input, context, self._steps)
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]

            # Stream reasoning tokens — images only sent on first pass
            iter_images = images if iteration == 0 else None
            raw_response = ""
            async for token in self.llm.stream(messages, images=iter_images, max_new_tokens=1024):
                raw_response += token
                yield token

            # Parse
            step = self._parse_react_response(raw_response)
            self._steps.append(step)

            # Final answer
            if step.is_final or step.action == "FINAL_ANSWER":
                duration = int((time.time() - start_time) * 1000)
                yield f"\n\n__DONE__:{json.dumps({'steps': steps_count, 'duration_ms': duration, 'run_id': self.session_id})}"

                # Persist run to DB in background
                await self._save_run_record(
                    user_input=user_input,
                    result=step.thought,
                    steps=self._steps,
                    duration_ms=duration,
                )
                return

            # Execute tool
            if step.tool_name and step.tool_name in tools:
                yield f"\n\n[Using tool: {step.tool_name}...]\n"
                try:
                    tool_fn = tools[step.tool_name]
                    from agents.base import asyncio_is_coroutine
                    if asyncio_is_coroutine(tool_fn):
                        observation = await tool_fn(**step.tool_input)
                    else:
                        observation = tool_fn(**step.tool_input)
                    step.observation = str(observation)
                    yield f"[Observation: {step.observation[:150]}{'...' if len(step.observation) > 150 else ''}]\n\n"
                except Exception as e:
                    step.observation = f"Tool error: {e}"
                    logger.warning(f"Tool {step.tool_name} failed: {e}")
            elif step.tool_name:
                step.observation = f"Unknown tool: {step.tool_name}"

        # Max iterations
        duration = int((time.time() - start_time) * 1000)
        yield "\n\nI've reached my reasoning limit. Here's what I've gathered so far.\n"
        yield f"\n\n__DONE__:{json.dumps({'steps': steps_count, 'duration_ms': duration, 'run_id': self.session_id})}"

    async def _save_run_record(
        self,
        user_input: str,
        result: str,
        steps: list[AgentStep],
        duration_ms: int,
    ) -> None:
        """Persist agent run to database for auditing and feedback."""
        try:
            from database.connection import AsyncSessionLocal
            from database.models import AgentRun
            async with AsyncSessionLocal() as db:
                run = AgentRun(
                    session_id=self.session_id,
                    agent_type=self.agent_type,
                    user_input=user_input,
                    result=result,
                    steps_json=[
                        {
                            "thought": s.thought,
                            "tool_name": s.tool_name,
                            "tool_input": s.tool_input,
                            "observation": s.observation,
                            "is_final": s.is_final,
                        }
                        for s in steps
                    ],
                    steps_count=len(steps),
                    duration_ms=duration_ms,
                    success=True,
                )
                db.add(run)
                await db.commit()
        except Exception as e:
            logger.warning(f"Could not save agent run record: {e}")
