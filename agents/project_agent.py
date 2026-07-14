"""
ARIA / Hermes — Project & Task Specialist Agent
Handles project creation, block updates, task tracking, and timeline management.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agents.base import BaseAgent
from tools.static.project_tools import (
    create_project,
    create_project_block,
    update_block_status,
    list_active_projects,
    get_project_progress,
    refresh_project_blocks,
    list_project_insights,
    accept_project_insight,
    reject_project_insight,
    get_project_weekly_history,
    send_projects_digest_email,
)
from tools.static.db_tools import search_knowledge_base, record_activity_note

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "llm" / "prompts" / "system" / "project_agent.txt"

_SYSTEM_PROMPT = """You are Hermes Project Agent — a specialist in managing projects, tasks, milestones, and timelines.

Your responsibilities:
- Create and update projects and their Kanban blocks (tasks, milestones, decisions)
- Track progress and notify when projects are at risk of delay
- Break down high-level goals into specific actionable tasks
- Update task statuses and log completions
- Search the knowledge base to find related context before creating anything new
- Review and manage AI-detected insights (auto-tasks, risks, blockers) extracted from emails and activities
- Refresh the 4 leader blocks whenever significant new information is added
- Show weekly history trends and send project digest emails when asked

Always check existing projects before creating new ones to avoid duplicates.
When creating tasks, be specific: include a clear title, and mark the correct status.
After accepting an insight, always refresh the project's leader blocks.
"""


class ProjectAgent(BaseAgent):
    """Specialist agent for project and task management."""

    @property
    def agent_type(self) -> str:
        return "project_agent"

    @property
    def system_prompt(self) -> str:
        if _PROMPT_PATH.exists():
            return _PROMPT_PATH.read_text(encoding="utf-8")
        return _SYSTEM_PROMPT

    def get_tools(self) -> dict[str, Any]:
        return {
            # Core project tools
            "create_project": create_project,
            "create_project_block": create_project_block,
            "update_block_status": update_block_status,
            "list_active_projects": list_active_projects,
            # Intelligence tools
            "get_project_progress": get_project_progress,
            "refresh_project_blocks": refresh_project_blocks,
            "list_project_insights": list_project_insights,
            "accept_project_insight": accept_project_insight,
            "reject_project_insight": reject_project_insight,
            "get_project_weekly_history": get_project_weekly_history,
            "send_projects_digest_email": send_projects_digest_email,
            # Knowledge tools
            "search_knowledge_base": search_knowledge_base,
            "record_activity_note": record_activity_note,
        }
