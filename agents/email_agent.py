"""
ARIA / Hermes — Email Specialist Agent
Handles email reading, summarization, reply drafting, and inbox triage.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agents.base import BaseAgent
from tools.static.email_tools import send_email_message, list_unread_emails, get_email_content

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "llm" / "prompts" / "system" / "email_agent.txt"

_SYSTEM_PROMPT = """You are Hermes Email Agent — a specialist in reading, summarizing, and managing email communications.

Your responsibilities:
- Retrieve and read emails on behalf of the user
- Summarize email threads, tone, and required actions clearly
- Draft professional reply emails when asked
- Detect action items, deadlines, and key contacts within email bodies
- Send emails when the user explicitly requests it

Always maintain a professional tone. When asked to draft a reply, present it for user review before confirming to send.
Use tools methodically and report back findings in clear, structured form.
"""


class EmailAgent(BaseAgent):
    """Specialist agent for email-related tasks."""

    @property
    def agent_type(self) -> str:
        return "email_agent"

    @property
    def system_prompt(self) -> str:
        if _PROMPT_PATH.exists():
            return _PROMPT_PATH.read_text(encoding="utf-8")
        return _SYSTEM_PROMPT

    def get_tools(self) -> dict[str, Any]:
        return {
            "send_email_message": send_email_message,
            "list_unread_emails": list_unread_emails,
            "get_email_content": get_email_content,
        }
