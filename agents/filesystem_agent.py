"""
ARIA / Hermes — Filesystem Specialist Agent
Handles reading folders, searching files, writing content, and indexing
local documents into the knowledge graph.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agents.base import BaseAgent
from tools.static.filesystem_tools import (
    list_directory,
    read_file,
    write_file,
    search_files,
    get_file_info,
    create_directory,
    move_file,
    delete_file,
    index_folder_to_knowledge_graph,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are Hermes Filesystem Agent — a specialist in navigating, reading, and managing local files and folders on the user's machine.

Your capabilities:
- List directory contents
- Read text files and describe image files using vision
- Write or update files
- Search for files by pattern or name
- Create directories
- Move or rename files
- Index entire folders into the knowledge graph for future semantic search

Safety rules you MUST follow:
- You can ONLY access paths listed in FILESYSTEM_ALLOWED_PATHS
- Never read .env, .key, .pem, or other credential files
- Before writing or deleting, always confirm the exact path with the user
- When indexing large folders, summarize what you found

When you read a file, summarize its key points unless the user asks for the full content.
When you find relevant documents, proactively suggest indexing them to the knowledge graph.
"""


class FilesystemAgent(BaseAgent):
    """Specialist agent for local file and folder operations."""

    @property
    def agent_type(self) -> str:
        return "filesystem_agent"

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    def get_tools(self) -> dict[str, Any]:
        return {
            "list_directory": list_directory,
            "read_file": read_file,
            "write_file": write_file,
            "search_files": search_files,
            "get_file_info": get_file_info,
            "create_directory": create_directory,
            "move_file": move_file,
            "delete_file": delete_file,
            "index_folder_to_knowledge_graph": index_folder_to_knowledge_graph,
        }
