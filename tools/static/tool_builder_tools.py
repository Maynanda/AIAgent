"""
ARIA / Hermes — Static Tool Builder Tools
Allows Hermes to dynamically create new tools for itself when requested in chat.
"""
from __future__ import annotations

import logging
from tools.registry import tool
from agents.tool_builder import create_dynamic_tool as _create_tool

logger = logging.getLogger(__name__)


@tool
async def create_new_dynamic_tool(name: str, description: str, python_code: str) -> str:
    """
    Dynamically compile, validate, and register a new Python tool for Hermes to use.
    The code must import the @tool decorator: `from tools.registry import tool`
    and decorate the entry point function.
    Banned modules like 'os', 'subprocess', 'sys', 'socket' are not allowed for security reasons.
    Args:
        name: Unique snake_case name of the tool (e.g. 'fetch_bitcoin_price')
        description: Clear explanation of what the tool does
        python_code: The Python code containing the tool function definition
    """
    try:
        res = await _create_tool(name=name, description=description, code=python_code, author="hermes_agent")
        if res["success"]:
            return f"✅ Successfully created and hot-loaded tool '{name}'. I can now use it in my reasoning."
        return f"❌ Failed to create tool: {res['error']}"
    except Exception as e:
        logger.error(f"Error creating dynamic tool via static tool wrapper: {e}")
        return f"❌ Error: {str(e)}"
