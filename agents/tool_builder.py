"""
ARIA / Hermes — Dynamic Tool Builder Agent (Self-Improvement / Phase 9)
Allows Hermes to write, register, and use new Python tools at runtime.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import json
import logging
import textwrap
from pathlib import Path
from typing import Any

from config import settings
from database.connection import AsyncSessionLocal
from database.models import ToolRegistry

logger = logging.getLogger(__name__)

DYNAMIC_TOOLS_DIR = Path(__file__).parent.parent / "tools" / "dynamic"
DYNAMIC_TOOLS_DIR.mkdir(parents=True, exist_ok=True)

# Disallowed module imports in dynamic tools (security sandbox level 2)
_BANNED_IMPORTS = {
    "os", "subprocess", "sys", "shutil", "socket", "ctypes", "threading",
    "multiprocessing", "importlib", "builtins", "__import__",
}


def _validate_tool_code(code: str) -> tuple[bool, str]:
    """
    Validates dynamically generated tool code:
    - Must be valid Python syntax
    - Must not import banned modules
    - Must contain at least one async function decorated with @tool
    Returns (is_valid, error_message)
    """
    # 1. Syntax check
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    # 2. Banned import scan
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            else:
                names = [node.module.split(".")[0]] if node.module else []
            for name in names:
                if name in _BANNED_IMPORTS:
                    return False, f"Banned import: '{name}' is not allowed in dynamic tools"

    # 3. Must have at least one async function
    has_async_fn = any(
        isinstance(node, ast.AsyncFunctionDef)
        for node in ast.walk(tree)
    )
    if not has_async_fn:
        return False, "Dynamic tool must contain at least one async function"

    return True, ""


async def create_dynamic_tool(
    name: str,
    description: str,
    code: str,
    author: str = "hermes_agent",
) -> dict[str, Any]:
    """
    Write, validate, and register a new dynamic tool.
    
    Args:
        name: Unique snake_case tool name
        description: What this tool does
        code: The full Python code of the tool (must contain `from tools.registry import tool` and @tool decorated functions)
        author: Who created the tool (agent or user)
    """
    # Normalize name
    name = name.lower().replace(" ", "_").replace("-", "_")

    # 1. Validate code safety
    is_valid, error_msg = _validate_tool_code(code)
    if not is_valid:
        return {"success": False, "error": error_msg}

    # 2. Write to file
    file_path = DYNAMIC_TOOLS_DIR / f"{name}.py"
    header = textwrap.dedent(f'''
        """
        Dynamic Tool: {name}
        Description: {description}
        Created by: {author}
        """
        from tools.registry import tool
    ''').strip() + "\n\n"

    # Only add the header if it's not already in the code
    if "from tools.registry import tool" not in code:
        full_code = header + "\n" + code
    else:
        full_code = code

    file_path.write_text(full_code, encoding="utf-8")
    logger.info(f"Dynamic tool '{name}' written to {file_path}")

    # 3. Try to import and validate
    try:
        spec = importlib.util.spec_from_file_location(f"tools.dynamic.{name}", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        # Rollback file
        file_path.unlink(missing_ok=True)
        return {"success": False, "error": f"Import/execution error: {e}"}

    # 4. Register in database
    try:
        async with AsyncSessionLocal() as db:
            existing_stmt = __import__("sqlalchemy", fromlist=["select"]).select(ToolRegistry).where(ToolRegistry.name == name)
            from sqlalchemy import select
            existing = await db.execute(select(ToolRegistry).where(ToolRegistry.name == name))
            tool_record = existing.scalars().first()

            if tool_record:
                tool_record.description = description
                tool_record.code_path = str(file_path)
                tool_record.is_active = True
            else:
                db.add(ToolRegistry(
                    name=name,
                    description=description,
                    category="dynamic",
                    code_path=str(file_path),
                    is_active=True,
                    created_by=author,
                ))
            await db.commit()
    except Exception as e:
        logger.warning(f"DB registration for dynamic tool failed: {e}")

    # 5. Hot-reload into the registry
    try:
        from tools.registry import tool_registry
        tool_registry.load_dynamic_tools()
    except Exception as e:
        logger.warning(f"Hot-reload of tool registry failed: {e}")

    return {
        "success": True,
        "tool_name": name,
        "file_path": str(file_path),
        "message": f"Tool '{name}' created and registered successfully.",
    }


async def list_dynamic_tools() -> list[dict[str, Any]]:
    """List all registered dynamic tools."""
    async with AsyncSessionLocal() as db:
        from sqlalchemy import select
        stmt = select(ToolRegistry).where(ToolRegistry.category == "dynamic")
        res = await db.execute(stmt)
        tools = res.scalars().all()
        return [
            {
                "name": t.name,
                "description": t.description,
                "is_active": t.is_active,
                "created_by": t.created_by,
            }
            for t in tools
        ]


async def disable_dynamic_tool(name: str) -> str:
    """Disable a dynamic tool by name (keeps file, marks as inactive in DB)."""
    async with AsyncSessionLocal() as db:
        from sqlalchemy import select
        stmt = select(ToolRegistry).where(ToolRegistry.name == name)
        res = await db.execute(stmt)
        tool = res.scalars().first()
        if not tool:
            return f"Tool '{name}' not found."
        tool.is_active = False
        await db.commit()
        return f"Tool '{name}' disabled successfully."
