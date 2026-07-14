"""
ARIA / Hermes — Tool Registry
Manages both static (shipped) and dynamic (agent-created) tools.
"""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

DYNAMIC_TOOLS_DIR = Path(__file__).parent / "dynamic"
DYNAMIC_TOOLS_DIR.mkdir(exist_ok=True)


class ToolRegistry:
    """
    Central registry for all tools available to agents.

    - Static tools: imported at startup from tools/static/
    - Dynamic tools: written by Tool Builder Agent, loaded from tools/dynamic/
    - DB sync: usage_count and last_used_at are tracked in the DB
    """

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}
        self._load_static_tools()
        self._load_dynamic_tools()

    def _load_static_tools(self) -> None:
        """Import all static tool modules."""
        static_dir = Path(__file__).parent / "static"
        for module_path in static_dir.glob("*.py"):
            if module_path.name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(
                    f"tools.static.{module_path.stem}", module_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Register all functions decorated with @tool
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if callable(attr) and hasattr(attr, "_is_hermes_tool"):
                        self.register(attr, category="static")

            except Exception as e:
                logger.error(f"Failed to load static tool module {module_path}: {e}")

    def load_dynamic_tools(self) -> None:
        """Expose public trigger to hot-reload dynamically created tools."""
        self._load_dynamic_tools()

    def _load_dynamic_tools(self) -> None:
        """Load dynamically created tools from tools/dynamic/."""
        for tool_path in DYNAMIC_TOOLS_DIR.glob("*.py"):
            if tool_path.name.startswith("_"):
                continue
            self._load_tool_file(tool_path, category="dynamic")

    def _load_tool_file(self, path: Path, category: str = "dynamic") -> bool:
        """Load a single tool file and register its tools."""
        try:
            spec = importlib.util.spec_from_file_location(f"tools.dynamic.{path.stem}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if callable(attr) and hasattr(attr, "_is_hermes_tool"):
                    self.register(attr, category=category)
            return True
        except Exception as e:
            logger.error(f"Failed to load tool file {path}: {e}")
            return False

    def register(
        self,
        fn: Callable,
        category: str = "dynamic",
        description: str | None = None,
        name: str | None = None,
    ) -> None:
        """Register a tool function."""
        tool_name = name or fn.__name__
        tool_desc = description or (fn.__doc__ or "").strip().splitlines()[0]

        self._tools[tool_name] = {
            "fn": fn,
            "name": tool_name,
            "description": tool_desc,
            "category": category,
            "schema": getattr(fn, "_schema", {}),
            "usage_count": 0,
        }
        logger.debug(f"Registered tool: {tool_name} [{category}]")

    def register_dynamic_tool(
        self,
        name: str,
        description: str,
        source_code: str,
        schema: dict[str, Any],
    ) -> bool:
        """
        Register a new tool created by the Tool Builder Agent.
        Writes the source code to tools/dynamic/, then loads it.
        """
        tool_path = DYNAMIC_TOOLS_DIR / f"{name}.py"
        tool_path.write_text(source_code, encoding="utf-8")
        success = self._load_tool_file(tool_path, category="dynamic")
        if not success:
            tool_path.unlink(missing_ok=True)
        return success

    def get(self, name: str) -> dict[str, Any] | None:
        return self._tools.get(name)

    def get_all_active(self) -> dict[str, Callable]:
        """Return {name: fn} for all active tools."""
        return {name: info["fn"] for name, info in self._tools.items()}

    def list_tools(self) -> list[dict[str, Any]]:
        """Return tool metadata for display/API."""
        return [
            {
                "name": info["name"],
                "description": info["description"],
                "category": info["category"],
                "usage_count": info["usage_count"],
                "schema": info["schema"],
            }
            for info in self._tools.values()
        ]

    def increment_usage(self, name: str) -> None:
        if name in self._tools:
            self._tools[name]["usage_count"] += 1


def tool(fn: Callable) -> Callable:
    """Decorator to mark a function as a Hermes tool."""
    fn._is_hermes_tool = True
    return fn


# Global singleton
tool_registry = ToolRegistry()
