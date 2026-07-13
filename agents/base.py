"""
ARIA / Hermes — Base Agent
All specialist agents inherit from this. Implements the core ReAct loop.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from config import settings
from llm.client import HermesLLM

logger = logging.getLogger(__name__)


@dataclass
class AgentStep:
    """One step in the ReAct loop."""

    thought: str = ""
    action: str = ""
    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    is_final: bool = False


@dataclass
class AgentResult:
    """Final result of an agent run."""

    session_id: str
    agent_type: str
    user_input: str
    steps: list[AgentStep]
    result: str
    tokens_used: int = 0
    duration_ms: int = 0
    success: bool = True
    error: str | None = None


class BaseAgent(ABC):
    """
    Abstract base agent implementing the ReAct (Reason + Act) loop.

    Subclasses must implement:
    - system_prompt: str property — the agent's system prompt
    - get_tools() -> dict[str, callable] — available tools
    - format_tools_description() -> str — how tools are described to the LLM
    """

    def __init__(self, llm: HermesLLM) -> None:
        self.llm = llm
        self.session_id: str = str(uuid.uuid4())
        self._steps: list[AgentStep] = []

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Identifier for this agent type."""
        ...

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt for this agent."""
        ...

    @abstractmethod
    def get_tools(self) -> dict[str, Any]:
        """Return dict of tool_name -> tool_function."""
        ...

    def format_tools_description(self) -> str:
        """Format available tools for the LLM prompt."""
        tools = self.get_tools()
        lines = ["Available tools:"]
        for name, tool in tools.items():
            doc = getattr(tool, "__doc__", "No description") or "No description"
            lines.append(f"  - {name}: {doc.strip().splitlines()[0]}")
        return "\n".join(lines)

    def _build_react_prompt(
        self,
        user_input: str,
        context: str,
        history: list[AgentStep],
    ) -> str:
        """Build the ReAct format prompt including history."""
        history_text = ""
        for step in history:
            history_text += f"\nThought: {step.thought}"
            if step.tool_name:
                history_text += f"\nAction: {step.tool_name}({json.dumps(step.tool_input)})"
                history_text += f"\nObservation: {step.observation}"

        return f"""
{self.format_tools_description()}

Context from knowledge base:
{context}

User request: {user_input}
{history_text}

Respond in this exact format:
Thought: <your reasoning>
Action: <tool_name>(<json_args>) OR Action: FINAL_ANSWER
Answer: <your answer — only when Action is FINAL_ANSWER>
"""

    async def run(
        self,
        user_input: str,
        context: str = "",
        session_id: str | None = None,
    ) -> AgentResult:
        """Execute the ReAct loop until completion or max iterations."""
        if session_id:
            self.session_id = session_id

        start_time = time.time()
        self._steps = []
        tools = self.get_tools()

        for iteration in range(settings.agent_max_iterations):
            # Build prompt with full history
            prompt = self._build_react_prompt(user_input, context, self._steps)

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]

            # Generate next step
            raw_response = await self.llm.generate(messages, max_new_tokens=1024)
            logger.debug(f"[{self.agent_type}] Iteration {iteration}: {raw_response[:200]}")

            # Parse the response
            step = self._parse_react_response(raw_response)
            self._steps.append(step)

            # Check for final answer
            if step.is_final or step.action == "FINAL_ANSWER":
                duration = int((time.time() - start_time) * 1000)
                return AgentResult(
                    session_id=self.session_id,
                    agent_type=self.agent_type,
                    user_input=user_input,
                    steps=self._steps,
                    result=step.thought,  # final answer is in thought when is_final
                    duration_ms=duration,
                    success=True,
                )

            # Execute tool call
            if step.tool_name and step.tool_name in tools:
                try:
                    tool_fn = tools[step.tool_name]
                    if asyncio_is_coroutine(tool_fn):
                        observation = await tool_fn(**step.tool_input)
                    else:
                        observation = tool_fn(**step.tool_input)
                    step.observation = str(observation)
                except Exception as e:
                    step.observation = f"Tool error: {e}"
                    logger.warning(f"Tool {step.tool_name} failed: {e}")
            elif step.tool_name:
                step.observation = f"Unknown tool: {step.tool_name}"

        # Max iterations reached
        duration = int((time.time() - start_time) * 1000)
        return AgentResult(
            session_id=self.session_id,
            agent_type=self.agent_type,
            user_input=user_input,
            steps=self._steps,
            result="I reached the maximum number of reasoning steps. Here's what I found so far: "
                   + (self._steps[-1].thought if self._steps else "No progress made."),
            duration_ms=duration,
            success=False,
            error="Max iterations reached",
        )

    def _parse_react_response(self, response: str) -> AgentStep:
        """Parse a ReAct-formatted LLM response into an AgentStep."""
        step = AgentStep()
        lines = response.strip().split("\n")

        for i, line in enumerate(lines):
            if line.startswith("Thought:"):
                step.thought = line[len("Thought:"):].strip()
            elif line.startswith("Action:"):
                action_text = line[len("Action:"):].strip()
                if action_text == "FINAL_ANSWER":
                    step.is_final = True
                    step.action = "FINAL_ANSWER"
                    # Answer is in the next line
                    for j in range(i + 1, len(lines)):
                        if lines[j].startswith("Answer:"):
                            step.thought = lines[j][len("Answer:"):].strip()
                            break
                else:
                    # Parse tool call: tool_name({"key": "value"})
                    try:
                        paren_idx = action_text.index("(")
                        step.tool_name = action_text[:paren_idx].strip()
                        step.action = step.tool_name
                        json_str = action_text[paren_idx + 1:].rstrip(")")
                        step.tool_input = json.loads(json_str) if json_str.strip() else {}
                    except (ValueError, json.JSONDecodeError) as e:
                        step.tool_name = action_text
                        step.tool_input = {}
                        logger.warning(f"Could not parse tool call: {action_text} — {e}")

        return step

    async def stream_run(
        self,
        user_input: str,
        context: str = "",
        session_id: str | None = None,
    ):
        """
        Streaming version of run() — yields tokens as they are generated.
        Suitable for WebSocket connections.
        """
        # For streaming, we run the full loop but stream the final response
        result = await self.run(user_input, context, session_id)

        # Stream the final answer character by character for smooth UX
        for char in result.result:
            yield char

        # Yield the metadata at the end
        yield f"\n\n__DONE__:{json.dumps({'steps': len(result.steps), 'duration_ms': result.duration_ms})}"


def asyncio_is_coroutine(fn: Any) -> bool:
    """Check if a function is a coroutine function."""
    import asyncio
    return asyncio.iscoroutinefunction(fn)
