"""
ARIA / Hermes — Prompt Versioning & Self-Improvement Engine
Tracks prompt performance and evolves prompts based on feedback metrics.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import AsyncSessionLocal
from database.models import PromptVersion, ResponseFeedback, AgentRun

logger = logging.getLogger(__name__)


class PromptEvolutionEngine:
    """
    Analyzes feedback scores to identify underperforming prompts
    and uses the LLM to draft improved versions.
    """

    async def run_weekly_analysis(self) -> dict[str, Any]:
        """
        Analyze the last 7 days of agent runs and feedback scores.
        If a prompt_key averages below threshold, propose an improvement.
        """
        async with AsyncSessionLocal() as db:
            results = await self._analyze_performance(db)
            improvements = []

            for prompt_key, avg_score in results.items():
                if avg_score < 3.5:  # Scale 1-5, below 3.5 triggers improvement
                    logger.info(f"Prompt '{prompt_key}' underperforming (avg score: {avg_score:.2f}). Drafting improvement...")
                    improved = await self._draft_improved_prompt(db, prompt_key)
                    if improved:
                        improvements.append({"prompt_key": prompt_key, "avg_score": avg_score, "new_version": improved})

            return {
                "analyzed_prompts": len(results),
                "improvements_triggered": len(improvements),
                "improvements": improvements,
            }

    async def _analyze_performance(self, db: AsyncSession) -> dict[str, float]:
        """
        Calculate average feedback rating for each prompt key over the last 7 days.
        """
        week_ago = datetime.utcnow() - timedelta(days=7)

        # Join AgentRun with ResponseFeedback
        stmt = (
            select(AgentRun.agent_type, func.avg(ResponseFeedback.rating).label("avg_rating"))
            .join(ResponseFeedback, ResponseFeedback.run_id == AgentRun.id)
            .where(AgentRun.created_at >= week_ago)
            .group_by(AgentRun.agent_type)
        )

        res = await db.execute(stmt)
        return {row.agent_type: float(row.avg_rating) for row in res.all()}

    async def _draft_improved_prompt(self, db: AsyncSession, prompt_key: str) -> str | None:
        """Use the LLM to draft an improved system prompt based on bad outcomes."""
        # Get the current active prompt
        stmt = select(PromptVersion).where(
            PromptVersion.prompt_key == prompt_key,
            PromptVersion.is_active == True,
        )
        res = await db.execute(stmt)
        current_prompt = res.scalars().first()
        if not current_prompt:
            return None

        # Get sample of bad runs for context
        bad_runs_stmt = (
            select(AgentRun.user_input, AgentRun.result, ResponseFeedback.notes)
            .join(ResponseFeedback, ResponseFeedback.run_id == AgentRun.id)
            .where(AgentRun.agent_type == prompt_key, ResponseFeedback.rating <= 2)
            .limit(5)
        )
        bad_runs_res = await db.execute(bad_runs_stmt)
        bad_examples = bad_runs_res.all()

        if not bad_examples:
            return None

        # Build improvement prompt
        examples_text = "\n".join(
            f"User asked: '{r.user_input}'\nAgent said: '{r.result[:200]}'\nUser feedback: '{r.notes}'"
            for r in bad_examples
        )

        improvement_prompt = f"""You are a prompt engineer. Here is a system prompt that is underperforming:

CURRENT PROMPT:
{current_prompt.content}

EXAMPLES WHERE IT FAILED:
{examples_text}

Please write an improved version of this system prompt that would handle these cases better.
Output ONLY the improved prompt text, nothing else."""

        try:
            from llm.client import llm
            messages = [
                {"role": "system", "content": "You are an expert prompt engineer."},
                {"role": "user", "content": improvement_prompt},
            ]
            improved_text = await llm.generate(messages, max_new_tokens=512)

            # Save new version (inactive by default — requires human review)
            new_version_number = current_prompt.version + 1
            new_version = PromptVersion(
                prompt_key=prompt_key,
                version=new_version_number,
                content=improved_text.strip(),
                is_active=False,  # Inactive until reviewed
                created_by="hermes_self_improvement",
                performance_score=0.0,
                notes=f"Auto-generated improvement v{new_version_number} based on low feedback scores",
            )
            db.add(new_version)
            await db.commit()
            logger.info(f"Drafted improved prompt v{new_version_number} for '{prompt_key}' (pending review)")
            return improved_text.strip()

        except Exception as e:
            logger.error(f"Prompt improvement draft failed for {prompt_key}: {e}")
            return None

    async def activate_prompt_version(self, prompt_key: str, version: int) -> str:
        """Activate a specific prompt version and deactivate all others for that key."""
        async with AsyncSessionLocal() as db:
            # Deactivate all versions for this key
            all_stmt = select(PromptVersion).where(PromptVersion.prompt_key == prompt_key)
            all_res = await db.execute(all_stmt)
            for p in all_res.scalars().all():
                p.is_active = (p.version == version)

            await db.commit()
            return f"Activated prompt version {version} for key '{prompt_key}'."


# Singleton
prompt_evolution = PromptEvolutionEngine()
