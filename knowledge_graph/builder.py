"""
ARIA / Hermes — Knowledge Graph Builder and Entity Merger
Uses LLM-based entity-relation extraction and SpaCy fallback.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Any

import spacy
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database.models import Entity, Relation, Task, Project
from llm.client import llm
from rag.embedder import embedder

logger = logging.getLogger(__name__)

# Load SpaCy model for quick entity extraction fallback
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None
    logger.warning("SpaCy model 'en_core_web_sm' not found. Fallback extraction disabled.")

_PROMPT_PATH = Path(__file__).parent.parent / "llm" / "prompts" / "tasks" / "extract_entities.txt"


class KnowledgeGraphBuilder:
    """
    Extracts entities and relations from arbitrary text (emails, notes, activities),
    resolves conflicts, merges duplicate nodes, and updates the knowledge graph.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self._prompt_template: str | None = None

    @property
    def prompt_template(self) -> str:
        if self._prompt_template is None:
            self._prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")
        return self._prompt_template

    async def ingest_document(self, text_content: str, source_entity_id: str | None = None) -> dict[str, Any]:
        """
        Ingest text content, extract graph nodes/edges, and save to database.
        Optionally link extracted nodes back to the source_entity_id (e.g., an email or note).
        """
        if not text_content.strip():
            return {"entities_added": 0, "relations_added": 0}

        # 1. Run LLM Entity Extraction
        extraction = await self._extract_via_llm(text_content)

        entities_added = 0
        relations_added = 0
        processed_entities: dict[str, str] = {}  # name -> db_entity_id

        # 2. Process Extracted Entities
        for ent_data in extraction.get("entities", []):
            name = ent_data.get("name", "").strip()
            ent_type = ent_data.get("type", "topic").strip()
            desc = ent_data.get("description")
            confidence = ent_data.get("confidence", 1.0)

            if not name or confidence < 0.6:
                continue

            # Merge or Create
            entity_id = await self.get_or_create_entity(name, ent_type, desc)
            processed_entities[name.lower()] = entity_id
            entities_added += 1

            # Link back to source document/email if provided
            if source_entity_id and entity_id != source_entity_id:
                await self.create_relation_if_not_exists(
                    from_id=source_entity_id,
                    to_id=entity_id,
                    rel_type="mentions",
                )

        # 3. Process Extracted Relations
        for rel_data in extraction.get("relations", []):
            from_name = rel_data.get("from", "").strip().lower()
            to_name = rel_data.get("to", "").strip().lower()
            rel_type = rel_data.get("type", "related_to").strip()

            from_id = processed_entities.get(from_name)
            to_id = processed_entities.get(to_name)

            if from_id and to_id:
                await self.create_relation_if_not_exists(from_id, to_id, rel_type)
                relations_added += 1

        # 4. Process Action Items (Auto-create tasks)
        for act in extraction.get("action_items", []):
            title = act.get("text", "").strip()
            if not title:
                continue
            due_date_str = act.get("due_date")
            assignee_name = act.get("assignee")

            # Check if task already exists in DB
            task_stmt = select(Task).where(Task.title == title)
            task_res = await self.db.execute(task_stmt)
            if not task_res.scalars().first():
                # Resolve assignee entity
                assignee_id = None
                if assignee_name:
                    assignee_id = processed_entities.get(assignee_name.lower())
                    if not assignee_id:
                        assignee_id = await self.get_or_create_entity(assignee_name, "person")

                # Create Task Node
                task_ent_id = await self.get_or_create_entity(title, "task")
                new_task = Task(
                    entity_id=task_ent_id,
                    title=title,
                    status="todo",
                    due_date=date.fromisoformat(due_date_str) if due_date_str else None,
                    assignee_id=assignee_id,
                    source="agent",
                )
                self.db.add(new_task)
                logger.info(f"Auto-created task: {title}")

        # 5. Process New Project Signals (Auto-create projects)
        for proj_sig in extraction.get("new_project_signals", []):
            proj_name = proj_sig.get("name", "").strip()
            confidence = proj_sig.get("confidence", 0.0)

            if proj_name and confidence >= settings.auto_project_confidence_threshold:
                # Check duplicate project
                proj_stmt = select(Project).where(Project.title == proj_name)
                proj_res = await self.db.execute(proj_stmt)
                if not proj_res.scalars().first():
                    proj_ent_id = await self.get_or_create_entity(proj_name, "project")
                    new_proj = Project(
                        entity_id=proj_ent_id,
                        title=proj_name,
                        description=proj_sig.get("evidence"),
                        status="active",
                        auto_created=True,
                        confidence_score=confidence,
                    )
                    self.db.add(new_proj)
                    logger.info(f"Auto-created project: {proj_name}")

        await self.db.commit()
        return {"entities_added": entities_added, "relations_added": relations_added}

    async def get_or_create_entity(self, name: str, entity_type: str, description: str | None = None) -> str:
        """Find matching entity (exact name or vector similarity merge) or create one."""
        # 1. Check exact match
        stmt = select(Entity).where(Entity.name == name)
        res = await self.db.execute(stmt)
        entity = res.scalars().first()

        if entity:
            # Update description if it's longer
            if description and (not entity.description or len(description) > len(entity.description)):
                entity.description = description
            return entity.id

        # 2. Check Vector similarity for duplicate entity merging
        # If vector score is > 0.95, merge.
        embedding_val = None
        try:
            embedding_val = embedder.embed_document(name)
        except Exception:
            pass

        if embedding_val:
            # Query top matching entity
            try:
                # Ordering by pgvector distance
                sim_stmt = (
                    select(Entity, Entity.embedding.cosine_distance(embedding_val).label("dist"))
                    .where(Entity.type == entity_type)
                    .order_by(text("dist"))
                    .limit(1)
                )
                sim_res = await self.db.execute(sim_stmt)
                match = sim_res.first()
                if match:
                    sim_entity, dist = match
                    score = 1.0 - float(dist)
                    if score > 0.94:  # high similarity merge threshold
                        logger.info(f"Merging duplicate entity '{name}' into existing '{sim_entity.name}' (score: {score:.2f})")
                        return sim_entity.id
            except Exception as e:
                logger.warning(f"Vector deduplication lookup failed: {e}")

        # 3. Create new entity
        new_ent = Entity(
            type=entity_type,
            name=name,
            description=description,
            embedding=embedding_val,
        )
        self.db.add(new_ent)
        await self.db.flush()
        return new_ent.id

    async def create_relation_if_not_exists(self, from_id: str, to_id: str, rel_type: str) -> None:
        """Add directed relation between two entity IDs if it doesn't already exist."""
        stmt = select(Relation).where(
            Relation.from_entity_id == from_id,
            Relation.to_entity_id == to_id,
            Relation.relation_type == rel_type,
        )
        res = await self.db.execute(stmt)
        if not res.scalars().first():
            rel = Relation(
                from_entity_id=from_id,
                to_entity_id=to_id,
                relation_type=rel_type,
                weight=1.0,
            )
            self.db.add(rel)

    async def _extract_via_llm(self, text: str) -> dict[str, Any]:
        """Call Qwen2.5-VL to extract structured json graph components."""
        try:
            response = await llm.json_chat(
                system=self.prompt_template,
                user=f"Analyze this text:\n\n{text}",
            )
            # Find JSON block
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end != -1:
                return json.loads(response[start:end])
        except Exception as e:
            logger.error(f"LLM graph extraction parsing failed: {e}")

        # Fallback to SpaCy NER if LLM fails
        return self._extract_via_spacy(text)

    def _extract_via_spacy(self, text: str) -> dict[str, Any]:
        """Lightweight NER fallback extractor using local SpaCy."""
        entities = []
        relations = []

        if not nlp:
            return {"entities": [], "relations": []}

        doc = nlp(text)
        # Map spacy type labels to Hermes types
        spacy_map = {
            "PERSON": "person",
            "ORG": "company",
            "GPE": "location",
            "DATE": "date",
        }

        seen = set()
        for ent in doc.ents:
            t = spacy_map.get(ent.label_)
            if t and ent.text.strip() not in seen:
                seen.add(ent.text.strip())
                entities.append({
                    "name": ent.text.strip(),
                    "type": t,
                    "confidence": 0.8,
                })

        return {"entities": entities, "relations": relations}
from sqlalchemy import text
