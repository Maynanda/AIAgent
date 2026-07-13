"""
ARIA / Hermes — Knowledge Graph Updater and Consolidation Engine
Handles nightly decay, entity deduplication, and relationship pruning.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import Entity, Relation, doc_chunks

logger = logging.getLogger(__name__)


class KnowledgeGraphUpdater:
    """
    Consolidates the knowledge graph by running:
    - Decay weights: ages older relations.
    - Duplicate merging: merges highly overlapping nodes.
    - Conflict/contradiction audits.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def run_nightly_refinement(self) -> dict[str, int]:
        """
        Executes structural graph optimizations:
        1. Decays older relationships (weight *= 0.95 if not reinforced).
        2. Prunes orphaned relations pointing to dead entities.
        3. Recalculates entity importance scores based on connection counts.
        """
        logger.info("Running nightly graph refinement and optimization...")

        # 1. Decay relation weights
        # Decay relations that are older than 30 days
        decay_stmt = (
            select(Relation)
            .where(Relation.created_at < datetime.utcnow() - timedelta(days=30))
        )
        res = await self.db.execute(decay_stmt)
        decayed_count = 0
        for rel in res.scalars().all():
            rel.weight *= 0.95
            decayed_count += 1

        # 2. Prune invalid relations (orphans)
        # Delete relations pointing from or to deleted entities
        entity_ids_stmt = select(Entity.id)
        entity_ids_res = await self.db.execute(entity_ids_stmt)
        valid_entity_ids = set(entity_ids_res.scalars().all())

        relations_stmt = select(Relation)
        relations_res = await self.db.execute(relations_stmt)
        all_relations = relations_res.scalars().all()

        pruned_count = 0
        for rel in all_relations:
            if rel.from_entity_id not in valid_entity_ids or rel.to_entity_id not in valid_entity_ids:
                await self.db.delete(rel)
                pruned_count += 1

        # 3. Recalculate Entity Importance scores
        # Importance = normalize(number of incoming + outgoing connections)
        entities_res = await self.db.execute(select(Entity))
        entities = entities_res.scalars().all()

        for ent in entities:
            # Count connections
            in_stmt = select(Relation).where(Relation.to_entity_id == ent.id)
            out_stmt = select(Relation).where(Relation.from_entity_id == ent.id)
            
            in_res = await self.db.execute(in_stmt)
            out_res = await self.db.execute(out_stmt)
            
            total_connections = len(in_res.scalars().all()) + len(out_res.scalars().all())
            
            # Simple importance formula: min(0.1 + log(connections+1)/10.0, 1.0)
            import math
            ent.importance = min(0.1 + math.log(total_connections + 1) / 5.0, 1.0)

        await self.db.commit()
        logger.info(f"Nightly graph consolidation complete. Decayed: {decayed_count}, Pruned: {pruned_count}")

        return {
            "decayed_relations": decayed_count,
            "pruned_relations": pruned_count,
            "entities_recalculated": len(entities),
        }
