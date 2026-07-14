"""
ARIA / Hermes — Complete ORM Models
Single source of truth: all tables in one place.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


# ════════════════════════════════════════════════════════════════
#  KNOWLEDGE GRAPH — Core
# ════════════════════════════════════════════════════════════════


class Entity(Base):
    """
    Central node in the knowledge graph.
    Every object in the system (person, project, email, task, ...) is an Entity.
    """

    __tablename__ = "entities"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    type = Column(String(64), nullable=False, index=True)
    # person | project | company | task | email | activity | topic | document | tool | community
    name = Column(Text, nullable=False)
    description = Column(Text)
    embedding = Column(Vector(768))  # nomic-embed-text-v1.5 dimension
    importance = Column(Float, default=0.5)  # 0-1, auto-adjusted
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relations
    outgoing_relations = relationship(
        "Relation", foreign_keys="Relation.from_entity_id", back_populates="from_entity", lazy="dynamic"
    )
    incoming_relations = relationship(
        "Relation", foreign_keys="Relation.to_entity_id", back_populates="to_entity", lazy="dynamic"
    )
    doc_chunks = relationship("DocChunk", back_populates="entity", lazy="dynamic")

    __table_args__ = (
        Index("ix_entities_embedding", "embedding", postgresql_using="ivfflat",
              postgresql_with={"lists": 100}, postgresql_ops={"embedding": "vector_cosine_ops"}),
        Index("ix_entities_name_trgm", "name", postgresql_using="gin",
              postgresql_ops={"name": "gin_trgm_ops"}),
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "description": self.description,
            "importance": self.importance,
            "metadata": self.metadata_,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Relation(Base):
    """Directed edge in the knowledge graph."""

    __tablename__ = "relations"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    from_entity_id = Column(UUID(as_uuid=False), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False)
    to_entity_id = Column(UUID(as_uuid=False), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False)
    relation_type = Column(String(64), nullable=False)
    # works_on | assigned_to | mentioned_in | owns | reports_to | related_to
    # sent_by | received_by | part_of | blocks | depends_on | created_by | member_of
    weight = Column(Float, default=1.0)  # strengthens with usage, decays over time
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    from_entity = relationship("Entity", foreign_keys=[from_entity_id], back_populates="outgoing_relations")
    to_entity = relationship("Entity", foreign_keys=[to_entity_id], back_populates="incoming_relations")

    __table_args__ = (
        Index("ix_relations_from", "from_entity_id"),
        Index("ix_relations_to", "to_entity_id"),
        Index("ix_relations_type", "relation_type"),
    )


# ════════════════════════════════════════════════════════════════
#  PROJECTS
# ════════════════════════════════════════════════════════════════


class Project(Base):
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    entity_id = Column(UUID(as_uuid=False), ForeignKey("entities.id", ondelete="SET NULL"), nullable=True)
    title = Column(Text, nullable=False)
    description = Column(Text)
    status = Column(String(32), default="active")  # active|paused|completed|archived
    priority = Column(Integer, default=3)  # 1=critical … 5=low
    progress = Column(Integer, default=0)  # 0-100 %
    start_date = Column(Date)
    target_date = Column(Date)
    completed_at = Column(DateTime(timezone=True))
    auto_created = Column(Boolean, default=False)  # True = created by AI
    confidence_score = Column(Float)  # AI confidence when auto-created
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    blocks = relationship("ProjectBlock", back_populates="project", lazy="dynamic", order_by="ProjectBlock.order_index")
    tasks = relationship("Task", back_populates="project", lazy="dynamic")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "entity_id": self.entity_id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "progress": self.progress,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "target_date": self.target_date.isoformat() if self.target_date else None,
            "auto_created": self.auto_created,
            "confidence_score": self.confidence_score,
            "metadata": self.metadata_,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ProjectBlock(Base):
    """A block within a project — kanban item, milestone, note, decision, etc."""

    __tablename__ = "project_blocks"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    project_id = Column(UUID(as_uuid=False), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    entity_id = Column(UUID(as_uuid=False), ForeignKey("entities.id", ondelete="SET NULL"))
    title = Column(Text, nullable=False)
    block_type = Column(String(32), default="task")  # task|milestone|note|decision|risk|update
    status = Column(String(32), default="todo")  # todo|in_progress|done|blocked
    content = Column(Text)
    assignee_id = Column(UUID(as_uuid=False), ForeignKey("entities.id", ondelete="SET NULL"))
    due_date = Column(Date)
    completed_at = Column(DateTime(timezone=True))
    order_index = Column(Integer, default=0)
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    project = relationship("Project", back_populates="blocks")

    __table_args__ = (Index("ix_project_blocks_project", "project_id"),)


# ════════════════════════════════════════════════════════════════
#  TASKS
# ════════════════════════════════════════════════════════════════


class Task(Base):
    __tablename__ = "tasks"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    entity_id = Column(UUID(as_uuid=False), ForeignKey("entities.id", ondelete="SET NULL"))
    title = Column(Text, nullable=False)
    description = Column(Text)
    status = Column(String(32), default="todo")  # todo|in_progress|done|cancelled
    priority = Column(Integer, default=3)
    project_id = Column(UUID(as_uuid=False), ForeignKey("projects.id", ondelete="SET NULL"))
    assignee_id = Column(UUID(as_uuid=False), ForeignKey("entities.id", ondelete="SET NULL"))
    due_date = Column(Date)
    completed_at = Column(DateTime(timezone=True))
    source = Column(String(32), default="manual")  # manual|email|activity|agent
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    project = relationship("Project", back_populates="tasks")

    __table_args__ = (Index("ix_tasks_project", "project_id"),)


class ProjectWeeklySnapshot(Base):
    """
    Monday snapshot of a project's live state.
    Captured automatically each week and used to power the weekly history chart.
    Also stores the 4 AI-computed leader blocks as of that week.
    """

    __tablename__ = "project_weekly_snapshots"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    project_id = Column(UUID(as_uuid=False), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    week_start = Column(Date, nullable=False)          # Monday of the captured week

    # ── Computed KPIs ─────────────────────────────
    progress_pct = Column(Integer, default=0)          # 0-100
    tasks_total = Column(Integer, default=0)
    tasks_done = Column(Integer, default=0)
    tasks_in_progress = Column(Integer, default=0)
    tasks_blocked = Column(Integer, default=0)
    new_tasks_this_week = Column(Integer, default=0)
    completed_tasks_this_week = Column(Integer, default=0)

    # ── 4 Leader Blocks (AI-written, refreshed each week) ──
    block_progress = Column(Text)    # "67% complete — 12 of 18 tasks done"
    block_highlights = Column(Text)  # "This week: shipped API v2, unblocked auth issue"
    block_blockers = Column(Text)    # "2 blocked tasks: payment gateway, legal review"
    block_next_steps = Column(Text)  # "Next: finalize UI, user testing, send proposal"

    # ── Raw summary ────────────────────────────────
    summary = Column(Text)           # Full LLM narrative of the week
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("project_id", "week_start", name="uq_project_weekly_snapshot"),
        Index("ix_project_snapshots_project", "project_id"),
    )


class ProjectInsight(Base):
    """
    AI-detected action item, risk, or update extracted from an email,
    activity, or other incoming information.

    Status flow: pending → accepted (becomes real Task) | rejected (dismissed)
    """

    __tablename__ = "project_insights"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    project_id = Column(UUID(as_uuid=False), ForeignKey("projects.id", ondelete="CASCADE"), nullable=True)
    # null project_id = unmatched insight (shown in global inbox)

    insight_type = Column(String(32), default="auto_task")
    # auto_task | risk | blocker | update | milestone

    title = Column(Text, nullable=False)
    content = Column(Text)                             # Full extracted text
    suggested_due_date = Column(Date)                  # If AI detected a deadline

    source_type = Column(String(32))                   # email | activity | note | chat
    source_id = Column(UUID(as_uuid=False))            # FK to the source record

    confidence = Column(Float, default=0.8)            # AI match confidence (0-1)
    status = Column(String(32), default="pending")     # pending | accepted | rejected
    accepted_task_id = Column(UUID(as_uuid=False), ForeignKey("tasks.id", ondelete="SET NULL"))
    # set when accepted → promoted to a real task

    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("ix_project_insights_project", "project_id"),
        Index("ix_project_insights_status", "status"),
    )




# ════════════════════════════════════════════════════════════════
#  EMAILS
# ════════════════════════════════════════════════════════════════


class Email(Base):
    __tablename__ = "emails"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    entity_id = Column(UUID(as_uuid=False), ForeignKey("entities.id", ondelete="SET NULL"))
    subject = Column(Text)
    sender = Column(Text)
    recipients = Column(ARRAY(Text), default=list)
    body = Column(Text)
    summary = Column(Text)  # AI-generated summary
    sentiment = Column(String(16))  # positive|neutral|negative
    thread_id = Column(Text)
    message_id = Column(Text, unique=True)
    is_read = Column(Boolean, default=False)
    is_processed = Column(Boolean, default=False)  # has the agent processed it?
    received_at = Column(DateTime(timezone=True))
    processed_at = Column(DateTime(timezone=True))
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_emails_thread", "thread_id"),
        Index("ix_emails_processed", "is_processed"),
        Index("ix_emails_received", "received_at"),
    )


# ════════════════════════════════════════════════════════════════
#  ACTIVITIES
# ════════════════════════════════════════════════════════════════


class Activity(Base):
    __tablename__ = "activities"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    entity_id = Column(UUID(as_uuid=False), ForeignKey("entities.id", ondelete="SET NULL"))
    type = Column(String(32), default="note")  # note|meeting|call|milestone|decision|observation|voice_note
    content = Column(Text, nullable=False)
    source = Column(String(32), default="manual")  # manual|email|voice|import|agent
    related_entities = Column(ARRAY(UUID(as_uuid=False)), default=list)
    occurred_at = Column(DateTime(timezone=True), server_default=func.now())
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_activities_occurred", "occurred_at"),)


# ════════════════════════════════════════════════════════════════
#  PERSONS
# ════════════════════════════════════════════════════════════════


class Person(Base):
    __tablename__ = "persons"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    entity_id = Column(UUID(as_uuid=False), ForeignKey("entities.id", ondelete="SET NULL"), unique=True)
    name = Column(Text, nullable=False)
    email = Column(Text, index=True)
    phone = Column(Text)
    company_id = Column(UUID(as_uuid=False), ForeignKey("entities.id", ondelete="SET NULL"))
    role = Column(Text)
    notes = Column(Text)
    last_interacted = Column(DateTime(timezone=True))
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# ════════════════════════════════════════════════════════════════
#  DOCUMENTS & REPORTS
# ════════════════════════════════════════════════════════════════


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    entity_id = Column(UUID(as_uuid=False), ForeignKey("entities.id", ondelete="SET NULL"))
    title = Column(Text, nullable=False)
    content = Column(Text)
    doc_type = Column(String(32), default="note")  # report|note|wiki|weekly_report|spec
    project_id = Column(UUID(as_uuid=False), ForeignKey("projects.id", ondelete="SET NULL"))
    version = Column(Integer, default=1)
    period_start = Column(Date)  # for weekly reports
    period_end = Column(Date)
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# ════════════════════════════════════════════════════════════════
#  RAG — Document Chunks
# ════════════════════════════════════════════════════════════════


class DocChunk(Base):
    """Chunked text for vector retrieval (RAG)."""

    __tablename__ = "doc_chunks"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    entity_id = Column(UUID(as_uuid=False), ForeignKey("entities.id", ondelete="CASCADE"))
    content = Column(Text, nullable=False)
    embedding = Column(Vector(768))
    chunk_index = Column(Integer, default=0)
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    entity = relationship("Entity", back_populates="doc_chunks")

    __table_args__ = (
        Index("ix_doc_chunks_embedding", "embedding", postgresql_using="ivfflat",
              postgresql_with={"lists": 100}, postgresql_ops={"embedding": "vector_cosine_ops"}),
        Index("ix_doc_chunks_entity", "entity_id"),
    )


# ════════════════════════════════════════════════════════════════
#  MEMORY
# ════════════════════════════════════════════════════════════════


class AgentMemory(Base):
    __tablename__ = "agent_memory"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    session_id = Column(Text, index=True)
    agent_type = Column(String(64))
    memory_type = Column(String(32))  # short_term|episodic|semantic|procedural
    content = Column(Text)
    importance = Column(Float, default=0.5)
    access_count = Column(Integer, default=0)
    expires_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_memory_session", "session_id"),
        Index("ix_memory_type", "memory_type"),
    )


# ════════════════════════════════════════════════════════════════
#  TOOL REGISTRY (Self-Evolving)
# ════════════════════════════════════════════════════════════════


class ToolRegistry(Base):
    __tablename__ = "tool_registry"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    name = Column(Text, unique=True, nullable=False)
    description = Column(Text, nullable=False)
    category = Column(String(32), default="dynamic")  # static|dynamic
    source_code = Column(Text)
    file_path = Column(Text)
    schema_ = Column("schema", JSON, default=dict)  # {input_schema, output_schema}
    version = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)
    created_by = Column(String(32), default="agent")  # system|agent|user
    usage_count = Column(Integer, default=0)
    last_used_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# ════════════════════════════════════════════════════════════════
#  SELF-IMPROVEMENT — Prompt Versioning
# ════════════════════════════════════════════════════════════════


class PromptVersion(Base):
    __tablename__ = "prompt_versions"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    prompt_key = Column(Text, nullable=False)  # e.g. 'orchestrator_system'
    version = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    score = Column(Float, default=0.5)  # 0-1, avg quality
    usage_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=False)
    created_by = Column(String(32), default="system")  # system|agent|user
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("prompt_key", "version", name="uq_prompt_key_version"),
        Index("ix_prompt_versions_key", "prompt_key"),
    )


class ResponseFeedback(Base):
    __tablename__ = "response_feedback"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    run_id = Column(UUID(as_uuid=False), ForeignKey("agent_runs.id", ondelete="SET NULL"))
    prompt_key = Column(Text)
    prompt_version = Column(Integer)
    user_rating = Column(Integer)  # 1-5, null if not rated
    auto_score = Column(Float)  # automated heuristic
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# ════════════════════════════════════════════════════════════════
#  AGENT AUDIT LOG
# ════════════════════════════════════════════════════════════════


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    session_id = Column(Text, index=True)
    user_input = Column(Text)
    agent_type = Column(String(64))
    plan = Column(JSON)  # orchestrator plan
    steps = Column(JSON)  # [{thought, action, tool, input, observation}, ...]
    result = Column(Text)
    tokens_used = Column(Integer)
    duration_ms = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    feedback = relationship("ResponseFeedback", backref="run", lazy="dynamic")

    __table_args__ = (Index("ix_agent_runs_created", "created_at"),)
