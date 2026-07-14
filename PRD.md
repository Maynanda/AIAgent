# Product Requirements Document
# ARIA — Hermes: Autonomous Personal AI Agent
**Version:** 1.0  
**Status:** Active Development  
**Last Updated:** 2026-07-14  

---

## 1. Executive Summary

ARIA (Autonomous Reasoning & Intelligence Agent), internally codenamed **Hermes**, is a privacy-first, locally-hosted personal AI second brain. It combines a multi-agent reasoning system, a self-growing knowledge graph, multimodal vision, local filesystem access, email integration, and project management into a single unified assistant that runs entirely on the user's own hardware.

The product is designed for **knowledge workers, developers, and researchers** who want an AI assistant that learns from their personal data over time — without sending anything to external cloud providers.

---

## 2. Problem Statement

Modern knowledge workers juggle emails, project tasks, notes, files, and meetings across fragmented tools. Existing AI assistants (ChatGPT, Copilot, etc.):

- **Require cloud uploads** — personal and confidential data leaves the machine
- **Have no persistent memory** — every session starts from zero
- **Are siloed** — they cannot access local files, emails, or project state
- **Don't learn** — they cannot improve their own tools or prompts based on feedback

**Hermes solves this** by running 100% locally, maintaining a persistent knowledge graph from the user's own data, and improving itself over time.

---

## 3. Goals & Non-Goals

### Goals
- ✅ Run fully on user hardware (GPU-accelerated, 4-bit quantized)
- ✅ Build and maintain a persistent personal knowledge graph
- ✅ Support multimodal inputs (text + images in the same conversation)
- ✅ Integrate with email (IMAP/SMTP and macOS Outlook desktop)
- ✅ Provide filesystem navigation and document reading/writing
- ✅ Expose a beautiful, responsive dark-mode web UI
- ✅ Self-improve: create new tools, evolve prompts based on feedback
- ✅ Privacy guarantee: no data leaves the local machine

### Non-Goals
- ❌ Mobile app (web-first; mobile-responsive UI is acceptable)
- ❌ Multi-user / team collaboration (single-user personal assistant)
- ❌ Browser automation or web scraping (separate future scope)
- ❌ Fine-tuning the model (inference-only, no local training)

---

## 4. Target Users

| User Type | Description |
|---|---|
| **Primary** | Solo developer / researcher who wants a local AI that knows their codebase, emails, and notes |
| **Secondary** | Knowledge worker (consultant, PM) tracking projects, meetings, and follow-ups |
| **Tertiary** | Privacy-conscious user who refuses to use cloud AI for personal data |

---

## 5. System Architecture

### 5.1 LLM Layer (Layer 0 API)

Hermes does **not** load models directly. It calls a local model server that exposes a clean API:

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | Text-only LLM generation (OpenAI-compatible) |
| `POST /v1/multimodal` | Vision + text generation (image inputs) |
| `POST /v1/embeddings` | Single or small-batch text embedding |
| `POST /v1/embeddings/batch` | Large-batch text embedding |

The model behind these endpoints is **Qwen2.5-VL-7B-Instruct** — a capable 7B multimodal model that handles both pure text and image-grounded reasoning.

### 5.2 Multi-Agent System

Hermes uses a **ReAct (Reason + Act)** loop orchestrated by the `OrchestratorAgent`:

```
User Input
    │
    ▼
OrchestratorAgent
    ├─► EmailAgent         — inbox, draft, reply, sync
    ├─► ProjectAgent       — kanban, milestones, deadlines
    ├─► KnowledgeAgent     — entity search, graph traversal
    └─► FilesystemAgent    — file read/write/search/index
```

Each agent has access to a typed tool set. The orchestrator routes by keyword intent and can delegate mid-reasoning.

### 5.3 Knowledge Graph

Every piece of information Hermes processes — emails, notes, files, conversations — is parsed into:
- **Entities** (people, companies, projects, concepts)
- **Relations** (linked edges: "works at", "related to", "mentioned in")
- **Activities** (timestamped log of events)

Stored in **PostgreSQL + pgvector**. Queried via **Hybrid RAG**:
1. Vector similarity search (embeddings)
2. Trigram keyword match (`pg_trgm`)
3. 1-hop graph traversal (linked entity expansion)

### 5.4 Memory System

| Layer | Storage | TTL |
|---|---|---|
| Short-term | In-process list (session) | Duration of session |
| Episodic | PostgreSQL `EpisodicMemory` table | Permanent |
| Semantic | Knowledge graph entities | Permanent |

---

## 6. Feature Requirements

### 6.1 Chat Interface
**Priority: P0**

| ID | Requirement |
|---|---|
| CHAT-01 | WebSocket streaming — tokens stream live as Hermes reasons |
| CHAT-02 | HTTP fallback — `POST /api/chat/message` for non-WS clients |
| CHAT-03 | Image upload in chat — binary WS frame or base64 JSON field |
| CHAT-04 | Multimodal routing — images automatically sent to `/v1/multimodal` |
| CHAT-05 | Session memory — conversation context maintained per session |
| CHAT-06 | Feedback — thumbs up/down rating stored per response |

### 6.2 Email Integration
**Priority: P0**

| ID | Requirement |
|---|---|
| EMAIL-01 | IMAP inbox sync (Gmail, Outlook.com, any IMAP server) |
| EMAIL-02 | SMTP send (Gmail, any SMTP server) |
| EMAIL-03 | macOS Outlook desktop sync via AppleScript (no IMAP needed) |
| EMAIL-04 | macOS Outlook send via AppleScript |
| EMAIL-05 | Automatic background sync on configurable interval |
| EMAIL-06 | Email content indexed into knowledge graph on sync |
| EMAIL-07 | Hermes can draft and send email on user's behalf |

### 6.3 Project Management
**Priority: P1**

| ID | Requirement |
|---|---|
| PROJ-01 | Create / update / delete projects |
| PROJ-02 | Kanban blocks (todo → in-progress → done) |
| PROJ-03 | Milestones with deadlines |
| PROJ-04 | Block status updates via chat or UI |
| PROJ-05 | Projects linked to entities in knowledge graph |

### 6.4 Activity Logging
**Priority: P1**

| ID | Requirement |
|---|---|
| ACT-01 | Manual text note recording |
| ACT-02 | Voice note recording → Whisper STT transcription |
| ACT-03 | Image-annotated activity — Qwen describes image, merges with note |
| ACT-04 | All activities indexed into knowledge graph |
| ACT-05 | Activity feed with chronological display |

### 6.5 Filesystem Access
**Priority: P1**

| ID | Requirement |
|---|---|
| FS-01 | List directory contents (name, size, modified date) |
| FS-02 | Read text files (code, markdown, CSV, logs) |
| FS-03 | Read image files → Qwen vision describes contents |
| FS-04 | Write / create files within allowed paths |
| FS-05 | Search files by glob pattern recursively |
| FS-06 | Move / rename files within allowed paths |
| FS-07 | Delete files (directories blocked for safety) |
| FS-08 | Index folder contents → knowledge graph |
| FS-09 | Path allow-list enforcement — no access outside `FILESYSTEM_ALLOWED_PATHS` |
| FS-10 | Blocked extensions — `.env`, `.key`, `.pem` never readable |

### 6.6 Knowledge Graph
**Priority: P0**

| ID | Requirement |
|---|---|
| KG-01 | Automatic entity extraction from any text source |
| KG-02 | Relation building between entities |
| KG-03 | Hybrid search: vector + trigram + graph hop |
| KG-04 | Nightly consolidation + relation decay job |
| KG-05 | D3.js live graph visualization in UI |

### 6.7 Self-Improvement
**Priority: P2**

| ID | Requirement |
|---|---|
| SI-01 | Dynamic tool creation: Hermes writes and hot-loads new Python tools |
| SI-02 | Prompt evolution: prompts updated based on feedback patterns |
| SI-03 | Tool sandboxed execution with timeout |
| SI-04 | Tool registry with enable/disable |

### 6.8 Reports
**Priority: P2**

| ID | Requirement |
|---|---|
| RPT-01 | Weekly HTML report synthesized every Sunday at 8AM |
| RPT-02 | Report covers: emails, activities, project progress, people mentioned |
| RPT-03 | Manual trigger via UI or API |
| RPT-04 | Reports stored in DB and viewable in UI |

### 6.9 People / Contacts
**Priority: P2**

| ID | Requirement |
|---|---|
| PPL-01 | Person entities auto-created from email and activity mentions |
| PPL-02 | People page showing known contacts + interaction history |
| PPL-03 | Hermes can answer "Who is [name]?" from graph |

---

## 7. Non-Functional Requirements

| Category | Requirement |
|---|---|
| **Privacy** | Zero data egress — all inference, embedding, storage on local machine |
| **Performance** | Chat first token < 2s on CUDA; full response < 30s for complex queries |
| **Reliability** | Background jobs must not crash the main server on failure (isolated) |
| **Security** | Filesystem path allow-list; blocked extension enforcement; no shell injection in dynamic tools |
| **Scalability** | Single-user design; no horizontal scaling needed |
| **Extensibility** | New agents, tools, and routes can be added without modifying core |

---

## 8. Technical Stack

| Component | Technology |
|---|---|
| **Backend** | Python 3.11 + FastAPI + uvicorn |
| **LLM** | Qwen2.5-VL-7B-Instruct via Layer 0 API |
| **Vision** | Qwen2.5-VL multimodal endpoint |
| **Embeddings** | Layer 0 API (`/v1/embeddings`, `/v1/embeddings/batch`) |
| **STT** | OpenAI Whisper (local) |
| **Database** | PostgreSQL 16 + pgvector + pg_trgm |
| **ORM** | SQLAlchemy 2 async |
| **Migrations** | Alembic |
| **Scheduler** | APScheduler |
| **Frontend** | Vanilla HTML/CSS/JS — no framework |
| **Graph Viz** | D3.js |
| **Email** | Python `imaplib` / `smtplib` + macOS AppleScript |
| **NLP** | spaCy `en_core_web_sm` |
| **Container** | Docker Compose (PostgreSQL only; app runs natively) |

---

## 9. Data Model (Summary)

| Table | Purpose |
|---|---|
| `entities` | Knowledge graph nodes (people, companies, concepts, projects) |
| `relations` | Edges between entities with type and confidence score |
| `activities` | Timestamped events (notes, emails, file reads) |
| `projects` | Project records with status and metadata |
| `project_blocks` | Kanban cards linked to projects |
| `emails` | Synced email records |
| `episodic_memory` | Hermes's personal event memory |
| `agent_runs` | Full log of every orchestrator execution |
| `tool_registry` | Static and dynamic tool metadata |
| `prompt_versions` | Versioned system and task prompts |
| `response_feedback` | User thumbs up/down ratings |

---

## 10. API Surface (Key Endpoints)

| Method | Path | Description |
|---|---|---|
| `WS` | `/api/chat/ws` | Streaming chat WebSocket |
| `POST` | `/api/chat/message` | HTTP chat (multipart, supports image upload) |
| `GET` | `/api/projects` | List projects |
| `POST` | `/api/projects` | Create project |
| `GET` | `/api/emails` | List synced emails |
| `POST` | `/api/emails/sync` | Trigger email sync |
| `POST` | `/api/emails/send` | Send email |
| `GET` | `/api/activities` | List activities |
| `POST` | `/api/activities` | Record activity |
| `POST` | `/api/activities/with-image` | Record activity + vision description |
| `POST` | `/api/activities/transcribe` | Whisper voice transcription |
| `GET` | `/api/knowledge/search` | Hybrid semantic search |
| `GET` | `/api/knowledge/graph` | Graph nodes + edges for D3.js |
| `GET` | `/api/dashboard` | Summary metrics |
| `GET` | `/api/reports` | List weekly reports |

---

## 11. UI Pages

| Page | URL Path | Description |
|---|---|---|
| Dashboard | `/` | Activity feed, project summary, recent emails |
| Chat | `/pages/chat.html` | Streaming chat with multimodal support |
| Projects | `/pages/projects.html` | Kanban boards + milestones |
| Tasks | `/pages/tasks.html` | Task list view |
| Emails | `/pages/emails.html` | Inbox manager |
| Activities | `/pages/activities.html` | Activity log + voice recorder |
| People | `/pages/people.html` | Contacts and interaction history |
| Graph | `/pages/graph.html` | D3.js live knowledge graph |
| Reports | `/pages/reports.html` | Weekly report viewer |
| Tools | `/pages/tools.html` | Tool registry viewer |

---

## 12. Configuration Reference

All configuration lives in `.env` (based on `.env.example`):

```
LLM_PROVIDER=openai
LLM_API_BASE=http://localhost:8080
LLM_CHAT_PATH=/v1/chat/completions
LLM_MULTIMODAL_PATH=/v1/multimodal

EMBED_PROVIDER=api
EMBED_API_BASE=http://localhost:8080
EMBED_API_PATH=/v1/embeddings
EMBED_BATCH_API_PATH=/v1/embeddings/batch

EMAIL_CLIENT=outlook              # or imap
FILESYSTEM_ALLOWED_PATHS=/Users/alpha/Documents,/Users/alpha/Projects
```

---

## 13. Future Roadmap

| Priority | Feature |
|---|---|
| P3 | Browser automation tools (Playwright-based web agent) |
| P3 | Calendar integration (macOS Calendar / Google Calendar) |
| P3 | PDF and DOCX reading in filesystem agent |
| P3 | Slack / Teams integration |
| P4 | Voice output (TTS responses) |
| P4 | Mobile-optimized PWA |
| P4 | Knowledge graph export (GraphML / JSON-LD) |
| P4 | Multi-model routing (use different models for different tasks) |

---

## 14. Success Metrics

| Metric | Target |
|---|---|
| Chat first-token latency | < 2 seconds |
| Email sync latency | < 5 seconds per batch |
| Knowledge graph queries | < 500ms |
| Filesystem file read | < 200ms |
| Agent accuracy (task completion) | > 85% on representative prompts |
| User feedback positive rate | > 80% thumbs-up |

---

*ARIA / Hermes — Personal AI. Local. Private. Yours.*
