# ⚡ ARIA — Hermes

> **A**utonomous **R**easoning & **I**ntelligence **A**gent — Your Personal AI Second Brain

ARIA (codename: **Hermes**) is a **fully local, privacy-first personal AI assistant** built around a multi-agent ReAct loop. It reads your emails, tracks your projects, records your activities, navigates your local file system, and grows a knowledge graph from everything it learns — all powered by your own hardware via a layer 0 model API.

---

## 🧠 Capabilities at a Glance

| Capability                        | Description                                                                              |
| --------------------------------- | ---------------------------------------------------------------------------------------- |
| **Multi-Agent ReAct Loop**  | Orchestrator delegates to specialist agents — email, project, knowledge, and filesystem |
| **Multimodal Vision**       | Qwen2.5-VL understands images sent in chat or attached to activities                     |
| **Local Filesystem Access** | Read folders, search files, write content, and describe images via vision                |
| **Knowledge Graph**         | Auto-grows from every email, note, file, and activity using PostgreSQL + pgvector        |
| **Hybrid RAG**              | Vector similarity + trigram keyword + 1-hop graph traversal                              |
| **Email Integration**       | IMAP/SMTP OR local macOS Outlook desktop app (AppleScript sync + send)                   |
| **Project Tracker**         | Kanban boards, milestones, and deadlines                                                 |
| **Activity Logger**         | Manual notes, image-annotated entries, and Whisper voice transcription                   |
| **Self-Improvement**        | Dynamic tool sandbox + prompt evolution engine                                           |
| **Weekly Reports**          | LLM-synthesized HTML reports of the week's work                                          |
| **Live Streaming UI**       | WebSocket token streaming, D3.js knowledge graph, glassmorphic dark frontend             |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       FastAPI + WebSocket                        │
├──────────┬─────────────┬──────────┬──────────────┬──────────────┤
│ Chat API │ Projects API│Emails API│Knowledge API │Filesystem API│
└──────────┴─────────────┴──────────┴──────────────┴──────────────┘
                     ↓ OrchestratorAgent (ReAct loop)
       ┌──────────┬───────────┬──────────────┬──────────────┐
       │  Email   │  Project  │  Knowledge   │  Filesystem  │  ← Specialists
       │  Agent   │   Agent   │    Agent     │    Agent     │
       └──────────┴───────────┴──────────────┴──────────────┘
                     ↓ Tools Registry (static + dynamic)
┌──────────────────────────────────────────────────────────────────┐
│               PostgreSQL + pgvector  (Single DB)                 │
│  Entities · Relations · Projects · Emails · Activities           │
│  AgentRuns · ToolRegistry · PromptVersions · Feedback            │
└──────────────────────────────────────────────────────────────────┘
                     ↓ Background Scheduler (APScheduler)
     • 30-min email sync         • Nightly graph refinement
     • Sunday weekly report      • Weekly prompt evolution
```

### LLM Endpoint Routing (Layer 0 API)

| Request Type          | Endpoint                                      |
| --------------------- | --------------------------------------------- |
| Text-only generation  | `POST {LLM_API_BASE}/v1/chat/completions`   |
| Vision (image + text) | `POST {LLM_API_BASE}/v1/multimodal`         |
| Single embedding      | `POST {EMBED_API_BASE}/v1/embeddings`       |
| Batch embeddings      | `POST {EMBED_API_BASE}/v1/embeddings/batch` |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Layer 0 model API running locally (provides `/v1/chat/completions`, `/v1/multimodal`, `/v1/embeddings`, `/v1/embeddings/batch`)

### 1. Clone & Configure

```bash
git clone https://github.com/Maynanda/AIAgent.git
cd AIAgent
cp .env.example .env
# Edit .env with your settings
```

### 2. Start Database

```bash
docker-compose up -d
# PostgreSQL with pgvector + pg_trgm
```

### 3. Install Python Dependencies

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Run Migrations & Seed

```bash
alembic upgrade head
python database/seed.py
```

### 5. Launch Hermes

```bash
python main.py
# → Open http://localhost:8000
```

---

## 📁 Project Structure

```
AIAgent/
├── agents/
│   ├── base.py                  # ReAct loop foundation + AgentStep/AgentResult
│   ├── orchestrator.py          # Top-level routing + WebSocket streaming
│   ├── email_agent.py           # Email read/send specialist
│   ├── project_agent.py         # Kanban + milestones specialist
│   ├── knowledge_agent.py       # Graph search + entity specialist
│   ├── filesystem_agent.py      # Local file read/write/search specialist
│   ├── tool_builder.py          # Dynamic tool sandbox (code execution)
│   └── prompt_evolution.py      # Self-improvement engine
├── tools/
│   ├── registry.py              # Static + dynamic tool loader
│   └── static/
│       ├── db_tools.py          # Knowledge graph CRUD tools
│       ├── email_tools.py       # Email send/list tools
│       ├── project_tools.py     # Project + block tools
│       └── filesystem_tools.py  # File read/write/search/index tools
├── llm/
│   ├── client.py                # LLM client — chat vs multimodal endpoint routing
│   └── prompts/                 # Versioned system & task prompts
├── rag/
│   ├── embedder.py              # Embedding client — /v1/embeddings + /v1/embeddings/batch
│   ├── retriever.py             # Hybrid vector + trigram + graph retrieval
│   └── context_builder.py       # Context window assembly
├── memory/
│   └── manager.py               # Short-term + episodic memory
├── knowledge_graph/
│   ├── builder.py               # Entity extraction + graph building
│   └── updater.py               # Nightly consolidation + relation decay
├── database/
│   ├── models.py                # Single-source ORM (SQLAlchemy async)
│   ├── connection.py            # Async engine + session factory
│   ├── seed.py                  # Initial data seeder
│   └── migrations/              # Alembic schema migrations
├── services/
│   ├── email_service.py         # IMAP/SMTP + macOS Outlook AppleScript
│   ├── whisper_service.py       # Local Whisper STT
│   ├── report_service.py        # Weekly report synthesis
│   └── scheduler.py             # APScheduler background jobs
├── routes/
│   ├── chat.py                  # WebSocket streaming + HTTP multipart chat
│   ├── projects.py              # Projects + Kanban
│   ├── tasks.py                 # Task management
│   ├── emails.py                # Inbox CRUD + sync + send
│   ├── activities.py            # Activity log + voice + image-annotated entries
│   ├── knowledge.py             # Graph viz + hybrid search
│   ├── dashboard.py             # Metrics summary
│   ├── tools.py                 # Tool registry management
│   └── reports.py               # Weekly report API
├── frontend/
│   ├── index.html               # Dashboard
│   └── pages/
│       ├── chat.html            # Streaming chat (WebSocket + multimodal)
│       ├── projects.html        # Kanban board
│       ├── tasks.html           # Task list
│       ├── activities.html      # Activity recorder
│       ├── emails.html          # Inbox manager
│       ├── people.html          # People / contacts view
│       ├── graph.html           # D3.js live knowledge graph
│       ├── reports.html         # Weekly reports viewer
│       └── tools.html           # Tool registry viewer
├── main.py                      # FastAPI entrypoint + lifespan
├── config.py                    # Settings (pydantic-settings + .env)
├── requirements.txt
├── docker-compose.yml
└── alembic.ini
```

---

## ⚙️ Environment Variables

Copy `.env.example` to `.env`. Key variables:

### LLM (Layer 0 API)

| Variable                | Default                         | Description                                              |
| ----------------------- | ------------------------------- | -------------------------------------------------------- |
| `LLM_PROVIDER`        | `openai`                      | `local` (Transformers GPU) or `openai` (layer 0 API) |
| `LLM_MODEL_ID`        | `Qwen/Qwen2.5-VL-7B-Instruct` | Model name sent in API payload                           |
| `LLM_API_BASE`        | `http://localhost:8080`       | Base URL of your layer 0 server                          |
| `LLM_CHAT_PATH`       | `/v1/chat/completions`        | Text-only generation endpoint                            |
| `LLM_MULTIMODAL_PATH` | `/v1/multimodal`              | Vision + text generation endpoint                        |

### Embeddings (Layer 0 API)

| Variable                 | Default                   | Description                                              |
| ------------------------ | ------------------------- | -------------------------------------------------------- |
| `EMBED_PROVIDER`       | `api`                   | `local` (SentenceTransformer) or `api` (layer 0 API) |
| `EMBED_API_BASE`       | `http://localhost:8080` | Base URL of your embedding server                        |
| `EMBED_API_PATH`       | `/v1/embeddings`        | Single / small batch endpoint                            |
| `EMBED_BATCH_API_PATH` | `/v1/embeddings/batch`  | Large batch endpoint                                     |

### Email

| Variable           | Description                                                   |
| ------------------ | ------------------------------------------------------------- |
| `EMAIL_CLIENT`   | `imap` or `outlook` (macOS local Outlook via AppleScript) |
| `EMAIL_ADDRESS`  | Your email address                                            |
| `EMAIL_PASSWORD` | App password (IMAP only)                                      |

### Filesystem

| Variable                          | Description                                        |
| --------------------------------- | -------------------------------------------------- |
| `FILESYSTEM_ALLOWED_PATHS`      | Comma-separated absolute paths Hermes can access   |
| `FILESYSTEM_MAX_FILE_BYTES`     | Max file size to read in full (default 10MB)       |
| `FILESYSTEM_BLOCKED_EXTENSIONS` | Extensions always blocked (`.env,.key,.pem,...`) |

---

## 🤖 Talking to Hermes

Navigate to `http://localhost:8000/pages/chat.html`:

```
# Projects
"What are my active projects?"
"Create a new project called Product Launch"
"Mark the design task as done"

# Email
"Summarize my unread emails"
"Draft a reply to the budget email from John"

# Knowledge
"What do you know about Alpha Technologies?"
"Who did I last talk to about the API redesign?"

# Filesystem
"List my Documents folder"
"Read the README in ~/Projects/Backend"
"Search for all PDF files in ~/Documents"
"Index my meeting notes folder into the knowledge graph"

# Multimodal
"What's in this screenshot?" + attach image
"Add this whiteboard photo as a meeting note" + attach image
```

---

## 🔒 Security Model

- **Filesystem**: Path allow-list enforced at every call — Hermes cannot read outside `FILESYSTEM_ALLOWED_PATHS`
- **Blocked extensions**: `.env`, `.key`, `.pem`, `.crt` files are never readable
- **Dynamic tools**: Sandboxed execution with configurable timeout
- **API keys**: Stored in `.env` only, never logged or stored in DB

---

## 🏃 Development

```bash
# Run with hot reload
APP_ENV=development python main.py

# API docs
open http://localhost:8000/api/docs

# Create a new migration
alembic revision --autogenerate -m "describe change"
alembic upgrade head
```

---

## 📄 License

MIT License — Built for personal productivity. Run locally. Own your data.

---

*Built with ❤️ by Hermes — your AI second brain that grows with you.*
