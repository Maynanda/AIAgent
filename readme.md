# ⚡ ARIA — Hermes

> **A**utonomous **R**easoning & **I**ntelligence **A**gent — Your Personal AI Second Brain

ARIA (codename: **Hermes**) is a fully local, self-improving personal AI assistant built on top of `Qwen2.5-VL-7B-Instruct`. It reads your emails, tracks your projects, records your activities, grows a knowledge graph from everything it sees, and runs as a local-first multi-agent system.

---

## 🧠 Features

| Capability | Description |
|---|---|
| **Multi-Agent ReAct Loop** | Orchestrator delegates to email, project, and knowledge specialist agents |
| **Local LLM** | Qwen2.5-VL-7B-Instruct via HuggingFace `transformers` with 4-bit GPU quantization |
| **Knowledge Graph** | PostgreSQL + pgvector — auto-grows from every email, note, and activity |
| **Hybrid RAG** | Vector search + trigram keyword + 1-hop graph traversal |
| **Email Integration** | IMAP sync + SMTP send OR local macOS Outlook desktop app sync + send via AppleScript |
| **Project Tracker** | Kanban boards, milestones, and progress tracking |
| **Activity Logger** | Manual notes + Whisper voice-to-text transcription |
| **Self-Improvement** | Dynamic tool creation sandbox + prompt evolution engine |
| **Weekly Reports** | LLM-synthesized HTML reports every Sunday |
| **Live Streaming UI** | WebSocket streaming chat, D3.js knowledge graph, glassmorphic dark UI |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     FastAPI + WebSocket                      │
├────────────┬─────────────┬────────────┬──────────────────────┤
│  Chat API  │ Projects API│ Emails API │  Knowledge/RAG API   │
└────────────┴─────────────┴────────────┴──────────────────────┘
              ↓ Orchestrator Agent (ReAct loop)
    ┌─────────┬────────────┬──────────────┐
    │ Email   │  Project   │  Knowledge   │  ← Specialist Agents
    │  Agent  │   Agent    │    Agent     │
    └─────────┴────────────┴──────────────┘
              ↓ Tools Registry (static + dynamic)
┌──────────────────────────────────────────────────────────────┐
│              PostgreSQL + pgvector (Single DB)                │
│   Entities · Relations · Projects · Emails · Activities      │
│   AgentRuns · ToolRegistry · PromptVersions · Feedback       │
└──────────────────────────────────────────────────────────────┘
              ↓ Background Scheduler (APScheduler)
  • Nightly graph refinement & relation decay
  • 30-min IMAP email sync
  • Sunday weekly report synthesis
  • Weekly prompt evolution analysis
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- NVIDIA GPU (recommended, ~8GB VRAM for 4-bit Qwen2.5-VL-7B)
- CUDA 12.x

### 1. Clone & Configure

```bash
git clone <your-repo-url>
cd AIAgent
cp .env.example .env
# Edit .env with your settings
```

### 2. Start Database

```bash
docker-compose up -d
# PostgreSQL with pgvector, pg_trgm enabled
```

### 3. Install Python Dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
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
├── agents/                    # Multi-agent system
│   ├── base.py               # ReAct loop foundation
│   ├── orchestrator.py       # Top-level routing + streaming
│   ├── email_agent.py        # Email specialist
│   ├── project_agent.py      # Project/task specialist
│   ├── knowledge_agent.py    # Graph/search specialist
│   ├── tool_builder.py       # Dynamic tool sandbox
│   └── prompt_evolution.py   # Self-improvement engine
├── database/
│   ├── models.py             # Single source of truth ORM
│   ├── connection.py         # Async engine + sessions
│   ├── seed.py               # Initial data seeding
│   └── migrations/           # Alembic schema migrations
├── knowledge_graph/
│   ├── builder.py            # Entity extraction + graph building
│   └── updater.py            # Nightly consolidation
├── llm/
│   ├── client.py             # Qwen2.5-VL singleton (4-bit)
│   └── prompts/              # Versioned system prompts
├── rag/
│   ├── embedder.py           # nomic-embed-text-v1.5
│   ├── retriever.py          # Hybrid search engine
│   └── context_builder.py   # Context assembly
├── memory/
│   └── manager.py            # Short-term + episodic memory
├── routes/                   # FastAPI API routers
│   ├── chat.py               # WebSocket + HTTP chat
│   ├── projects.py           # Projects + Kanban blocks
│   ├── tasks.py              # Task management
│   ├── emails.py             # Email CRUD + sync + send
│   ├── activities.py         # Activity logging + transcribe
│   ├── knowledge.py          # Graph viz + hybrid search
│   ├── dashboard.py          # Summary metrics
│   ├── tools.py              # Tool registry management
│   └── reports.py            # Weekly report API
├── services/
│   ├── email_service.py      # IMAP/SMTP integration
│   ├── whisper_service.py    # Local STT (Whisper)
│   ├── report_service.py     # Weekly report synthesis
│   └── scheduler.py          # APScheduler background jobs
├── tools/
│   ├── registry.py           # Static + dynamic tool loader
│   ├── static/               # Built-in agent tools
│   └── dynamic/              # Agent-created tools (hot-loaded)
├── frontend/
│   ├── index.html            # Dashboard
│   └── pages/
│       ├── chat.html         # Streaming chat UI
│       ├── projects.html     # Kanban board
│       ├── activities.html   # Activity recorder
│       ├── emails.html       # Inbox manager
│       ├── graph.html        # D3.js knowledge graph
│       └── reports.html      # Weekly reports viewer
├── main.py                   # FastAPI entrypoint
├── config.py                 # Settings (pydantic-settings)
├── requirements.txt          # Dependencies
├── docker-compose.yml        # PostgreSQL + pgvector
└── alembic.ini               # Migration config
```

---

## 🔧 Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL async connection string |
| `LLM_MODEL_NAME` | HuggingFace model ID (default: Qwen/Qwen2.5-VL-7B-Instruct) |
| `LLM_DEVICE` | `cuda` or `cpu` |
| `LLM_LOAD_IN_4BIT` | Enable 4-bit quantization (true/false) |
| `EMBEDDING_MODEL` | nomic-embed-text-v1.5 |
| `EMAIL_CLIENT` | Email source client (`imap` or `outlook` for local macOS desktop app) |
| `EMAIL_ADDRESS` | Your Gmail / IMAP email address (required for IMAP) |
| `EMAIL_PASSWORD` | App password (required for IMAP) |
| `EMAIL_IMAP_HOST` | IMAP server (e.g. imap.gmail.com) |
| `EMAIL_SMTP_HOST` | SMTP server (e.g. smtp.gmail.com) |
| `EMAIL_SMTP_PORT` | SMTP port (e.g. 587) |
| `EMAIL_SMTP_USE_TLS` | SMTP TLS enabled (true/false) |
| `WHISPER_MODEL_SIZE` | base / small / medium / large |

---

## 🤖 Using Hermes

### Chat Interface
Navigate to `http://localhost:8000/chat` and talk to Hermes:

```
"What are my active projects?"
"Summarize my emails from this week"
"Create a new project called 'Product Launch'"
"What tasks are blocked?"
"Draft a reply to John's email about the budget"
```

### Dynamic Tool Creation
Hermes can create its own tools:

```
"Create a tool that fetches the current Bitcoin price"
```

### Weekly Reports
Generated every Sunday at 8AM, or trigger manually at `http://localhost:8000/reports`

---

## 🏃 Development

```bash
# Run with hot reload
APP_ENV=development python main.py

# Access API docs
open http://localhost:8000/api/docs

# Create a new migration after model changes
alembic revision --autogenerate -m "describe change"
alembic upgrade head
```

---

## 📄 License

MIT License — Built for personal productivity. Grow it with your own data.

---

*Built with ❤️ by Hermes — your AI second brain that grows with you.*
