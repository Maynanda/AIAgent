# ARIA — Hermes Build Tasks

## Phase 1: Foundation
- [x] Project scaffold & infrastructure
  - [x] task.md
  - [x] docker-compose.yml
  - [x] requirements.txt
  - [x] .env.example
  - [x] config.py
- [/] Database layer
  - [x] database/__init__.py
  - [x] database/connection.py
  - [x] database/models.py (full schema)
  - [ ] alembic.ini + migrations/env.py
  - [ ] migrations/001_initial.py
- [/] LLM layer
  - [x] llm/__init__.py
  - [x] llm/client.py (Transformers GPU singleton)
  - [ ] llm/tokenizer.py
  - [x] llm/prompts/system/orchestrator.txt
- [x] Core agent skeleton
  - [x] agents/__init__.py
  - [x] agents/base.py (ReAct loop base)
- [x] Memory layer (skeleton)
  - [x] memory/__init__.py
  - [x] memory/manager.py
- [x] Tools layer (skeleton)
  - [x] tools/__init__.py
  - [x] tools/registry.py
- [x] RAG layer (skeleton)
  - [x] rag/__init__.py
  - [x] rag/embedder.py (nomic-embed-text-v1.5)
- [/] API routes
  - [x] routes/__init__.py
  - [x] routes/chat.py (WebSocket streaming)
  - [x] routes/health.py (implemented in main.py)
- [x] FastAPI main app
  - [x] main.py
- [x] Frontend
  - [x] frontend/assets/css/main.css (design system)
  - [x] frontend/assets/js/api.js (unified client)
  - [x] frontend/index.html (dashboard)
  - [x] frontend/pages/chat.html (streaming chat UI)

## Phase 2: Core Data (complete)
- [x] CRUD routes for all entities
- [x] Alembic migration runner
- [x] Seed data / fixtures

## Phase 3: Knowledge Engine (next)
- [ ] SPAcy NER entity extractor
- [ ] Knowledge graph builder and merger
- [ ] Entity relationship sync logic

## Phase 4: RAG
- [ ] Hybrid search retriever (vector + keyword + graph traversal)
- [ ] Context window pack builder
- [ ] Short-term / episodic memory integrations

## Phase 5: Projects UI
- [ ] Projects Board (Kanban frontend view)
- [ ] Timeline chart visualization
- [ ] Detail view for projects, tasks, and notes

## Phase 6: Activities
- [ ] Activity recording manual logger
- [ ] Local Whisper transcription setup
- [ ] Automatic project-linking extraction

## Phase 7: Email
- [ ] SMTP service link
- [ ] IMAP scanner and processor
- [ ] Subject sentiment analyser

## Phase 8: Multi-Agent
- [ ] Orchestrator reasoning loop
- [ ] Specialized email, task, knowledge agents
- [ ] Sub-agent delegation controls

## Phase 9: Self-Improvement
- [ ] Dynamic Tool Builder agent sandbox
- [ ] Prompt Versioning metrics logger
- [ ] Nightly graph refinement worker

## Phase 10: Weekly Reports
- [ ] Report synthesis agent template
- [ ] Weekly scheduled cron task
- [ ] HTML report renderer

## Phase 11: Polish
- [ ] D3.js knowledge graph view
- [ ] Contact/people profiles and analytics
- [ ] Quantization tuning & execution checks

