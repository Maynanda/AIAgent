# ARIA — Hermes Edition

### Product Requirements Document — FINAL (v3)

#### All Decisions Resolved · Ready to Build

---

## ✅ Resolved Stack

| Decision | Choice |

|---|---|

| **LLM Runtime** | HuggingFace Transformers + GPU (CUDA) |

| **LLM Model** | Qwen2.5-VL-7B-Instruct (4-bit quantized via bitsandbytes) |

| **Embedding Model** | `nomic-ai/nomic-embed-text-v1.5` (local, 768-dim) |

| **Tool Sandboxing** | Level 2 — subprocess isolation with timeout |

| **Voice Input** | Yes — `openai/whisper-base` (local) |

| **Database** | PostgreSQL + pgvector + pg_trgm |

| **Backend** | FastAPI (async) |

| **Frontend** | Modular HTML + Vanilla CSS + Vanilla JS |

| **Graph Viz** | D3.js force-directed |

| **Background Jobs** | APScheduler |

| **Email** | IMAP/SMTP (generic, works with any provider) |

---

## 1. Auto-Project Detection

Hermes **automatically detects and creates projects** when it encounters enough signals.

### 1.1 Detection Sources

```

Email arrives → "Hey, can we discuss the Alpha Platform launch?"

                     │

                     ▼

           Knowledge Agent: "Alpha Platform" is new entity

           No project exists for it

                     │

                     ▼

           Auto-Project Trigger checks:

           - Is it a named initiative?

           - Is it mentioned by ≥2 people?

           - Does it have a deadline signal?

           - Is it a community / group / org?

                     │

                ───────────

               │           │

              Yes          No

               │           │

               ▼           ▼

        Create Project   Tag as Topic

        entity           entity only

        + notify user    (monitor for

        in dashboard     more signals)
```

### 1.2 Auto-Project Schema Hydration

When a project is auto-created, Hermes immediately populates what it knows:

```

Auto-created Project: "Alpha Platform Launch"

├── Status: active (inferred)

├── People: [person who sent email] → assigned as stakeholder

├── Source emails: [linked email entities]

├── Topics extracted: ["launch", "platform", "alpha"]

├── Timeline: extracted deadline if found in email

├── First block: "Initial email context — [date]"

└── Confidence score: 0.82 (shown in UI so user can review)
```

User sees a **"Review AI-created project"** badge — can confirm, rename, merge, or discard.

### 1.3 Community / Group Detection

If the signal is a **group or community** (not a deliverable project):

- Creates a `Topic` entity with type `community`
- Tracks all related people, emails, discussions
- Can be promoted to a full Project at any time

---

## 2. Self-Improvement Architecture

This is the core of what makes Hermes truly grow. Three layers of self-improvement:

```

┌─────────────────────────────────────────────────────────┐

│                  SELF-IMPROVEMENT LAYERS                │

│                                                         │

│  Layer 1: Tool Evolution                                │

│  ┌─────────────────────────────────────────────────┐   │

│  │ Tool Builder Agent writes new tools             │   │

│  │ Tools improve with usage (version history)      │   │

│  │ Unused tools are archived                       │   │

│  └─────────────────────────────────────────────────┘   │

│                         ↓                               │

│  Layer 2: Prompt Evolution                              │

│  ┌─────────────────────────────────────────────────┐   │

│  │ Hermes tracks which prompts produce good output │   │

│  │ Runs A/B tests between prompt variants          │   │

│  │ Keeps best-performing version                   │   │

│  │ User feedback signals "good" vs "bad" responses │   │

│  └─────────────────────────────────────────────────┘   │

│                         ↓                               │

│  Layer 3: Knowledge Refinement                          │

│  ┌─────────────────────────────────────────────────┐   │

│  │ Entity deduplication & merging runs weekly      │   │

│  │ Relation weights decay if not reinforced        │   │

│  │ Importance scores adjust based on usage         │   │

│  │ Contradictions are flagged for review           │   │

│  └─────────────────────────────────────────────────┘   │

└─────────────────────────────────────────────────────────┘
```

### 2.1 Prompt Evolution (Detailed)

```sql

-- Prompt versioning table

prompt_versions (

  id           uuid PRIMARY KEY,

  prompt_key   text,           -- e.g. 'extract_entities', 'orchestrator_system'

versionint,

  content      text,           -- the actual prompt text

  score        float,          -- average quality score (0-1)

  usage_count  int,

  is_active    bool,

  created_by   text,           -- 'system' | 'agent' | 'user'

  created_at   timestamptz

)


-- Response quality tracking

response_feedback (

  id           uuid PRIMARY KEY,

  run_id       uuid REFERENCES agent_runs(id),

  prompt_key   text,

  prompt_version int,

  user_rating  int,            -- 1-5 if user rated it

  auto_score   float,          -- automated quality heuristic

  notes        text,

  created_at   timestamptz

)
```

**How it works**:

1. Every agent run records which prompt version was used
2. User can 👍/👎 any Hermes response in the chat UI
3. Weekly: Hermes analyzes low-scoring runs, uses LLM to suggest improved prompts
4. New prompt version is tested on a sample → if better, promoted to active
5. Prompt history is preserved forever (full audit trail)

### 2.2 Knowledge Refinement (Nightly Job)

```

Every night at 2am:


1. DEDUPLICATION

   Find entities with cosine similarity > 0.92

   LLM judges: same entity? → merge with combined metadata


2. DECAY

   Relations not referenced in 60 days → weight *= 0.95

   Importance scores recalculated from access frequency


3. CONTRADICTION DETECTION

   Find conflicting facts (e.g. person at two companies)

   Flag for user review in dashboard


4. GRAPH HEALTH REPORT

   Total entities, relations, orphan nodes

   Growth rate, density, clusters

   Stored as weekly metric
```

### 2.3 RAG Self-Optimization

Hermes tracks which retrieved chunks actually helped answer questions:

```python

# After each generation, we check:

# - Did the retrieved context contain the answer?

# - Which chunks were most referenced in the response?

# - Were any chunks irrelevant (hallucination risk)?

# 

# This data tunes the retrieval weights over time:

# vector_weight, bm25_weight, graph_weight — auto-adjusted
```

---

## 3. GPU Acceleration Strategy

```python

# config.py

DEVICE = "cuda"# full GPU mode

TORCH_DTYPE = torch.bfloat16      # Qwen2.5-VL native dtype

QUANTIZATION = "4bit"# bitsandbytes 4-bit NF4

# ~4-5GB VRAM for 7B model

EMBEDDING_DEVICE = "cuda"# nomic-embed on GPU too

WHISPER_DEVICE = "cuda"# Whisper STT on GPU


# Model loading (startup singleton)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(

"Qwen/Qwen2.5-VL-7B-Instruct",

torch_dtype=torch.bfloat16,

device_map="auto",

quantization_config=BitsAndBytesConfig(

load_in_4bit=True,

bnb_4bit_compute_dtype=torch.bfloat16,

bnb_4bit_use_double_quant=True,

bnb_4bit_quant_type="nf4"

    )

)
```

**Expected VRAM usage:**

- Qwen2.5-VL-7B @ 4-bit: ~5 GB
- nomic-embed-text-v1.5: ~0.3 GB
- Whisper-base: ~0.15 GB

-**Total**: ~5.5 GB VRAM (fits on 8GB+ GPU)

---

## 4. Voice Input Pipeline

```

User speaks → Browser MediaRecorder API → WebSocket → FastAPI

                                                          │

                                                          ▼

                                               Whisper (local) STT

                                                          │

                                                          ▼

                                               Text transcript

                                                          │

                                            ┌─────────────┴──────────┐

                                            ▼                        ▼

                                    Create Activity          Send to Chat

                                    (type: voice_note)       (if in chat mode)

                                            │

                                            ▼

                                    Knowledge Agent

                                    extracts entities

                                    updates projects
```

---

## 5. Complete Feature List

### 5.1 Data Ingestion

- [X] Manual text activity recording
- [X] Voice activity recording (Whisper STT)
- [X] Email reading (IMAP) with AI parsing
- [X] Email sending (SMTP)
- [X] Image/attachment understanding (Qwen2.5-VL vision)

### 5.2 Intelligence

- [X] Named entity recognition (people, companies, projects, topics, dates)
- [X] Knowledge graph auto-population from all sources
- [X] Project auto-detection and creation
- [X] Community/group detection
- [X] Action item extraction
- [X] Deadline extraction
- [X] Sentiment analysis on emails

### 5.3 Project Management

- [X] Multi-project dashboard
- [X] Kanban-style project blocks
- [X] Task management with priorities
- [X] Timeline tracking
- [X] Progress tracking (0-100%)
- [X] Auto-status updates from activity/email context
- [X] Person-project relation mapping

### 5.4 Retrieval & Memory

- [X] Hybrid RAG (vector + graph + BM25)
- [X] Short-term session memory
- [X] Episodic memory (past interactions)
- [X] Semantic long-term memory
- [X] Self-optimizing retrieval weights

### 5.5 Self-Improvement (Hermes Core)

- [X] Dynamic tool creation & registration
- [X] Prompt versioning & evolution
- [X] Knowledge graph refinement (nightly)
- [X] Entity deduplication & merging
- [X] RAG weight auto-tuning
- [X] Contradiction detection

### 5.6 Reporting

- [X] Weekly Intelligence Report (auto Sunday 8am)
- [X] On-demand project reports
- [X] Daily digest (optional)
- [X] Knowledge graph growth metrics
- [X] Prompt performance analytics

### 5.7 Frontend

- [X] Dashboard — project overview, activity feed
- [X] Projects — kanban board, timeline
- [X] Chat — streaming WebSocket, voice input
- [X] Activities — log + date browser
- [X] Emails — inbox with AI summaries
- [X] Knowledge Graph — D3.js interactive
- [X] Reports — weekly/monthly viewer
- [X] People — profiles + relation map
- [X] Tools — registry viewer + dynamic tool manager

---

## 6. Build Order (Confirmed Phases)

| Phase | Focus | Est. Sessions |

|---|---|---|

| **1 — Foundation** | FastAPI skeleton, DB init, Transformers LLM client, GPU setup, basic `/chat` | 1 |

| **2 — Core Data** | All DB models, Alembic migrations, CRUD routes | 1 |

| **3 — Knowledge Engine** | nomic-embed + pgvector, entity extraction, graph builder | 1-2 |

| **4 — RAG** | Hybrid retrieval, context assembly, memory layers | 1 |

| **5 — Projects UI** | Dashboard, projects board, tasks — full frontend | 1-2 |

| **6 — Activities** | Recording, voice (Whisper), auto-project linking | 1 |

| **7 — Email** | IMAP reader, parser, entity linker, SMTP sender | 1 |

| **8 — Multi-Agent** | Orchestrator + all specialist agents, ReAct loop | 2 |

| **9 — Self-Improvement** | Tool Builder Agent, prompt versioning, nightly refinement | 2 |

| **10 — Weekly Reports** | Report Agent, scheduler, report viewer | 1 |

| **11 — Polish** | D3.js graph, person profiles, analytics, UX polish | 1-2 |

---

## 7. Immediate Next Step — Phase 1

**What we'll build first:**

```

aria/

├── docker-compose.yml          ← PostgreSQL + pgvector

├── requirements.txt            ← all deps

├── config.py                   ← settings

├── main.py                     ← FastAPI app

├── database/

│   ├── connection.py

│   └── models.py               ← all tables

├── llm/

│   └── client.py               ← Transformers singleton (GPU)

└── routes/

    └── chat.py                 ← /chat WebSocket (streaming)
```

Plus a basic `frontend/chat.html` so you can talk to Hermes immediately.

> [!NOTE]

> **Approve this plan to start building Phase 1.**

> We'll have a working local LLM chat interface connected to your GPU in the first session.
>
> u
