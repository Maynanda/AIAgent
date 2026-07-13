"""
ARIA / Hermes — FastAPI Application Entry Point
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import settings

# ── Logging setup ────────────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer() if settings.is_dev else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
)
logging.basicConfig(level=logging.INFO if settings.is_dev else logging.WARNING)
logger = structlog.get_logger()

FRONTEND_DIR = Path(__file__).parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: initialize DB, LLM, embedder.
    Shutdown: clean up resources.
    """
    logger.info("🚀 Starting ARIA / Hermes...")

    # 1. Initialize database tables
    logger.info("Initializing database...")
    from database.connection import init_db
    await init_db()
    logger.info("✅ Database ready")

    # 2. Load embedding model (fast, ~300MB)
    logger.info("Loading embedding model...")
    from rag.embedder import embedder
    embedder.initialize()

    # 3. Load LLM (slow, ~5GB — done last)
    logger.info("Loading LLM (this may take a few minutes on first run)...")
    from llm.client import llm
    llm.initialize()

    # 4. Load tool registry
    logger.info("Loading tool registry...")
    from tools.registry import tool_registry
    tools = tool_registry.list_tools()
    logger.info(f"✅ {len(tools)} tools registered")

    logger.info(f"✅ ARIA / Hermes is ready — http://{settings.app_host}:{settings.app_port}")

    yield

    # Shutdown
    logger.info("Shutting down ARIA / Hermes...")


# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="ARIA — Hermes",
    description="Autonomous Reasoning & Intelligence Agent — Your Personal AI Second Brain",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs" if settings.is_dev else None,
    redoc_url=None,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.is_dev else ["http://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files ─────────────────────────────────────────────────────────────
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

# ── API Routes ────────────────────────────────────────────────────────────────
from routes.chat import router as chat_router  # noqa: E402

app.include_router(chat_router, prefix="/api")


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health() -> dict:
    from llm.client import llm
    from rag.embedder import embedder
    return {
        "status": "ok",
        "app": settings.app_name,
        "env": settings.app_env,
        "llm_loaded": llm._initialized,
        "embedder_loaded": embedder._initialized,
    }


# ── Frontend page serving ─────────────────────────────────────────────────────
@app.get("/")
async def serve_dashboard():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/chat")
async def serve_chat():
    return FileResponse(FRONTEND_DIR / "pages" / "chat.html")


@app.get("/projects")
async def serve_projects():
    index = FRONTEND_DIR / "pages" / "projects.html"
    if index.exists():
        return FileResponse(index)
    return FileResponse(FRONTEND_DIR / "index.html")


# Catch-all for SPA-style navigation
@app.get("/{path:path}")
async def serve_frontend(path: str):
    page = FRONTEND_DIR / "pages" / f"{path}.html"
    if page.exists():
        return FileResponse(page)
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.is_dev,
        log_level="info",
    )
