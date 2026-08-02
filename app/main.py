"""FastAPI application entry point for the AI Recruitment Suite.

Creates the FastAPI app, registers the recruitment router, exposes a
``/health`` endpoint, and runs ``init_db()`` on startup via the lifespan
context manager.

Out of scope for this pass
--------------------------
* JWT auth wiring (config slots exist in ``app/config.py`` but no
  dependency/middleware is implemented yet).
* Alembic migrations (``init_db()`` uses ``create_all`` for dev; replace
  before production).
* Resume PDF parsing endpoint (``pdf_parser.py`` and ``ner_engine.py``
  are retained from the legacy codebase for a future
  ``POST /api/v1/resumes/upload`` endpoint).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .database import init_db
from .routers.recruitment import router as recruitment_router

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: startup/shutdown hooks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run dev-only table creation on startup.

    .. warning::
        ``init_db()`` uses ``Base.metadata.create_all`` which only creates
        missing tables — it does not handle schema evolution. Before
        production, replace with Alembic migrations.
    """
    logger.info("Starting AI Recruitment Suite — initializing database...")
    await init_db()
    logger.info("Database initialized. Ready to serve requests.")
    yield
    logger.info("Shutting down AI Recruitment Suite.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Recruitment & Hiring Suite",
    description=(
        "An end-to-end AI recruitment backend: LLM-powered job generation, "
        "interview question banks, conversational AI interviews with "
        "automated scoring, and hybrid resume ranking (TF-IDF + keyword + "
        "vector similarity)."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Include the recruitment router (all four endpoints under /api/v1).
app.include_router(recruitment_router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["health"])
async def health_check():
    """Liveness probe — returns 200 if the process is alive.

    This does **not** check DB or Chroma connectivity. A deeper readiness
    probe would ping the DB and Chroma, but for a simple liveness check
    (e.g. Kubernetes livenessProbe) we only need to know the process is
    serving HTTP.
    """
    return {
        "status": "ok",
        "service": "AI Recruitment & Hiring Suite",
        "version": "2.0.0",
    }


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

@app.get("/", tags=["root"])
async def root():
    """Root endpoint with service info and links."""
    return {
        "service": "AI Recruitment & Hiring Suite",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "api": "/api/v1",
    }