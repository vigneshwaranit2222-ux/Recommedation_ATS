"""FastAPI application entry point for the AI Recruitment Suite.

Creates the FastAPI app, registers the recruitment router, exposes a
``/health`` endpoint, and runs ``init_db()`` on startup via the lifespan
context manager.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from .database import init_db
from .routers.auth import router as auth_router
from .routers.chatbot import router as chatbot_router
from .routers.recruitment import router as recruitment_router

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: startup/shutdown hooks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run dev-only table creation on startup."""
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

# Include the authentication router (registration, login, user info).
app.include_router(auth_router)

# Include the recruitment router (jobs, questions, interview chat, ranking).
app.include_router(recruitment_router)

# Include the general-purpose chatbot router.
app.include_router(chatbot_router)

# Mount static directory if it exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Root Endpoint (Serves Frontend SPA or JSON Info)
# ---------------------------------------------------------------------------

@app.get("/", tags=["root"])
async def root():
    """Root endpoint serving static/index.html via FileResponse if available."""
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    
    return JSONResponse(
        content={
            "service": "AI Recruitment & Hiring Suite",
            "version": "2.0.0",
            "docs": "/docs",
            "health": "/health",
            "api": "/api/v1",
        }
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"])
async def health_check():
    """Liveness probe — returns 200 if the process is alive."""
    return {
        "status": "ok",
        "service": "AI Recruitment & Hiring Suite",
        "version": "2.0.0",
    }