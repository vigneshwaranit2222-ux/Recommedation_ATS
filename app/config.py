"""Application configuration via pydantic-settings.

All runtime configuration is centralized here so that no module hardcodes
secrets, URLs, or model identifiers. Values are loaded from environment
variables and/or a local ``.env`` file (git-ignored).

Key design decisions
--------------------
* ``DATABASE_URL`` uses the ``postgresql+asyncpg`` driver because the entire
  data layer is async (FastAPI + SQLAlchemy 2.0 async ORM). Using the sync
  ``psycopg2`` driver would force ``run_sync`` wrappers around every query,
  negating the throughput benefits of async I/O on a managed Postgres pool.
* ``HF_ROUTER_BASE_URL`` defaults to Hugging Face's unified OpenAI-compatible
  router (``https://router.huggingface.co/v1``). As of 2026, the legacy
  per-model ``api-inference.huggingface.co/models/<model>`` endpoint is
  deprecated; the router is the single entry point for free-tier chat
  completions.
* ``HF_CHAT_MODEL`` is a config value (not hardcoded) because free-tier model
  availability rotates. The default ``meta-llama/Llama-3.1-8B-Instruct`` is
  a commonly available instruction-tuned model, but **verify it is live on
  the free tier** at
  https://huggingface.co/models?inference_provider=all&pipeline_tag=text-generation
  before deploying.
* JWT settings are placeholders — auth wiring is out of scope for this pass
  but the config slots exist so a future auth middleware can consume them
  without touching this file.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application settings loaded from env / .env file.

    Every field has either a default or is required (no default). Required
    fields will raise a ``ValidationError`` at startup if missing, which is
    preferable to a silent misconfiguration discovered at runtime.
    """

    # ------------------------------------------------------------------
    # Database (PostgreSQL via asyncpg)
    # ------------------------------------------------------------------
    # Example: postgresql+asyncpg://user:password@localhost:5432/ats_suite
    DATABASE_URL: str
    SQL_ECHO: bool = False  # Echo SQL statements to stdout (dev/debug only)

    # ------------------------------------------------------------------
    # Hugging Face free-tier inference (OpenAI-compatible router)
    # ------------------------------------------------------------------
    # Required: obtain from https://huggingface.co/settings/tokens
    HF_API_TOKEN: str
    # The unified router base URL. Do NOT use the legacy
    # api-inference.huggingface.co/models/<model> endpoint.
    HF_ROUTER_BASE_URL: str = "https://router.huggingface.co/v1"
    # Chat completions model id. Swappable without code changes.
    # VERIFY this model is live on the free tier before deploying.
    HF_CHAT_MODEL: str = "meta-llama/Llama-3.1-8B-Instruct"
    # Max tokens for chat completions. Tuned for structured JSON outputs
    # (job descriptions, question banks, interview turns) which rarely
    # exceed 1024 tokens.
    HF_CHAT_MAX_TOKENS: int = 1024
    # Temperature for generation. Low temperature for structured JSON
    # outputs to reduce hallucination of field names.
    HF_CHAT_TEMPERATURE: float = 0.3
    # Request timeout in seconds. HF free tier can be slow under load.
    HF_REQUEST_TIMEOUT: int = 60

    # ------------------------------------------------------------------
    # ChromaDB (local persistent vector store)
    # ------------------------------------------------------------------
    # Directory where Chroma persists its HNSW index + metadata to disk.
    CHROMA_PERSIST_DIR: str = "./chroma_data"
    # Collection name for job documents (title + description + keywords).
    CHROMA_COLLECTION_JOBS: str = "company_jobs"

    # ------------------------------------------------------------------
    # JWT auth (placeholders — wiring is out of scope for this pass)
    # ------------------------------------------------------------------
    JWT_SECRET_KEY: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Allow extra fields in .env without crashing (forward-compat).
        extra="ignore",
    )


# Module-level singleton. Importing modules do `from app.config import settings`.
# pydantic-settings reads the .env file once at instantiation.
settings = Settings()