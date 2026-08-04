from __future__ import annotations

from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from .env
    """

    # -------------------------
    # PostgreSQL
    # -------------------------
    DATABASE_URL: str
    SQL_ECHO: bool = False

    # -------------------------
    # Hugging Face
    # -------------------------
    HF_TOKEN: str = Field(
        ...,
        validation_alias=AliasChoices("HF_TOKEN", "HF_API_TOKEN", "HUGGINGFACE_API_KEY"),
    )
    HF_ROUTER_BASE_URL: str = "https://router.huggingface.co/v1"

    HF_CHAT_MODEL_PRIMARY: str = "Qwen/Qwen2.5-Coder-32B-Instruct"
    HF_CHAT_MODEL_INTERVIEW: str = "meta-llama/Llama-3.1-8B-Instruct"
    HF_CHAT_MODEL_SCORING: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

    HF_CHAT_MAX_TOKENS: int = 1024
    HF_CHAT_TEMPERATURE: float = 0.3
    HF_REQUEST_TIMEOUT: int = 60

    # -------------------------
    # ChromaDB
    # -------------------------
    CHROMA_PERSIST_DIR: str = "./chroma_data"
    CHROMA_COLLECTION_JOBS: str = "company_jobs"
    CHROMA_COLLECTION_RESUMES: str = "student_resumes"

    # -------------------------
    # JWT
    # -------------------------
    JWT_SECRET_KEY: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()