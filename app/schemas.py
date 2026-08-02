"""Pydantic v2 request/response schemas for the recruitment API.

Schemas are kept **separate** from ORM models (``app/models.py``) so that:

1. **Security** — ``hashed_password`` and other internal columns are never
   accidentally serialized over the wire. The ORM model has the column;
   the schema does not, so it can never leak.
2. **Stability** — the API contract is decoupled from the DB schema. Adding
   a column to a table doesn't change the API response; adding a field to
   a schema doesn't require a migration.
3. **Validation** — Pydantic validates and coerces input before it reaches
   the ORM layer, so malformed data is rejected with a 422 before any DB
   query is attempted.

All schemas use Pydantic v2's ``model_config = ConfigDict(from_attributes=True)``
so they can be constructed from ORM objects via ``ModelOut.model_validate(orm_obj)``.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# ===========================================================================
# Shared / common schemas
# ===========================================================================

class ORMBase(BaseModel):
    """Base schema with ORM compatibility enabled.

    ``from_attributes=True`` (Pydantic v2 equivalent of v1's ``orm_mode``)
    allows constructing a schema instance from an ORM object by reading
    its attributes, e.g. ``JobOut.model_validate(job_orm_obj)``.
    """

    model_config = ConfigDict(from_attributes=True)


# ===========================================================================
# User schemas
# ===========================================================================

class UserOut(ORMBase):
    """User response — never includes ``hashed_password``.

    The ``hashed_password`` column exists on the ``User`` ORM model but is
    deliberately absent here so it can never be serialized into an API
    response, even by accident.
    """

    id: uuid.UUID
    email: str
    full_name: str
    role: str
    is_active: bool
    created_at: datetime


# ===========================================================================
# Job schemas
# ===========================================================================

class JobGenerateRequest(BaseModel):
    """Request body for ``POST /api/v1/jobs/generate``.

    The ``raw_input`` is a short natural-language prompt (e.g. "Need 2 YOE
    React Developer with Tailwind & GraphQL") that the LLM expands into a
    full job description with structured keywords.
    """

    raw_input: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Short natural-language job description prompt.",
        examples=["Need 2 YOE React Developer with Tailwind & GraphQL"],
    )
    created_by: Optional[uuid.UUID] = Field(
        default=None,
        description="Optional recruiter user id to attribute the job to.",
    )


class JobOut(ORMBase):
    """Job response — the canonical job representation over the wire."""

    id: uuid.UUID
    title: str
    description: str
    keywords: list[str] = Field(default_factory=list)
    extra_metadata: Optional[dict[str, Any]] = None
    chroma_doc_id: Optional[str] = None
    created_at: datetime


class JobGenerateResponse(JobOut):
    """Response for ``POST /api/v1/jobs/generate``.

    Extends ``JobOut`` with a confirmation message.
    """

    message: str = Field(
        default="Job generated and indexed successfully.",
        description="Human-readable confirmation.",
    )


# ===========================================================================
# Interview question schemas
# ===========================================================================

class QuestionOut(ORMBase):
    """A single interview question in the response."""

    id: uuid.UUID
    question_text: str
    category: str


class QuestionGenerateResponse(BaseModel):
    """Response for ``POST /api/v1/jobs/{job_id}/questions``."""

    job_id: uuid.UUID
    questions: list[QuestionOut] = Field(default_factory=list)
    total: int = Field(..., description="Number of questions generated.")


# ===========================================================================
# Interview chat schemas
# ===========================================================================

class ChatTurn(BaseModel):
    """A single turn in the chat history.

    ``role`` is ``"assistant"`` (the LLM/interviewer) or ``"user"``
    (the candidate). This mirrors the OpenAI chat message format so it
    can be passed directly to the HF router.
    """

    role: str = Field(..., pattern="^(assistant|user)$")
    content: str


class InterviewChatRequest(BaseModel):
    """Request body for ``POST /api/v1/interview/chat``.

    If ``session_id`` is omitted, a new ``InterviewSession`` is created.
    If ``candidate_message`` is omitted, the session starts with the first
    question (or resumes if the session already exists).
    """

    session_id: Optional[uuid.UUID] = Field(
        default=None,
        description="Omit to start a new interview session.",
    )
    candidate_id: uuid.UUID = Field(
        ...,
        description="The candidate (User) id for this interview.",
    )
    job_id: uuid.UUID = Field(
        ...,
        description="The job being interviewed for.",
    )
    candidate_message: Optional[str] = Field(
        default=None,
        max_length=5000,
        description="The candidate's answer to the previous question. "
        "Omit on the first turn to start the interview.",
    )


class InterviewChatResponse(BaseModel):
    """Response for ``POST /api/v1/interview/chat``.

    Returns the assistant's latest message and the full session state so
    the frontend can render the conversation and know when the interview
    is complete.
    """

    session_id: uuid.UUID
    assistant_message: str = Field(
        ...,
        description="The interviewer's latest message (acknowledgment + next question, or wrap-up).",
    )
    is_complete: bool = Field(
        ...,
        description="True when all questions have been asked and scored.",
    )
    final_score: Optional[float] = Field(
        default=None,
        description="Average of all per-turn scores (0–10). Present only when is_complete=True.",
    )
    feedback: Optional[str] = Field(
        default=None,
        description="Overall feedback summary. Present only when is_complete=True.",
    )
    chat_history: list[ChatTurn] = Field(
        default_factory=list,
        description="Full conversation history for client-side rendering.",
    )


# ===========================================================================
# Ranking schemas
# ===========================================================================

class CandidateResumeInput(BaseModel):
    """A single candidate's resume text for ranking.

    The ``resume_text`` is raw text (not a PDF) — the PDF parsing endpoint
    is out of scope for this pass. In a future iteration, a
    ``POST /api/v1/resumes/upload`` endpoint will parse PDFs via
    ``pdfplumber`` + ``spaCy`` and store the extracted text.
    """

    candidate_id: uuid.UUID
    resume_text: str = Field(..., min_length=1)


class RankCandidatesRequest(BaseModel):
    """Request body for ``POST /api/v1/jobs/{job_id}/rank``."""

    candidates: list[CandidateResumeInput] = Field(
        ...,
        min_length=1,
        description="List of candidate resumes to rank against the job.",
    )


class CandidateRankResult(BaseModel):
    """A single candidate's ranking result with full score breakdown.

    All four scores are returned (never collapsed to a single opaque
    number) so recruiters can understand **why** a candidate ranked where
    they did — e.g., a candidate might have high keyword match but low
    semantic similarity, indicating keyword-stuffing.
    """

    candidate_id: uuid.UUID
    tfidf_score: float = Field(
        ...,
        description="TF-IDF cosine similarity × 100 (0–100). 50% weight.",
    )
    keyword_score: float = Field(
        ...,
        description="Keyword match percentage (0–100). 35% weight.",
    )
    vector_score: float = Field(
        ...,
        description="ChromaDB cosine similarity (0–100). 15% weight.",
    )
    final_score: float = Field(
        ...,
        description="Weighted total: 0.50×tfidf + 0.35×keyword + 0.15×vector (0–100).",
    )


class RankCandidatesResponse(BaseModel):
    """Response for ``POST /api/v1/jobs/{job_id}/rank``.

    Candidates are sorted descending by ``final_score`` by the ranking
    service before this response is constructed.
    """

    job_id: uuid.UUID
    total_candidates: int
    results: list[CandidateRankResult] = Field(default_factory=list)


# ===========================================================================
# Error schema (used by FastAPI's default exception handler)
# ===========================================================================

class ErrorResponse(BaseModel):
    """Standard error envelope for non-2xx responses.

    FastAPI's default ``HTTPException`` already returns ``{"detail": "..."}``;
    this schema documents that shape so the frontend has a typed contract.
    """

    detail: str