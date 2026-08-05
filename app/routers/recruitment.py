"""Recruitment API router — four endpoints wired to services.

Endpoints
---------
1. ``POST /api/v1/jobs/generate``           — LLM-generate a job from a short prompt.
2. ``POST /api/v1/jobs/{job_id}/questions`` — LLM-generate interview questions.
3. ``POST /api/v1/interview/chat``          — Conduct/scoring an interview turn.
4. ``POST /api/v1/jobs/{job_id}/rank``      — Hybrid-rank candidates against a job.

Design principles
-----------------
* **Thin router** — endpoints do only validation, I/O orchestration, and
  error translation. All business logic lives in the service layer.
* **502 for HF failures** — when an HF router call fails, the failure is
  in an upstream dependency, not in application code. A 500 would imply
  a bug in our own code. 502 (Bad Gateway) is the correct semantic.
* **Scoring failure ≠ interview abort** — if the per-turn scoring HF
  call fails, the interview continues. The candidate's answer is still
  appended to chat_history; only the score is missing.
* **Sync→async bridge** — the HF service uses plain ``requests`` (sync).
  We wrap calls in ``asyncio.to_thread()`` so the event loop isn't
  blocked during the HTTP round-trip.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import (
    InterviewQuestion,
    InterviewSession,
    JobRequirement,
    QuestionCategory,
    User,
    UserRole,
)
from ..schemas import (
    CandidateRankResult,
    ChatTurn,
    CreateSessionRequest,
    InterviewChatRequest,
    InterviewChatResponse,
    InterviewSessionSummary,
    JobGenerateRequest,
    JobGenerateResponse,
    JobOut,
    QuestionGenerateResponse,
    QuestionOut,
    RankCandidatesRequest,
    RankCandidatesResponse,
)
from ..services import hf_service
from ..services.hf_service import HFServiceError
from ..services.ranking_service import rank_candidates
from ..services.vector_service import vector_service
from ..short_id import format_short_id, get_short_id, resolve_id

router = APIRouter(prefix="/api/v1")


# ===========================================================================
# Helper: run sync service functions in a thread
# ===========================================================================

async def _run_sync(func, *args, **kwargs):
    """Run a sync function in a thread pool to avoid blocking the event loop.

    The HF service uses plain ``requests`` (sync). Calling it directly in
    an ``async def`` endpoint would block the event loop for the duration
    of the HTTP round-trip (potentially 30–60s on the free tier).
    ``asyncio.to_thread`` runs the call in a worker thread.
    """
    return await asyncio.to_thread(func, *args, **kwargs)


# ===========================================================================
# 1. POST /api/v1/jobs/generate
# ===========================================================================

@router.post("/jobs/generate", response_model=JobGenerateResponse, status_code=status.HTTP_201_CREATED, tags=["Jobs"])
async def generate_job(
    request: JobGenerateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Generate a structured job description from a short natural-language prompt.

    Calls the HF router with a system prompt instructing JSON-only output
    matching ``{"title", "description", "keywords"}``. Saves to PostgreSQL,
    then indexes into ChromaDB, storing the returned doc id on the row.
    """
    # --- 1. Call HF router (sync → thread) ------------------------------
    try:
        job_data = await _run_sync(hf_service.generate_job_description, request.raw_input)
    except HFServiceError as exc:
        # 502: upstream dependency failure, not app code failure.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Job generation failed (upstream LLM error): {exc}",
        )

    # --- 2. Persist to PostgreSQL ---------------------------------------
    job = JobRequirement(
        title=job_data["title"],
        description=job_data["description"],
        keywords=job_data["keywords"],
        created_by=None,  # TODO: wire in auth
    )
    try:
        db.add(job)
        await db.flush()  # populates job.id without committing
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save job to database: {exc.__class__.__name__}",
        )

    # --- 3. Index into ChromaDB -----------------------------------------
    job_id_str = str(job.id)
    try:
        chroma_doc_id = await _run_sync(
            vector_service.index_job,
            job_id=job_id_str,
            title=job.title,
            description=job.description,
            keywords=job.keywords,
        )
        job.chroma_doc_id = chroma_doc_id
        await db.flush()
    except Exception as exc:
        # Chroma indexing failure is not fatal — the job is still in Postgres.
        # We log the error but don't fail the request. The chroma_doc_id
        # stays None, and vector similarity will return 0.0 for this job.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Job was not indexed in ChromaDB: {exc}",
        ) from exc

    # --- 4. Return response ---------------------------------------------
    return JobGenerateResponse(
        id=job.id,
        title=job.title,
        description=job.description,
        keywords=job.keywords,
        chroma_doc_id=job.chroma_doc_id,
        created_at=job.created_at,
    )


# ===========================================================================
# 1b. GET /api/v1/jobs
# ===========================================================================

@router.get("/jobs", response_model=List[JobOut], tags=["Jobs"])
async def list_jobs(
    db: AsyncSession = Depends(get_db),
):
    """List all available jobs with their UUIDs and short_ids (j001, j002)."""
    try:
        result = await db.execute(
            select(JobRequirement).order_by(JobRequirement.created_at.asc())
        )
        jobs = result.scalars().all()
        resp = []
        for idx, j in enumerate(jobs, 1):
            j_out = JobOut.model_validate(j)
            j_out.short_id = format_short_id("j", idx)
            resp.append(j_out)
        return resp
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {exc.__class__.__name__}",
        )


# ===========================================================================
# 2. POST /api/v1/jobs/{job_id}/questions
# ===========================================================================

@router.post("/jobs/{job_id}/questions", response_model=QuestionGenerateResponse, status_code=status.HTTP_201_CREATED, tags=["Questions"])
async def generate_questions(
    job_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Generate 5–10 interview questions for a job via the HF router."""
    # --- 1. Fetch job from PostgreSQL -----------------------------------
    try:
        resolved_job_id = await resolve_id(job_id, JobRequirement, "j", db)
        job = await db.get(JobRequirement, resolved_job_id) if resolved_job_id else None
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {exc.__class__.__name__}",
        )

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with id {job_id} not found.",
        )

    # --- 2. Call HF router for questions --------------------------------
    try:
        raw_questions = await _run_sync(
            hf_service.generate_interview_questions,
            job_title=job.title,
            job_description=job.description,
            keywords=job.keywords,
            num_questions=7,
        )
    except HFServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Question generation failed (upstream LLM error): {exc}",
        )

    # --- 3. Bulk-insert questions ---------------------------------------
    # The HF service already normalized categories to strings. We map them
    # to the QuestionCategory enum here. If an unexpected value slips
    # through, default to technical — never drop the question text.
    category_map = {
        "technical": QuestionCategory.technical,
        "behavioral": QuestionCategory.behavioral,
        "experience": QuestionCategory.experience,
    }

    question_models: List[InterviewQuestion] = []
    for q in raw_questions:
        cat_str = q.get("category", "technical")
        category = category_map.get(cat_str, QuestionCategory.technical)
        question = InterviewQuestion(
            job_id=job.id,
            question_text=q["question_text"],
            category=category,
        )
        question_models.append(question)

    try:
        db.add_all(question_models)
        await db.flush()
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save questions: {exc.__class__.__name__}",
        )

    # --- 4. Return response ---------------------------------------------
    return QuestionGenerateResponse(
        job_id=job.id,
        questions=[
            QuestionOut(
                id=q.id,
                question_text=q.question_text,
                category=q.category.value,
            )
            for q in question_models
        ],
        total=len(question_models),
    )


# ===========================================================================
# 3. POST /api/v1/interview/chat
# ===========================================================================

@router.post("/interview/chat", response_model=InterviewChatResponse)
async def interview_chat(
    request: InterviewChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """Conduct one turn of an AI interview.

    If ``session_id`` is omitted, a new ``InterviewSession`` is created.
    If ``candidate_message`` is provided and there's a prior question, the
    response is scored via a separate HF call (0–10 scale) before the next
    turn is generated. A scoring failure does **not** abort the interview.
    The next turn acknowledges the candidate's answer and asks the next
    question verbatim, or wraps up if the bank is exhausted.
    """
    # Resolve input IDs (UUID or short_id j001, c001, s001)
    job_uuid = await resolve_id(request.job_id, JobRequirement, "j", db)
    if not job_uuid:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job requirement {request.job_id} not found.",
        )

    # --- 1. Create or fetch session -------------------------------------
    session_uuid = await resolve_id(request.session_id, InterviewSession, "s", db) if request.session_id else None
    if session_uuid is not None:
        # Fetch existing session.
        try:
            session = await db.get(InterviewSession, session_uuid)
        except SQLAlchemyError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error: {exc.__class__.__name__}",
            )
        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Interview session {request.session_id} not found.",
            )
    else:
        # Create new session.
        cand_id = await resolve_id(request.candidate_id, User, "c", db) if request.candidate_id else None
        if cand_id is None:
            user_res = await db.execute(select(User.id).limit(1))
            cand_id = user_res.scalar_one_or_none()
            if cand_id is None:
                guest_user = User(
                    email="guest_candidate@ats.local",
                    hashed_password="guest_password_123",
                    full_name="Guest Candidate",
                    role=UserRole.candidate,
                )
                db.add(guest_user)
                await db.flush()
                cand_id = guest_user.id

        session = InterviewSession(
            candidate_id=cand_id,
            job_id=job_uuid,
            chat_history=[],
            is_complete=False,
        )
        try:
            db.add(session)
            await db.flush()
        except SQLAlchemyError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create session: {exc.__class__.__name__}",
            )

    # If the interview is already complete, return the final state.
    if session.is_complete:
        s_short = await get_short_id(session.id, InterviewSession, "s", db)
        c_short = await get_short_id(session.candidate_id, User, "c", db)
        j_short = await get_short_id(session.job_id, JobRequirement, "j", db)
        return InterviewChatResponse(
            short_id=s_short,
            candidate_short_id=c_short,
            job_short_id=j_short,
            session_id=session.id,
            candidate_id=session.candidate_id,
            assistant_message="This interview session is already complete.",
            is_complete=True,
            final_score=session.final_score,
            feedback=session.feedback,
            chat_history=[ChatTurn(**t) for t in session.chat_history],
        )

    # --- 2. Fetch the job's question bank -------------------------------
    try:
        job = await db.get(JobRequirement, job_uuid)
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {exc.__class__.__name__}",
        )
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {request.job_id} not found.",
        )

    # Load questions via relationship (lazy load triggers a query).
    try:
        result = await db.execute(
            select(InterviewQuestion)
            .where(InterviewQuestion.job_id == job.id)
            .order_by(InterviewQuestion.created_at)
        )
        all_questions = [q.question_text for q in result.scalars().all()]
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load questions: {exc.__class__.__name__}",
        )

    if not all_questions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This job has no interview questions. Generate questions first.",
        )

    # --- 3. Track asked questions ---------------------------------------
    # Scan chat_history for assistant turns and check which questions
    # from the bank have been asked (substring match).
    chat_history: list[dict[str, Any]] = list(session.chat_history or [])

    asked_questions: set[str] = set()
    for turn in chat_history:
        if turn.get("role") == "assistant":
            for q in all_questions:
                if q in turn.get("content", ""):
                    asked_questions.add(q)

    remaining_questions = [q for q in all_questions if q not in asked_questions]

    # --- 4. Score the candidate's last response (if applicable) ---------
    per_turn_scores: list[float] = []

    # Collect existing scores from chat_history (for final_score computation).
    for turn in chat_history:
        if turn.get("role") == "user" and "score" in turn:
            try:
                per_turn_scores.append(float(turn["score"]))
            except (ValueError, TypeError):
                continue

    last_question: Optional[str] = None
    if chat_history and chat_history[-1].get("role") == "assistant" and not request.candidate_message:
        last_assistant_message = str(chat_history[-1].get("content", ""))
        if any(q in last_assistant_message for q in all_questions):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="candidate_message is required to answer the current interview question.",
            )

    if request.candidate_message and asked_questions:
        # Find the last question asked (the one the candidate is answering).
        for turn in reversed(chat_history):
            if turn.get("role") == "assistant":
                for q in all_questions:
                    if q in turn.get("content", ""):
                        last_question = q
                        break
                if last_question:
                    break

        if last_question:
            try:
                score_result = await _run_sync(
                    hf_service.score_interview_response,
                    question=last_question,
                    answer=request.candidate_message,
                )
                # Append user turn WITH score.
                chat_history.append(
                    {
                        "role": "user",
                        "content": request.candidate_message,
                        "score": score_result["score"],
                        "feedback": score_result["feedback"],
                    }
                )
                per_turn_scores.append(score_result["score"])
            except HFServiceError:
                # Scoring failure does NOT abort the interview.
                # Append user turn WITHOUT score and continue.
                chat_history.append(
                    {
                        "role": "user",
                        "content": request.candidate_message,
                    }
                )
        else:
            # No prior question found — just append the user message.
            chat_history.append(
                {"role": "user", "content": request.candidate_message}
            )
    elif request.candidate_message:
        # New session with a candidate_message but no prior question.
        chat_history.append(
            {"role": "user", "content": request.candidate_message}
        )

    # --- 5. Generate the next interviewer turn --------------------------
    try:
        assistant_message = await _run_sync(
            hf_service.run_interview_turn,
            chat_history=chat_history,
            next_question=remaining_questions[0] if remaining_questions else "",
        )
    except HFServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Interview turn generation failed (upstream LLM error): {exc}",
        )

    # Append assistant turn.
    chat_history.append({"role": "assistant", "content": assistant_message})

    # --- 6. Check if interview is complete ------------------------------
    # Re-scan for asked questions (the new assistant turn may have asked
    # the last remaining question).
    for q in all_questions:
        if q in assistant_message:
            asked_questions.add(q)

    remaining_after = [q for q in all_questions if q not in asked_questions]

    # The last question must be answered before the interview can finish.
    is_complete = bool(request.candidate_message and last_question) and len(remaining_after) == 0
    final_score: Optional[float] = None
    feedback: Optional[str] = None

    if is_complete:
        session.is_complete = True
        if per_turn_scores:
            final_score = round(sum(per_turn_scores) / len(per_turn_scores), 2)
            session.final_score = final_score
        # Generate simple feedback summary.
        if per_turn_scores:
            avg = final_score or 0
            if avg >= 7:
                feedback = f"Strong performance (avg score: {avg}/10)."
            elif avg >= 5:
                feedback = f"Average performance (avg score: {avg}/10)."
            else:
                feedback = f"Below-average performance (avg score: {avg}/10)."
            session.feedback = feedback

    # --- 7. Save session ------------------------------------------------
    session.chat_history = chat_history
    try:
        await db.flush()
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save session: {exc.__class__.__name__}",
        )

    # --- 8. Return response ---------------------------------------------
    s_short = await get_short_id(session.id, InterviewSession, "s", db)
    c_short = await get_short_id(session.candidate_id, User, "c", db)
    j_short = await get_short_id(session.job_id, JobRequirement, "j", db)
    return InterviewChatResponse(
        short_id=s_short,
        candidate_short_id=c_short,
        job_short_id=j_short,
        session_id=session.id,
        candidate_id=session.candidate_id,
        assistant_message=assistant_message,
        is_complete=is_complete,
        final_score=final_score,
        feedback=feedback,
        chat_history=[ChatTurn(role=t["role"], content=t["content"]) for t in chat_history],
    )


# ===========================================================================
# 3b. POST & GET /api/v1/interview/sessions
# ===========================================================================

@router.post("/interview/sessions", response_model=InterviewSessionSummary, status_code=status.HTTP_201_CREATED, tags=["Interview Chat"])
async def create_interview_session(
    request: CreateSessionRequest,
    db: AsyncSession = Depends(get_db),
):
    """Explicitly create a new interview session and return the generated session_id."""
    job_uuid = await resolve_id(request.job_id, JobRequirement, "j", db)
    if job_uuid is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job requirement {request.job_id} not found.",
        )

    cand_id = await resolve_id(request.candidate_id, User, "c", db) if request.candidate_id else None
    if cand_id is None:
        user_res = await db.execute(select(User.id).limit(1))
        cand_id = user_res.scalar_one_or_none()
        if cand_id is None:
            guest_user = User(
                email="guest_candidate@ats.local",
                hashed_password="guest_password_123",
                full_name="Guest Candidate",
                role=UserRole.candidate,
            )
            db.add(guest_user)
            await db.flush()
            cand_id = guest_user.id

    session = InterviewSession(
        candidate_id=cand_id,
        job_id=job_uuid,
        chat_history=[],
        is_complete=False,
    )
    db.add(session)
    await db.flush()

    s_short = await get_short_id(session.id, InterviewSession, "s", db)
    c_short = await get_short_id(session.candidate_id, User, "c", db)
    j_short = await get_short_id(session.job_id, JobRequirement, "j", db)

    return InterviewSessionSummary(
        short_id=s_short,
        candidate_short_id=c_short,
        job_short_id=j_short,
        session_id=session.id,
        candidate_id=session.candidate_id,
        job_id=session.job_id,
        is_complete=session.is_complete,
        final_score=session.final_score,
        feedback=session.feedback,
        created_at=session.created_at,
    )


@router.get("/interview/sessions", response_model=List[InterviewSessionSummary], tags=["Interview Chat"])
async def list_interview_sessions(
    candidate_id: Optional[str] = None,
    job_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Retrieve interview sessions with UUIDs and short IDs (s001, c001, j001)."""
    try:
        cand_uuid = await resolve_id(candidate_id, User, "c", db) if candidate_id else None
        job_uuid = await resolve_id(job_id, JobRequirement, "j", db) if job_id else None

        query = select(InterviewSession).order_by(InterviewSession.created_at.asc())
        if cand_uuid:
            query = query.where(InterviewSession.candidate_id == cand_uuid)
        if job_uuid:
            query = query.where(InterviewSession.job_id == job_uuid)

        result = await db.execute(query)
        sessions = result.scalars().all()

        resp = []
        for idx, s in enumerate(sessions, 1):
            c_short = await get_short_id(s.candidate_id, User, "c", db)
            j_short = await get_short_id(s.job_id, JobRequirement, "j", db)
            resp.append(
                InterviewSessionSummary(
                    short_id=format_short_id("s", idx),
                    candidate_short_id=c_short,
                    job_short_id=j_short,
                    session_id=s.id,
                    candidate_id=s.candidate_id,
                    job_id=s.job_id,
                    is_complete=s.is_complete,
                    final_score=s.final_score,
                    feedback=s.feedback,
                    created_at=s.created_at,
                )
            )
        return resp
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {exc.__class__.__name__}",
        )


# ===========================================================================
# 4. POST /api/v1/jobs/{job_id}/rank
# ===========================================================================

@router.post("/jobs/{job_id}/rank", response_model=RankCandidatesResponse, tags=["Resume Ranking"])
async def rank_candidates_endpoint(
    job_id: str,
    request: RankCandidatesRequest,
    db: AsyncSession = Depends(get_db),
):
    """Rank candidates against a job using the hybrid scoring strategy."""
    # --- 1. Fetch job from PostgreSQL -----------------------------------
    try:
        resolved_job_id = await resolve_id(job_id, JobRequirement, "j", db)
        job = await db.get(JobRequirement, resolved_job_id) if resolved_job_id else None
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {exc.__class__.__name__}",
        )

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with id {job_id} not found.",
        )

    # --- 2. Rank candidates (sync → thread) ------------------------------
    candidates_data = [
        (str(c.candidate_id), c.resume_text) for c in request.candidates
    ]

    try:
        results = await _run_sync(
            rank_candidates,
            job_id=str(job.id),
            job_description=job.description,
            keywords=job.keywords,
            candidates=candidates_data,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ranking failed: {exc}",
        )

    # --- 3. Return response ---------------------------------------------
    return RankCandidatesResponse(
        job_id=job.id,
        total_candidates=len(results),
        results=[
            CandidateRankResult(
                candidate_id=uuid.UUID(r["candidate_id"]),
                tfidf_score=r["tfidf_score"],
                keyword_score=r["keyword_score"],
                vector_score=r["vector_score"],
                final_score=r["final_score"],
            )
            for r in results
        ],
    )
