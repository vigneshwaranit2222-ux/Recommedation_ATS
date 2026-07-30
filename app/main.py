"""FastAPI application: endpoints, error handling and static file serving.

Endpoints
---------
* POST /post_job/        - create a job (SQLite + Chroma) with 207 partial-success.
* GET  /jobs/            - list all jobs for the frontend dropdown.
* POST /upload_resume/   - upload a PDF resume, extract text + NER, index in Chroma.
* POST /rank_candidates/ - rank all resumes against a job with full breakdown.

Error handling
--------------
Every I/O boundary (PDF parsing, DB writes, vector store writes) is wrapped
so the API returns structured JSON errors instead of crashing the worker.
"""

from __future__ import annotations

import os
import traceback
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from .database import JobRequirement, get_db, init_db
from .ner_engine import ner_engine
from .pdf_parser import PDFParseError, extract_text_from_pdf
from .scorer import CandidateResult, rank_resumes
from .vector_db import vector_db

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class JobOut(BaseModel):
    """Job representation returned by GET /jobs/ and POST /post_job/."""

    id: int
    title: str
    description: str
    required_keywords: str


class JobCreateResponse(BaseModel):
    """Response for POST /post_job/ including indexing status."""

    message: str
    job: JobOut
    indexed_in_chroma: bool
    error: Optional[str] = Field(
        default=None,
        description="Present only when Chroma indexing failed (207 response).",
    )


class ResumeUploadResponse(BaseModel):
    """Response for POST /upload_resume/."""

    message: str
    resume_id: str
    student_id: str
    student_name: str
    extracted_skills: List[str]
    organizations: List[str]
    degrees: List[str]
    locations: List[str]
    text_length: int


class RankResponse(BaseModel):
    """Response for POST /rank_candidates/."""

    job_id: int
    job_title: str
    total_candidates: int
    candidates: List[CandidateResult]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ATS Resume Ranking System",
    description=(
        "A production-ready FastAPI Applicant Tracking System that ranks "
        "student resumes against posted jobs using a hybrid scoring "
        "strategy: 50% TF-IDF cosine similarity, 35% keyword match, and "
        "15% ChromaDB vector similarity."
    ),
    version="1.0.0",
)

# Resolve the static directory (sibling of the `app` package).
_STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def _on_startup() -> None:
    """Initialize the SQLite schema on app start.

    We deliberately do NOT pre-load the spaCy model here so that a missing
    model doesn't prevent the server from starting; the error surfaces only
    when a resume is actually uploaded (with a clear download command).
    """
    init_db()


# ---------------------------------------------------------------------------
# Custom exception handler for PDF parse errors
# ---------------------------------------------------------------------------

@app.exception_handler(PDFParseError)
async def pdf_parse_error_handler(request: Request, exc: PDFParseError):
    """Map PDFParseError to a 422 with a clear message."""
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    """Health check / root redirect info."""
    return {
        "service": "ATS Resume Ranking System",
        "status": "ok",
        "docs": "/docs",
        "ui": "/static/index.html",
    }


@app.post("/post_job/", response_model=JobCreateResponse)
def post_job(
    title: str = Form(...),
    description: str = Form(...),
    required_keywords: str = Form(""),
    db: Session = Depends(get_db),
):
    """Create a job in SQLite AND index it into Chroma.

    Per the spec: if Chroma indexing fails AFTER the SQLite write succeeds,
    we do NOT roll back the SQLite write. Instead we return a 207
    partial-success response describing what succeeded and what failed.
    """
    # --- 1. Persist to SQLite ---------------------------------------------
    job = JobRequirement(
        title=title.strip(),
        description=description.strip(),
        required_keywords=required_keywords.strip(),
    )
    try:
        db.add(job)
        db.commit()
        db.refresh(job)  # populates job.id
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save job to database: {exc.__class__.__name__}",
        )

    job_out = JobOut(
        id=job.id,
        title=job.title,
        description=job.description,
        required_keywords=job.required_keywords,
    )

    # --- 2. Index into Chroma (best-effort, no rollback on failure) -------
    try:
        vector_db.index_job(
            job_id=job.id,
            title=job.title,
            description=job.description,
            required_keywords=job.required_keywords,
        )
        return JobCreateResponse(
            message="Job created and indexed successfully.",
            job=job_out,
            indexed_in_chroma=True,
        )
    except Exception as exc:  # noqa: BLE001 - any Chroma failure
        # 207 Multi-Status: SQLite write succeeded, Chroma indexing failed.
        # We intentionally do NOT roll back the SQLite transaction.
        return JSONResponse(
            status_code=207,
            content=JobCreateResponse(
                message=(
                    "Job saved to database, but indexing into the vector "
                    "store failed. The job is stored and usable for ranking "
                    "via TF-IDF + keyword matching, but semantic vector "
                    "search may be degraded."
                ),
                job=job_out,
                indexed_in_chroma=False,
                error=f"{exc.__class__.__name__}: {exc}",
            ).model_dump(),
        )


@app.get("/jobs/", response_model=List[JobOut])
def list_jobs(db: Session = Depends(get_db)):
    """List all saved jobs, newest first, for the frontend dropdown."""
    try:
        jobs = db.query(JobRequirement).order_by(JobRequirement.id.desc()).all()
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch jobs: {exc.__class__.__name__}",
        )
    return [
        JobOut(
            id=j.id,
            title=j.title,
            description=j.description,
            required_keywords=j.required_keywords,
        )
        for j in jobs
    ]


@app.post("/upload_resume/", response_model=ResumeUploadResponse)
async def upload_resume(
    student_id: str = Form(...),
    student_name: str = Form(...),
    file: UploadFile = File(...),
):
    """Upload a PDF resume, extract text + entities, and index into Chroma.

    Error handling:
      * Non-PDF / corrupt / scanned PDF -> 422 via PDFParseError handler.
      * spaCy model missing -> 500 with the exact download command.
      * Chroma write failure -> 500.
    """
    # --- 1. Validate file type --------------------------------------------
    filename = file.filename or "resume.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted. Please upload a .pdf resume.",
        )

    # --- 2. Read bytes ----------------------------------------------------
    try:
        pdf_bytes = await file.read()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read uploaded file: {exc}",
        )
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # --- 3. Extract text (raises PDFParseError -> 422) --------------------
    resume_text = extract_text_from_pdf(pdf_bytes)

    # --- 4. NER extraction (may raise RuntimeError for missing model) -----
    try:
        entities = ner_engine.extract_entities(resume_text)
    except RuntimeError:
        # Re-raise with 500 so the download command is visible to the user.
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Entity extraction failed: {exc}",
        )

    # --- 5. Index into Chroma ---------------------------------------------
    # Use a deterministic-ish resume id: student_id + filename hash fallback.
    # We use student_id as the primary id so re-uploads upsert instead of
    # creating duplicates.
    resume_id = f"resume_{student_id}"

    metadata = {
        "student_id": student_id,
        "student_name": student_name,
        "filename": filename,
        "skills": entities["skills"],
        "organizations": entities["organizations"],
        "degrees": entities["degrees"],
        "locations": entities["locations"],
        "text_length": len(resume_text),
    }

    try:
        vector_db.index_resume(
            resume_id=resume_id,
            text=resume_text,
            metadata=metadata,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Failed to index resume in vector store: {exc}",
        )

    return ResumeUploadResponse(
        message="Resume uploaded, parsed and indexed successfully.",
        resume_id=resume_id,
        student_id=student_id,
        student_name=student_name,
        extracted_skills=entities["skills"],
        organizations=entities["organizations"],
        degrees=entities["degrees"],
        locations=entities["locations"],
        text_length=len(resume_text),
    )


@app.post("/rank_candidates/", response_model=RankResponse)
def rank_candidates(
    job_id: int = Form(...),
    db: Session = Depends(get_db),
):
    """Rank all stored resumes against the given job.

    Returns candidates sorted highest-to-lowest with full score breakdowns
    and matched/missing keyword lists.
    """
    # --- 1. Fetch the job from SQLite -------------------------------------
    try:
        job = db.query(JobRequirement).filter(JobRequirement.id == job_id).first()
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Database error while fetching job: {exc.__class__.__name__}",
        )

    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job with id {job_id} not found.",
        )

    # --- 2. Rank ----------------------------------------------------------
    try:
        candidates = rank_resumes(
            job_id=job.id,
            job_title=job.title,
            job_description=job.description,
            required_keywords_raw=job.required_keywords,
        )
    except RuntimeError:
        # Propagate spaCy model errors (shouldn't normally happen here, but
        # be defensive).
        raise
    except Exception as exc:  # noqa: BLE001
        # Log the traceback for debugging; return a clean 500.
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Ranking failed: {exc}",
        )

    return RankResponse(
        job_id=job.id,
        job_title=job.title,
        total_candidates=len(candidates),
        candidates=candidates,
    )


# ---------------------------------------------------------------------------
# Static file serving (frontend)
# ---------------------------------------------------------------------------

# Mount the static directory so /static/index.html serves the Bootstrap UI.
# This must come after the API routes so they take precedence.
app.mount("/static", StaticFiles(directory=_STATIC_DIR, html=True), name="static")