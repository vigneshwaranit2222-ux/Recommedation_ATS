"""ATS Resume Ranking System - application package.

This package contains all backend modules for the FastAPI-based Applicant
Tracking System (ATS) that ranks student resumes against posted jobs using a
hybrid scoring strategy:

  * 50% TF-IDF cosine similarity (job text vs resume text)
  * 35% keyword match percentage (required keywords found in resume text or
        extracted skills, via case-insensitive word-boundary regex)
  * 15% ChromaDB vector similarity (cosine distance -> 0-100 score)

Modules
-------
database  : SQLAlchemy engine/session + ORM models for SQLite persistence.
vector_db : Thin wrapper around a ChromaDB PersistentClient for two
            collections (`company_jobs`, `student_resumes`).
pdf_parser: Robust PDF text extraction with pdfplumber.
ner_engine: spaCy NER + PhraseMatcher skill extraction, plus org/degree/
            location extraction.
scorer    : Hybrid scoring logic combining TF-IDF, keyword match and vector
            similarity.
main      : FastAPI application, endpoints and static file mounting.
"""

__all__ = [
    "database",
    "vector_db",
    "pdf_parser",
    "ner_engine",
    "scorer",
    "main",
]