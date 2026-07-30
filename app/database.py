"""SQLAlchemy + SQLite persistence layer.

Defines the database engine, a session factory, the declarative base and the
ORM model for the `job_requirements` table. Only job metadata is persisted in
SQLite; resume data lives entirely in ChromaDB (see `vector_db.py`).

Design notes
------------
* We use a file-based SQLite DB located next to this package
  (`ats_db.sqlite`) so data survives restarts.
* `check_same_thread=False` is required because FastAPI handles requests in a
  thread pool while SQLAlchemy connections are created in the main thread.
* A `get_db` dependency is provided for FastAPI route injection; it ensures
  the session is always closed, even on exception.
"""

from __future__ import annotations

import os
from typing import Generator

from sqlalchemy import Column, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ---------------------------------------------------------------------------
# Engine / session setup
# ---------------------------------------------------------------------------

# Resolve the SQLite file path relative to this module so the DB always lives
# inside the project directory regardless of the current working directory.
_DB_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_DB_DIR)  # ats_ranking_system/
DB_PATH = os.path.join(_PROJECT_DIR, "ats_db.sqlite")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

# `check_same_thread=False` lets the engine be shared across FastAPI worker
# threads. SQLite handles concurrency at the file level via its own locking.
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)

# `sessionmaker` is a factory; `autocommit=False` + `autoflush=False` gives us
# explicit transaction control which is safer for write endpoints.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Single declarative base shared by all ORM models in this module.
Base = declarative_base()


# ---------------------------------------------------------------------------
# ORM model
# ---------------------------------------------------------------------------

class JobRequirement(Base):
    """A posted job requirement persisted in SQLite.

    The same record is also indexed into Chroma's `company_jobs` collection
    (see `vector_db.py`) so we can perform semantic similarity later.
    """

    __tablename__ = "job_requirements"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    title = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=False)
    # Comma-separated required keywords, stored verbatim as provided by the
    # user. We keep the raw string so the original casing/format is preserved.
    required_keywords = Column(Text, nullable=False, default="")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<JobRequirement id={self.id} title={self.title!r}>"


# ---------------------------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create all tables if they do not already exist.

    Safe to call multiple times; `create_all` is idempotent.
    """
    Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

def get_db() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session and guarantee it is closed afterwards.

    Usage in a route::

        @app.post("/foo")
        def foo(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()