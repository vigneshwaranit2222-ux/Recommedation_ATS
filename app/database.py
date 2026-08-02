"""Async SQLAlchemy 2.0 engine and session setup for PostgreSQL.

This module replaces the legacy SQLite sync engine with an async engine
backed by ``asyncpg``. The entire data layer is async because FastAPI is
async-first and blocking DB calls in an event-loop thread starve other
requests (a single slow query can stall the whole worker).

Key design decisions
--------------------
* ``pool_pre_ping=True`` — emits a lightweight ``SELECT 1`` before
  handing out a pooled connection. Managed Postgres providers (RDS,
  Cloud SQL, Supabase) kill idle connections after 5–30 min of inactivity;
  without pre-ping the pool hands out dead sockets and the first query
  raises ``OperationalError: server closed the connection unexpectedly``.
* ``pool_recycle=1800`` — proactively recycle connections every 30 min
  (1800 s) even if they appear healthy. This stays below the typical
  managed-Postgres idle timeout (often 300 s for Supabase, 5 min for RDS
  proxy) so a connection is refreshed before the server reaps it.
* ``async def get_db()`` — a FastAPI dependency that yields an
  ``AsyncSession``, commits on success, rolls back on exception, and
  always closes. This is the single transaction boundary for a request.
* ``init_db()`` — **dev-only** table creation via
  ``run_sync(Base.metadata.create_all)``. This is fine for local
  development and CI, but **Alembic migrations should replace this before
  production** because ``create_all`` does not handle schema evolution
  (adding/dropping columns on existing tables).
"""

from __future__ import annotations

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .config import settings

# ---------------------------------------------------------------------------
# Async engine
# ---------------------------------------------------------------------------

# ``create_async_engine`` returns an ``AsyncEngine`` whose ``connect()`` and
# ``begin()`` methods return awaitables. The underlying DBAPI driver is
# ``asyncpg`` (specified in the DATABASE_URL scheme: ``postgresql+asyncpg``).
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.SQL_ECHO,
    # Pre-ping avoids handing out connections that the managed-Postgres
    # server has already closed during idle time.
    pool_pre_ping=True,
    # Recycle connections every 30 minutes. This must be shorter than the
    # managed-Postgres idle-connection timeout (varies by provider; 300 s
    # for Supabase, 5 min for RDS proxy). 1800 s is a safe default that
    # balances connection churn against stale-socket risk.
    pool_recycle=1800,
    # ``pool_size`` and ``max_overflow`` default to 5 and 10 respectively.
    # For a single-process FastAPI deployment this is usually sufficient;
    # tune up if you see ``TimeoutError`` under load.
)

# ``async_sessionmaker`` is the async equivalent of ``sessionmaker``. It
# produces ``AsyncSession`` objects. ``expire_on_commit=False`` is important:
# by default SQLAlchemy expires all ORM objects after commit, so the next
# attribute access triggers a lazy refresh query. In an async context that
# lazy refresh raises ``MissingGreenlet`` because it runs outside the
# session's event loop. Disabling it keeps objects usable post-commit.
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async DB session with automatic commit/rollback/close.

    Usage in a route::

        @router.post("/foo")
        async def foo(db: AsyncSession = Depends(get_db)):
            ...

    The session is committed if the route body completes without raising.
    On any exception the transaction is rolled back. The session is always
    closed in ``finally`` to return the connection to the pool.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Dev-only schema initialization
# ---------------------------------------------------------------------------

async def init_db() -> None:
    """Create all tables if they do not exist (dev/CI only).

    .. warning::
        This uses ``Base.metadata.create_all`` which only creates tables
        that are **missing** — it does not alter existing tables. Before
        production, replace this with **Alembic** migrations so that schema
        changes (new columns, index additions, type changes) are applied
        incrementally and reversibly.

    This is called from the FastAPI ``lifespan`` startup hook in
    ``app/main.py``.
    """
    # Import here to avoid a circular import at module load time:
    # ``models.py`` imports ``Base`` from this module, and we need to
    # import ``models`` so that all ORM classes are registered with the
    # ``Base.metadata`` before ``create_all`` runs.
    from .models import Base  # noqa: WPS433 - deferred import is intentional

    async with engine.begin() as conn:
        # ``run_sync`` bridges the async engine to the sync DDL executor.
        # DDL (CREATE TABLE) is not truly async in asyncpg; this is the
        # correct way to run it.
        await conn.run_sync(Base.metadata.create_all)