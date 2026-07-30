"""ChromaDB PersistentClient wrapper.

Provides a thin, typed wrapper around a persistent ChromaDB instance with two
collections:

  * `company_jobs`      - one document per posted job (job text).
  * `student_resumes`   - one document per uploaded resume (resume text).

Important gotchas handled here
------------------------------
1. **Cosine space**: collections MUST be created with
   ``metadata={"hnsw:space": "cosine"}`` or the default (L2) distance is used
   and similarity math will be wrong. We pass this metadata on every
   `get_or_create_collection` call.
2. **Metadata value types**: Chroma only accepts `str`, `int`, `float` or
   `bool` as metadata values. Any list (skills, organizations, ...) must be
   joined into a comma-separated string before storage. The helper
   `_coerce_metadata` enforces this defensively.
3. **Partial success**: Indexing failures are raised to the caller so the
   FastAPI layer can decide whether to roll back. Per the spec, job indexing
   failure after a successful SQLite write should NOT roll back SQLite; the
   route returns a 207 instead.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

# ---------------------------------------------------------------------------
# Persistent client setup
# ---------------------------------------------------------------------------

# Chroma persists its data to disk under this directory (next to the project).
_DB_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_DB_DIR)
CHROMA_PATH = os.path.join(_PROJECT_DIR, "chroma_data")

# Collection names - exposed as constants so other modules don't hardcode
# strings.
JOBS_COLLECTION = "company_jobs"
RESUMES_COLLECTION = "student_resumes"

# HNSW cosine metadata. This is the critical config that makes Chroma use
# cosine distance for similarity queries.
_COSINE_METADATA = {"hnsw:space": "cosine"}


def _coerce_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure every metadata value is a Chroma-accepted scalar type.

    Chroma rejects `list`/`dict`/`None` values. We:
      * join lists/tuples into comma-separated strings,
      * drop keys whose value is None,
      * stringify anything else as a safety net.

    This is called before every `add`/`update` so callers can pass rich
    Python objects without worrying about Chroma's type constraints.
    """
    clean: Dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            # Join list members into a single CSV string. Cast each member to
            # str so non-string lists (e.g. ints) don't blow up.
            clean[key] = ", ".join(str(v) for v in value)
        elif isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            # Fallback: stringify unknown types (e.g. datetime).
            clean[key] = str(value)
    return clean


class VectorDB:
    """Lazy-initialized wrapper around a Chroma PersistentClient.

    The client and collections are created on first use (not at import time)
    so that importing this module never triggers disk I/O or heavy model
    loading. This keeps test/import paths fast and side-effect free.
    """

    def __init__(self, persist_path: str = CHROMA_PATH) -> None:
        self.persist_path = persist_path
        self._client: Optional[chromadb.PersistentClient] = None
        self._jobs_collection = None
        self._resumes_collection = None

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------

    def _ensure_client(self) -> chromadb.PersistentClient:
        """Create the PersistentClient once and cache it."""
        if self._client is None:
            # `Settings(anonymized_telemetry=False)` disables Chroma's
            # anonymous usage telemetry for privacy in local dev.
            self._client = chromadb.PersistentClient(
                path=self.persist_path,
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    def _ensure_collection(self, name: str):
        """Get-or-create a collection with cosine HNSW space."""
        return self._ensure_client().get_or_create_collection(
            name=name,
            metadata=_COSINE_METADATA,
        )

    @property
    def jobs_collection(self):
        if self._jobs_collection is None:
            self._jobs_collection = self._ensure_collection(JOBS_COLLECTION)
        return self._jobs_collection

    @property
    def resumes_collection(self):
        if self._resumes_collection is None:
            self._resumes_collection = self._ensure_collection(RESUMES_COLLECTION)
        return self._resumes_collection

    # ------------------------------------------------------------------
    # Job indexing
    # ------------------------------------------------------------------

    def index_job(
        self,
        job_id: int,
        title: str,
        description: str,
        required_keywords: str,
    ) -> None:
        """Upsert a job document into the `company_jobs` collection.

        The document text is the concatenation of title + description +
        keywords so semantic search has the full context. We use the SQLite
        primary key (as a string) as the Chroma id to keep the two stores
        linked.

        Raises ChromaDB errors on failure; the caller decides how to handle
        (e.g. return a 207 partial-success response).
        """
        doc_text = f"{title}\n{description}\n{required_keywords}"
        metadata = _coerce_metadata(
            {
                "job_id": job_id,
                "title": title,
                "required_keywords": required_keywords,
            }
        )
        self.jobs_collection.upsert(
            ids=[str(job_id)],
            documents=[doc_text],
            metadatas=[metadata],
        )

    # ------------------------------------------------------------------
    # Resume indexing
    # ------------------------------------------------------------------

    def index_resume(
        self,
        resume_id: str,
        text: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Upsert a resume document into the `student_resumes` collection.

        `metadata` may contain lists (skills, organizations, locations,
        degrees); `_coerce_metadata` joins them into CSV strings before
        storage. Raises on failure.
        """
        clean_meta = _coerce_metadata(metadata)
        self.resumes_collection.upsert(
            ids=[resume_id],
            documents=[text],
            metadatas=[clean_meta],
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_all_resumes(self) -> List[Dict[str, Any]]:
        """Return every resume document with its metadata.

        Each item is a dict with keys: id, document, metadata. Returns an
        empty list if the collection is empty.
        """
        # `get()` with no ids/filters returns all rows. We request embeddings
        # explicitly False because we don't need the raw vectors here.
        result = self.resumes_collection.get(include=["documents", "metadatas"])
        items: List[Dict[str, Any]] = []
        ids = result.get("ids", []) or []
        docs = result.get("documents", []) or []
        metas = result.get("metadatas", []) or []
        for rid, doc, meta in zip(ids, docs, metas):
            items.append({"id": rid, "document": doc, "metadata": meta or {}})
        return items

    def query_resumes_by_job_text(
        self,
        job_text: str,
        n_results: Optional[int] = None,
    ) -> Dict[str, float]:
        """Query ALL resumes against the job text in a SINGLE Chroma call.

        Returns a mapping of {resume_id: similarity_score} where the score is
        a 0-100 cosine similarity derived from Chroma's cosine distance:

            similarity = (1 - distance) * 100

        Because the collection uses `hnsw:space: cosine`, the `distances`
        field returned by Chroma is the cosine *distance* (0 = identical,
        2 = opposite). Converting to similarity with `(1 - d)` maps 0->1 and
        2->-1; we clamp to [0, 100] after scaling.

        If `n_results` is None we request as many results as possible. Chroma
        doesn't expose a true "all" flag reliably across versions, so we pass
        a large number; the caller can also pass an explicit count.
        """
        # Request a generous number of results so we effectively get all
        # resumes in one round-trip. 10_000 is well above any realistic
        # resume count for a single deployment.
        if n_results is None:
            n_results = 10_000

        result = self.resumes_collection.query(
            query_texts=[job_text],
            n_results=n_results,
            include=["distances"],
        )

        # Chroma returns parallel lists grouped by query. We issued one
        # query, so index 0 holds our results.
        ids_batch = (result.get("ids") or [[]])[0]
        dist_batch = (result.get("distances") or [[]])[0]

        scores: Dict[str, float] = {}
        for rid, dist in zip(ids_batch, dist_batch):
            # Cosine distance -> similarity in [0,1] (clamped), then *100.
            sim = 1.0 - float(dist)
            sim = max(0.0, min(1.0, sim))  # clamp to [0,1]
            scores[rid] = round(sim * 100.0, 4)
        return scores

    def get_job_text(self, job_id: int) -> Optional[str]:
        """Fetch the indexed job document text for a given job id."""
        result = self.jobs_collection.get(
            ids=[str(job_id)], include=["documents"]
        )
        docs = result.get("documents") or []
        return docs[0] if docs else None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

# A single shared instance is fine for a single-process deployment. Chroma's
# PersistentClient is thread-safe for our usage pattern.
vector_db = VectorDB()