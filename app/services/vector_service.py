"""ChromaDB persistent vector store service.

Wraps a ChromaDB ``PersistentClient`` with two operations:
* ``index_job`` — upsert a job document (title + description + keywords).
* ``get_job_similarity`` — query the collection restricted to a specific
  job id, returning cosine distance converted to a 0–1 similarity score.

Embeddings
----------
We use ChromaDB's **local default embedding function** (``all-MiniLM-L6-v2``
via ``sentence-transformers``) rather than a hosted HF embeddings endpoint.
HF's free router does not currently expose a stable embeddings API the way
it does chat completions. The local model runs on-CPU, adds no network
dependency, and is sufficient for the 15% vector-similarity signal.

Cosine distance → similarity
----------------------------
ChromaDB's ``hnsw:space: cosine`` metadata makes the ``distances`` field
a cosine *distance* (0 = identical, 2 = opposite). We convert to
similarity with ``sim = 1 - distance``, clamped to [0, 1].
"""

from __future__ import annotations

from typing import Any, Optional

import chromadb
from chromadb.config import Settings

from ..config import settings


# ---------------------------------------------------------------------------
# Cosine HNSW metadata
# ---------------------------------------------------------------------------

# This is the critical config that makes Chroma use cosine distance for
# similarity queries. Without it, the default is L2 (Euclidean), which
# produces incorrect similarity rankings.
_COSINE_METADATA = {"hnsw:space": "cosine"}


def _coerce_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Ensure every metadata value is a Chroma-accepted scalar type.

    Chroma rejects ``list``/``dict``/``None`` values. We:
      * join lists/tuples into comma-separated strings,
      * drop keys whose value is None,
      * stringify anything else as a safety net.
    """
    clean: dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            clean[key] = ", ".join(str(v) for v in value)
        elif isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


# ---------------------------------------------------------------------------
# VectorDB wrapper
# ---------------------------------------------------------------------------

class VectorService:
    """Lazy-initialized wrapper around a Chroma PersistentClient.

    The client and collection are created on first use (not at import time)
    so that importing this module never triggers disk I/O or model loading.
    """

    def __init__(self, persist_path: str = settings.CHROMA_PERSIST_DIR) -> None:
        self.persist_path = persist_path
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[Any] = None  # chromadb.Collection

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------

    def _ensure_client(self) -> chromadb.PersistentClient:
        """Create the PersistentClient once and cache it."""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=self.persist_path,
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    @property
    def collection(self) -> Any:
        """Get-or-create the jobs collection with cosine HNSW space."""
        if self._collection is None:
            self._collection = self._ensure_client().get_or_create_collection(
                name=settings.CHROMA_COLLECTION_JOBS,
                metadata=_COSINE_METADATA,
            )
        return self._collection

    # ------------------------------------------------------------------
    # Job indexing
    # ------------------------------------------------------------------

    def index_job(
        self,
        job_id: str,
        title: str,
        description: str,
        keywords: list[str],
    ) -> str:
        """Upsert a job document into the Chroma collection.

        The document text is the concatenation of title + description +
        keywords so semantic search has the full context. We use the
        job's UUID (as a string) as the Chroma id to keep Postgres and
        Chroma linked.

        Returns the Chroma document id (same as ``job_id``).
        """
        doc_text = f"{title}\n{description}\n{', '.join(keywords)}"
        metadata = _coerce_metadata(
            {
                "job_id": job_id,
                "title": title,
                "keywords": keywords,
            }
        )
        self.collection.upsert(
            ids=[job_id],
            documents=[doc_text],
            metadatas=[metadata],
        )
        return job_id

    # ------------------------------------------------------------------
    # Similarity query
    # ------------------------------------------------------------------

    def get_job_similarity(self, job_id: str, text: str) -> float:
        """Query the collection for similarity between ``text`` and the job.

        Uses ``ids=[job_id]`` in ``collection.query()`` to restrict the
        search to the specific job document. Returns a 0–1 similarity
        score derived from Chroma's cosine distance:

            similarity = max(0.0, min(1.0, 1.0 - distance))

        Returns 0.0 if the job is not found in the collection.
        """
        result = self.collection.query(
            query_texts=[text],
            ids=[job_id],
            n_results=1,
            include=["distances"],
        )

        # Chroma returns parallel lists grouped by query. We issued one
        # query, so index 0 holds our results.
        ids_batch = (result.get("ids") or [[]])[0]
        dist_batch = (result.get("distances") or [[]])[0]

        if not ids_batch or not dist_batch:
            # Job not found in Chroma — return 0 similarity.
            return 0.0

        distance = float(dist_batch[0])
        # Cosine distance (0=identical, 2=opposite) → similarity [0, 1].
        sim = 1.0 - distance
        return max(0.0, min(1.0, sim))


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

vector_service = VectorService()