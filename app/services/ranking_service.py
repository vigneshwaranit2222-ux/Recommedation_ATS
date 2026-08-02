"""Hybrid resume ranking service.

Combines three signals into a single 0–100 score per candidate:

  1. TF-IDF cosine similarity (50%) — lexical overlap, batch-fitted.
  2. Keyword match percentage (35%) — word-boundary regex.
  3. ChromaDB vector similarity (15%) — semantic similarity.

Key design decisions
--------------------
* **Batch TF-IDF** — ONE ``TfidfVectorizer`` is fit over
  ``[job_description, *all_resumes]`` in a single call, then pairwise
  cosine similarity is computed. Fitting per-candidate in a loop would
  destroy score comparability (each candidate gets a different vocabulary)
  and waste compute (re-fitting the vectorizer N times).
* **Word-boundary regex** — ``\\b<keyword>\\b`` (case-insensitive) prevents
  "Java" from matching inside "JavaScript". Plain substring matching would
  inflate keyword scores for candidates whose resumes contain
  superstring matches.
* **Explainable scores** — all four numbers (tfidf, keyword, vector,
  final) are returned per candidate, never collapsed to a single opaque
  score. Recruiters need to see *why* a candidate ranked where they did.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .vector_service import vector_service

# ---------------------------------------------------------------------------
# Scoring weights (must sum to 1.0)
# ---------------------------------------------------------------------------

TFIDF_WEIGHT: float = 0.50
KEYWORD_WEIGHT: float = 0.35
VECTOR_WEIGHT: float = 0.15


# ---------------------------------------------------------------------------
# TF-IDF batch scoring
# ---------------------------------------------------------------------------

def tfidf_cosine_scores_batch(
    job_description: str,
    resume_texts: List[str],
) -> List[float]:
    """Compute TF-IDF cosine similarity of the job against each resume.

    Fits ONE ``TfidfVectorizer`` over ``[job_description, *resume_texts]``
    so all documents share the same vocabulary. This is critical: fitting
    per-candidate would give each candidate a different vocabulary, making
    scores non-comparable.

    Returns a list of similarity scores in [0, 1], aligned with
    ``resume_texts``.
    """
    if not resume_texts:
        return []

    # Combine job + resumes into one corpus for a shared vocabulary.
    corpus = [job_description] + resume_texts

    # ``sublinear_tf=True`` dampens frequent terms (standard for doc
    # similarity). ``ngram_range=(1, 2)`` captures multiword skills like
    # "Machine Learning". ``stop_words="english"`` removes noise words.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        sublinear_tf=True,
        ngram_range=(1, 2),
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        # ``fit_transform`` raises if the corpus is empty after stop-word
        # removal. Fall back to all zeros.
        return [0.0] * len(resume_texts)

    # Row 0 = job; rows 1..n = resumes. Take pairwise cosine similarity.
    sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    # Clamp to [0, 1] to guard against floating-point overshoot.
    return [max(0.0, min(1.0, float(s))) for s in sims]


# ---------------------------------------------------------------------------
# Keyword matching
# ---------------------------------------------------------------------------

def keyword_match_score(resume_text: str, keywords: List[str]) -> float:
    """Compute keyword match percentage using word-boundary regex.

    Uses ``\\b<keyword>\\b`` (case-insensitive) so "Java" doesn't match
    inside "JavaScript". Returns the fraction of keywords found, as a
    0–100 percentage. If no keywords are required, returns 100.0
    (vacuously true).
    """
    if not keywords:
        return 100.0

    matched = 0
    for kw in keywords:
        # Escape special regex chars (e.g. C++, C#) and wrap in \b.
        pattern = rf"\b{re.escape(kw)}\b"
        if re.search(pattern, resume_text, re.IGNORECASE):
            matched += 1

    return (matched / len(keywords)) * 100.0


# ---------------------------------------------------------------------------
# Combined ranking
# ---------------------------------------------------------------------------

def rank_candidates(
    job_id: str,
    job_description: str,
    keywords: List[str],
    candidates: List[Tuple[str, str]],
) -> List[dict]:
    """Rank candidates against a job using the hybrid scoring strategy.

    Parameters
    ----------
    job_id:
        The job's UUID as a string (for ChromaDB id-restricted query).
    job_description:
        The job description text.
    keywords:
        The job's required keywords (list of strings).
    candidates:
        List of ``(candidate_id_str, resume_text)`` tuples.

    Returns
    -------
    list of dicts, each with keys:
        - ``candidate_id`` (str)
        - ``tfidf_score`` (float, 0–100)
        - ``keyword_score`` (float, 0–100)
        - ``vector_score`` (float, 0–100)
        - ``final_score`` (float, 0–100)
    Sorted descending by ``final_score``.
    """
    if not candidates:
        return []

    resume_texts = [text for _, text in candidates]

    # --- 1. TF-IDF (batch) --------------------------------------------
    tfidf_sims = tfidf_cosine_scores_batch(job_description, resume_texts)

    # --- 2. Keyword match (per candidate) ------------------------------
    keyword_scores = [
        keyword_match_score(text, keywords) for _, text in candidates
    ]

    # --- 3. Vector similarity (per candidate, id-restricted query) -----
    # Each candidate's resume text is queried against the job's Chroma
    # document, restricted to the job's id.
    vector_scores = [
        vector_service.get_job_similarity(job_id, text) * 100.0
        for _, text in candidates
    ]

    # --- 4. Weighted combination ---------------------------------------
    results: List[dict] = []
    for idx, (candidate_id, _) in enumerate(candidates):
        tfidf_score = tfidf_sims[idx] * 100.0
        keyword_score = keyword_scores[idx]
        vector_score = vector_scores[idx]

        final_score = (
            TFIDF_WEIGHT * tfidf_score
            + KEYWORD_WEIGHT * keyword_score
            + VECTOR_WEIGHT * vector_score
        )

        results.append(
            {
                "candidate_id": candidate_id,
                "tfidf_score": round(tfidf_score, 2),
                "keyword_score": round(keyword_score, 2),
                "vector_score": round(vector_score, 2),
                "final_score": round(final_score, 2),
            }
        )

    # --- 5. Sort descending by final score -----------------------------
    results.sort(key=lambda r: r["final_score"], reverse=True)
    return results