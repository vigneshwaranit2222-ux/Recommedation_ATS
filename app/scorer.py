"""Hybrid resume scoring engine.

Combines three signals into a single 0-100 score for each resume against a
given job:

  1. TF-IDF cosine similarity (50%)  - lexical overlap of job vs resume text.
  2. Keyword match percentage (35%)  - fraction of required keywords found
     in the resume text OR in the extracted skills list, via case-insensitive
     word-boundary regex.
  3. ChromaDB vector similarity (15%) - semantic similarity from Chroma's
     cosine distance, retrieved in a SINGLE query for all resumes.

The module exposes a `ScoreBreakdown` Pydantic model and a `rank_resumes`
function that returns ranked candidates with full score breakdowns and
matched/missing keyword lists.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .vector_db import vector_db

# ---------------------------------------------------------------------------
# Scoring weights (must sum to 1.0)
# ---------------------------------------------------------------------------

WEIGHT_TFIDF: float = 0.50
WEIGHT_KEYWORD: float = 0.35
WEIGHT_VECTOR: float = 0.15

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class ScoreBreakdown(BaseModel):
    """Detailed per-candidate score breakdown returned by the ranker."""

    tfidf_score: float = Field(..., description="TF-IDF cosine similarity * 100 (0-100)")
    keyword_score: float = Field(..., description="Keyword match percentage (0-100)")
    vector_score: float = Field(..., description="Chroma cosine similarity (0-100)")
    weighted_total: float = Field(..., description="Final weighted score (0-100)")
    matched_keywords: List[str] = Field(
        default_factory=list, description="Required keywords found in the resume"
    )
    missing_keywords: List[str] = Field(
        default_factory=list, description="Required keywords NOT found in the resume"
    )


class CandidateResult(BaseModel):
    """A single ranked candidate in the leaderboard response."""

    resume_id: str
    student_id: str
    student_name: str
    skills: List[str] = Field(default_factory=list)
    organizations: List[str] = Field(default_factory=list)
    degrees: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    score_breakdown: ScoreBreakdown


# ---------------------------------------------------------------------------
# Keyword matching
# ---------------------------------------------------------------------------


def _parse_required_keywords(raw: str) -> List[str]:
    """Split a comma-separated keyword string into a clean list.

    Empty entries and surrounding whitespace are removed. Returns [] for
    empty/None input.
    """
    if not raw:
        return []
    return [kw.strip() for kw in raw.split(",") if kw.strip()]


def _keyword_match_info(
    resume_text: str,
    skills_csv: str,
    required_keywords: List[str],
) -> Tuple[float, List[str], List[str]]:
    """Compute keyword match percentage and matched/missing lists.

    A keyword is considered "found" if it appears in the resume text OR in the
    extracted skills (stored as a CSV string in Chroma metadata) using a
    case-insensitive word-boundary regex. Word boundaries prevent partial
    matches like "Java" inside "JavaScript".

    Returns (match_percentage_0_100, matched, missing).
    """
    if not required_keywords:
        # No requirements -> 100% match by convention (vacuously true).
        return 100.0, [], []

    # Combine resume text + skills into one haystack for matching. We join
    # with a separator so a keyword can't accidentally span the boundary.
    haystack = f"{resume_text}\n{skills_csv}"

    matched: List[str] = []
    missing: List[str] = []
    for kw in required_keywords:
        # Escape the keyword for regex special chars (e.g. C++, C#) and wrap
        # in word boundaries. `\b` doesn't work well with non-word chars, so
        # we use a lookaround that treats the keyword as a standalone token.
        escaped = re.escape(kw)
        # Use lookarounds instead of \b so symbols like + and # are handled.
        # (?<![A-Za-z0-9]) ensures no alphanumeric immediately before, and
        # (?![A-Za-z0-9]) ensures none immediately after. This prevents
        # "Java" matching inside "JavaScript" while still matching "C++".
        pattern = rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"
        if re.search(pattern, haystack, re.IGNORECASE):
            matched.append(kw)
        else:
            missing.append(kw)

    match_pct = (len(matched) / len(required_keywords)) * 100.0
    return match_pct, matched, missing


# ---------------------------------------------------------------------------
# TF-IDF scoring
# ---------------------------------------------------------------------------


def _tfidf_cosine(job_text: str, resume_texts: List[str]) -> List[float]:
    """Compute TF-IDF cosine similarity of the job against each resume.

    Returns a list of similarity scores in [0, 1], aligned with
    `resume_texts`. We fit the vectorizer on the job + all resumes together
    so the vocabulary is shared.
    """
    if not resume_texts:
        return []

    # Combine job + resumes into one corpus for a shared vocabulary.
    corpus = [job_text] + resume_texts
    # `sublinear_tf=True` dampens the effect of very frequent terms, which
    # is standard for document similarity.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        sublinear_tf=True,
        ngram_range=(1, 2),  # unigrams + bigrams capture multiword skills
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        # `fit_transform` raises ValueError if the corpus is empty after
        # stop-word removal (e.g. all stopwords). Fall back to all zeros.
        return [0.0] * len(resume_texts)

    # Row 0 is the job; rows 1..n are resumes. cosine_similarity returns a
    # matrix; we take the first row (job vs each resume) and drop the
    # self-similarity at index 0.
    sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    # Clamp to [0, 1] to guard against tiny floating-point overshoot.
    return [max(0.0, min(1.0, float(s))) for s in sims]


# ---------------------------------------------------------------------------
# Main ranking function
# ---------------------------------------------------------------------------


def rank_resumes(
    job_id: int,
    job_title: str,
    job_description: str,
    required_keywords_raw: str,
) -> List[CandidateResult]:
    """Rank all stored resumes against the given job.

    Steps:
      1. Fetch all resumes from Chroma (`get_all_resumes`).
      2. Query Chroma ONCE for vector similarity of all resumes vs job text.
      3. Compute TF-IDF cosine similarity for job vs each resume.
      4. Compute keyword match % for each resume.
      5. Combine with weights 50/35/15 into a final 0-100 score.
      6. Sort candidates highest-to-lowest by weighted total.

    Returns a list of `CandidateResult` sorted descending by score. Returns
    an empty list if no resumes are stored.
    """
    # --- 1. Fetch all resumes ----------------------------------------------
    resumes = vector_db.get_all_resumes()
    if not resumes:
        return []

    required_keywords = _parse_required_keywords(required_keywords_raw)
    job_text = f"{job_title}\n{job_description}\n{required_keywords_raw}"

    # --- 2. Vector similarity (single Chroma query) -----------------------
    # This is the critical efficiency requirement: ONE query for all resumes,
    # not one per resume.
    vector_scores = vector_db.query_resumes_by_job_text(job_text)

    # --- 3. TF-IDF similarity ---------------------------------------------
    resume_texts = [r["document"] or "" for r in resumes]
    tfidf_sims = _tfidf_cosine(job_text, resume_texts)

    # --- 4 & 5. Keyword match + weighted combination ----------------------
    results: List[CandidateResult] = []
    for idx, r in enumerate(resumes):
        meta = r["metadata"] or {}
        resume_text = r["document"] or ""
        skills_csv = str(meta.get("skills", ""))

        # Keyword match
        kw_pct, matched, missing = _keyword_match_info(
            resume_text, skills_csv, required_keywords
        )

        # Individual 0-100 scores
        tfidf_score = tfidf_sims[idx] * 100.0
        keyword_score = kw_pct
        vector_score = vector_scores.get(r["id"], 0.0)

        # Weighted total
        weighted_total = (
            WEIGHT_TFIDF * tfidf_score
            + WEIGHT_KEYWORD * keyword_score
            + WEIGHT_VECTOR * vector_score
        )

        # Reconstruct list-typed metadata from CSV strings stored in Chroma.
        def _split_csv(val: str) -> List[str]:
            if not val:
                return []
            return [v.strip() for v in val.split(",") if v.strip()]

        breakdown = ScoreBreakdown(
            tfidf_score=round(tfidf_score, 2),
            keyword_score=round(keyword_score, 2),
            vector_score=round(vector_score, 2),
            weighted_total=round(weighted_total, 2),
            matched_keywords=matched,
            missing_keywords=missing,
        )

        results.append(
            CandidateResult(
                resume_id=str(r["id"]),
                student_id=str(meta.get("student_id", "")),
                student_name=str(meta.get("student_name", "")),
                skills=_split_csv(skills_csv),
                organizations=_split_csv(str(meta.get("organizations", ""))),
                degrees=_split_csv(str(meta.get("degrees", ""))),
                locations=_split_csv(str(meta.get("locations", ""))),
                score_breakdown=breakdown,
            )
        )

    # --- 6. Sort descending by weighted total -----------------------------
    results.sort(key=lambda c: c.score_breakdown.weighted_total, reverse=True)
    return results