"""Focused unit tests for recruitment workflow boundaries.

External services are mocked: tests never call Hugging Face, PostgreSQL, or a
real Chroma server. PostgreSQL integration is covered by deployment smoke tests
against the configured database separately.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi import HTTPException

from app.models import InterviewQuestion, InterviewSession, JobRequirement, QuestionCategory
from app.routers import recruitment
from app.schemas import InterviewChatRequest, JobGenerateRequest
from app.services import hf_service, ranking_service
from app.services.vector_service import VectorService


class _ScalarResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _QuestionDB:
    def __init__(self, job):
        self.job = job
        self.added = []

    async def get(self, model, identifier):
        return self.job if model is JobRequirement and identifier == self.job.id else None

    def add_all(self, rows):
        self.added.extend(rows)

    async def flush(self):
        for row in self.added:
            if row.id is None:
                row.id = uuid.uuid4()


class _InterviewDB:
    def __init__(self, session, job, questions):
        self.session = session
        self.job = job
        self.questions = questions

    async def get(self, model, identifier):
        if model is InterviewSession:
            return self.session if identifier == self.session.id else None
        if model is JobRequirement:
            return self.job if identifier == self.job.id else None
        return None

    async def execute(self, statement):
        return _ScalarResult(self.questions)

    async def flush(self):
        return None


def test_question_service_rejects_invalid_question_count(monkeypatch):
    monkeypatch.setattr(
        hf_service,
        "_call_hf_router",
        lambda *args: '{"questions": [{"question_text": "One?", "category": "technical"}]}',
    )

    with pytest.raises(hf_service.HFServiceError, match="between 5 and 10"):
        hf_service.generate_interview_questions("Engineer", "Build APIs", ["Python"])


def test_interview_turn_appends_exact_stored_question(monkeypatch):
    monkeypatch.setattr(hf_service, "_call_hf_router", lambda *args: "Good explanation. What is caching?")

    result = hf_service.run_interview_turn(
        [{"role": "user", "content": "I use indexes."}],
        "Describe how you would design a database index.",
    )

    assert result.endswith("Describe how you would design a database index.")
    assert result.count("Describe how you would design a database index.") == 1


@pytest.mark.asyncio
async def test_question_creation_flow_persists_valid_questions(monkeypatch):
    job = JobRequirement(
        id=uuid.uuid4(), title="Backend Engineer", description="Python APIs", keywords=["Python"]
    )
    db = _QuestionDB(job)
    questions = [
        {"question_text": f"Question {number}?", "category": "technical"}
        for number in range(1, 6)
    ]
    monkeypatch.setattr(recruitment.hf_service, "generate_interview_questions", lambda **kwargs: questions)

    response = await recruitment.generate_questions(job.id, db)

    assert response.total == 5
    assert len(db.added) == 5
    assert all(question.category is QuestionCategory.technical for question in db.added)


@pytest.mark.asyncio
async def test_job_generation_translates_hf_errors_to_http_status(monkeypatch):
    request = JobGenerateRequest(raw_input="Need a Python backend engineer")

    monkeypatch.setattr(
        recruitment.hf_service,
        "generate_job_description",
        lambda raw_input: (_ for _ in ()).throw(hf_service.HFServiceError("upstream unavailable")),
    )
    with pytest.raises(HTTPException) as failure:
        await recruitment.generate_job(request, object())
    assert failure.value.status_code == 502

    monkeypatch.setattr(
        recruitment.hf_service,
        "generate_job_description",
        lambda raw_input: (_ for _ in ()).throw(hf_service.HFServiceError("rate limited")),
    )
    with pytest.raises(HTTPException) as rate_limited:
        await recruitment.generate_job(request, object())
    assert rate_limited.value.status_code == 502


@pytest.mark.asyncio
async def test_final_answer_is_scored_before_interview_completion(monkeypatch):
    candidate_id, job_id, session_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    final_question = "How would you monitor a production API?"
    job = JobRequirement(id=job_id, title="SRE", description="Operate APIs", keywords=["Python"])
    question = InterviewQuestion(
        id=uuid.uuid4(), job_id=job_id, question_text=final_question, category=QuestionCategory.technical
    )
    session = InterviewSession(
        id=session_id,
        candidate_id=candidate_id,
        job_id=job_id,
        chat_history=[{"role": "assistant", "content": final_question}],
        is_complete=False,
    )
    db = _InterviewDB(session, job, [question])
    monkeypatch.setattr(recruitment.hf_service, "score_interview_response", lambda **kwargs: {"score": 8.0, "feedback": "Strong answer."})
    monkeypatch.setattr(recruitment.hf_service, "run_interview_turn", lambda **kwargs: "Thank you. The interview is complete.")

    response = await recruitment.interview_chat(
        InterviewChatRequest(session_id=session_id, candidate_id=candidate_id, job_id=job_id, candidate_message="I use metrics and alerts."),
        db,
    )

    assert response.is_complete is True
    assert response.final_score == 8.0
    assert session.chat_history[-2]["score"] == 8.0


def test_vector_similarity_uses_metadata_filter_and_ranking_is_explainable(monkeypatch):
    class Collection:
        def __init__(self):
            self.where = None

        def query(self, **kwargs):
            self.where = kwargs["where"]
            return {"ids": [["job-1"]], "distances": [[0.2]]}

    collection = Collection()
    service = VectorService(persist_path="unused")
    service._collection = collection
    assert service.get_job_similarity("job-1", "Python FastAPI") == pytest.approx(0.8)
    assert collection.where == {"job_id": "job-1"}

    monkeypatch.setattr(ranking_service.vector_service, "get_job_similarity", lambda job_id, text: 0.5)
    ranked = ranking_service.rank_candidates(
        "job-1",
        "Python FastAPI PostgreSQL",
        ["Python", "FastAPI"],
        [("candidate-a", "Python FastAPI PostgreSQL"), ("candidate-b", "Java")],
    )
    assert ranked[0]["candidate_id"] == "candidate-a"
    assert set(ranked[0]) == {"candidate_id", "tfidf_score", "keyword_score", "vector_score", "final_score"}
