"""Comprehensive end-to-end endpoint tests for AI Recruitment & Hiring Suite."""
import uuid
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from app.main import app
from app.models import JobRequirement
from app.routers import recruitment
from app.services import hf_service

from datetime import datetime, timezone

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code in {200, 404}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "AI Recruitment" in data["service"]


def test_chatbot_chat_endpoint(monkeypatch):
    monkeypatch.setattr(
        "app.routers.chatbot.hf_service.chat_bot",
        lambda message, chat_history=None: "Hello! How can I assist you with recruiting today?",
    )
    response = client.post(
        "/api/v1/chatbot/chat",
        json={"message": "Hi, what can you do?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert "assist" in data["reply"]


@pytest.mark.asyncio
async def test_job_generation_endpoint(monkeypatch):
    monkeypatch.setattr(
        recruitment.hf_service,
        "generate_job_description",
        lambda prompt: {
            "title": "Senior Python Engineer",
            "description": "Develop high-scale async Python microservices using FastAPI and PostgreSQL.",
            "keywords": ["Python", "FastAPI", "PostgreSQL", "AsyncIO"],
        },
    )
    monkeypatch.setattr(
        recruitment.vector_service,
        "index_job",
        lambda job_id, title, description, keywords: "doc-123",
    )

    fake_id = uuid.uuid4()

    class FakeJob:
        id = fake_id
        title = "Senior Python Engineer"
        description = "Develop high-scale async Python microservices using FastAPI and PostgreSQL."
        keywords = ["Python", "FastAPI", "PostgreSQL", "AsyncIO"]
        chroma_doc_id = "doc-123"
        created_at = datetime.now(timezone.utc)

    class FakeDB:
        def __init__(self):
            self.items = []

        def add(self, item):
            self.items.append(item)

        async def flush(self):
            for item in self.items:
                if getattr(item, "id", None) is None:
                    item.id = uuid.uuid4()
                if getattr(item, "created_at", None) is None:
                    item.created_at = datetime.now(timezone.utc)

    async def mock_get_db():
        yield FakeDB()

    app.dependency_overrides[recruitment.get_db] = mock_get_db

    try:
        response = client.post(
            "/api/v1/jobs/generate",
            json={"raw_input": "Need a Senior Python developer with FastAPI experience."},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Senior Python Engineer"
        assert "FastAPI" in data["keywords"]
    finally:
        app.dependency_overrides.clear()
