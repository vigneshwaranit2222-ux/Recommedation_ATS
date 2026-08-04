# AI Recruitment & Hiring Suite

Production-oriented FastAPI backend for LLM-assisted job creation, interview
question generation, conversational interviews, and explainable resume
ranking.

## Architecture

- Python 3.11+, FastAPI, and Uvicorn
- PostgreSQL with SQLAlchemy 2.0 async ORM and `asyncpg`
- ChromaDB local persistent embeddings (`all-MiniLM-L6-v2` default embedding)
- Hugging Face OpenAI-compatible router for chat completion calls
- TF-IDF, exact keyword matching, and vector similarity for rankings

The API intentionally does not include JWT authentication, Alembic migrations,
or PDF-upload resume parsing yet. Configure them before exposing the service to
untrusted users.

## Configuration

Copy `.env.example` to `.env` and set the required values:

```powershell
Copy-Item .env.example .env
```

`DATABASE_URL` must use the asyncpg form:

```text
postgresql+asyncpg://USER:PASSWORD@HOST:5432/DATABASE
```

Set `HF_API_TOKEN` to a Hugging Face token and verify that `HF_CHAT_MODEL` is
currently available through the Hugging Face inference router before deployment.
Never commit `.env`.

## Local run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://127.0.0.1:8000/docs` for OpenAPI documentation and
`http://127.0.0.1:8000/health` for the liveness endpoint.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| POST | `/api/v1/jobs/generate` | Generate, save, and vector-index a job from `raw_input`. |
| POST | `/api/v1/jobs/{job_id}/questions` | Generate and save 5-10 interview questions. |
| POST | `/api/v1/interview/chat` | Start or continue a scored interview session. |
| POST | `/api/v1/jobs/{job_id}/rank` | Rank supplied candidate resumes with a score breakdown. |
| GET | `/health` | Process liveness probe. |

Hugging Face upstream errors, including provider rate limits, return `502`.
A job is only reported as created after its Chroma document is indexed.

## Chroma reset after dependency changes

The old Chroma SQLite layout is incompatible with the configured ChromaDB
version. Stop the API, then remove only the local vector index and restart:

```powershell
Remove-Item -LiteralPath .\chroma_data -Recurse -Force
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

This deletes local vector embeddings only. It does not delete PostgreSQL job
records; regenerate/re-index those jobs before relying on vector ranking.

## Docker

Build from the repository root:

```powershell
docker build -t ai-recruitment-suite .
docker run --rm -p 8000:8000 --env-file .env -v "${PWD}\chroma_data:/app/chroma_data" ai-recruitment-suite
```

For managed deployments, use a persistent volume for `CHROMA_PERSIST_DIR` and
a managed PostgreSQL database. The container does not run migrations; replace
development `create_all` with Alembic before a production schema change.

## Tests

```powershell
pytest -q
```

The suite uses mocked external services and validates question constraints,
HTTP error translation, interview completion after the final answer, and
explainable ranking calculations.
