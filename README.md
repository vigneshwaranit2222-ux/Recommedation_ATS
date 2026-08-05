# AI Recruitment & Hiring Suite

Production-ready FastAPI backend for LLM-assisted job creation, interview question generation, conversational interviews, hybrid resume ranking, and ChatGPT-style assistant.

---

## Architecture Stack

- **Framework:** Python 3.11+, FastAPI, Uvicorn
- **Database:** PostgreSQL with SQLAlchemy 2.0 async ORM (`asyncpg`) & Alembic migrations
- **Authentication:** JWT Access & Refresh tokens with bcrypt password hashing
- **Vector DB:** ChromaDB local persistent embeddings (`all-MiniLM-L6-v2`)
- **LLM Integration:** Hugging Face OpenAI-compatible serverless inference router
- **Ranking Engine:** Batch TF-IDF (50%) + Regex Keyword Matching (35%) + Vector Cosine Similarity (15%)
- **Frontend:** HTML5, CSS3, Bootstrap 5, ES6+ JavaScript

---

## Environment Configuration

Create a `.env` file in the root directory:

```ini
# PostgreSQL Connection URL
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/ats_db
SQL_ECHO=false

# JWT Security
JWT_SECRET_KEY=super-secret-jwt-key-change-in-production
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60

# Hugging Face Inference Router
HF_TOKEN=your_huggingface_api_token
HF_ROUTER_BASE_URL=https://router.huggingface.co/v1
HF_CHAT_MODEL_PRIMARY=Qwen/Qwen2.5-Coder-32B-Instruct
HF_CHAT_MODEL_INTERVIEW=meta-llama/Llama-3.1-8B-Instruct
HF_CHAT_MODEL_SCORING=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B

# Vector DB
CHROMA_PERSIST_DIR=./chroma_data
CHROMA_COLLECTION_JOBS=company_jobs
CHROMA_COLLECTION_RESUMES=student_resumes
```

---

## Quick Start & Local Execution

1. **Activate Virtual Environment & Install Dependencies:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Database Migration (Alembic):**
   ```powershell
   alembic upgrade head
   ```

3. **Launch Server:**
   ```powershell
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Access Applications:**
   - Web App UI: `http://localhost:8000/`
   - Interactive Swagger API Documentation: `http://localhost:8000/docs`
   - Liveness Probe: `http://localhost:8000/health`

---

## Docker Deployment

Build and run using Docker Compose:

```powershell
docker-compose up --build -d
```

Or build manually:

```powershell
docker build -t ai-recruitment-suite .
docker run --rm -p 8000:8000 --env-file .env -v "${PWD}\chroma_data:/app/chroma_data" ai-recruitment-suite
```

---

## API Endpoints Overview

| Tag | Method | Endpoint | Description |
| --- | --- | --- | --- |
| **Authentication** | POST | `/api/v1/auth/register` | Register Candidate, Company (HR), or Admin account |
| **Authentication** | POST | `/api/v1/auth/login` | Authenticate & retrieve Access & Refresh tokens |
| **Authentication** | POST | `/api/v1/auth/refresh` | Refresh expired access tokens |
| **Authentication** | GET | `/api/v1/auth/me` | Fetch current user details & role |
| **Jobs** | POST | `/api/v1/jobs/generate` | Generate structured job & index in ChromaDB |
| **Jobs** | GET | `/api/v1/jobs` | List all available job requirements |
| **Questions** | POST | `/api/v1/jobs/{job_id}/questions` | Generate interview question bank |
| **Interview Chat** | POST | `/api/v1/interview/chat` | Conduct & score AI interview turn |
| **Interview Chat** | GET | `/api/v1/interview/sessions` | List active interview sessions |
| **Resume Ranking**| POST | `/api/v1/jobs/{job_id}/rank` | Perform hybrid resume ranking & leaderboard |
| **Chatbot** | POST | `/api/v1/chatbot/chat` | Multi-turn conversational recruiter chatbot |
| **Health** | GET | `/health` | Health & liveness check |

---

## Automated Test Suite

Run unit and integration tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/
```

---

## Manual Setup Steps Required for Production Deployment

1. **Obtain Hugging Face API Token:**
   - Sign up at [huggingface.co](https://huggingface.co) and generate an API Access Token under Account Settings -> Access Tokens.
   - Paste the token into your `.env` file as `HF_TOKEN=hf_...`.

2. **Provision PostgreSQL Database:**
   - Set up a PostgreSQL instance (or local Docker PostgreSQL container).
   - Create a database (e.g. `ats_db`) and update `DATABASE_URL` in `.env`.

3. **Configure JWT Production Secret:**
   - Replace `JWT_SECRET_KEY` in `.env` with a strong random 64-character secret.
