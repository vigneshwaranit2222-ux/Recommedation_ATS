# ATS Resume Ranking System

A production-ready FastAPI Applicant Tracking System (ATS) that ranks student
resumes against posted jobs using a **hybrid scoring strategy**:

| Component | Weight | Description |
|-----------|--------|-------------|
| TF-IDF cosine similarity | **50%** | Lexical overlap of job text vs resume text (`TfidfVectorizer` + `cosine_similarity`) |
| Keyword match % | **35%** | Fraction of required keywords found via case-insensitive word-boundary regex in resume text **or** extracted skills |
| ChromaDB vector similarity | **15%** | Semantic similarity from Chroma's cosine distance (single query for all resumes) |

## Tech Stack

- **FastAPI** + **uvicorn** — async REST API
- **SQLAlchemy** + **SQLite** — job metadata persistence
- **ChromaDB** (PersistentClient) — vector store for jobs & resumes (cosine HNSW)
- **pdfplumber** — robust PDF text extraction
- **spaCy** (`en_core_web_sm`) + **PhraseMatcher** — NER for skills, orgs, degrees, locations
- **scikit-learn** — TF-IDF + cosine similarity
- **Bootstrap 5** + vanilla JS — frontend served via `StaticFiles`

## Project Structure

```
ats_ranking_system/
├── app/
│   ├── __init__.py        # package marker
│   ├── database.py        # SQLAlchemy engine/session + JobRequirement model
│   ├── vector_db.py       # ChromaDB PersistentClient wrapper (cosine collections)
│   ├── pdf_parser.py      # pdfplumber extraction with error handling
│   ├── ner_engine.py      # spaCy NER + PhraseMatcher skill extraction
│   ├── scorer.py          # Hybrid scoring (TF-IDF + keyword + vector)
│   └── main.py            # FastAPI app, endpoints, static mounting
├── static/
│   └── index.html         # Bootstrap 5 UI (3 tabs + live leaderboard)
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Install Python dependencies
cd ats_ranking_system
pip install -r requirements.txt

# 2. Download the spaCy English model (required for NER)ll
python -m spacy download en_core_web_sm

# 3. Run the server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open:
- **UI**: http://127.0.0.1:8000/static/index.html
- **Swagger docs**: http://127.0.0.1:8000/docs

## API Endpoints

### `POST /post_job/`
Form fields: `title`, `description`, `required_keywords` (comma-separated).
Saves to SQLite `job_requirements` **and** indexes into Chroma `company_jobs`.
Returns `200` on full success, `207` if SQLite succeeded but Chroma failed
(partial success — SQLite is **not** rolled back).

### `GET /jobs/`
Returns all saved jobs (newest first) for the frontend dropdown.

### `POST /upload_resume/`
Form fields: `student_id`, `student_name`, `file` (PDF).
Extracts text with pdfplumber, runs spaCy NER (skills via PhraseMatcher, orgs
via `ORG`, locations via `GPE`, degrees via regex), and indexes into Chroma
`student_resumes`. Corrupt/scanned PDFs return `422` with a clear message.

### `POST /rank_candidates/`
Form field: `job_id`. Fetches **all** resumes from Chroma, scores each against
the job, and returns candidates ranked highest-to-lowest with full score
breakdowns and matched/missing keyword lists.

## Key Design Decisions & Gotchas Handled

1. **Cosine HNSW space** — Chroma collections are created with
   `metadata={"hnsw:space": "cosine"}` so distance is cosine (not L2).
2. **Metadata type coercion** — Chroma only accepts `str/int/float/bool`;
   lists (skills, orgs) are joined into CSV strings before storage.
3. **spaCy model guard** — missing `en_core_web_sm` raises a `RuntimeError`
   with the exact `python -m spacy download` command.
4. **207 partial success** — if Chroma indexing fails after SQLite succeeds,
   SQLite is **not** rolled back; a `207` response is returned instead.
5. **Single Chroma query** — vector similarity for all resumes is fetched in
   **one** `query()` call, not one per resume.
6. **Word-boundary keyword matching** — uses lookarounds
   `(?<![A-Za-z0-9])...(?![A-Za-z0-9])` so "Java" doesn't match inside
   "JavaScript" while still matching "C++" / "C#".
7. **numpy 1.x** — pinned to `1.26.4` for spaCy 3.7.x / thinc 8.2.x binary
   compatibility (numpy 2.x causes a dtype size mismatch crash).

## Error Handling

Every I/O boundary is wrapped:
- PDF parsing → `PDFParseError` → `422`
- SQLite writes → `SQLAlchemyError` → `500` (with rollback)
- Chroma writes → exception → `500` (or `207` for job indexing)
- Missing spaCy model → `RuntimeError` → `500` with download command