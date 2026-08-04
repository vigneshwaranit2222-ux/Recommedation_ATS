"""Hugging Face free-tier inference service.

Calls Hugging Face's unified OpenAI-compatible router
(``https://router.huggingface.co/v1/chat/completions``) using plain
``requests`` — **not** the ``huggingface_hub`` SDK and **not** the legacy
``api-inference.huggingface.co/models/<model>`` endpoint.

As of 2026, HF's free serverless tier is unified behind this router. The
model ids are task-specific config values so they can be swapped without a
code change — free-tier model availability rotates, and the deployer
must confirm the current live model at
https://huggingface.co/models?inference_provider=all&pipeline_tag=text-generation

Key design decisions
--------------------
* **Plain ``requests``** — the ``huggingface_hub`` SDK adds a dependency
  and abstracts away the HTTP layer, making error handling harder. The
  router is a standard OpenAI-compatible REST API; ``requests`` is the
  simplest, most transparent way to call it.
* **``HFServiceError`` on any failure** — network errors, non-2xx HTTP,
  malformed JSON, and missing required keys all raise a single exception
  type. The router layer catches this and returns HTTP 502 (upstream
  dependency failure, not app code failure).
* **HTTP 429 handled explicitly** — HF free-tier rate limits are common.
  They are translated to ``HFServiceError(status_code=502)`` so API clients
  receive one stable upstream-dependency error contract.
* **Markdown fence stripping** — LLMs often wrap JSON in `````json``
  fences despite instructions not to. We strip them defensively before
  ``json.loads`` rather than failing.
* **Never fabricate data** — if JSON parsing fails or required keys are
  missing, we raise an error. We never substitute defaults for the
  *content* of the response (e.g., we don't invent a job title if the
  model didn't return one). The only normalization we do is on the
  ``category`` field for questions (defaulting to ``technical``), because
  the question text itself is always preserved.

Embeddings note
---------------
For vector embeddings, we use ChromaDB's local default embedding function
(``all-MiniLM-L6-v2`` via ``sentence-transformers``) rather than a hosted
HF embeddings endpoint — HF's free router does not currently expose a
stable embeddings API the way it does chat completions. See
``app/services/vector_service.py``.
"""

from __future__ import annotations

import json
import re
from typing import Any

import requests

from ..config import settings


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class HFServiceError(Exception):
    """Raised on any HF router failure (network, non-2xx, malformed JSON).

    The router layer catches this and returns HTTP 502 to the client,
    because the failure is in an upstream dependency, not in application
    code. A 500 would imply a bug in our own code.
    """

    def __init__(self, detail: str, status_code: int = 502) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Markdown fence stripping
# ---------------------------------------------------------------------------

# Matches: ```json\n...\n```  or  ```\n...\n```
# The (?:json)? makes the language tag optional. DOTALL lets . match \n.
_FENCE_RE = re.compile(r"^```(?:json)?\s*\n(.*?)\n```\s*$", re.DOTALL)


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences if the LLM wrapped the output in them.

    LLMs frequently return JSON wrapped in `````json ... ````` despite
    explicit instructions not to. This function strips that wrapper so
    ``json.loads`` can parse the inner content. If no fence is present,
    the text is returned unchanged.
    """
    text = text.strip()
    match = _FENCE_RE.match(text)
    if match:
        return match.group(1).strip()
    return text


# ---------------------------------------------------------------------------
# Core router call
# ---------------------------------------------------------------------------

def _call_hf_router(messages: list[dict[str, str]], model: str) -> str:
    """Call the HF OpenAI-compatible router and return the content string.

    Parameters
    ----------
    messages:
        OpenAI-format message list: ``[{"role": "system"|"user"|"assistant",
        "content": "..."}]``.

    Returns
    -------
    str
        The ``content`` field from the first choice's message.

    Raises
    ------
    HFServiceError
        On network errors, non-2xx HTTP (except 429), or unexpected
        response structure.
    """
    url = f"{settings.HF_ROUTER_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.HF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": settings.HF_CHAT_TEMPERATURE,
        "max_tokens": settings.HF_CHAT_MAX_TOKENS,
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=settings.HF_REQUEST_TIMEOUT,
        )
    except requests.exceptions.Timeout as exc:
        raise HFServiceError(
            f"HF router request timed out after {settings.HF_REQUEST_TIMEOUT}s. "
            f"The free tier may be under load; retry shortly."
        ) from exc
    except requests.RequestException as exc:
        raise HFServiceError(f"Network error calling HF router: {exc}") from exc

    # Handle 429 explicitly — it's the most common HF free-tier error.
    if response.status_code == 429:
        raise HFServiceError(
            "Hugging Face router rate limited this request (HTTP 429). "
            "Retry after the provider's cooldown period."
        )

    if response.status_code in {500, 502}:
        raise HFServiceError(
            f"Hugging Face router is unavailable (HTTP {response.status_code}): "
            f"{response.text[:500]}"
        )

    # Any other non-2xx is an upstream failure.
    if not response.ok:
        raise HFServiceError(
            f"HF router returned HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise HFServiceError(
            f"HF router returned non-JSON response: {exc}"
        ) from exc

    # OpenAI-compatible response: {"choices": [{"message": {"content": "..."}}]}
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HFServiceError(
            f"Unexpected HF router response structure: "
            f"{json.dumps(data)[:500]}"
        ) from exc

    if not content or not content.strip():
        raise HFServiceError("HF router returned an empty content string.")

    return content


def _parse_json_response(content: str, context: str = "") -> dict[str, Any]:
    """Strip markdown fences and parse JSON from the LLM response.

    Parameters
    ----------
    content:
        Raw content string from the HF router.
    context:
        Human-readable label for the error message (e.g. "job description").

    Returns
    -------
    dict
        Parsed JSON object.

    Raises
    ------
    HFServiceError
        If the content cannot be parsed as JSON after fence stripping.
    """
    cleaned = _strip_markdown_fences(content)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        label = f" ({context})" if context else ""
        raise HFServiceError(
            f"Failed to parse JSON from HF response{label}: {exc}. "
            f"Raw content: {content[:500]}"
        ) from exc


# ---------------------------------------------------------------------------
# Public service functions
# ---------------------------------------------------------------------------

def generate_job_description(raw_input: str) -> dict[str, Any]:
    """Generate a structured job description from a short natural-language prompt.

    Parameters
    ----------
    raw_input:
        Short prompt like "Need 2 YOE React Developer with Tailwind & GraphQL".

    Returns
    -------
    dict with keys:
        - ``title`` (str)
        - ``description`` (str)
        - ``keywords`` (list[str])

    Raises
    ------
    HFServiceError
        If the HF call fails, JSON is malformed, or required keys are missing.
        We never fabricate a fallback job — the caller (router) returns 502.
    """
    system_prompt = (
        "You are a job description generator. Given a short prompt, produce "
        "a complete job posting. Return ONLY valid JSON (no markdown, no "
        "explanation) with this exact schema:\n"
        '{"title": string, "description": string, "keywords": [string, ...]}\n'
        "The description should be 2-3 paragraphs covering responsibilities, "
        "tech stack, and experience requirements. Keywords should be 5-10 "
        "technical/professional terms relevant to the role."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": raw_input},
    ]

    content = _call_hf_router(messages, settings.HF_CHAT_MODEL_PRIMARY)
    result = _parse_json_response(content, "job description")

    # --- Validate required keys & types --------------------------------
    if not isinstance(result, dict):
        raise HFServiceError(
            f"HF job response is not a JSON object. Got: {type(result).__name__}"
        )

    missing = [k for k in ("title", "description", "keywords") if k not in result]
    if missing:
        raise HFServiceError(
            f"HF job response missing required keys: {missing}. "
            f"Got keys: {list(result.keys())}"
        )

    title = result["title"]
    description = result["description"]
    keywords = result["keywords"]

    if not isinstance(title, str) or not title.strip():
        raise HFServiceError("HF job response 'title' is empty or not a string.")
    if not isinstance(description, str) or not description.strip():
        raise HFServiceError(
            "HF job response 'description' is empty or not a string."
        )
    if not isinstance(keywords, list):
        raise HFServiceError(
            f"HF job response 'keywords' is not a list. "
            f"Got: {type(keywords).__name__}"
        )

    # Clean keywords: strip whitespace, remove empties, dedupe (preserve order)
    seen: set[str] = set()
    clean_keywords: list[str] = []
    for kw in keywords:
        if isinstance(kw, str):
            kw_stripped = kw.strip()
            if kw_stripped and kw_stripped.lower() not in seen:
                seen.add(kw_stripped.lower())
                clean_keywords.append(kw_stripped)

    if not clean_keywords:
        raise HFServiceError(
            "HF job response 'keywords' list is empty after cleaning."
        )

    return {
        "title": title.strip(),
        "description": description.strip(),
        "keywords": clean_keywords,
    }


def generate_interview_questions(
    job_title: str,
    job_description: str,
    keywords: list[str],
    num_questions: int = 7,
) -> list[dict[str, str]]:
    """Generate interview questions for a job across three categories.

    Parameters
    ----------
    job_title:
        The job title (for context).
    job_description:
        The job description (for context).
    num_questions:
        Number of questions to generate (5–10 recommended).

    Returns
    -------
    list of dicts, each with keys:
        - ``question_text`` (str)
        - ``category`` (str: "technical" | "behavioral" | "experience")

    Raises
    ------
    HFServiceError
        If the HF call fails or the response contains no valid questions.
    """
    system_prompt = (
        f"You are an interview question generator. Given the job title and "
        f"description below, generate {num_questions} interview questions "
        f"across three categories: technical, behavioral, and experience. "
        f"Return ONLY valid JSON (no markdown, no explanation) with this "
        f"exact schema:\n"
        f'{{"questions": [{{"question_text": string, "category": string}}, ...]}}\n'
        f'Categories must be one of: "technical", "behavioral", "experience". '
        f"Distribute questions across all three categories.\n\n"
        f"Job Title: {job_title}\n"
        f"Job Description: {job_description}\n"
        f"Required keywords: {', '.join(keywords)}"
    )
    messages = [{"role": "user", "content": system_prompt}]

    content = _call_hf_router(messages, settings.HF_CHAT_MODEL_PRIMARY)
    result = _parse_json_response(content, "interview questions")

    if not isinstance(result, dict) or "questions" not in result:
        raise HFServiceError(
            f"HF questions response missing 'questions' key. "
            f"Got keys: {list(result.keys()) if isinstance(result, dict) else type(result)}"
        )

    raw_questions = result["questions"]
    if not isinstance(raw_questions, list):
        raise HFServiceError(
            f"HF 'questions' is not a list. Got: {type(raw_questions).__name__}"
        )
    if not 5 <= len(raw_questions) <= 10:
        raise HFServiceError(
            "HF questions response must contain between 5 and 10 questions; "
            f"received {len(raw_questions)}."
        )

    # --- Validate & normalize each question ---------------------------
    # Valid categories. If the model returns something unexpected, we
    # default to "technical" — we NEVER drop or fabricate the question
    # text itself, only normalize the category label.
    valid_categories = {"technical", "behavioral", "experience"}

    questions: list[dict[str, str]] = []
    for q in raw_questions:
        if not isinstance(q, dict):
            continue
        question_text = q.get("question_text")
        if not question_text or not isinstance(question_text, str):
            continue
        question_text = question_text.strip()
        if not question_text:
            continue

        # Normalize category: lowercase, strip, validate, default to technical.
        category = str(q.get("category", "technical")).strip().lower()
        if category not in valid_categories:
            category = "technical"

        questions.append(
            {"question_text": question_text, "category": category}
        )

    if not 5 <= len(questions) <= 10:
        raise HFServiceError(
            "HF questions response must contain 5 to 10 valid question objects "
            f"after validation; received {len(questions)}."
        )

    return questions


def run_interview_turn(
    chat_history: list[dict[str, str]],
    next_question: str,
) -> str:
    """Generate the interviewer's next message.

    Parameters
    ----------
    chat_history:
        The full conversation so far as a list of ``{"role", "content"}``
        dicts (OpenAI message format). The last entry may be a user
        message (the candidate's answer) or empty (first turn).
    next_question:
        The exact next question from the persisted job bank. An empty string
        signals that the final answer was received and a wrap-up is needed.

    Returns
    -------
    str
        The interviewer's next message: a one-sentence acknowledgment of
        the candidate's last answer (if any) followed by the next question,
        or a wrap-up message if no questions remain.

    Raises
    ------
    HFServiceError
        If the HF call fails.
    """
    # The exact database question is appended in Python below. LLMs are good
    # at a brief acknowledgement, but they must not be trusted to preserve a
    # persisted question verbatim.
    if next_question:
        has_candidate_answer = bool(
            chat_history and chat_history[-1].get("role") == "user"
        )
        if not has_candidate_answer:
            return next_question

        system_prompt = (
            "You are an AI interviewer conducting a job interview. "
            "acknowledge the candidate's immediately preceding answer in one "
            "short sentence. Do not ask a question, do not use markdown, and "
            "do not add any other sentence."
        )
    else:
        system_prompt = (
            "You are an AI interviewer conducting a job interview. All "
            "questions have been asked. Thank the candidate for their time "
            "and let them know the interview is complete. Return ONLY your "
            "message text (no JSON, no markdown fences)."
        )

    # Build the messages array: system prompt + full chat history.
    # The chat_history is already in {role, content} format, so we can
    # extend the messages list directly.
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]
    messages.extend(chat_history)

    content = _call_hf_router(messages, settings.HF_CHAT_MODEL_INTERVIEW).strip()
    if next_question:
        # Keep at most one acknowledgement sentence, then append the exact
        # stored question. This guarantees the question bank wording survives
        # every model response unchanged.
        acknowledgement = re.split(r"(?<=[.!?])\s+", content, maxsplit=1)[0].strip()
        return f"{acknowledgement}\n\n{next_question}" if acknowledgement else next_question
    return content


def score_interview_response(
    question: str,
    answer: str,
) -> dict[str, Any]:
    """Score a candidate's response to an interview question (0–10 scale).

    Parameters
    ----------
    question:
        The interview question that was asked.
    answer:
        The candidate's answer.

    Returns
    -------
    dict with keys:
        - ``score`` (float, clamped to 0–10)
        - ``feedback`` (str, 1–2 sentences)

    Raises
    ------
    HFServiceError
        If the HF call fails or the response is malformed. The router
        catches this and continues the interview without aborting — a
        scoring failure should not prevent the next question from being
        asked.
    """
    system_prompt = (
        "You are an interview response evaluator. Given a question and "
        "the candidate's answer, score the answer on a 0-10 scale. Return "
        "ONLY valid JSON (no markdown, no explanation) with this exact "
        "schema:\n"
        '{"score": number, "feedback": string}\n'
        "The score should reflect technical accuracy, completeness, and "
        "clarity. Feedback should be 1-2 sentences explaining the score."
    )
    user_message = (
        f"Question: {question}\n\nCandidate's answer: {answer}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    content = _call_hf_router(messages, settings.HF_CHAT_MODEL_SCORING)
    result = _parse_json_response(content, "response scoring")

    if not isinstance(result, dict) or "score" not in result:
        raise HFServiceError(
            f"HF scoring response missing 'score' key. "
            f"Got: {list(result.keys()) if isinstance(result, dict) else type(result)}"
        )

    # Parse and clamp score to [0, 10].
    try:
        score = float(result["score"])
    except (ValueError, TypeError) as exc:
        raise HFServiceError(
            f"HF scoring response 'score' is not a number: {result['score']}"
        ) from exc

    score = max(0.0, min(10.0, score))

    feedback = str(result.get("feedback", "")).strip()

    return {"score": score, "feedback": feedback}


# ---------------------------------------------------------------------------
# General-purpose chatbot
# ---------------------------------------------------------------------------

def chat_bot(
    message: str,
    chat_history: list[dict[str, str]] | None = None,
) -> str:
    """Return a conversational reply from the HF router.

    Parameters
    ----------
    message:
        The user's latest message.
    chat_history:
        Optional prior conversation as a list of ``{"role", "content"}``
        dicts (OpenAI message format). The provided history is appended
        before the new user message so the model has full context.

    Returns
    -------
    str
        The assistant's reply text.

    Raises
    ------
    HFServiceError
        If the HF call fails or the response is empty.
    """
    system_prompt = (
        "You are a helpful, friendly AI assistant embedded in the AI "
        "Recruitment & Hiring Suite. You can answer general questions, "
        "explain the recruitment features, help debug usage, and hold a "
        "natural conversation. Be concise, accurate, and helpful."
    )

    # Build the OpenAI-style message list: system prompt + history + new user turn.
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]

    if chat_history:
        # Only keep well-formed turns to avoid sending malformed data upstream.
        for turn in chat_history:
            role = turn.get("role")
            content = turn.get("content")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": str(content)})

    messages.append({"role": "user", "content": message})

    content = _call_hf_router(messages, settings.HF_CHAT_MODEL_PRIMARY)
    return content.strip()
