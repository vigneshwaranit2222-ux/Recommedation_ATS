"""General-purpose chatbot router.

Endpoints
---------
1. ``POST /api/v1/chatbot/chat`` — API to run a general-purpose bot API.

Design principles
-----------------
* **Thin router** — the endpoint does only validation, I/O orchestration,
  and error translation. All business logic lives in the service layer.
* **502 for HF failures** — when an HF router call fails, the failure is
  in an upstream dependency, not in application code. A 500 would imply a
  bug in our own code. 502 (Bad Gateway) is the correct semantic.
* **Sync→async bridge** — the HF service uses plain ``requests`` (sync).
  We wrap calls in ``asyncio.to_thread()`` so the event loop isn't blocked
  during the HTTP round-trip.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, status

from ..schemas import ChatbotRequest, ChatbotResponse
from ..services import hf_service
from ..services.hf_service import HFServiceError

router = APIRouter(prefix="/api/v1", tags=["Chatbot"])


# ===========================================================================
# Helper: run sync service functions in a thread
# ===========================================================================

async def _run_sync(func, *args, **kwargs):
    """Run a sync function in a thread pool to avoid blocking the event loop.

    The HF service uses plain ``requests`` (sync). Calling it directly in
    an ``async def`` endpoint would block the event loop for the duration
    of the HTTP round-trip (potentially 30–60s on the free tier).
    ``asyncio.to_thread`` runs the call in a worker thread.
    """
    return await asyncio.to_thread(func, *args, **kwargs)


# ===========================================================================
# 1. POST /api/v1/chatbot/chat
# ===========================================================================

@router.post("/chatbot/chat", response_model=ChatbotResponse, tags=["Chatbot"])
async def chatbot_chat(
    request: ChatbotRequest,
):
    """Generate a conversational reply from the chatbot.

    Accepts the user's latest ``message`` and an optional ``chat_history``
    (list of prior ``{"role", "content"}`` turns). Calls the HF router with
    a friendly system prompt plus the accumulated history, then returns the
    assistant's reply and the updated conversation.
    """
    # Convert the validated history into plain dicts for the service layer.
    history = [
        {"role": turn.role, "content": turn.content}
        for turn in request.chat_history
    ]

    try:
        reply = await _run_sync(
            hf_service.chat_bot,
            message=request.message,
            chat_history=history,
        )
    except HFServiceError as exc:
        # 502: upstream dependency failure, not app code failure.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Chatbot request failed (upstream LLM error): {exc}",
        )

    # Build the updated conversation history.
    updated_history = history + [
        {"role": "user", "content": request.message},
        {"role": "assistant", "content": reply},
    ]

    return ChatbotResponse(
        reply=reply,
        chat_history=updated_history,
    )
