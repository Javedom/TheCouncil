"""Gemini client and robust generation helpers.

Every model call in the Council goes through `safe_generate` or
`safe_generate_structured`. These never raise: they retry transient errors,
handle blocked/empty responses, and always return something usable so the
graph can keep flowing to a final answer.
"""
import os
import time
from typing import List, Optional, Type, TypeVar

from google import genai
from google.genai import types
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

import config

_client = None
T = TypeVar("T", bound=BaseModel)


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        _client = genai.Client(api_key=api_key)
    return _client


# Relaxed safety so creative/security analysis tasks aren't spuriously blocked.
_SAFETY = [
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH"),
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_ONLY_HIGH"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_ONLY_HIGH"),
]


def render_transcript(messages: List[BaseMessage]) -> str:
    """Flatten the message history into a readable, role-labelled transcript.

    Gemini rejects alternating same-role turns, so a multi-agent debate is best
    delivered as a single annotated transcript inside one user turn.
    """
    lines = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue
        if isinstance(msg, HumanMessage):
            label = "User"
        elif isinstance(msg, AIMessage):
            label = getattr(msg, "name", None) or "Agent"
        else:
            label = "Note"
        content = (msg.content or "").strip()
        if content:
            lines.append(f"[{label}]:\n{content}")
    return "\n\n".join(lines) if lines else "(no prior discussion)"


def build_contents(transcript: str, directive: str) -> List[types.Content]:
    """Compose the single user-turn payload: full transcript + the task."""
    text = (
        f"=== COUNCIL TRANSCRIPT SO FAR ===\n{transcript}\n\n"
        f"=== YOUR TASK RIGHT NOW ===\n{directive}"
    )
    return [types.Content(role="user", parts=[types.Part(text=text)])]


def _extract_text(response) -> str:
    """Pull text out of a response, tolerating empty/blocked candidates."""
    text = getattr(response, "text", None)
    if text:
        return text.strip()
    try:
        cand = response.candidates[0]
        parts = getattr(cand.content, "parts", None) or []
        joined = "".join(getattr(p, "text", "") or "" for p in parts).strip()
        if joined:
            return joined
        reason = getattr(cand, "finish_reason", "unknown")
        return f"*[No content generated. Finish reason: {reason}.]*"
    except Exception:
        return "*[No content generated.]*"


def safe_generate(
    model: str,
    system_instruction: str,
    contents,
    tools: Optional[list] = None,
) -> str:
    """Generate free-form text. Never raises; returns a usable string."""
    client = get_client()
    cfg_kwargs = {"system_instruction": system_instruction, "safety_settings": _SAFETY}
    if tools:
        cfg_kwargs["tools"] = tools
    cfg = types.GenerateContentConfig(**cfg_kwargs)

    last_err = None
    for attempt in range(config.MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(model=model, contents=contents, config=cfg)
            return _extract_text(response)
        except Exception as e:  # noqa: BLE001 - intentional catch-all for robustness
            last_err = e
            if attempt < config.MAX_RETRIES:
                time.sleep(1.5 * (attempt + 1))
    return f"*[The Council hit a technical issue reaching the model: {last_err}]*"


def safe_generate_structured(
    model: str,
    system_instruction: str,
    contents,
    schema: Type[T],
) -> Optional[T]:
    """Generate a validated pydantic object, or None if it cannot be produced."""
    client = get_client()
    cfg = types.GenerateContentConfig(
        system_instruction=system_instruction,
        safety_settings=_SAFETY,
        response_mime_type="application/json",
        response_json_schema=schema.model_json_schema(),
    )

    for attempt in range(config.MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(model=model, contents=contents, config=cfg)
            raw = _extract_text(response)
            return schema.model_validate_json(raw)
        except Exception:  # noqa: BLE001
            if attempt < config.MAX_RETRIES:
                time.sleep(1.5 * (attempt + 1))
    return None
