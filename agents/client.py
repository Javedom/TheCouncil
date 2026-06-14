"""Gemini client and robust generation helpers.

Every model call in the Council goes through `safe_generate` or
`safe_generate_structured`. These never raise: they retry transient errors,
handle blocked/empty responses, and always return something usable so the
graph can keep flowing to a final answer.
"""
import os
import time
import threading
from typing import List, Optional, Type, TypeVar

from google import genai
from google.genai import types
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

import config

_client = None
T = TypeVar("T", bound=BaseModel)

# --- Usage / cost tracking --------------------------------------------------
# A per-run log of model calls. The UI resets it before a run and reads the
# summary afterwards. (Single-threaded execution today; revisit when parallel
# branches land — usage would then be attached to messages instead.)
_usage_log: list = []
_usage_lock = threading.Lock()


def reset_usage():
    with _usage_lock:
        _usage_log.clear()


def _record_usage(label, model, response, seconds, ok):
    inp = out = 0
    meta = getattr(response, "usage_metadata", None)
    if meta is not None:
        inp = getattr(meta, "prompt_token_count", 0) or 0
        out = getattr(meta, "candidates_token_count", 0) or 0
    with _usage_lock:
        _usage_log.append({
            "label": label or "?",
            "model": model,
            "input_tokens": inp,
            "output_tokens": out,
            "seconds": seconds,
            "ok": ok,
        })


def usage_summary() -> dict:
    """Aggregate the current run's usage into totals + a per-label breakdown."""
    import config
    with _usage_lock:
        log = list(_usage_log)
    total_in = total_out = 0
    total_cost = 0.0
    by_label: dict = {}
    for r in log:
        in_p, out_p = config.price_for(r["model"])
        cost = r["input_tokens"] / 1e6 * in_p + r["output_tokens"] / 1e6 * out_p
        total_in += r["input_tokens"]
        total_out += r["output_tokens"]
        total_cost += cost
        b = by_label.setdefault(r["label"], {"calls": 0, "input": 0, "output": 0, "cost": 0.0, "seconds": 0.0})
        b["calls"] += 1
        b["input"] += r["input_tokens"]
        b["output"] += r["output_tokens"]
        b["cost"] += cost
        b["seconds"] += r["seconds"]
    return {
        "calls": len(log),
        "input_tokens": total_in,
        "output_tokens": total_out,
        "total_tokens": total_in + total_out,
        "cost": total_cost,
        "by_label": by_label,
    }


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
    """Pull text out of a response, tolerating empty/blocked candidates.

    Returns "" when the model produced no usable content so callers can detect
    failure, rather than a placeholder string that would otherwise flow into the
    transcript and the final answer as if it were a real contribution.
    """
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
        print(f"[Council] Model returned no content. Finish reason: {reason}")
    except Exception as e:  # noqa: BLE001
        print(f"[Council] Could not extract text from response: {e}")
    return ""


def safe_generate(
    model: str,
    system_instruction: str,
    contents,
    tools: Optional[list] = None,
    label: str = "",
) -> str:
    """Generate free-form text. Never raises.

    Returns the generated text, or "" if the model could not be reached or
    produced nothing. An empty string is the failure signal — callers must treat
    it as "this step produced no output" rather than as content.
    """
    client = get_client()
    cfg_kwargs = {"system_instruction": system_instruction, "safety_settings": _SAFETY}
    if tools:
        cfg_kwargs["tools"] = tools
    cfg = types.GenerateContentConfig(**cfg_kwargs)

    last_err = None
    for attempt in range(config.MAX_RETRIES + 1):
        start = time.time()
        try:
            response = client.models.generate_content(model=model, contents=contents, config=cfg)
            _record_usage(label, model, response, time.time() - start, ok=True)
            return _extract_text(response)
        except Exception as e:  # noqa: BLE001 - intentional catch-all for robustness
            last_err = e
            if attempt < config.MAX_RETRIES:
                time.sleep(1.5 * (attempt + 1))
    print(f"[Council] Generation failed after retries: {last_err}")
    return ""


def safe_generate_structured(
    model: str,
    system_instruction: str,
    contents,
    schema: Type[T],
    label: str = "",
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
        start = time.time()
        try:
            response = client.models.generate_content(model=model, contents=contents, config=cfg)
            _record_usage(label, model, response, time.time() - start, ok=True)
            raw = _extract_text(response)
            if not raw:
                raise ValueError("empty response")
            return schema.model_validate_json(raw)
        except Exception:  # noqa: BLE001
            if attempt < config.MAX_RETRIES:
                time.sleep(1.5 * (attempt + 1))
    return None


def bullets(items, empty: str = "- (none specified)") -> str:
    """Render a list as a markdown bullet list, with a placeholder when empty."""
    cleaned = [str(i).strip() for i in (items or []) if str(i).strip()]
    return "\n".join(f"- {i}" for i in cleaned) if cleaned else empty


def generate_reasoned(
    model: str,
    system_instruction: str,
    contents,
    schema: Type[T],
    fallback_reasoning: str = "",
    label: str = "",
):
    """Structured {reasoning, content, ...} generation with a free-form fallback.

    Returns ``(obj, content, reasoning, ok)``:
      - obj: the validated schema instance, or None if we fell back to text.
      - content / reasoning: the deliverable and its rationale.
      - ok: False only when the model produced no usable content at all.

    Centralizes the "try structured, else plain text, else fail" pattern shared
    by the worker and synthesizer nodes. The schema must expose ``content`` and
    ``reasoning`` fields.
    """
    obj = safe_generate_structured(model, system_instruction, contents, schema, label=label)
    if obj is not None and (getattr(obj, "content", "") or "").strip():
        reasoning = (getattr(obj, "reasoning", "") or "").strip() or fallback_reasoning
        return obj, obj.content, reasoning, True
    text = safe_generate(model, system_instruction, contents, label=label)
    if text:
        return None, text, fallback_reasoning, True
    return None, "", fallback_reasoning, False
