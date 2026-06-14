"""Gemini client and robust generation helpers.

Every model call in the Council goes through `safe_generate` or
`safe_generate_structured`. These never raise: they retry transient errors,
handle blocked/empty responses, and always return something usable so the
graph can keep flowing to a final answer.
"""
import os
import time
import threading
import contextvars
from typing import List, Optional, Type, TypeVar

from google import genai
from google.genai import types
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

import config

T = TypeVar("T", bound=BaseModel)

# --- Per-session API key isolation ------------------------------------------
# The Gemini API key is held in a context-local variable, NOT a process global.
# This is essential for the multi-user web deployment: each Streamlit session
# (and each graph run inside it) sees only its own key, so one visitor can never
# use — or be billed against — another visitor's key. The key is never written
# to disk or to os.environ.
_API_KEY: "contextvars.ContextVar[str]" = contextvars.ContextVar("council_api_key", default="")

# Cache one genai.Client per distinct key so we don't rebuild it on every call.
_clients: dict = {}
_clients_lock = threading.Lock()

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


def _byok_only() -> bool:
    """Whether to refuse any server-side env key (strict bring-your-own-key).

    Set COUNCIL_BYOK_ONLY=1 in the deployment so each web visitor must supply
    their own key and an accidental server GOOGLE_API_KEY is never used.
    """
    return os.environ.get("COUNCIL_BYOK_ONLY", "").strip().lower() in ("1", "true", "yes", "on")


def _resolve_key() -> str:
    """The key for the current context: this session's key, else a server env
    key (unless BYOK-only mode forbids it — e.g. on the public web deployment)."""
    key = (_API_KEY.get() or "").strip()
    if key:
        return key
    if _byok_only():
        return ""
    return (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip()


def get_client():
    key = _resolve_key()
    if not key:
        raise RuntimeError(
            "No Gemini API key for this session. Enter your own key in the sidebar settings."
        )
    with _clients_lock:
        client = _clients.get(key)
        if client is None:
            # Bound cache: avoid unbounded growth from many distinct visitors.
            if len(_clients) > 256:
                _clients.clear()
            client = genai.Client(api_key=key)
            _clients[key] = client
        return client


def set_api_key(key: str):
    """Bind the Gemini API key for the current execution context (this session).

    Stored only in a context-local variable — isolated per session/run, never
    written to disk or os.environ, and never shared between web visitors.
    """
    _API_KEY.set((key or "").strip())


def has_api_key() -> bool:
    return bool(_resolve_key())


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
    """Assemble usable content from a response, tolerating empty/blocked output.

    Handles plain text and tool outputs — in particular code execution, whose
    executed code and results live in dedicated parts rather than `.text`.
    Returns "" when nothing usable was produced so callers can detect failure
    instead of receiving a placeholder that would pollute the transcript.
    """
    try:
        cand = response.candidates[0]
        parts = getattr(cand.content, "parts", None) or []
        chunks = []
        for p in parts:
            if getattr(p, "text", None):
                chunks.append(p.text)
            ec = getattr(p, "executable_code", None)
            if ec is not None and getattr(ec, "code", None):
                lang = (getattr(ec, "language", "") or "python").lower()
                chunks.append(f"```{lang}\n{ec.code}\n```")
            cr = getattr(p, "code_execution_result", None)
            if cr is not None and getattr(cr, "output", None):
                chunks.append(f"**Execution output:**\n```\n{cr.output}\n```")
        joined = "\n\n".join(c for c in chunks if c).strip()
        if joined:
            return joined
        reason = getattr(cand, "finish_reason", "unknown")
        print(f"[Council] Model returned no content. Finish reason: {reason}")
    except Exception as e:  # noqa: BLE001
        # Last resort: the simple text accessor.
        try:
            text = getattr(response, "text", None)
            if text:
                return text.strip()
        except Exception:  # noqa: BLE001
            pass
        print(f"[Council] Could not extract text from response: {e}")
    return ""


_THINKING_LEVELS = {"minimal", "low", "medium", "high"}


def _thinking_config():
    """Build a ThinkingConfig from config.THINKING_LEVEL, or None for default."""
    level = (getattr(config, "THINKING_LEVEL", "") or "").strip().lower()
    if level not in _THINKING_LEVELS:
        return None
    try:
        return types.ThinkingConfig(thinking_level=level)
    except Exception:  # noqa: BLE001 - unsupported SDK -> ignore
        return None


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
    thinking = _thinking_config()
    if thinking is not None:
        cfg_kwargs["thinking_config"] = thinking
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
    cfg_kwargs = {
        "system_instruction": system_instruction,
        "safety_settings": _SAFETY,
        "response_mime_type": "application/json",
        "response_json_schema": schema.model_json_schema(),
    }
    thinking = _thinking_config()
    if thinking is not None:
        cfg_kwargs["thinking_config"] = thinking
    cfg = types.GenerateContentConfig(**cfg_kwargs)

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
