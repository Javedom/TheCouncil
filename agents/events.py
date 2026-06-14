"""Helper for emitting UI-rich messages from nodes.

Every message the Council produces carries metadata in `additional_kwargs` so
the UI can render phases, per-agent reasoning and verdicts without re-parsing
content. `kind` tells the UI how to render it.
"""
from langchain_core.messages import AIMessage


def council_message(
    role: str,
    content: str,
    kind: str = "agent",
    phase: str = "",
    reasoning: str = "",
    extra: dict | None = None,
) -> AIMessage:
    meta = {"kind": kind, "phase": phase, "reasoning": reasoning, "role": role}
    if extra:
        meta.update(extra)
    return AIMessage(content=content or "", name=role, additional_kwargs=meta)
