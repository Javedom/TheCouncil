"""Export a Council session to Markdown.

Turns the rendered message history (with its phase/role/reasoning metadata) into
a self-contained Markdown document the user can download or share.
"""
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, AIMessage


def transcript_to_markdown(history, confidence=None) -> str:
    lines = [
        "# The Council — Session Transcript",
        f"_Exported {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
    ]
    for msg in history:
        if isinstance(msg, HumanMessage):
            lines += ["## 🧑 User", "", msg.content or "", ""]
            continue
        if not isinstance(msg, AIMessage):
            continue
        meta = getattr(msg, "additional_kwargs", {}) or {}
        kind = meta.get("kind", "agent")
        role = meta.get("role") or getattr(msg, "name", "Council")
        phase = meta.get("phase", "")
        reasoning = meta.get("reasoning", "")

        if kind == "plan":
            header = "## 🧭 Chairman — Plan"
        elif kind == "final":
            badge = f" (confidence {confidence}%)" if confidence is not None else ""
            header = f"## ⚖️ Final Answer{badge}"
        elif kind == "critique":
            header = f"## 🕵️ Critic — {phase}"
        elif kind == "error":
            header = f"## ⚠️ {role} — {phase} (step failed)"
        else:
            header = f"## {role}" + (f" — {phase}" if phase else "")

        lines.append(header)
        lines.append("")
        if reasoning and kind not in ("plan", "critique"):
            lines += [f"> 🧠 _{reasoning}_", ""]
        lines += [msg.content or "", ""]

    return "\n".join(lines).strip() + "\n"
