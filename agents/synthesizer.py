"""The Synthesizer node — delivers the Council's final answer.

Reconciles every contribution in the transcript into one clean, complete answer
addressed directly to the user. If synthesis itself fails, it surfaces the
strongest real contribution rather than an empty or fabricated answer.
"""
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage

import config
from .client import generate_reasoned, render_transcript, build_contents, bullets
from .prompts import SYNTHESIZER_PROMPT
from .events import council_message


class _Synthesis(BaseModel):
    reasoning: str
    content: str
    confidence: int = Field(default=0, description="0-100: how confident the answer meets the success criteria")


def _last_good_output(messages) -> str:
    """The most recent successful worker contribution, for fallback."""
    for m in reversed(messages):
        meta = getattr(m, "additional_kwargs", {}) or {}
        if isinstance(m, AIMessage) and meta.get("kind") == "agent" and (m.content or "").strip():
            return m.content
    return ""


def synthesizer_node(state):
    problem = state.get("problem", "")
    criteria_md = bullets(state.get("success_criteria", []))
    messages = state.get("messages", [])
    transcript = render_transcript(messages)

    system = SYNTHESIZER_PROMPT.format(problem=problem, criteria=criteria_md)
    documents = state.get("documents", "")
    if documents:
        system += (
            "\n\n=== USER-PROVIDED DOCUMENTS (the answer must respect these) ===\n"
            f"{documents}"
        )
    contents = build_contents(transcript, "Deliver the Council's final answer to the user.")

    obj, final, reasoning, ok = generate_reasoned(
        config.SYNTH_MODEL, system, contents, _Synthesis,
        fallback_reasoning="Integrated the Council's contributions into a single answer.",
        label="Exec",
    )
    confidence = max(0, min(100, obj.confidence)) if (obj is not None and ok) else None

    if not ok or not final.strip():
        # Last resort: give the user the best real work produced, not an empty
        # box or an invented answer.
        salvage = _last_good_output(messages)
        if salvage:
            final = (
                "_(The Council could not run a final synthesis pass; presenting the "
                "most complete contribution below.)_\n\n" + salvage
            )
            reasoning = "Synthesis failed; surfaced the strongest available contribution."
        else:
            final = (
                "The Council was unable to produce an answer due to repeated model "
                "errors. Please try again in a moment."
            )
            reasoning = "Synthesis failed and no usable contributions were available."

    msg = council_message(
        role="Exec",
        content=final,
        kind="final",
        phase=config.PHASE_SYNTHESIS,
        reasoning=reasoning,
        extra={"confidence": confidence},
    )
    return {
        "phase": config.PHASE_SYNTHESIS,
        "final_answer": final,
        "confidence": confidence,
        "messages": [msg],
    }
