"""The Synthesizer node — delivers the Council's final answer.

Reconciles every contribution in the transcript into one clean, complete answer
addressed directly to the user.
"""
from pydantic import BaseModel

import config
from .client import (
    safe_generate,
    safe_generate_structured,
    render_transcript,
    build_contents,
)
from .prompts import SYNTHESIZER_PROMPT
from .events import council_message


class _Synthesis(BaseModel):
    reasoning: str
    content: str


def synthesizer_node(state):
    problem = state.get("problem", "")
    criteria = state.get("success_criteria", []) or []
    criteria_md = "\n".join(f"- {c}" for c in criteria) or "- (none specified)"
    transcript = render_transcript(state.get("messages", []))

    system = SYNTHESIZER_PROMPT.format(problem=problem, criteria=criteria_md)
    contents = build_contents(transcript, "Deliver the Council's final answer to the user.")

    result = safe_generate_structured(config.SYNTH_MODEL, system, contents, _Synthesis)
    if result is not None and result.content:
        final = result.content
        reasoning = result.reasoning
    else:
        final = safe_generate(config.SYNTH_MODEL, system, contents)
        reasoning = "Integrated the Council's contributions into a single answer."

    msg = council_message(
        role="Exec",
        content=final,
        kind="final",
        phase=config.PHASE_SYNTHESIS,
        reasoning=reasoning,
    )
    return {
        "phase": config.PHASE_SYNTHESIS,
        "final_answer": final,
        "messages": [msg],
    }
