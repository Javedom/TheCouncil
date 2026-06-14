"""The generic Worker node.

A single node executes whichever plan step the cursor points at, adopting that
step's dynamic role and capability. Reasoning + content are captured separately
so the UI can show the agent's thinking. The cursor and counters advance here.
"""
from pydantic import BaseModel

import config
from .client import (
    safe_generate,
    safe_generate_structured,
    render_transcript,
    build_contents,
)
from .prompts import WORKER_PROMPT, RESEARCH_WORKER_PROMPT
from .events import council_message


class _WorkerOutput(BaseModel):
    reasoning: str
    content: str


def _model_for(capability: str) -> str:
    if capability in ("reason", "code"):
        return config.REASONING_WORKER_MODEL
    return config.WORKER_MODEL


def worker_node(state):
    plan = state.get("plan", [])
    cursor = state.get("cursor", 0)

    # Defensive guard: nothing valid to execute -> no-op advance.
    if cursor >= len(plan):
        return {"cursor": cursor + 1}

    step = plan[cursor]
    role = step["role"]
    phase = step["phase"]
    capability = step["capability"]
    problem = state.get("problem", "")
    transcript = render_transcript(state.get("messages", []))
    scratch = state.get("scratchpad", "")

    if capability == "research":
        system = RESEARCH_WORKER_PROMPT.format(role=role, phase=phase, objective=step["objective"], problem=problem)
        if scratch:
            system += f"\n\n=== SHARED SCRATCHPAD ===\n{scratch}"
        directive = f"Carry out your objective: {step['objective']}"
        contents = build_contents(transcript, directive)
        content = safe_generate(config.WORKER_MODEL, system, contents, tools=[{"google_search": {}}])
        reasoning = "Gathered current, sourced information from the web for the Council."
    else:
        system = WORKER_PROMPT.format(role=role, phase=phase, objective=step["objective"], problem=problem)
        if scratch:
            system += f"\n\n=== SHARED SCRATCHPAD ===\n{scratch}"
        directive = f"Carry out your objective for this step: {step['objective']}"
        contents = build_contents(transcript, directive)
        result = safe_generate_structured(_model_for(capability), system, contents, _WorkerOutput)
        if result is not None and result.content:
            content = result.content
            reasoning = result.reasoning
        else:
            # Fall back to free-form text so the step still contributes.
            content = safe_generate(_model_for(capability), system, contents)
            reasoning = f"Worked on: {step['objective']}"

    # Advance plan bookkeeping (last-write-wins fields).
    new_plan = [dict(s) for s in plan]
    new_plan[cursor]["status"] = "done"

    msg = council_message(
        role=role,
        content=content,
        kind="agent",
        phase=phase,
        reasoning=reasoning,
        extra={"step_id": step["id"]},
    )

    return {
        "plan": new_plan,
        "cursor": cursor + 1,
        "phase": phase,
        "steps_executed": state.get("steps_executed", 0) + 1,
        "messages": [msg],
    }
