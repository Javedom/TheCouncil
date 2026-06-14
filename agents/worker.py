"""The generic Worker node.

A single node executes whichever plan step the cursor points at, adopting that
step's dynamic role and capability. Reasoning + content are captured separately
so the UI can show the agent's thinking, and durable facts the step surfaces are
appended to the shared scratchpad for later steps. The cursor and counters
advance here. Steps that produce no model output are marked failed and flagged
(rather than letting an error string masquerade as a real contribution).
"""
from typing import List
from pydantic import BaseModel, Field

import config
from .client import (
    safe_generate,
    generate_reasoned,
    render_transcript,
    build_contents,
)
from .prompts import WORKER_PROMPT, RESEARCH_WORKER_PROMPT
from .events import council_message


class _WorkerOutput(BaseModel):
    reasoning: str
    content: str
    notes: List[str] = Field(default_factory=list)


def _model_for(capability: str) -> str:
    # Single source of truth for model selection across all capabilities.
    if capability in ("reason", "code"):
        return config.REASONING_WORKER_MODEL
    return config.WORKER_MODEL  # research, write, anything else


def _append_notes(scratch: str, notes, role: str) -> str:
    """Append a step's durable notes under the scratchpad's facts section."""
    clean = [n.strip() for n in (notes or []) if n and str(n).strip()]
    if not clean:
        return scratch
    lines = "\n".join(f"- {n}  _(— {role})_" for n in clean)
    return f"{scratch}\n{lines}" if scratch else lines


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
    model = _model_for(capability)

    # Build the brief (research uses a tool-oriented prompt; everything else the
    # standard worker frame). Scratchpad injection is shared across both.
    if capability == "research":
        system = RESEARCH_WORKER_PROMPT.format(role=role, phase=phase, objective=step["objective"], problem=problem)
    else:
        system = WORKER_PROMPT.format(role=role, phase=phase, objective=step["objective"], problem=problem)
    if scratch:
        system += f"\n\n=== SHARED SCRATCHPAD (durable facts from earlier steps) ===\n{scratch}"

    directive = f"Carry out your objective for this step: {step['objective']}"
    contents = build_contents(transcript, directive)

    notes: List[str] = []
    failed = False
    if capability == "research":
        # Tool use cannot be combined with structured JSON output, so research
        # steps use free-form text. Empty result == failure.
        content = safe_generate(model, system, contents, tools=[{"google_search": {}}])
        if content:
            reasoning = "Gathered current, sourced information from the web for the Council."
        else:
            failed = True
            reasoning = "Step failed: the research call returned no output."
    else:
        obj, content, reasoning, ok = generate_reasoned(
            model, system, contents, _WorkerOutput,
            fallback_reasoning=f"Worked on: {step['objective']}",
        )
        if not ok:
            failed = True
            reasoning = "Step failed: the model returned no output."
        elif obj is not None:
            notes = obj.notes

    new_plan = [dict(s) for s in plan]
    if failed:
        new_plan[cursor]["status"] = "failed"
        content = (
            f"_(System: {role} could not produce output for this step — the model "
            f"returned nothing. The Council will proceed without it.)_"
        )
        kind = "error"
    else:
        new_plan[cursor]["status"] = "done"
        kind = "agent"

    updates = {
        "plan": new_plan,
        "cursor": cursor + 1,
        "phase": phase,
        "steps_executed": state.get("steps_executed", 0) + 1,
        "messages": [council_message(
            role=role,
            content=content,
            kind=kind,
            phase=phase,
            reasoning=reasoning,
            extra={"step_id": step["id"], "failed": failed},
        )],
    }

    new_scratch = _append_notes(scratch, notes, role)
    if new_scratch != scratch:
        updates["scratchpad"] = new_scratch
    return updates
