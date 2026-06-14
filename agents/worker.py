"""The Worker layer — concurrent, dependency-aware execution.

Execution is wave-based instead of a single cursor: on each turn the executor
runs every plan step whose dependencies are already satisfied, in parallel
(bounded by config.MAX_PARALLEL). Steps within a wave are independent, so they
deliberate off the same shared transcript without seeing each other.

Each step adopts its dynamic role and capability, captures reasoning + content
separately, and appends durable facts to the shared scratchpad. Steps that
produce no model output are marked failed and flagged rather than letting an
error string masquerade as a real contribution.
"""
from concurrent.futures import ThreadPoolExecutor
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


def ready_steps(state) -> list:
    """Pending steps whose dependencies have all resolved (done or failed)."""
    plan = state.get("plan", [])
    resolved = {s["id"] for s in plan if s["status"] in ("done", "failed")}
    return [
        s for s in plan
        if s["status"] == "pending" and all(d in resolved for d in s.get("depends_on", []))
    ]


def execute_step(step, problem, transcript, scratch, documents="") -> dict:
    """Run a single plan step. Returns its message, resulting status and notes."""
    role = step["role"]
    phase = step["phase"]
    capability = step["capability"]
    model = _model_for(capability)

    if capability == "research":
        system = RESEARCH_WORKER_PROMPT.format(role=role, phase=phase, objective=step["objective"], problem=problem)
    else:
        system = WORKER_PROMPT.format(role=role, phase=phase, objective=step["objective"], problem=problem)
    if scratch:
        system += f"\n\n=== SHARED SCRATCHPAD (durable facts from earlier steps) ===\n{scratch}"
    if documents:
        system += (
            "\n\n=== USER-PROVIDED DOCUMENTS (ground your work in these; cite them) ===\n"
            f"{documents}"
        )

    directive = f"Carry out your objective for this step: {step['objective']}"
    contents = build_contents(transcript, directive)

    notes: List[str] = []
    failed = False
    if capability in ("research", "code"):
        # Tool use cannot be combined with structured JSON output.
        if capability == "research":
            tools = [{"google_search": {}}]
            ok_reason = "Gathered current, sourced information from the web for the Council."
        else:
            tools = [{"code_execution": {}}]
            ok_reason = "Wrote and ran code in a sandbox to produce and verify the result."
        content = safe_generate(model, system, contents, tools=tools, label=role)
        if content:
            reasoning = ok_reason
        else:
            failed = True
            reasoning = f"Step failed: the {capability} call returned no output."
    else:
        obj, content, reasoning, ok = generate_reasoned(
            model, system, contents, _WorkerOutput,
            fallback_reasoning=f"Worked on: {step['objective']}",
            label=role,
        )
        if not ok:
            failed = True
            reasoning = "Step failed: the model returned no output."
        elif obj is not None:
            notes = obj.notes

    if failed:
        status, kind = "failed", "error"
        content = (
            f"_(System: {role} could not produce output for this step — the model "
            f"returned nothing. The Council will proceed without it.)_"
        )
    else:
        status, kind = "done", "agent"

    msg = council_message(
        role=role, content=content, kind=kind, phase=phase, reasoning=reasoning,
        extra={"step_id": step["id"], "failed": failed},
    )
    return {"step_id": step["id"], "message": msg, "status": status, "notes": notes, "role": role}


def worker_node(state):
    """Execute one wave: all ready steps, concurrently, within the step budget."""
    plan = state.get("plan", [])
    budget = config.MAX_STEPS - state.get("steps_executed", 0)
    ready = ready_steps(state)[:max(0, budget)]
    if not ready:
        return {}  # routing will send us to the Critic

    problem = state.get("problem", "")
    transcript = render_transcript(state.get("messages", []))
    scratch = state.get("scratchpad", "")
    documents = state.get("documents", "")

    if len(ready) == 1:
        results = [execute_step(ready[0], problem, transcript, scratch, documents)]
    else:
        with ThreadPoolExecutor(max_workers=min(config.MAX_PARALLEL, len(ready))) as pool:
            results = list(pool.map(
                lambda s: execute_step(s, problem, transcript, scratch, documents), ready
            ))
    results.sort(key=lambda r: r["step_id"])  # deterministic transcript order

    status_by_id = {r["step_id"]: r["status"] for r in results}
    new_plan = [dict(s) for s in plan]
    for s in new_plan:
        if s["id"] in status_by_id:
            s["status"] = status_by_id[s["id"]]

    new_scratch = scratch
    for r in results:
        new_scratch = _append_notes(new_scratch, r["notes"], r["role"])

    updates = {
        "plan": new_plan,
        "phase": ready[0]["phase"],
        "steps_executed": state.get("steps_executed", 0) + len(results),
        "messages": [r["message"] for r in results],
    }
    if new_scratch != scratch:
        updates["scratchpad"] = new_scratch
    return updates
