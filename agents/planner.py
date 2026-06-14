"""The Planner node — entry point of the Council.

Analyzes the user's problem and produces a dynamic roster + ordered, phased
plan. This is what makes the Council adaptive: the team is invented per problem.
"""
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

import config
from .client import safe_generate_structured, render_transcript, build_contents, bullets
from .prompts import PLANNER_PROMPT
from .events import council_message


class _PlanStep(BaseModel):
    role: str = Field(description="The expert persona for this step")
    objective: str = Field(description="The single clear objective for this step")
    phase: str = Field(description="Named phase this step belongs to")
    capability: Literal["reason", "research", "code", "write"] = "reason"
    depends_on: List[int] = Field(
        default_factory=list,
        description="0-based indices of EARLIER steps that must finish first; "
                    "leave empty for steps that can run in parallel.",
    )


class _Plan(BaseModel):
    understanding: str
    success_criteria: List[str]
    steps: List[_PlanStep]


def _fallback_plan(problem: str) -> _Plan:
    """Used only if structured planning fails — keeps the flow alive."""
    return _Plan(
        understanding=f"Solve the user's request: {problem}",
        success_criteria=["Directly and completely addresses the user's request."],
        steps=[
            _PlanStep(role="Lead Analyst", objective="Analyze the problem and outline the solution approach.", phase="Analysis", capability="reason"),
            _PlanStep(role="Specialist", objective="Produce the core deliverable that solves the problem.", phase="Execution", capability="reason", depends_on=[0]),
        ],
    )


def planner_node(state):
    problem = state.get("problem") or (state["messages"][-1].content if state.get("messages") else "")
    transcript = render_transcript(state.get("messages", []))
    contents = build_contents(transcript, f"Plan the Council's work for this problem:\n{problem}")

    plan = safe_generate_structured(config.PLANNER_MODEL, PLANNER_PROMPT, contents, _Plan, label="Planner")
    if plan is None or not plan.steps:
        plan = _fallback_plan(problem)

    # Enforce the plan-size circuit breaker.
    steps = plan.steps[: config.MAX_PLAN_STEPS]

    plan_steps = []
    for i, s in enumerate(steps):
        # Keep only backward references so the dependency graph is always an
        # acyclic DAG (no cycles, no forward/self deadlocks).
        deps = sorted({d for d in (s.depends_on or []) if 0 <= d < i})
        plan_steps.append({
            "id": i,
            "role": s.role,
            "objective": s.objective,
            "phase": s.phase,
            "capability": s.capability,
            "depends_on": deps,
            "status": "pending",
        })

    # Build a human-readable plan summary for the UI / transcript.
    criteria_md = bullets(plan.success_criteria)
    def _dep_note(s):
        if not s["depends_on"]:
            return ""
        return "  _(after " + ", ".join(f"#{d+1}" for d in s["depends_on"]) + ")_"
    roster_md = "\n".join(
        f"{i+1}. **{s['role']}** — {s['objective']}  _(phase: {s['phase']})_{_dep_note(s)}"
        for i, s in enumerate(plan_steps)
    )
    summary = (
        f"**Understanding**\n{plan.understanding}\n\n"
        f"**Success criteria**\n{criteria_md}\n\n"
        f"**Plan**\n{roster_md}"
    )

    # Seed the shared scratchpad so it is immediately useful; workers append
    # durable facts under "Key facts & decisions" as the work progresses.
    scratchpad = (
        f"## Understanding\n{plan.understanding}\n\n"
        f"## Success criteria\n{criteria_md}\n\n"
        f"## Key facts & decisions"
    )

    msg = council_message(
        role="Chairman",
        content=summary,
        kind="plan",
        phase=config.PHASE_PLAN,
        reasoning="Assembled a problem-specific team and ordered the steps so each builds on the last.",
        extra={
            "understanding": plan.understanding,
            "success_criteria": plan.success_criteria,
            "plan": plan_steps,
        },
    )

    return {
        "problem": problem,
        "understanding": plan.understanding,
        "success_criteria": plan.success_criteria,
        "plan": plan_steps,
        "cursor": 0,
        "phase": config.PHASE_PLAN,
        "steps_executed": 0,
        "revisions": 0,
        "scratchpad": scratchpad,
        "messages": [msg],
    }
