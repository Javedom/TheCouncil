"""The Planner node — entry point of the Council.

Analyzes the user's problem and produces a dynamic roster + ordered, phased
plan. This is what makes the Council adaptive: the team is invented per problem.
"""
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

import config
from .client import safe_generate_structured, render_transcript, build_contents
from .prompts import PLANNER_PROMPT
from .events import council_message


class _PlanStep(BaseModel):
    role: str = Field(description="The expert persona for this step")
    objective: str = Field(description="The single clear objective for this step")
    phase: str = Field(description="Named phase this step belongs to")
    capability: Literal["reason", "research", "code", "write"] = "reason"


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
            _PlanStep(role="Specialist", objective="Produce the core deliverable that solves the problem.", phase="Execution", capability="reason"),
        ],
    )


def planner_node(state):
    problem = state.get("problem") or (state["messages"][-1].content if state.get("messages") else "")
    transcript = render_transcript(state.get("messages", []))
    contents = build_contents(transcript, f"Plan the Council's work for this problem:\n{problem}")

    plan = safe_generate_structured(config.PLANNER_MODEL, PLANNER_PROMPT, contents, _Plan)
    if plan is None or not plan.steps:
        plan = _fallback_plan(problem)

    # Enforce the plan-size circuit breaker.
    steps = plan.steps[: config.MAX_PLAN_STEPS]

    plan_steps = []
    for i, s in enumerate(steps):
        plan_steps.append({
            "id": i,
            "role": s.role,
            "objective": s.objective,
            "phase": s.phase,
            "capability": s.capability,
            "status": "pending",
        })

    # Build a human-readable plan summary for the UI / transcript.
    criteria_md = "\n".join(f"- {c}" for c in plan.success_criteria) or "- (none specified)"
    roster_md = "\n".join(
        f"{i+1}. **{s['role']}** — {s['objective']}  _(phase: {s['phase']})_"
        for i, s in enumerate(plan_steps)
    )
    summary = (
        f"**Understanding**\n{plan.understanding}\n\n"
        f"**Success criteria**\n{criteria_md}\n\n"
        f"**Plan**\n{roster_md}"
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
        "scratchpad": state.get("scratchpad", ""),
        "messages": [msg],
    }
