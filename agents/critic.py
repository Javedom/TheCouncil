"""The Critic node — the Council's quality gate.

Judges the team's work against the success criteria and decides whether to
approve (move to synthesis) or request one bounded revision. When it asks for a
revision it appends a targeted fix step to the plan and points the cursor at it.

If the structured assessment cannot be produced, the gate fails *open* (approve)
so the Council still delivers an answer — but it says so visibly rather than
silently waving the work through.
"""
from typing import List, Literal, Optional
from pydantic import BaseModel

import config
from .client import safe_generate_structured, render_transcript, build_contents, bullets
from .prompts import CRITIC_PROMPT
from .events import council_message


class _Critique(BaseModel):
    reasoning: str
    verdict: Literal["approve", "revise"]
    issues: List[str] = []
    revision_role: Optional[str] = None
    revision_objective: Optional[str] = None


def critic_node(state):
    problem = state.get("problem", "")
    criteria_md = bullets(state.get("success_criteria", []))
    transcript = render_transcript(state.get("messages", []))

    system = CRITIC_PROMPT.format(problem=problem, criteria=criteria_md)
    contents = build_contents(transcript, "Evaluate the Council's work and return your structured verdict.")
    critique = safe_generate_structured(config.CRITIC_MODEL, system, contents, _Critique)

    revisions = state.get("revisions", 0)
    budget_left = revisions < config.MAX_REVISIONS and state.get("steps_executed", 0) < config.MAX_STEPS

    gate_skipped = critique is None
    if gate_skipped:
        verdict = "approve"
        reasoning = (
            "⚠️ The quality gate could not run (the Critic returned no valid "
            "assessment); approving the current work so the Council can still "
            "deliver an answer."
        )
        issues: List[str] = []
    else:
        verdict = critique.verdict  # Literal -> already "approve" or "revise"
        reasoning = critique.reasoning
        issues = critique.issues or []

    updates = {"phase": config.PHASE_CRITIQUE}

    if verdict == "revise" and budget_left and not gate_skipped:
        rev_role = critique.revision_role or "Reviser"
        rev_obj = critique.revision_objective or "Address the issues raised by the Critic."
        plan = [dict(s) for s in state.get("plan", [])]
        plan.append({
            "id": len(plan),
            "role": rev_role,
            "objective": rev_obj,
            "phase": config.PHASE_REVISION,
            "capability": "reason",
            "status": "pending",
        })
        updates["plan"] = plan
        # Point the worker at the new step. Only load-bearing on the MAX_STEPS
        # early-exit path; on the normal path the cursor already sits here.
        updates["cursor"] = len(plan) - 1
        updates["revisions"] = revisions + 1
        updates["critic_verdict"] = "revise"
        verdict_label = "↻ Revision requested"
    else:
        updates["critic_verdict"] = "approve"
        verdict_label = "✓ Approved for synthesis"
        if verdict == "revise" and not budget_left:
            reasoning += " (Revision budget exhausted — proceeding with current work.)"

    issues_md = ("\n\n**Issues identified:**\n" + bullets(issues)) if issues else ""
    content = f"**{verdict_label}**\n\n{reasoning}{issues_md}"

    updates["messages"] = [council_message(
        role="Critic",
        content=content,
        kind="critique",
        phase=config.PHASE_CRITIQUE,
        reasoning=reasoning,
        extra={"verdict": updates["critic_verdict"], "issues": issues, "gate_skipped": gate_skipped},
    )]
    return updates
