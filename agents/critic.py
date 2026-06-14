"""The Critic node — the Council's quality gate.

Judges the team's work against the success criteria and decides whether to
approve (move to synthesis) or request one bounded revision. When it asks for a
revision it appends a targeted fix step to the plan and points the cursor at it.
"""
from typing import List, Optional
from pydantic import BaseModel

import config
from .client import safe_generate_structured, render_transcript, build_contents
from .prompts import CRITIC_PROMPT
from .events import council_message


class _Critique(BaseModel):
    reasoning: str
    verdict: str  # "approve" | "revise"
    issues: List[str] = []
    revision_role: Optional[str] = None
    revision_objective: Optional[str] = None


def critic_node(state):
    problem = state.get("problem", "")
    criteria = state.get("success_criteria", []) or []
    criteria_md = "\n".join(f"- {c}" for c in criteria) or "- (none specified)"
    transcript = render_transcript(state.get("messages", []))

    system = CRITIC_PROMPT.format(problem=problem, criteria=criteria_md)
    contents = build_contents(transcript, "Evaluate the Council's work and return your structured verdict.")
    critique = safe_generate_structured(config.CRITIC_MODEL, system, contents, _Critique)

    revisions = state.get("revisions", 0)
    budget_left = revisions < config.MAX_REVISIONS and state.get("steps_executed", 0) < config.MAX_STEPS

    # Default to approve if parsing failed or no budget remains — guarantees progress.
    if critique is None:
        verdict = "approve"
        reasoning = "Could not produce a structured critique; proceeding to synthesis with the work in hand."
        issues: List[str] = []
    else:
        verdict = "revise" if critique.verdict.lower().startswith("rev") else "approve"
        reasoning = critique.reasoning
        issues = critique.issues or []

    updates = {"phase": config.PHASE_CRITIQUE}

    if verdict == "revise" and budget_left and critique is not None:
        rev_role = critique.revision_role or "Reviser"
        rev_obj = critique.revision_objective or "Address the issues raised by the Critic."
        plan = [dict(s) for s in state.get("plan", [])]
        new_step = {
            "id": len(plan),
            "role": rev_role,
            "objective": rev_obj,
            "phase": config.PHASE_REVISION,
            "capability": "reason",
            "status": "pending",
        }
        plan.append(new_step)
        updates["plan"] = plan
        updates["cursor"] = len(plan) - 1  # point the worker at the new step
        updates["revisions"] = revisions + 1
        updates["critic_verdict"] = "revise"
        verdict_label = "↻ Revision requested"
    else:
        updates["critic_verdict"] = "approve"
        verdict_label = "✓ Approved for synthesis"
        if verdict == "revise" and not budget_left:
            reasoning += " (Revision budget exhausted — proceeding with current work.)"

    issues_md = ""
    if issues:
        issues_md = "\n\n**Issues identified:**\n" + "\n".join(f"- {i}" for i in issues)
    content = f"**{verdict_label}**\n\n{reasoning}{issues_md}"

    updates["messages"] = [council_message(
        role="Critic",
        content=content,
        kind="critique",
        phase=config.PHASE_CRITIQUE,
        reasoning=reasoning,
        extra={"verdict": updates["critic_verdict"], "issues": issues},
    )]
    return updates
