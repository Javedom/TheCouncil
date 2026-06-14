"""Shared graph state for The Council.

The state is the single source of truth that flows between every node. It holds
the dynamic plan, an execution cursor, the running transcript, a shared
scratchpad (the Council's working memory) and the bounded counters that keep
the whole flow robust and terminating.
"""
import operator
from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage


class PlanStep(TypedDict):
    """A single, dynamically-assigned unit of work in the Council's plan."""
    id: int
    role: str          # dynamic persona, e.g. "Senior Tax Lawyer"
    objective: str     # what this step must accomplish
    phase: str         # phase label this step belongs to, e.g. "Research"
    capability: str    # "reason" | "research" | "code" | "write"
    status: str        # "pending" | "active" | "done"


class CouncilState(TypedDict, total=False):
    # The running transcript. `operator.add` makes node returns append.
    messages: Annotated[List[BaseMessage], operator.add]

    # The original problem as stated by the user (kept verbatim for grounding).
    problem: str

    # Planner output -------------------------------------------------------
    understanding: str           # restatement of the problem + constraints
    success_criteria: List[str]  # what a good answer must satisfy
    plan: List[PlanStep]         # the dynamic, ordered execution plan

    # Execution bookkeeping ------------------------------------------------
    cursor: int            # index of the next plan step to execute
    phase: str             # current phase label (drives the UI)
    steps_executed: int    # total worker executions (circuit breaker)
    revisions: int         # number of critique->revise loops taken
    critic_verdict: str    # "approve" | "revise"

    # Shared working memory the whole Council reads and writes.
    scratchpad: str

    # The delivered answer.
    final_answer: str
