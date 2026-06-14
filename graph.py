"""The Council graph.

A robust, dynamic, multi-step flow:

    planner ─▶ worker ─▶ (more steps?) ─▶ worker ...
                  │
                  └─(plan done)─▶ critic ─▶ (revise & budget?) ─▶ worker
                                     │
                                     └─(approve / out of budget)─▶ synthesizer ─▶ END

The Planner builds a problem-specific team and ordered plan. A single generic
Worker executes each step in turn. The Critic gates quality and may trigger one
bounded revision loop. The Synthesizer always delivers a final answer.

All routing is deterministic and bounded (MAX_STEPS / MAX_REVISIONS /
RECURSION_LIMIT), so the flow cannot loop forever and always terminates.
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

import config  # noqa: E402
from state import CouncilState
from agents.planner import planner_node
from agents.worker import worker_node
from agents.critic import critic_node
from agents.synthesizer import synthesizer_node


def route_after_planner(state):
    return "worker" if state.get("plan") else "synthesizer"


def route_after_worker(state):
    # Circuit breaker: never exceed the global step budget.
    if state.get("steps_executed", 0) >= config.MAX_STEPS:
        return "critic"
    if state.get("cursor", 0) < len(state.get("plan", [])):
        return "worker"
    return "critic"


def route_after_critic(state):
    if state.get("critic_verdict") == "revise" and state.get("steps_executed", 0) < config.MAX_STEPS:
        return "worker"
    return "synthesizer"


def _make_checkpointer():
    """A durable SQLite checkpointer when COUNCIL_DB_PATH is set, else in-memory.

    Falls back to MemorySaver on any error so the app never fails to start.
    """
    if config.DB_PATH:
        try:
            import sqlite3
            from langgraph.checkpoint.sqlite import SqliteSaver
            conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
            return SqliteSaver(conn)
        except Exception as e:  # noqa: BLE001
            print(f"[Council] Durable checkpointer unavailable ({e}); using in-memory.")
    return MemorySaver()


def build_graph(interrupt_after=None):
    workflow = StateGraph(CouncilState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("worker", worker_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("synthesizer", synthesizer_node)

    workflow.set_entry_point("planner")

    workflow.add_conditional_edges("planner", route_after_planner, {
        "worker": "worker",
        "synthesizer": "synthesizer",
    })
    workflow.add_conditional_edges("worker", route_after_worker, {
        "worker": "worker",
        "critic": "critic",
    })
    workflow.add_conditional_edges("critic", route_after_critic, {
        "worker": "worker",
        "synthesizer": "synthesizer",
    })
    workflow.add_edge("synthesizer", END)

    return workflow.compile(
        checkpointer=_make_checkpointer(),
        interrupt_after=interrupt_after or [],
    )


# Default graph runs end-to-end. `app_review` pauses after planning so the user
# can review/edit the plan (human-in-the-loop) before execution begins.
app = build_graph()
app_review = build_graph(interrupt_after=["planner"])
