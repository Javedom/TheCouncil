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

import config
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


def build_graph():
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

    return workflow.compile(checkpointer=MemorySaver())


app = build_graph()
