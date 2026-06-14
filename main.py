"""CLI runner for The Council.

Streams the deliberation to the terminal so you can watch the phases, each
member's reasoning and the final answer without the UI.

Usage:
    python main.py "your problem here"
"""
import sys

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage

import config
from graph import app


def run_council(problem: str):
    print(f"\n{'='*70}\nPROBLEM: {problem}\n{'='*70}\n")

    initial_state = {"messages": [HumanMessage(content=problem)], "problem": problem}
    cfg = {
        "configurable": {"thread_id": "cli"},
        "recursion_limit": config.RECURSION_LIMIT,
    }

    for event in app.stream(initial_state, config=cfg, stream_mode="updates"):
        for node_name, update in event.items():
            if not isinstance(update, dict):
                continue
            for msg in update.get("messages", []) or []:
                if not isinstance(msg, AIMessage):
                    continue
                meta = getattr(msg, "additional_kwargs", {}) or {}
                role = meta.get("role", node_name)
                phase = meta.get("phase", "")
                reasoning = meta.get("reasoning", "")
                print(f"\n{'-'*70}")
                print(f"[{phase}] {role}")
                if reasoning:
                    print(f"  🧠 {reasoning}")
                print(f"{'-'*70}")
                print(msg.content)


if __name__ == "__main__":
    problem = " ".join(sys.argv[1:]) or "Design a fair, abuse-resistant rate limiter for a public API."
    run_council(problem)
