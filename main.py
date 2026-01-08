import os
from dotenv import load_dotenv
load_dotenv() # Load GOOGLE_API_KEY immediately

from langchain_core.messages import HumanMessage
from graph import app

def run_council(user_problem):
    print(f"--- User Problem: {user_problem} ---\n")
    
    initial_state = {"messages": [HumanMessage(content=user_problem)]}
    
    # Stream the events to see the Council thinking
    config = {"configurable": {"thread_id": "cli_test"}}
    for event in app.stream(initial_state, config=config):
        for key, value in event.items():
            if key == "Chairman":
                print(f"[Chairman]: Decided next step -> {value['next_agent']}")
            else:
                # Get the last message from the worker
                last_msg = value['messages'][-1].content
                print(f"\n[{key}]: {last_msg}") # Print full message
                print("-" * 30)

if __name__ == "__main__":
    problem = "I want to you to write a poem that resembles The Howl by Allen Ginsberg."
    run_council(problem)
