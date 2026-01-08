from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from state import CouncilState
from agents.supervisor import chairman_node
from agents.workers import (
    architect_node, 
    writer_node, 
    skeptic_node, 
    exec_node, 
    researcher_node, 
    coder_node,
    adhoc_node
)

# Initialize the Graph
workflow = StateGraph(CouncilState)

# 1. Add Nodes
workflow.add_node("Chairman", chairman_node)
workflow.add_node("Writer", writer_node)
workflow.add_node("Architect", architect_node)
workflow.add_node("Skeptic", skeptic_node)
workflow.add_node("Exec", exec_node)
workflow.add_node("Researcher", researcher_node)
workflow.add_node("Coder", coder_node)
workflow.add_node("AdHoc", adhoc_node)

# 2. Add Edges
# Entry point acts as the user input handling
workflow.set_entry_point("Chairman")

def route_chairman(state):
    proposed_next = state['next_agent']
    messages = state['messages']
    
    # Helper to see who spoke last
    # We check the last message. If it doesn't have a name, assume it's the User.
    last_msg = messages[-1]
    last_agent = getattr(last_msg, 'name', 'User')

    # Helper to count turns for circuit breaking
    def count_turns(agent_name):
        # 1. Etsi, missä kohtaa on käyttäjän VIIMEISIN viesti
        # Käymme listaa lopusta alkuun
        last_human_idx = 0
        for i, msg in enumerate(reversed(messages)):
            if isinstance(msg, HumanMessage):
                # Koska lista on käännetty, lasketaan oikea indeksi
                last_human_idx = len(messages) - 1 - i
                break
        
        # 2. Rajaa tarkastelu vain uusiin viesteihin (nykyinen "kierros")
        relevant_messages = messages[last_human_idx:]
        
        # 3. Laske osumat tästä rajatusta joukosta
        count = sum(1 for m in relevant_messages if getattr(m, 'name', '') == agent_name)
        
        # (AdHoc-logiikka pysyy samana, mutta käyttää relevant_messages)
        if agent_name == "AdHoc" and count == 0:
             known_agents = {"Architect", "Writer", "Skeptic", "Exec", "Researcher", "Coder", "Chairman", "User"}
             count = sum(1 for m in relevant_messages if getattr(m, 'name', '') not in known_agents)
        
        return count

    # --- RULE 1: THE SANDWICH (Architect -> Skeptic) ---
    # If the Architect just spoke, we ALWAYS force a critique.
    # We do not trust the Chairman to route this; we hardcode the "Sandwich" logic.
    if last_agent == "Architect":
        # Exception: If Skeptic has already yelled too much, let it go to Exec to resolve.
        if count_turns("Skeptic") > 2:
            return "Exec"
        print("--- ROUTING: Architect -> Skeptic (Mandatory Critique) ---")
        return "Skeptic"

    # --- RULE 2: CIRCUIT BREAKERS & LOOP PREVENTION ---
    
    # If the Skeptic just spoke...
    if last_agent == "Skeptic":
        # If the Chairman (LLM) is confused and tries to send it BACK to Skeptic,
        # we intervene and send it to Exec (or the Writer if you prefer) to break the loop.
        if proposed_next == "Skeptic":
            print("--- DETECTED STUTTER: Skeptic -> Skeptic. Forcing Exec. ---")
            return "Exec"
    
    # Standard Circuit Breakers (Max 2 turns per agent)
    # If an agent is proposed but has already spoken 2+ times, force Exec.
    if proposed_next == "Skeptic" and count_turns("Skeptic") >= 2:
        return "Exec"
    
    if proposed_next == "Writer" and count_turns("Writer") >= 2:
        return "Exec"
            
    if proposed_next == "Architect" and count_turns("Architect") >= 2:
        return "Exec"

    if proposed_next == "Researcher" and count_turns("Researcher") >= 2:
        return "Exec"
    
    if proposed_next == "AdHoc" and count_turns("AdHoc") >= 2:
        return "Exec"

    # --- RULE 3: DEFAULT PATH ---
    if proposed_next == "FINISH":
        return END
        
    return proposed_next

# Conditional logic based on Chairman's output
workflow.add_conditional_edges(
    "Chairman",
    route_chairman,
    {
        "Architect": "Architect",
        "Writer": "Writer",
        "Skeptic": "Skeptic",
        "Exec": "Exec",
        "Researcher": "Researcher",
        "Coder": "Coder",
        "AdHoc": "AdHoc",
        
        END: END
    }
)

# Workers always report back to the Chairman to decide the next step
workflow.add_edge("Architect", "Chairman")
workflow.add_edge("Writer", "Chairman")
workflow.add_edge("Skeptic", "Chairman")
workflow.add_edge("Exec", "Chairman")
workflow.add_edge("Researcher", "Chairman")
workflow.add_edge("Coder", "Chairman")
workflow.add_edge("AdHoc", "Chairman")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)