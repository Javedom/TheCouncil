import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage
from graph import app  # Your existing graph

st.set_page_config(page_title="The Council", page_icon="🏛️", layout="wide")
st.title("The Council 🏛️")
st.caption("A Multi-Agent System powered by Gemini 3")

import uuid
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memo" not in st.session_state:
    st.session_state.memo = "No notes yet."

# --- SIDEBAR: SHARED MEMORY ---
with st.sidebar:
    st.header("🧠 Shared Memory")
    st.info("The Council maintains a shared memo for key facts, decisions, and constraints.")
    st.text_area("Current Memo", value=st.session_state.memo, height=300, disabled=True)

# Display chat history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        name = getattr(msg, "name", "Assistant")
        avatars = {"Architect": "🏗️", "Skeptic": "🕵️", "Exec": "🏛️", "Researcher": "🔎", "Coder": "💻"}
        icon = avatars.get(name, "🤖")
        with st.chat_message("assistant", avatar=icon):
            st.write(f"**{name}**: {msg.content}")

# Handle New Input
if prompt := st.chat_input("State your problem for The Council"):
    # Add user message to state and display it
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    st.chat_message("user").write(prompt)

    # Run the Council
    with st.spinner("The Council is deliberating..."):
        # Pass existing memo to the graph state
        initial_state = {
            "messages": st.session_state.messages,
            "memo": st.session_state.memo 
        }
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        # Stream the graph
        for event in app.stream(initial_state, config=config):
            for key, value in event.items():
                
                # UPDATE MEMO IF CHANGED
                if "memo" in value:
                    st.session_state.memo = value["memo"]
                
                if "messages" in value:
                    last_msg = value["messages"][-1]
                    if isinstance(last_msg, list): last_msg = last_msg[-1]
                    
                    agent_name = getattr(last_msg, "name", key)
                    content = last_msg.content
                    
                    # IGNORE internal 'Chairman' routing steps
                    if agent_name == "Chairman":
                        continue

                    # Render immediately
                    avatars = {"Architect": "🏗️", "Skeptic": "🕵️", "Exec": "🏛️", "Researcher": "🔎", "Coder": "💻"}
                    icon = avatars.get(agent_name, "🤖")
                    
                    with st.chat_message("assistant", avatar=icon):
                        st.markdown(f"**{agent_name}**")
                        st.write(content)

                    # Append to history
                    st.session_state.messages.append(last_msg)
        
        # Force rerun to update sidebar immediately after processing
        st.rerun()