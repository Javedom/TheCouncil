"""The Council — Streamlit UI.

Surfaces the full multi-step deliberation live: the dynamic plan, each phase as
it happens, every member's contribution with its reasoning, the Critic's
verdict, and the final synthesized answer.
"""
import os
import uuid

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage

import config
from graph import app
from agents.roles import avatar_for

st.set_page_config(page_title="The Council", page_icon="🏛️", layout="wide")

# --- Session state ----------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []          # list[BaseMessage]
if "plan" not in st.session_state:
    st.session_state.plan = []
if "scratchpad" not in st.session_state:
    st.session_state.scratchpad = ""
if "exchanges" not in st.session_state:
    st.session_state.exchanges = []        # bounded Q/A recap for follow-ups


# --- Rendering helpers -------------------------------------------------------
def render_plan_board(container, plan):
    with container.container():
        if not plan:
            st.caption("No plan yet — pose a problem to convene the Council.")
            return
        icons = {"done": "✅", "active": "▶️", "pending": "⏳", "failed": "⚠️"}
        active_marked = False
        for i, step in enumerate(plan):
            status = step.get("status", "pending")
            # Derive the "currently running" indicator: the first not-yet-done
            # step is shown as active so progress is visible while it runs.
            if status == "pending" and not active_marked:
                status, active_marked = "active", True
            icon = icons.get(status, "⏳")
            st.markdown(
                f"{icon} **{i+1}. {step['role']}**  \n"
                f"<span style='color:gray;font-size:0.85em'>{step['phase']} — {step['objective']}</span>",
                unsafe_allow_html=True,
            )


def render_message(msg):
    """Render a single message (user or any Council role) into the chat feed."""
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
        return

    meta = getattr(msg, "additional_kwargs", {}) or {}
    kind = meta.get("kind", "agent")
    role = meta.get("role") or getattr(msg, "name", "Council")
    reasoning = meta.get("reasoning", "")

    if kind == "plan":
        with st.chat_message("assistant", avatar="🧭"):
            st.markdown("### 🧭 The Chairman convenes the Council")
            st.markdown(msg.content)
        return

    if kind == "final":
        with st.chat_message("assistant", avatar="⚖️"):
            st.markdown("### ⚖️ Final Answer")
            if reasoning:
                with st.expander("🧠 How the Council reconciled this"):
                    st.markdown(reasoning)
            st.markdown(msg.content)
        return

    if kind == "critique":
        with st.chat_message("assistant", avatar="🕵️"):
            st.markdown(f"**🕵️ Critic** · _{meta.get('phase','')}_")
            st.markdown(msg.content)
        return

    if kind == "error":
        with st.chat_message("assistant", avatar="⚠️"):
            st.markdown(f"**{role}** · _{meta.get('phase','')}_")
            st.warning(msg.content)
        return

    # Generic dynamic worker.
    with st.chat_message("assistant", avatar=avatar_for(role)):
        phase = meta.get("phase", "")
        st.markdown(f"**{role}**" + (f" · _{phase}_" if phase else ""))
        if reasoning:
            with st.expander("🧠 Reasoning"):
                st.markdown(reasoning)
        st.markdown(msg.content)


# --- Sidebar ----------------------------------------------------------------
with st.sidebar:
    st.header("🏛️ The Council")
    st.caption(f"Dynamic multi-step deliberation · {config.PRO_MODEL}")
    st.divider()
    st.subheader("📋 Plan board")
    plan_board = st.empty()
    render_plan_board(plan_board, st.session_state.plan)
    st.divider()
    with st.expander("🧠 Shared scratchpad"):
        st.text(st.session_state.scratchpad or "(empty)")
    if st.button("🔄 New session", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.session_state.plan = []
        st.session_state.scratchpad = ""
        st.session_state.exchanges = []
        st.rerun()


# --- Header -----------------------------------------------------------------
st.title("The Council 🏛️")
st.caption(
    "A dynamic, multi-step AI council. A Chairman assembles a problem-specific "
    "team, the members deliberate phase by phase, a Critic gates the quality, "
    "and an Executive delivers the final answer."
)

# Replay history.
for m in st.session_state.history:
    render_message(m)


# --- Handle input -----------------------------------------------------------
if prompt := st.chat_input("State your problem for The Council"):
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        st.error("No GOOGLE_API_KEY (or GEMINI_API_KEY) found. Set it in your environment or a .env file.")
        st.stop()

    user_msg = HumanMessage(content=prompt)
    st.session_state.history.append(user_msg)
    render_message(user_msg)

    # Each problem runs on a FRESH thread so the checkpointer never accumulates a
    # whole prior deliberation. For follow-ups we seed only a short, bounded
    # recap of recent Q/A pairs so context is preserved without token blow-up.
    seed_messages = []
    if st.session_state.exchanges:
        recap = "\n\n".join(
            f"Q: {e['problem']}\nA: {e['answer'][:600]}"
            for e in st.session_state.exchanges[-3:]
        )
        seed_messages.append(HumanMessage(content=f"(Context from earlier in this conversation)\n{recap}"))
    seed_messages.append(user_msg)

    initial_state = {"messages": seed_messages, "problem": prompt, "scratchpad": ""}
    cfg = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": config.RECURSION_LIMIT,
    }

    final_answer = ""
    status = st.status("🏛️ The Council is convening…", expanded=True)
    try:
        for event in app.stream(initial_state, config=cfg, stream_mode="updates"):
            for node_name, update in event.items():
                if not isinstance(update, dict):
                    continue

                # Keep the live plan board / scratchpad in sync.
                if update.get("plan") is not None:
                    st.session_state.plan = update["plan"]
                    render_plan_board(plan_board, st.session_state.plan)
                if update.get("scratchpad"):
                    st.session_state.scratchpad = update["scratchpad"]
                if update.get("final_answer"):
                    final_answer = update["final_answer"]

                for msg in update.get("messages", []) or []:
                    if isinstance(msg, AIMessage):
                        phase = (getattr(msg, "additional_kwargs", {}) or {}).get("phase", "")
                        if phase:
                            status.update(label=f"🏛️ {phase}…", state="running")
                        st.session_state.history.append(msg)
                        render_message(msg)
        status.update(label="✅ The Council has delivered its answer.", state="complete", expanded=False)
        if final_answer:
            st.session_state.exchanges.append({"problem": prompt, "answer": final_answer})
    except Exception as e:  # noqa: BLE001 - surface any unexpected failure to the user
        status.update(label="⚠️ The Council hit an error.", state="error")
        st.error(f"Something went wrong during deliberation: {e}")
