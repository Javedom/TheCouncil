"""The Council — Streamlit UI.

Surfaces the full multi-step deliberation live: the dynamic plan, each phase as
it happens, every member's contribution with its reasoning, the Critic's
verdict, and the final synthesized answer.
"""
import uuid

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage

import config
from graph import app, app_review
from agents.roles import avatar_for
from agents.client import reset_usage, usage_summary, set_api_key, has_api_key
from export import transcript_to_markdown
from documents import extract_text, build_grounding

CAPABILITIES = ["reason", "research", "code", "write"]

# Defaults for the editable settings, seeded from config/env on first load.
DEFAULT_SETTINGS = {
    "reasoning_model": config.PLANNER_MODEL,
    "fast_model": config.WORKER_MODEL,
    "max_steps": config.MAX_STEPS,
    "max_plan_steps": config.MAX_PLAN_STEPS,
    "max_revisions": config.MAX_REVISIONS,
    "max_parallel": config.MAX_PARALLEL,
    "recursion_limit": config.RECURSION_LIMIT,
}


def apply_settings():
    """Push the current session's settings into the live config + client.

    Re-applied every rerun so the process-wide config matches this session.
    """
    s = st.session_state.settings
    config.apply_overrides({
        "PRO_MODEL": s["reasoning_model"],
        "PLANNER_MODEL": s["reasoning_model"],
        "CRITIC_MODEL": s["reasoning_model"],
        "SYNTH_MODEL": s["reasoning_model"],
        "REASONING_WORKER_MODEL": s["reasoning_model"],
        "FLASH_MODEL": s["fast_model"],
        "WORKER_MODEL": s["fast_model"],
        "MAX_STEPS": int(s["max_steps"]),
        "MAX_PLAN_STEPS": int(s["max_plan_steps"]),
        "MAX_REVISIONS": int(s["max_revisions"]),
        "MAX_PARALLEL": int(s["max_parallel"]),
        "RECURSION_LIMIT": int(s["recursion_limit"]),
    })
    if st.session_state.api_key:
        set_api_key(st.session_state.api_key)

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
if "usage" not in st.session_state:
    st.session_state.usage = None          # last run's cost/usage summary
if "last_error" not in st.session_state:
    st.session_state.last_error = None     # persisted across the post-run rerun
if "approval_mode" not in st.session_state:
    st.session_state.approval_mode = False  # review/edit the plan before running
if "pending" not in st.session_state:
    st.session_state.pending = None         # run paused awaiting plan approval
if "doc_files" not in st.session_state:
    st.session_state.doc_files = []         # [{"name", "text"}] grounding docs
if "settings" not in st.session_state:
    st.session_state.settings = dict(DEFAULT_SETTINGS)
if "api_key" not in st.session_state:
    st.session_state.api_key = ""           # in-memory only; never persisted

# Apply this session's editable settings to the live config on every rerun.
apply_settings()


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
            conf = meta.get("confidence")
            badge = ""
            if conf is not None:
                tier = "🟢" if conf >= 75 else ("🟡" if conf >= 50 else "🔴")
                badge = f"  ·  {tier} confidence {conf}%"
            st.markdown(f"### ⚖️ Final Answer{badge}")
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


def stream_and_render(graph, inp, cfg, status):
    """Stream a (resumed) graph run, rendering messages and syncing the UI.

    `inp` is the initial state for a fresh run, or None to resume after a pause.
    Returns the final answer captured during the stream.
    """
    final_answer = ""
    for event in graph.stream(inp, config=cfg, stream_mode="updates"):
        for node_name, update in event.items():
            if not isinstance(update, dict):  # e.g. "__interrupt__" payloads
                continue
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
    return final_answer


def finalize_run(prompt, final_answer):
    if final_answer:
        st.session_state.exchanges.append({"problem": prompt, "answer": final_answer})
    st.session_state.usage = usage_summary()


# --- Sidebar ----------------------------------------------------------------
with st.sidebar:
    st.header("🏛️ The Council")
    st.caption(f"Dynamic multi-step deliberation · {config.PLANNER_MODEL}")
    st.divider()

    # --- Settings (editable config, including API key) ----------------------
    key_ok = has_api_key()
    with st.expander("⚙️ Settings" + ("" if key_ok else " · ⚠️ API key needed"), expanded=not key_ok):
        with st.form("settings_form", border=False):
            api_key = st.text_input(
                "Gemini API key", value=st.session_state.api_key, type="password",
                help="Kept in memory for this session only — never written to disk. "
                     "Overrides GOOGLE_API_KEY for this process.",
                placeholder="Set, or leave blank to use GOOGLE_API_KEY",
            )
            reasoning_model = st.text_input(
                "🧠 Reasoning model", value=st.session_state.settings["reasoning_model"],
                help="Used for planning, critique, synthesis and reasoning/code steps.",
            )
            fast_model = st.text_input(
                "⚡ Fast model", value=st.session_state.settings["fast_model"],
                help="Used for research and writing steps.",
            )
            c1, c2 = st.columns(2)
            max_steps = c1.number_input("Max steps", 1, 50, int(st.session_state.settings["max_steps"]))
            max_plan_steps = c2.number_input("Max plan steps", 1, 20, int(st.session_state.settings["max_plan_steps"]))
            max_revisions = c1.number_input("Max revisions", 0, 10, int(st.session_state.settings["max_revisions"]))
            max_parallel = c2.number_input("Max parallel", 1, 16, int(st.session_state.settings["max_parallel"]))
            recursion_limit = st.number_input("Recursion limit", 10, 500, int(st.session_state.settings["recursion_limit"]))

            cc1, cc2 = st.columns(2)
            if cc1.form_submit_button("💾 Save", use_container_width=True, type="primary"):
                st.session_state.api_key = api_key
                st.session_state.settings.update({
                    "reasoning_model": reasoning_model.strip() or DEFAULT_SETTINGS["reasoning_model"],
                    "fast_model": fast_model.strip() or DEFAULT_SETTINGS["fast_model"],
                    "max_steps": int(max_steps),
                    "max_plan_steps": int(max_plan_steps),
                    "max_revisions": int(max_revisions),
                    "max_parallel": int(max_parallel),
                    "recursion_limit": int(recursion_limit),
                })
                st.rerun()
            if cc2.form_submit_button("↩︎ Reset", use_container_width=True):
                st.session_state.settings = dict(DEFAULT_SETTINGS)
                st.rerun()
        st.caption("Note: model/budget changes apply process-wide (last save wins "
                   "across sessions). `COUNCIL_DB_PATH` is set at startup only.")

    st.divider()
    st.toggle(
        "✋ Review plan before running",
        key="approval_mode",
        help="Pause after planning so you can edit the roster and steps before the Council executes.",
    )
    st.divider()
    st.subheader("📋 Plan board")
    plan_board = st.empty()
    render_plan_board(plan_board, st.session_state.plan)
    st.divider()
    st.subheader("📎 Grounding documents")
    uploads = st.file_uploader(
        "Upload files the Council should ground its work in",
        type=["txt", "md", "pdf", "csv", "json", "py"],
        accept_multiple_files=True,
        help="Text/Markdown/PDF/CSV/JSON. Their relevant content is fed to the agents.",
    )
    if uploads is not None:
        st.session_state.doc_files = [
            {"name": f.name, "text": extract_text(f.name, f.getvalue())}
            for f in uploads
        ]
    usable = [d for d in st.session_state.doc_files if d["text"]]
    if usable:
        st.caption(f"✅ {len(usable)} document(s) loaded · {sum(len(d['text']) for d in usable):,} chars")
    elif uploads:
        st.caption("⚠️ Could not extract text from the upload(s).")

    st.divider()
    with st.expander("🧠 Shared scratchpad"):
        st.text(st.session_state.scratchpad or "(empty)")

    # Cost & usage panel (from the most recent run).
    if st.session_state.usage and st.session_state.usage.get("calls"):
        u = st.session_state.usage
        with st.expander(f"💸 Cost & usage · ~${u['cost']:.4f}", expanded=False):
            c1, c2 = st.columns(2)
            c1.metric("Model calls", u["calls"])
            c2.metric("Total tokens", f"{u['total_tokens']:,}")
            st.caption("Estimated cost per agent (input/output tokens):")
            for label, b in sorted(u["by_label"].items(), key=lambda kv: -kv[1]["cost"]):
                st.markdown(
                    f"- **{label}** — ~${b['cost']:.4f}  "
                    f"<span style='color:gray;font-size:0.85em'>"
                    f"({b['input']:,} in / {b['output']:,} out, {b['calls']} call(s))</span>",
                    unsafe_allow_html=True,
                )
            st.caption("Prices are estimates; configure in config.PRICING.")

    # Export the session transcript.
    if st.session_state.history:
        conf = next(
            (m.additional_kwargs.get("confidence")
             for m in reversed(st.session_state.history)
             if isinstance(m, AIMessage) and (m.additional_kwargs or {}).get("kind") == "final"),
            None,
        )
        st.download_button(
            "⬇️ Export transcript (Markdown)",
            data=transcript_to_markdown(st.session_state.history, confidence=conf),
            file_name="council-session.md",
            mime="text/markdown",
            use_container_width=True,
        )

    if st.button("🔄 New session", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.session_state.plan = []
        st.session_state.scratchpad = ""
        st.session_state.exchanges = []
        st.session_state.usage = None
        st.session_state.pending = None
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

# Surface an error from the previous run (persisted across the post-run rerun).
if st.session_state.last_error:
    st.error(st.session_state.last_error)


# --- Pending plan review (human-in-the-loop) --------------------------------
if st.session_state.pending:
    p = st.session_state.pending
    cfg = {"configurable": {"thread_id": p["thread_id"]}, "recursion_limit": config.RECURSION_LIMIT}

    st.info("✋ **Review the plan.** Edit roles, objectives, phases or capabilities; "
            "add or remove steps; then approve to convene the Council.")
    editable = [
        {"role": s["role"], "objective": s["objective"], "phase": s["phase"], "capability": s["capability"]}
        for s in p["plan"]
    ]
    edited = st.data_editor(
        editable,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "role": st.column_config.TextColumn("Role", required=True),
            "objective": st.column_config.TextColumn("Objective", width="large", required=True),
            "phase": st.column_config.TextColumn("Phase"),
            "capability": st.column_config.SelectboxColumn("Capability", options=CAPABILITIES, default="reason"),
        },
        key="plan_editor",
    )

    c1, c2 = st.columns(2)
    if c1.button("▶️ Approve & run", type="primary", use_container_width=True):
        # Rebuild a clean plan from the edited rows.
        new_plan = []
        for row in edited:
            role = (row.get("role") or "").strip()
            objective = (row.get("objective") or "").strip()
            if not role or not objective:
                continue
            idx = len(new_plan)
            new_plan.append({
                "id": idx,
                "role": role,
                "objective": objective,
                "phase": (row.get("phase") or "Execution").strip(),
                "capability": row.get("capability") or "reason",
                # Hand-edited plans run sequentially (each after the previous).
                "depends_on": [idx - 1] if idx > 0 else [],
                "status": "pending",
            })
        app_review.update_state(cfg, {"plan": new_plan, "cursor": 0})
        st.session_state.plan = new_plan
        st.session_state.pending = None

        reset_usage()
        status = st.status("🏛️ The Council is deliberating…", expanded=True)
        try:
            final_answer = stream_and_render(app_review, None, cfg, status)
            status.update(label="✅ The Council has delivered its answer.", state="complete", expanded=False)
            finalize_run(p["prompt"], final_answer)
        except Exception as e:  # noqa: BLE001
            status.update(label="⚠️ The Council hit an error.", state="error")
            st.session_state.last_error = f"Something went wrong during deliberation: {e}"
            st.session_state.usage = usage_summary()
        st.rerun()

    if c2.button("✖️ Cancel", use_container_width=True):
        st.session_state.pending = None
        st.rerun()


# --- Handle input -----------------------------------------------------------
if prompt := st.chat_input("State your problem for The Council", disabled=bool(st.session_state.pending)):
    if not has_api_key():
        st.error("No Gemini API key. Add one under **⚙️ Settings** in the sidebar "
                 "(or set GOOGLE_API_KEY in your environment).")
        st.stop()

    st.session_state.last_error = None
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

    thread_id = str(uuid.uuid4())
    documents = build_grounding(st.session_state.doc_files, prompt)
    initial_state = {
        "messages": seed_messages,
        "problem": prompt,
        "scratchpad": "",
        "documents": documents,
    }
    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": config.RECURSION_LIMIT}

    reset_usage()
    if st.session_state.approval_mode:
        # Run only as far as the plan, then pause for the user to review/edit.
        status = st.status("🧭 Drafting the plan…", expanded=True)
        try:
            stream_and_render(app_review, initial_state, cfg, status)
            snap = app_review.get_state(cfg)
            if snap.next:  # paused after the planner
                st.session_state.pending = {"thread_id": thread_id, "prompt": prompt, "plan": snap.values.get("plan", [])}
                status.update(label="✋ Plan ready — review it below.", state="complete", expanded=False)
            else:  # nothing to pause on; treat as finished
                finalize_run(prompt, snap.values.get("final_answer", ""))
        except Exception as e:  # noqa: BLE001
            status.update(label="⚠️ The Council hit an error.", state="error")
            st.session_state.last_error = f"Something went wrong while planning: {e}"
        st.session_state.usage = usage_summary()
        st.rerun()
    else:
        status = st.status("🏛️ The Council is convening…", expanded=True)
        try:
            final_answer = stream_and_render(app, initial_state, cfg, status)
            status.update(label="✅ The Council has delivered its answer.", state="complete", expanded=False)
            finalize_run(prompt, final_answer)
        except Exception as e:  # noqa: BLE001 - surface any unexpected failure to the user
            status.update(label="⚠️ The Council hit an error.", state="error")
            st.session_state.last_error = f"Something went wrong during deliberation: {e}"
            st.session_state.usage = usage_summary()
        st.rerun()
