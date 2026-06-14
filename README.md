# The Council 🏛️

A **dynamic, multi-step AI council** that solves a problem by assembling a
problem-specific team of expert agents, deliberating phase by phase, gating the
result through a critic, and synthesizing a final answer — with every phase and
each agent's reasoning surfaced live in the UI.

Built on **LangGraph** + **Google Gemini**, with a **Streamlit** front end.

## How it works

```
planner ─▶ worker ─▶ (more steps?) ─▶ worker ...
              │
              └─(plan done)─▶ critic ─▶ (revise & budget?) ─▶ worker
                                 │
                                 └─(approve / out of budget)─▶ synthesizer ─▶ END
```

1. **Planner (Chairman)** — reads the problem and invents a *dynamic roster* of
   expert roles plus an ordered, phased plan. Different problems get different
   teams. Emits its understanding + success criteria.
2. **Worker** — one generic node executes each plan step in turn, adopting that
   step's role and capability (`reason` / `research` / `code` / `write`).
   Research steps use Gemini's web search. Each worker reports its reasoning and
   its deliverable separately.
3. **Critic** — a red-teamer that checks the work against the success criteria
   and either approves or requests **one bounded revision** (a targeted fix step
   appended to the plan).
4. **Synthesizer (Exec)** — reconciles everything into one clean final answer.

### In-app settings

The sidebar **⚙️ Settings** panel edits config live — no restart, no `.env`:

- **Gemini API key** — each user supplies **their own** key. It is held only in
  context-local, per-session memory (never written to disk, never placed in the
  process environment, and never shared between users), so the app is safe to
  host publicly. Locally, if you don't enter one it falls back to
  `GOOGLE_API_KEY` (unless `COUNCIL_BYOK_ONLY=1`).
- **Per-role models** (🎛️ Models) — set the model for each role independently
  (Planner, Critic, Synthesizer, reasoning/code worker, research/writing
  worker), or "Apply to all roles" at once.
- **Thinking level** (Gemini 3.x) — `minimal`/`low`/`medium`/`high`, or the
  model default. (`minimal` isn't supported on Pro models.)
- **Budgets** — max steps, plan steps, revisions, parallelism, recursion limit.
- **Pricing** (💸) — editable USD/1M-token table that feeds the cost panel.

Defaults track the current Gemini 3.x line (`gemini-3.1-pro-preview`,
`gemini-3.5-flash`). Changes take effect on the next run because nodes read
`config` at execution time. Saving applies process-wide (last save wins across
sessions); `COUNCIL_DB_PATH` is the one setting bound at startup only.

### Document grounding (RAG) & code execution

- **Grounding documents** — upload txt/md/pdf/csv/json in the sidebar. Their
  most relevant content (ranked by overlap with your problem, char-budgeted) is
  injected into the agents' and synthesizer's context so answers are grounded in
  and cite your material.
- **Code execution** — `code`-capability steps use Gemini's built-in,
  sandboxed code execution: the agent writes code, runs it, and the executed
  code + output are captured into the transcript and verified by the Critic.

### Concurrent execution

The planner assigns each step a `depends_on` list. Execution is **wave-based**:
on each turn every step whose dependencies are already satisfied runs
**concurrently** (bounded by `COUNCIL_MAX_PARALLEL`), so independent research
angles or drafts proceed in parallel. Readiness is status-driven (a step runs
once its prerequisites are `done`/`failed`), which also keeps the revision loop
and circuit breakers simple and deadlock-free.

### Human-in-the-loop plan review

Toggle **"✋ Review plan before running"** in the sidebar to pause the Council
after planning. You can edit roles, objectives, phases and capabilities, add or
remove steps, then **Approve & run** (or **Cancel**). Implemented with a
LangGraph `interrupt_after=["planner"]` plus `update_state`, so execution
resumes from the edited plan — which is why the durable checkpointer matters.

### Persistence, export & cost

- **Durable checkpointer** — set `COUNCIL_DB_PATH` to a SQLite file to make
  sessions survive restarts (falls back to in-memory if unset/unavailable).
- **Per-agent cost & usage** — every model call's tokens are tracked; the
  sidebar shows estimated cost broken down by agent (prices in `config.PRICING`).
- **Confidence self-grade** — the Synthesizer scores the answer 0–100 against
  the success criteria; shown as a 🟢/🟡/🔴 badge.
- **Export** — download the full session (plan, contributions, reasoning, final
  answer) as Markdown.

### Robustness

The flow is built to **always terminate with an answer**:

- Deterministic, bounded routing with circuit breakers:
  `MAX_STEPS`, `MAX_PLAN_STEPS`, `MAX_REVISIONS`, `RECURSION_LIMIT`
  (all in `config.py`, env-overridable).
- Every model call goes through `safe_generate` / `safe_generate_structured`
  in `agents/client.py` — they retry transient errors and never raise.
- Structured outputs are validated with Pydantic; on any failure the node falls
  back to a safe default so the graph keeps flowing.

## Project layout

| File | Responsibility |
|------|----------------|
| `config.py` | Models, budgets, phase labels (env-overridable) |
| `state.py` | Shared graph state (plan, cursor, phase, counters, scratchpad) |
| `graph.py` | Node wiring + deterministic, bounded routing |
| `agents/planner.py` | Builds the dynamic team + plan |
| `agents/worker.py` | Executes one plan step in role |
| `agents/critic.py` | Quality gate + bounded revision loop |
| `agents/synthesizer.py` | Final answer |
| `agents/client.py` | Robust Gemini calls + transcript building |
| `agents/prompts.py` | System prompts for the structural roles |
| `agents/roles.py` | Avatars for dynamic roles |
| `agents/events.py` | UI-rich message helper (kind/phase/reasoning) |
| `app.py` | Streamlit UI — live phases, reasoning, plan board |
| `main.py` | CLI runner |

## Running

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Provide a Gemini API key (env var or a `.env` file):
   ```bash
   export GOOGLE_API_KEY="your-key"
   ```
3. Launch the UI:
   ```bash
   streamlit run app.py
   ```
   Or run from the terminal:
   ```bash
   python main.py "Design a fair, abuse-resistant rate limiter for a public API."
   ```

## Deploy to Railway (bring-your-own-key web app)

The app ships ready to deploy on [Railway](https://railway.app) as a public,
multi-user web app where **every visitor uses their own Gemini API key** — no
server-side key, so you never pay for anyone else's usage.

Included deploy files (in this `TheCouncil/` directory):

| File | Purpose |
|------|---------|
| `Procfile` / `railway.json` | Start command + health check for Railway |
| `.streamlit/config.toml` | Headless server config behind Railway's proxy |
| `.python-version` | Pins the Python version for the Nixpacks build |

Steps:

1. Push this repo to GitHub.
2. In Railway: **New Project → Deploy from GitHub repo** and pick it.
3. If the repo root is the parent folder, set the service **Root Directory** to
   `TheCouncil` (Settings → Source) so Railway sees `requirements.txt` and the
   deploy files. Skip this if `TheCouncil/` *is* the repo root.
4. Railway auto-detects Python (Nixpacks), installs `requirements.txt`, and runs
   the start command from `railway.json`/`Procfile`. No environment variables
   are required.
5. Open the generated URL (Settings → Networking → **Generate Domain**), then
   paste your own Gemini key into **⚙️ Settings** in the sidebar.

**Strict bring-your-own-key:** the start command sets `COUNCIL_BYOK_ONLY=1`,
which makes the app ignore any server `GOOGLE_API_KEY` so a user *must* provide
their own key. Each session's key is isolated via a context-local variable (see
`agents/client.py`), including across the worker's parallel threads.

> Note: the durable SQLite checkpointer is **off** by default (Railway's
> filesystem is ephemeral). Leave `COUNCIL_DB_PATH` unset to use in-memory
> sessions. Model/budget/pricing settings are still process-wide (last save
> wins across users); the per-user API key is the part that is fully isolated.

## Configuration

Override any default via environment variables, e.g.:

```bash
export COUNCIL_PRO_MODEL="gemini-3-pro-preview"
export COUNCIL_FLASH_MODEL="gemini-2.5-flash"
export COUNCIL_MAX_STEPS=12
export COUNCIL_MAX_REVISIONS=2
export COUNCIL_DB_PATH="council.db"   # durable, resumable sessions
export COUNCIL_BYOK_ONLY=1            # ignore server keys; users must bring their own
```
