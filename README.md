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

## Configuration

Override any default via environment variables, e.g.:

```bash
export COUNCIL_PRO_MODEL="gemini-3-pro-preview"
export COUNCIL_FLASH_MODEL="gemini-2.5-flash"
export COUNCIL_MAX_STEPS=12
export COUNCIL_MAX_REVISIONS=2
```
