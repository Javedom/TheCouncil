"""Central configuration for The Council.

Everything tunable lives here so models, budgets and behaviour can be changed
in one place (and overridden via environment variables) without touching the
graph or agent logic.
"""
import os


def _env(name: str, default: str) -> str:
    val = os.environ.get(name)
    return val if val else default


# --- Models -----------------------------------------------------------------
# The Council runs on Google Gemini. "Pro" handles reasoning-heavy roles
# (planning, critique, synthesis); "Flash" handles fast execution work.
PRO_MODEL = _env("COUNCIL_PRO_MODEL", "gemini-3-pro-preview")
FLASH_MODEL = _env("COUNCIL_FLASH_MODEL", "gemini-2.5-flash")

PLANNER_MODEL = _env("COUNCIL_PLANNER_MODEL", PRO_MODEL)
CRITIC_MODEL = _env("COUNCIL_CRITIC_MODEL", PRO_MODEL)
SYNTH_MODEL = _env("COUNCIL_SYNTH_MODEL", PRO_MODEL)
WORKER_MODEL = _env("COUNCIL_WORKER_MODEL", FLASH_MODEL)
# Reasoning/analysis steps benefit from the stronger model.
REASONING_WORKER_MODEL = _env("COUNCIL_REASONING_MODEL", PRO_MODEL)


# --- Budgets / circuit breakers --------------------------------------------
# Hard caps that guarantee the flow always terminates with an answer.
MAX_STEPS = int(_env("COUNCIL_MAX_STEPS", "12"))        # total worker executions
MAX_PLAN_STEPS = int(_env("COUNCIL_MAX_PLAN_STEPS", "6"))  # steps the planner may emit
MAX_REVISIONS = int(_env("COUNCIL_MAX_REVISIONS", "2"))  # critique -> revise loops
RECURSION_LIMIT = int(_env("COUNCIL_RECURSION_LIMIT", "60"))  # LangGraph safety net

# Retries for transient model/transport errors.
MAX_RETRIES = int(_env("COUNCIL_MAX_RETRIES", "2"))


# --- Persistence ------------------------------------------------------------
# Path to a SQLite file for a durable, resumable checkpointer. Empty => use the
# in-memory checkpointer (sessions are lost on restart).
DB_PATH = _env("COUNCIL_DB_PATH", "")


# --- Cost estimates ---------------------------------------------------------
# Approximate USD per 1,000,000 tokens as (input, output). These are estimates
# for the in-app cost panel only — override per model via env or edit here.
PRICING = {
    "gemini-3-pro-preview": (2.00, 12.00),
    "gemini-2.5-flash": (0.30, 2.50),
}
DEFAULT_PRICING = (1.00, 4.00)


def price_for(model: str):
    return PRICING.get(model, DEFAULT_PRICING)


# --- Phase labels -----------------------------------------------------------
PHASE_PLAN = "Planning"
PHASE_CRITIQUE = "Critique"
PHASE_REVISION = "Revision"
PHASE_SYNTHESIS = "Synthesis"
