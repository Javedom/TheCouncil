"""Helpers for presenting dynamic roles in the UI.

The roster is generated per-problem by the Planner, so roles are open-ended.
These helpers give every role a stable, friendly avatar based on keywords,
with a deterministic fallback so the same role always renders the same way.
"""

# Fixed icons for the structural roles the graph always uses.
SYSTEM_AVATARS = {
    "Chairman": "🏛️",
    "Planner": "🧭",
    "Critic": "🕵️",
    "Skeptic": "🕵️",
    "Exec": "⚖️",
    "Synthesizer": "⚖️",
}

# Keyword -> avatar, matched against the (lowercased) dynamic role name.
_KEYWORD_AVATARS = [
    (("architect", "design", "system"), "🏗️"),
    (("research", "analyst", "investig", "data"), "🔎"),
    (("engineer", "coder", "developer", "program"), "💻"),
    (("writer", "author", "copy", "poet", "editor"), "✍️"),
    (("lawyer", "legal", "counsel", "compliance"), "⚖️"),
    (("finance", "account", "tax", "economist"), "💰"),
    (("doctor", "medical", "clinic", "health"), "🩺"),
    (("security", "risk", "audit", "threat"), "🛡️"),
    (("market", "brand", "growth", "sales"), "📈"),
    (("scientist", "physic", "chem", "bio", "math"), "🔬"),
    (("teacher", "tutor", "educat", "professor"), "📚"),
    (("strateg", "consult", "advisor", "manager"), "📋"),
    (("product", "ux", "user"), "🧩"),
]

_FALLBACK_POOL = ["🧠", "🎓", "🛠️", "🗂️", "💡", "🔧", "🧪", "🗣️", "🤝", "🎯"]


def avatar_for(role: str) -> str:
    if not role:
        return "🤖"
    if role in SYSTEM_AVATARS:
        return SYSTEM_AVATARS[role]
    low = role.lower()
    for keywords, icon in _KEYWORD_AVATARS:
        if any(k in low for k in keywords):
            return icon
    return _FALLBACK_POOL[sum(ord(c) for c in role) % len(_FALLBACK_POOL)]
