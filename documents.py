"""Document ingestion + grounding for The Council.

Extracts text from uploaded files and assembles a bounded, relevance-ranked
grounding block to inject into the agents' context. Keeps a hard character
budget so large uploads never blow up the prompt (or the token bill).
"""
import io
import re

GROUNDING_BUDGET = 12000  # max characters of document context to inject
_CHUNK = 1200             # approximate chunk size for relevance ranking


def extract_text(name: str, data: bytes) -> str:
    """Best-effort plain-text extraction for txt/md/pdf. Never raises."""
    lower = (name or "").lower()
    try:
        if lower.endswith(".pdf"):
            try:
                from pypdf import PdfReader
            except Exception:  # noqa: BLE001
                return ""  # pypdf not installed
            reader = PdfReader(io.BytesIO(data))
            return "\n".join((page.extract_text() or "") for page in reader.pages).strip()
        # txt, md, csv, json, code, etc.
        return data.decode("utf-8", errors="ignore").strip()
    except Exception:  # noqa: BLE001
        return ""


def _chunks(text: str):
    paras = re.split(r"\n\s*\n", text)
    buf = ""
    for para in paras:
        if len(buf) + len(para) + 2 <= _CHUNK:
            buf = f"{buf}\n\n{para}" if buf else para
        else:
            if buf:
                yield buf
            buf = para
    if buf:
        yield buf


def build_grounding(docs, problem: str, budget: int = GROUNDING_BUDGET) -> str:
    """Assemble a grounding block from docs, ranked by overlap with the problem.

    `docs` is a list of {"name", "text"}. Returns a labelled, budget-bounded
    string (or "" if there is nothing usable).
    """
    docs = [d for d in (docs or []) if d.get("text")]
    if not docs:
        return ""

    keywords = set(re.findall(r"[a-z0-9]{4,}", (problem or "").lower()))

    # Score every chunk by keyword overlap; keep the best until the budget fills.
    scored = []
    for d in docs:
        for ch in _chunks(d["text"]):
            words = set(re.findall(r"[a-z0-9]{4,}", ch.lower()))
            score = len(keywords & words)
            scored.append((score, d["name"], ch))

    # Highest-scoring first; ties keep document order (stable sort on negated score).
    scored.sort(key=lambda t: -t[0])

    out, used = [], 0
    for _score, name, ch in scored:
        block = f"[{name}]\n{ch}"
        if used + len(block) > budget:
            continue
        out.append(block)
        used += len(block)
    return "\n\n".join(out)
