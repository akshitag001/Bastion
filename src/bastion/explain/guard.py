"""Verdict lock + LLM output validation.

The LLM can only ever contribute explanation TEXT. Its output is rejected if it:
  - is empty / too long / not in the expected format,
  - contradicts the detector's verdict (e.g. says "safe" for MALICIOUS),
  - invents URLs or phone numbers that were not in the input.
Rejected output -> deterministic template (and the event is counted).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bastion.schema import Evidence

MAX_EXPLANATION_CHARS = 600
_URLISH = re.compile(r"(https?://\S+|www\.\S+|\b[a-z0-9-]+\.(?:com|in|net|org|xyz|ly|io)\b)", re.I)
_PHONE = re.compile(r"\+?\d[\d\s-]{8,}\d")

_LINK = r"(is|looks|seems|appears)( to be| like)?( a| an)?( completely| totally| perfectly)?"
_SAFE_CLAIMS = re.compile(
    rf"\b{_LINK} (safe|legitimate|legit|genuine|harmless|trustworthy|real|official)\b|"
    r"\bnot (a |an )?(scam|phishing|malicious|dangerous|fraud)\b|\bno (risk|threat|danger)\b|"
    r"\bnothing to worry\b|\bdon'?t worry\b",
    re.I,
)
_DANGER_CLAIMS = re.compile(rf"\b{_LINK} (scam|phishing|malicious|dangerous|fraud)", re.I)


@dataclass
class GuardResult:
    ok: bool
    explanation: str = ""
    actions: list[str] = field(default_factory=list)
    reason: str = ""


def parse(text: str) -> tuple[str, list[str]]:
    text = text.strip()
    m = re.search(r"EXPLANATION:\s*(.*?)(?:\n\s*ACTIONS:|$)", text, re.S | re.I)
    explanation = (m.group(1) if m else text.split("ACTIONS:")[0]).strip()
    actions = []
    a = re.search(r"ACTIONS:\s*(.*)", text, re.S | re.I)
    if a:
        actions = [ln.strip(" -*•\t") for ln in a.group(1).splitlines() if ln.strip(" -*•\t")]
    return re.sub(r"\s+", " ", explanation), actions[:4]


class GuardStop(Exception):
    """Raised from a streaming token callback to abort generation as soon as the partial
    output violates the guard. The pipeline treats it like a rejection (template fallback)."""


def partial_violation(text: str, ev: Evidence) -> str | None:
    """Checks that are safe to run on INCOMPLETE output while it streams (so unsafe text never reaches
    the UI). Length/format checks only make sense on the final text and stay in validate()."""
    if ev.label in ("MALICIOUS", "SUSPICIOUS") and _SAFE_CLAIMS.search(text):
        return "contradicts verdict (claims safe)"
    if ev.label == "SAFE" and _DANGER_CLAIMS.search(text):
        return "contradicts verdict (claims dangerous)"
    src = ev.input.lower()
    # only judge links/numbers that are complete (followed by whitespace), not a half-streamed token
    for u in _URLISH.findall(text.rsplit(None, 1)[0] if " " in text.strip() else ""):
        if u.lower().rstrip(".,") not in src and not u.lower().rstrip(".,").endswith("gov.in"):
            return f"invented link: {u}"
    for p in _PHONE.findall(text):
        digits = re.sub(r"\D", "", p)
        if len(digits) >= 10 and digits not in re.sub(r"\D", "", src) and digits not in {"1930", "1909"}:
            return f"invented phone number: {p}"
    return None


def validate(raw: str, ev: Evidence) -> GuardResult:
    if not raw or not raw.strip():
        return GuardResult(False, reason="empty output")
    explanation, actions = parse(raw)
    if len(explanation) < 20:
        return GuardResult(False, reason="explanation too short / bad format")
    if len(explanation) > MAX_EXPLANATION_CHARS:
        explanation = explanation[:MAX_EXPLANATION_CHARS].rsplit(" ", 1)[0] + "..."

    body = explanation + " " + " ".join(actions)
    if ev.label in ("MALICIOUS", "SUSPICIOUS") and _SAFE_CLAIMS.search(body):
        return GuardResult(False, reason="contradicts verdict (claims safe)")
    if ev.label == "SAFE" and _DANGER_CLAIMS.search(body):
        return GuardResult(False, reason="contradicts verdict (claims dangerous)")

    src = ev.input.lower()
    for u in _URLISH.findall(body):
        if u.lower().rstrip(".,") not in src and not u.lower().rstrip(".,").endswith("gov.in"):
            return GuardResult(False, reason=f"invented link: {u}")
    for p in _PHONE.findall(body):
        digits = re.sub(r"\D", "", p)
        if digits not in re.sub(r"\D", "", src) and digits not in {"1930", "1909"}:
            return GuardResult(False, reason=f"invented phone number: {p}")

    return GuardResult(True, explanation=explanation, actions=actions)
