"""Deterministic explanation used when the LLM is unavailable or its output
fails validation. Guarantees the user always gets a correct, safe message."""

from __future__ import annotations

from edgeguard.schema import Evidence

_OPENERS = {
    ("url", "MALICIOUS"): "This link looks like a phishing or malicious website.",
    ("url", "SUSPICIOUS"): "This link has some warning signs and may not be what it claims to be.",
    ("url", "SAFE"): "This link does not show strong signs of phishing.",
    ("sms", "MALICIOUS"): "This message looks like a scam trying to steal money or personal details.",
    ("sms", "SUSPICIOUS"): "This message has some warning signs of a scam.",
    ("sms", "SAFE"): "This message does not show strong signs of a scam.",
}


def explain(ev: Evidence) -> str:
    opener = _OPENERS.get((ev.channel, ev.label), "Analysis complete.")
    if ev.indicators and ev.label != "SAFE":
        top = "; ".join(i[0].lower() + i[1:] for i in ev.indicators[:3])
        return f"{opener} Main reasons: {top}."
    return opener
