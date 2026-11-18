"""Deterministic remediation advice. Used directly in the output AND as a
grounding list for the LLM, so advice is always safe even if the LLM fails."""

from __future__ import annotations

from edgeguard.schema import Evidence

_URL = {
    "MALICIOUS": [
        "Do not open this link or enter any details on the page.",
        "If you already entered a password, change it now and enable 2-factor login.",
        "Report it: forward to your bank / the brand, or report at cybercrime.gov.in (or call 1930 in India).",
    ],
    "SUSPICIOUS": [
        "Do not log in through this link; type the official website address yourself.",
        "Check the exact domain spelling before trusting it.",
        "If unsure, ask someone you trust or contact the organisation through its official app.",
    ],
    "SAFE": [
        "No strong warning signs found, but stay alert if the page asks for passwords or payments.",
    ],
}

_SMS = {
    "MALICIOUS": [
        "Do not click links, call numbers or reply to this message.",
        "Never share OTP, PIN, CVV or passwords; banks never ask for them by SMS.",
        "Block the sender and report it (in India: forward to 1909, or report at sancharsaathi.gov.in / cybercrime.gov.in, helpline 1930).",
        "If you shared money or details, call your bank immediately to freeze the account.",
    ],
    "SUSPICIOUS": [
        "Do not act on this message directly; verify through the official app or website.",
        "Do not share OTPs or personal details.",
        "If it claims to be from a company, call its official number (not the one in the SMS).",
    ],
    "SAFE": [
        "No strong scam signals found. Still never share OTPs or passwords with anyone.",
    ],
}


def advice_for(ev: Evidence) -> list[str]:
    table = _SMS if ev.channel == "sms" else _URL
    return list(table.get(ev.label, table["SAFE"]))
