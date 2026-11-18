"""SMS cleaning and social-engineering indicator extraction (offline)."""

from __future__ import annotations

import re

URL_RE = re.compile(
    r"((?:https?://|www\.)[^\s<>\"']+|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|in|net|org|xyz|top|ly|io|co|info|link|click|me|tk|ml|ga|cf|gq|online|site|shop|app)(?:/[^\s<>\"']*)?)",
    re.I,
)
PHONE_RE = re.compile(r"(?:\+91[\s-]?)?\b[6-9]\d{9}\b|\b1800[\s-]?\d{3}[\s-]?\d{4}\b")
MONEY_RE = re.compile(r"(?:rs\.?|inr|₹|\$|£)\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*\s?(?:rs|rupees|lakh|crore)\b", re.I)

# (indicator_name, human-readable label, regex)
INDICATORS = [
    ("urgency", "Creates urgency / pressure",
     r"\b(urgent|immediately|within \d+ ?(?:hrs?|hours?|mins?|minutes?)|today only|last chance|expire[sd]?|act now|asap|final (?:notice|warning))\b"),
    ("threat", "Threatens suspension or penalty",
     r"\b(block(?:ed)?|suspend(?:ed)?|deactivat\w*|disconnect\w*|terminat\w*|legal action|penalty|arrest|fine of)\b"),
    ("credential_bait", "Asks for OTP, PIN, password or KYC",
     r"\b(otp|pin|cvv|password|passcode|kyc|aadhaa?r|pan card|verify your|update your|confirm your)\b"),
    ("reward_bait", "Promises prize, cashback or free money",
     r"\b(won|winner|congratulations|prize|lottery|cash ?back|reward|free|gift|bonus|claim|selected)\b"),
    ("financial", "Mentions bank, UPI or payment",
     r"\b(bank|account|a/c|upi|paytm|phonepe|gpay|credit card|debit card|loan|refund|payment|transaction)\b"),
    ("delivery_bait", "Fake delivery / courier / bill notice",
     r"\b(parcel|courier|delivery|shipment|package|electricity bill|power cut|customs)\b"),
    ("call_to_action", "Pushes you to click, call or reply",
     r"\b(click|tap|call now|call us|reply|download|install|visit|open the link)\b"),
    ("remote_access", "Asks to install an app / remote access",
     r"\b(anydesk|teamviewer|quicksupport|apk|install (?:this|the) app)\b"),
]
_COMPILED = [(n, lbl, re.compile(rx, re.I)) for n, lbl, rx in INDICATORS]
INDICATOR_LABELS = {n: lbl for n, lbl, _ in INDICATORS}


def clean(text: str) -> str:
    text = text.replace("​", "").replace("\r", " ")
    return re.sub(r"\s+", " ", text).strip()


def extract_urls(text: str) -> list[str]:
    urls = [m.group(0).rstrip(".,);!?") for m in URL_RE.finditer(text)]
    seen, out = set(), []
    for u in urls:
        if u.lower() not in seen:
            seen.add(u.lower())
            out.append(u)
    return out


def indicators(text: str) -> dict[str, list[str]]:
    """Return {indicator_name: [matched snippets]} for every indicator that fires."""
    t = clean(text)
    hits: dict[str, list[str]] = {}
    for name, _, rx in _COMPILED:
        found = sorted({m.group(0).lower() for m in rx.finditer(t)})
        if found:
            hits[name] = found
    if urls := extract_urls(t):
        hits["contains_url"] = urls
    if phones := PHONE_RE.findall(t):
        hits["contains_phone"] = phones
    if money := MONEY_RE.findall(t):
        hits["mentions_money"] = money
    letters = [c for c in t if c.isalpha()]
    if len(letters) >= 20 and sum(c.isupper() for c in letters) / len(letters) > 0.5:
        hits["shouting"] = ["mostly ALL CAPS"]
    return hits


INDICATOR_LABELS.update({
    "contains_url": "Contains a link",
    "contains_phone": "Contains a phone number",
    "mentions_money": "Mentions a money amount",
    "shouting": "Written mostly in capital letters",
})
