"""Lexical / structural URL features. Pure Python, fully offline.

No DNS, WHOIS or page fetching: everything is computed from the URL string,
so it runs anywhere (including an air-gapped Raspberry Pi) in microseconds.
"""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlsplit

SUSPICIOUS_TLDS = {
    "zip", "mov", "xyz", "top", "tk", "ml", "ga", "cf", "gq", "buzz", "click",
    "link", "work", "rest", "fit", "country", "kim", "loan", "men", "date",
    "racing", "review", "stream", "download", "icu", "cyou", "sbs", "monster",
}

SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "cutt.ly", "rb.gy", "shorturl.at", "tiny.cc", "rebrand.ly", "s.id", "t.ly",
}

SUSPICIOUS_WORDS = [
    "login", "signin", "sign-in", "verify", "verification", "secure", "account",
    "update", "confirm", "banking", "bank", "password", "wallet", "kyc", "upi",
    "otp", "suspend", "unlock", "reward", "prize", "free", "bonus", "refund",
    "webscr", "ebayisapi", "support", "billing", "invoice", "payment",
]

# Brands that phishers commonly imitate (global + India-relevant).
BRANDS = [
    "paypal", "apple", "icloud", "google", "microsoft", "office365", "outlook",
    "amazon", "netflix", "facebook", "instagram", "whatsapp", "linkedin",
    "sbi", "hdfc", "icici", "axis", "kotak", "paytm", "phonepe", "gpay",
    "irctc", "aadhaar", "uidai", "incometax", "epfo", "flipkart", "airtel", "jio",
]

FEATURE_NAMES = [
    "url_len", "host_len", "path_len", "query_len", "n_dots_host",
    "n_subdomains", "n_hyphens", "n_digits", "digit_ratio", "n_special",
    "has_at", "has_ip_host", "has_port", "is_https", "has_punycode",
    "is_shortener", "suspicious_tld", "n_suspicious_words", "brand_in_subdomain_or_path",
    "path_depth", "n_query_params", "host_entropy", "url_entropy",
    "double_slash_in_path", "has_hex_encoding", "longest_token_len",
    "brand_lookalike_domain",
]


def _ensure_scheme(url: str) -> str:
    url = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    return url


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        # decimal / hex encoded IPs like http://3232235777/
        return bool(re.fullmatch(r"(0x[0-9a-f]+|\d{8,10})", host, re.I))


def _registered_domain(host: str) -> str:
    """Naive eTLD+1 (no public-suffix list, so it stays dependency-free).
    Handles common 2-level suffixes like co.in / gov.in / co.uk."""
    parts = host.split(".")
    if len(parts) >= 3 and parts[-2] in {"co", "gov", "ac", "org", "net", "nic", "edu", "com"} and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def parse(url: str) -> dict:
    u = urlsplit(_ensure_scheme(url))
    host = (u.hostname or "").lower()
    return {
        "raw": url.strip(),
        "scheme": u.scheme.lower(),
        "host": host,
        "port": u.port if _safe_port(u) else None,
        "path": u.path or "",
        "query": u.query or "",
        "registered_domain": _registered_domain(host) if host and not _is_ip(host) else host,
        "tld": host.rsplit(".", 1)[-1] if "." in host else "",
        "has_at": "@" in u.netloc,
    }


def _safe_port(u) -> bool:
    try:
        return u.port is not None
    except ValueError:
        return False


def extract(url: str) -> dict[str, float]:
    p = parse(url)
    raw, host, path, query = p["raw"], p["host"], p["path"], p["query"]
    low = raw.lower()
    reg = p["registered_domain"]
    sub_and_path = (host[: -len(reg)] if reg and host.endswith(reg) else host) + path + "?" + query
    tokens = re.split(r"[/\.\?=&\-_:@]+", low)

    f = {
        "url_len": len(raw),
        "host_len": len(host),
        "path_len": len(path),
        "query_len": len(query),
        "n_dots_host": host.count("."),
        "n_subdomains": max(0, host.count(".") - reg.count(".")) if reg else 0,
        "n_hyphens": raw.count("-"),
        "n_digits": sum(c.isdigit() for c in raw),
        "digit_ratio": sum(c.isdigit() for c in raw) / max(1, len(raw)),
        "n_special": sum(raw.count(c) for c in "~%$!*,;+"),
        "has_at": int(p["has_at"]),
        "has_ip_host": int(_is_ip(host)) if host else 0,
        "has_port": int(p["port"] is not None),
        "is_https": int(p["scheme"] == "https"),
        "has_punycode": int("xn--" in host),
        "is_shortener": int(host in SHORTENERS or reg in SHORTENERS),
        "suspicious_tld": int(p["tld"] in SUSPICIOUS_TLDS),
        "n_suspicious_words": sum(w in low for w in SUSPICIOUS_WORDS),
        "brand_in_subdomain_or_path": int(any(b in sub_and_path for b in BRANDS)),
        "path_depth": path.count("/"),
        "n_query_params": len([q for q in query.split("&") if q]),
        "host_entropy": _entropy(host),
        "url_entropy": _entropy(raw),
        "double_slash_in_path": int("//" in path),
        "has_hex_encoding": int(bool(re.search(r"%[0-9a-fA-F]{2}", raw))),
        "longest_token_len": max((len(t) for t in tokens), default=0),
        # brand inside the registered domain, padded with hyphens / bait words / abused TLD
        # e.g. hdfc-netbanking-kyc.top, sbi-update.xyz  (but not hdfcbank.com)
        "brand_lookalike_domain": int(
            any(b in reg for b in BRANDS)
            and ("-" in reg or p["tld"] in SUSPICIOUS_TLDS
                 or any(w in reg for w in SUSPICIOUS_WORDS))
        ),
    }
    return {k: float(f[k]) for k in FEATURE_NAMES}


def vector(url: str) -> list[float]:
    f = extract(url)
    return [f[k] for k in FEATURE_NAMES]
