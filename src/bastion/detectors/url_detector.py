"""Phishing URL detector = rule engine (always on) + optional ML model."""

from __future__ import annotations

import time
from pathlib import Path

from bastion.features import url_features as uf
from bastion.fusion import risk as fusion
from bastion.paths import MODELS
from bastion.schema import Evidence

DEFAULT_MODEL = MODELS / "url_model.joblib"

# (feature, condition, weight, human-readable reason)
RULES = [
    ("has_ip_host",      lambda v: v >= 1, 0.35, "Uses a raw IP address instead of a domain name"),
    ("has_at",           lambda v: v >= 1, 0.30, "Contains '@', which can hide the real destination"),
    ("has_punycode",     lambda v: v >= 1, 0.30, "Uses punycode (look-alike characters) in the domain"),
    ("brand_in_subdomain_or_path", lambda v: v >= 1, 0.30, "Mentions a well-known brand outside its real domain"),
    ("brand_lookalike_domain", lambda v: v >= 1, 0.35, "Domain imitates a well-known brand (brand name plus extra words)"),
    ("suspicious_tld",   lambda v: v >= 1, 0.20, "Uses a top-level domain often abused for phishing"),
    ("is_shortener",     lambda v: v >= 1, 0.20, "Uses a link shortener that hides the real destination"),
    ("n_suspicious_words", lambda v: v >= 2, 0.25, "Contains several phishing keywords (login, verify, kyc...)"),
    ("n_suspicious_words", lambda v: v == 1, 0.10, "Contains a phishing keyword (login, verify, kyc...)"),
    ("n_subdomains",     lambda v: v >= 3, 0.15, "Has an unusually deep chain of subdomains"),
    ("has_port",         lambda v: v >= 1, 0.10, "Uses a non-standard port"),
    ("url_len",          lambda v: v >= 100, 0.10, "Is unusually long"),
    ("n_hyphens",        lambda v: v >= 4, 0.10, "Has many hyphens (common in fake domains)"),
    ("double_slash_in_path", lambda v: v >= 1, 0.10, "Contains a redirect-style '//' in the path"),
    ("has_hex_encoding", lambda v: v >= 1, 0.05, "Uses encoded characters to obscure the URL"),
    ("is_https",         lambda v: v == 0, 0.05, "Is not using HTTPS"),
]


_OMP_LIMIT = None


def _limit_openmp() -> None:
    """Scans are scored one input at a time. Measured on x86 (18 Nov 2026): HistGBDT single-sample predict_proba
    p50 6.2 ms with OpenMP on all cores vs 1.4 ms on one thread (thread-team spin-up dominates). A per-call limit
    costs ~2 ms of overhead, so the limit is set once for the process. ExecuTorch/XNNPACK (Llama) uses its own
    pthreadpool and is not affected. Override with BASTION_DETECTOR_THREADS."""
    global _OMP_LIMIT
    if _OMP_LIMIT is None:
        import os
        from threadpoolctl import threadpool_limits
        _OMP_LIMIT = threadpool_limits(int(os.environ.get("BASTION_DETECTOR_THREADS", "1")), user_api="openmp")


class UrlDetector:
    def __init__(self, model_path: str | Path | None = DEFAULT_MODEL,
                 thresholds: fusion.Thresholds = fusion.DEFAULT):
        self.t = thresholds
        self.model = None
        self.model_name = "rules-only"
        self.view = uf.VIEWS["full"]
        self.threshold = None              # recall-targeted operating point (calibrated prob scale)
        if model_path and Path(model_path).exists():
            import joblib  # only needed when a trained model exists
            bundle = joblib.load(model_path)  # trusted: produced by training/train_url.py
            if bundle.get("feature_names") != uf.FEATURE_NAMES:
                raise ValueError("URL model was trained with a different feature set; retrain it.")
            self.model = bundle["model"]
            self.model_name = bundle.get("name", "ml")
            self.view = uf.VIEWS[bundle.get("view", "full")]  # same URL view as in training
            self.threshold = bundle.get("threshold")
            _limit_openmp()

    def rule_score(self, feats: dict[str, float]) -> tuple[float, list[str]]:
        score, reasons = 0.0, []
        for name, cond, w, reason in RULES:
            if cond(feats[name]):
                score += w
                reasons.append(reason)
        return min(1.0, score), reasons

    def analyze(self, url: str) -> Evidence:
        t0 = time.perf_counter()
        feats = uf.extract(url)
        rule, reasons = self.rule_score(feats)
        ml = None
        if self.model is not None:
            mfeats = feats if self.view is uf.VIEWS["full"] else uf.extract(self.view(url))
            ml = float(self.model.predict_proba([[mfeats[k] for k in uf.FEATURE_NAMES]])[0][1])
        fused = fusion.combine(rule, None if ml is None else fusion.align_ml(ml, self.threshold), self.t)
        risk = int(round(fused * 100))
        label = fusion.label_for(risk, self.t)
        if ml is not None and ml >= (self.threshold or 0.5) and not reasons:
            reasons.append("ML model flags the overall URL structure as phishing-like")
        return Evidence(
            channel="url", input=url[:300], label=label, risk=risk,
            confidence=fusion.confidence_for(fused, self.t, ml, label), indicators=reasons,
            rule_score=round(rule, 3), ml_score=None if ml is None else round(ml, 3),
            model=self.model_name if self.model is not None else "rules-only",
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
        )
