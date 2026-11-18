"""Smoke test: trainers run end-to-end on SYNTHETIC data and produce loadable models.
Metrics here are meaningless and are written to a temp dir only."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(*args):
    subprocess.run([sys.executable, *map(str, args)], check=True, cwd=ROOT, capture_output=True, text=True)


def test_train_url_and_sms_smoke(tmp_path):
    _run(ROOT / "tests" / "make_synthetic.py", tmp_path)
    _run(ROOT / "training" / "train_url.py", "--csv", tmp_path / "synthetic_urls.csv", "--phish-value", "1",
         "--out", tmp_path / "url.joblib", "--results-dir", tmp_path / "res")
    _run(ROOT / "training" / "train_sms.py", "--uci", tmp_path / "synthetic_sms.tsv",
         "--out", tmp_path / "sms.joblib", "--results-dir", tmp_path / "res")

    from edgeguard.pipeline import EdgeGuard
    eg = EdgeGuard(use_llm=False, url_model=tmp_path / "url.joblib", sms_model=tmp_path / "sms.joblib")
    r = eg.scan_sms("URGENT verify KYC at http://sbi-kyc.xyz/login")
    assert r.evidence.ml_score is not None and r.evidence.sub_evidence[0].ml_score is not None
    assert len(list((tmp_path / "res").glob("*.json"))) == 2
