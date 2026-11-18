"""Command-line interface.

  bastion scan-url "http://..."
  bastion scan-sms "Your account will be blocked..."
  bastion demo                 # runs the handwritten demo samples
  bastion status               # shows which models / Llama runtime are available
Add --json for machine-readable output, --no-llm to skip the Llama explainer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bastion.pipeline import Bastion, Report

COLORS = {"SAFE": "\033[92m", "SUSPICIOUS": "\033[93m", "MALICIOUS": "\033[91m"}
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
from bastion.paths import ROOT  # noqa: E402


def _print(report: Report, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        return
    ev = report.evidence
    color = COLORS.get(ev.label, "") if sys.stdout.isatty() else ""
    reset = RESET if color else ""
    print(f"\n{BOLD if color else ''}[{ev.channel.upper()}]{reset} {ev.input[:90]}")
    print(f"  Verdict : {color}{ev.label}{reset}   risk {ev.risk}/100   confidence {ev.confidence}")
    print(f"  Model   : {ev.model}  (rule={ev.rule_score}, ml={ev.ml_score})")
    if ev.indicators:
        print("  Why     :")
        for i in ev.indicators[:6]:
            print(f"    - {i}")
    print(f"  Explain : {report.explanation}")
    print("  Do this :")
    for a in report.actions:
        print(f"    - {a}")
    t = ", ".join(f"{k} {v} ms" for k, v in report.timings_ms.items())
    print(f"  {DIM if color else ''}[{report.explainer}; {report.explainer_note}; {t}]{reset}")


def _status(eg: Bastion) -> None:
    from bastion import env
    d, n = env.detect(), env.network_status()
    print(f"Environment  : {d['environment']}  ({d['reason']})")
    print(f"  {d['timing_note']}")
    print(f"Network      : {'OFFLINE (no route)' if n['offline'] else 'online'}")
    print(f"URL detector : {eg.url.model_name if eg.url.model else 'rules-only (train with training/train_url.py)'}")
    print(f"SMS detector : {eg.sms.model_name if eg.sms.model else 'rules-only (train with training/train_sms.py)'}")
    if eg.llm:
        ok, why = eg.llm.available()
        print(f"Llama (ExecuTorch): {'READY' if ok else 'not ready - ' + why}")
        print(f"  runner={eg.llm.cfg.runner}\n  pte={eg.llm.cfg.model_path}\n  tokenizer={eg.llm.cfg.tokenizer_path}")


def _demo(eg: Bastion, as_json: bool, explain: bool) -> None:
    samples = ROOT / "data" / "samples"
    print(f"{BOLD}Demo samples are HANDWRITTEN for demonstration only - not benchmark data.{RESET}")
    for line in (samples / "demo_urls.txt").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            _print(eg.scan_url(line.strip(), explain), as_json)
    for line in (samples / "demo_sms.txt").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            _print(eg.scan_sms(line.strip(), explain), as_json)
    if eg.stats["llm_calls"]:
        print(f"\nLLM calls: {eg.stats['llm_calls']}, guard fallbacks: {eg.stats['llm_fallbacks']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bastion", description="Offline on-device phishing & scam detector")
    ap.add_argument("--json", action="store_true", help="print JSON output")
    ap.add_argument("--no-llm", action="store_true", help="skip Llama explainer (template explanation)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scan-url"); p.add_argument("url")
    p = sub.add_parser("scan-sms"); p.add_argument("text")
    sub.add_parser("demo")
    sub.add_parser("status")
    a = ap.parse_args(argv)

    eg = Bastion(use_llm=not a.no_llm)
    if a.cmd == "scan-url":
        _print(eg.scan_url(a.url), a.json)
    elif a.cmd == "scan-sms":
        _print(eg.scan_sms(a.text), a.json)
    elif a.cmd == "demo":
        _demo(eg, a.json, explain=True)
    elif a.cmd == "status":
        _status(eg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
