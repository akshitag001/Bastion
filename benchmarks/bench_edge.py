"""Edge benchmark: detector latency, RAM, model size, CPU, throughput, and
(optionally) Llama explainer latency. Run the SAME command on laptop and Pi.

  python benchmarks/bench_edge.py                 # detectors only
  python benchmarks/bench_edge.py --llm --llm-runs 5
Writes benchmarks/results/edge_<machine>_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edgeguard.pipeline import EdgeGuard  # noqa: E402


def pct(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(len(xs) * p))], 4)


def summarize(times):
    return {"runs": len(times), "mean_ms": round(statistics.mean(times), 4),
            "p50_ms": pct(times, 0.50), "p95_ms": pct(times, 0.95), "p99_ms": pct(times, 0.99),
            "throughput_per_s": round(1000 / statistics.mean(times), 1)}


def read_samples(name):
    lines = (ROOT / "data" / "samples" / name).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip() and not l.startswith("#")]


def cpu_temp():
    p = Path("/sys/class/thermal/thermal_zone0/temp")  # Raspberry Pi
    return round(int(p.read_text()) / 1000, 1) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--llm-runs", type=int, default=3)
    a = ap.parse_args()

    proc = psutil.Process(os.getpid())
    rss0 = proc.memory_info().rss
    t0 = time.perf_counter()
    eg = EdgeGuard(use_llm=a.llm)
    load_ms = (time.perf_counter() - t0) * 1000
    rss_loaded = proc.memory_info().rss

    urls, sms = read_samples("demo_urls.txt"), read_samples("demo_sms.txt")
    out = {"machine": {"platform": platform.platform(), "arch": platform.machine(),
                       "cpu_count": psutil.cpu_count(), "ram_total_mb": round(psutil.virtual_memory().total / 2**20),
                       "python": platform.python_version(), "cpu_temp_c_start": cpu_temp()},
           "models": {"url": eg.url.model_name if eg.url.model else "rules-only",
                      "sms": eg.sms.model_name if eg.sms.model else "rules-only"},
           "model_files_bytes": {p.name: p.stat().st_size for p in (ROOT / "models").glob("*.joblib")},
           "load_ms": round(load_ms, 1),
           "rss_mb": {"before_load": round(rss0 / 2**20, 1), "after_load": round(rss_loaded / 2**20, 1)},
           "note": "inputs are handwritten demo samples; latency only, not accuracy"}

    for name, fn, data in (("url_detect", eg.url.analyze, urls), ("sms_detect", eg.sms.analyze, sms)):
        for i in range(50):
            fn(data[i % len(data)])
        proc.cpu_percent(None)
        times = []
        for i in range(a.runs):
            t = time.perf_counter()
            fn(data[i % len(data)])
            times.append((time.perf_counter() - t) * 1000)
        out[name] = summarize(times) | {"cpu_percent_during": proc.cpu_percent(None)}
        print(f"{name}: {out[name]}")

    if a.llm:
        ok, why = eg.llm.available()
        if not ok:
            out["llm"] = {"available": False, "reason": why}
            print(f"LLM not available: {why}")
        else:
            walls, sources = [], []
            for i in range(a.llm_runs):
                r = eg.scan_sms(sms[(i + 2) % len(sms)])
                walls.append(r.timings_ms.get("explain", 0))
                sources.append(r.explainer)
            out["llm"] = {"available": True, "explain_wall_ms": summarize(walls),
                          "explainer_used": sources, "fallbacks": eg.stats["llm_fallbacks"],
                          "note": "wall time includes runner process start + model load per call "
                                  "(v1 subprocess design); see docs for a persistent-runner upgrade"}
            print(f"llm: {out['llm']}")

    out["rss_mb"]["peak_after_bench"] = round(proc.memory_info().rss / 2**20, 1)
    out["machine"]["cpu_temp_c_end"] = cpu_temp()
    res = ROOT / "benchmarks" / "results"
    res.mkdir(parents=True, exist_ok=True)
    path = res / f"edge_{platform.machine()}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
