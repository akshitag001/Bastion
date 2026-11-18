"""Bastion local web server (FastAPI). Binds 127.0.0.1 by default; `bastion serve --lan` binds 0.0.0.0.

Flow: POST /api/scan -> detector verdict returned instantly (template explanation)
      GET  /api/explain/{scan_id} -> Server-Sent Events: guarded Llama tokens, then a final event.
The verdict in every response is the detector's; the LLM only contributes guarded explanation text.
Fully offline: no external calls, all frontend assets served from src/bastion/web/.
"""

from __future__ import annotations

import collections
import json
import os
import queue
import re
import threading
import time
import uuid
from pathlib import Path

import psutil
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from bastion import __version__, env
from bastion.history import History
from bastion.paths import MODELS, RESULTS, SAMPLES
from bastion.pipeline import Bastion

MAX_URL = 2048
MAX_SMS = 2000
RATE_PER_MIN = int(os.environ.get("BASTION_RATE_PER_MIN", "60"))
WEB = Path(__file__).resolve().parent / "web"
_URL_ONLY = re.compile(r"^\s*(https?://|www\.)?[^\s]+\.[a-z]{2,}(/\S*)?\s*$", re.I)


class ScanIn(BaseModel):
    text: str = Field(..., min_length=1)
    channel: str | None = Field(None, description="url | sms | None = auto-detect")
    explain: bool = Field(False, description="True = block until the Llama explanation is ready")


def detect_channel(text: str) -> str:
    """A single token that looks like a link is a URL; anything else is treated as a message."""
    return "url" if _URL_ONLY.match(text) and " " not in text.strip() else "sms"


class _RateLimiter:
    def __init__(self, per_min: int):
        self.per_min, self.hits = per_min, collections.defaultdict(collections.deque)
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_min:
                return False
            q.append(now)
            return True


class State:
    """Process-wide state: models loaded once, Llama kept resident, live pipeline stage for the twin page."""

    def __init__(self, use_llm: bool, history: History, preload_llm: bool):
        self.eg = Bastion(use_llm=use_llm)
        self.history = history
        self.env = env.detect()
        self.stage = {"stage": "idle", "channel": None, "verdict": None, "since": time.time()}
        self.pending: collections.OrderedDict[str, object] = collections.OrderedDict()   # scan_id -> Report
        self.lock = threading.Lock()
        self.llm_lock = threading.Lock()
        self.limiter = _RateLimiter(RATE_PER_MIN)
        self.started = time.time()
        if preload_llm and self.eg.llm is not None and self.eg.llm.available()[0] \
                and hasattr(self.eg.llm, "load") and self.eg.llm.backend() in ("python", "runtime"):
            threading.Thread(target=self._preload, daemon=True).start()

    def _preload(self):
        self.set_stage("loading-llama")
        try:
            self.eg.llm.load()
        finally:
            self.set_stage("idle")

    def set_stage(self, stage: str, **kw):
        with self.lock:
            self.stage = {**self.stage, **kw, "stage": stage, "since": time.time()}

    def remember(self, scan_id: str, report) -> None:
        with self.lock:
            self.pending[scan_id] = report
            while len(self.pending) > 256:
                self.pending.popitem(last=False)


def _cgroup_mem() -> tuple[int | None, int | None]:
    def rd(p):
        try:
            v = Path(p).read_text().strip()
            return int(v) if v.isdigit() else None
        except OSError:
            return None
    return rd("/sys/fs/cgroup/memory.current"), rd("/sys/fs/cgroup/memory.max")


def create_app(use_llm: bool = True, history: History | None = None, preload_llm: bool = True) -> FastAPI:
    app = FastAPI(title="Bastion", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    st = State(use_llm, history or History(privacy=os.environ.get("BASTION_PRIVACY") == "1"), preload_llm)
    app.state.bastion = st

    def check_rate(request: Request):
        key = request.client.host if request.client else "local"
        if not st.limiter.allow(key):
            raise HTTPException(429, f"rate limit: {RATE_PER_MIN} scans per minute")

    def run_scan(text: str, channel: str | None, explain: bool) -> dict:
        text = text.strip()
        ch = channel or detect_channel(text)
        if ch not in ("url", "sms"):
            raise HTTPException(422, "channel must be 'url' or 'sms'")
        limit = MAX_URL if ch == "url" else MAX_SMS
        if len(text) > limit:
            raise HTTPException(413, f"{ch} too long: {len(text)} chars (max {limit})")
        st.set_stage("detector", channel=ch, verdict=None)
        report = st.eg.detect(ch, text)
        st.set_stage("fusion", verdict=report.evidence.label)
        scan_id = uuid.uuid4().hex
        if explain:
            st.set_stage("llama")
            with st.llm_lock:
                report = st.eg.explain(report.evidence, report.actions, report.timings_ms)
        st.set_stage("advice")
        st.remember(scan_id, report)
        d = report.to_dict()
        st.history.add(scan_id, d, st.env["environment"])
        st.set_stage("idle")
        llm_ok = st.eg.llm is not None and st.eg.llm.available()[0]
        return {"scan_id": scan_id, "channel": ch, **d, "environment": st.env["environment"],
                "explain_stream": f"/api/explain/{scan_id}" if (llm_ok and not explain) else None}

    # ------------------------------------------------------------------ scans
    @app.post("/api/scan")
    def scan(body: ScanIn, request: Request):
        check_rate(request)
        return run_scan(body.text, body.channel, body.explain)

    @app.post("/api/scan/url")
    def scan_url(body: ScanIn, request: Request):
        check_rate(request)
        return run_scan(body.text, "url", body.explain)

    @app.post("/api/scan/sms")
    def scan_sms(body: ScanIn, request: Request):
        check_rate(request)
        return run_scan(body.text, "sms", body.explain)

    @app.get("/api/explain/{scan_id}")
    def explain_stream(scan_id: str):
        report = st.pending.get(scan_id)
        if report is None:
            raise HTTPException(404, "unknown or expired scan id")
        q: queue.Queue = queue.Queue()

        def work():
            with st.llm_lock:                       # one resident model: one generation at a time
                st.set_stage("llama", verdict=report.evidence.label)
                try:
                    final = st.eg.explain(report.evidence, report.actions, report.timings_ms,
                                          on_token=lambda s: q.put(("token", s)))
                    st.history.update_explanation(scan_id, final.explainer, final.explanation)
                    q.put(("final", {"explanation": final.explanation, "actions": final.actions,
                                     "explainer": final.explainer, "explainer_note": final.explainer_note,
                                     "timings_ms": final.timings_ms, "verdict": final.evidence.label}))
                except Exception as e:              # never leave the client hanging
                    q.put(("final", {"explanation": report.explanation, "actions": report.actions,
                                     "explainer": "template", "explainer_note": f"error: {e}",
                                     "verdict": report.evidence.label}))
                finally:
                    st.set_stage("idle")
                    q.put(None)

        threading.Thread(target=work, daemon=True).start()

        def events():
            yield "retry: 0\n\n"
            while True:
                item = q.get()
                if item is None:
                    return
                kind, data = item
                yield f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ------------------------------------------------------------------ status / metrics
    @app.get("/api/health")
    def health():
        return {"ok": True, "version": __version__, "uptime_s": round(time.time() - st.started, 1)}

    @app.get("/api/status")
    def status():
        llm = st.eg.llm
        llm_info = {"enabled": llm is not None}
        if llm is not None:
            ok, why = llm.available()
            llm_info.update({"available": ok, "detail": why, "backend": llm.backend() if ok else None,
                             "loaded": bool(getattr(llm, "loaded", False)), "load_ms": getattr(llm, "load_ms", None),
                             "model": Path(llm.cfg.model_path).name})
        return {
            "version": __version__, "environment": st.env, "network": env.network_status(),
            "models": {"url": st.eg.url.model_name, "sms": st.eg.sms.model_name,
                       "files": {p.name: p.stat().st_size for p in MODELS.glob("*.joblib")}},
            "llm": llm_info, "llm_stats": st.eg.stats, "privacy_mode": st.history.privacy,
            "limits": {"url_chars": MAX_URL, "sms_chars": MAX_SMS, "scans_per_min": RATE_PER_MIN},
        }

    @app.get("/api/metrics/live")
    def metrics_live():
        proc = psutil.Process()
        cur, mx = _cgroup_mem()
        vm = psutil.virtual_memory()
        with st.lock:
            stage = dict(st.stage)
        return {
            "ts": time.time(), "environment": st.env["environment"],
            "cpu_percent_per_core": psutil.cpu_percent(percpu=True),
            "cpu_limit": st.env["cpu_limit"],
            "mem": {"process_rss": proc.memory_info().rss, "used": cur if cur is not None else vm.used,
                    "limit": mx if mx is not None else vm.total},
            "stage": stage, "network": env.network_status(),
            "llm_loaded": bool(getattr(st.eg.llm, "loaded", False)) if st.eg.llm else False,
        }

    # ------------------------------------------------------------------ history
    @app.get("/api/history")
    def history_list(verdict: str | None = Query(None), limit: int = 200):
        return {"privacy_mode": st.history.privacy, "items": st.history.list(verdict, limit)}

    @app.get("/api/history.csv")
    def history_csv(verdict: str | None = Query(None)):
        return PlainTextResponse(st.history.to_csv(verdict), media_type="text/csv",
                                 headers={"Content-Disposition": "attachment; filename=bastion_history.csv"})

    @app.delete("/api/history")
    def history_clear():
        return {"deleted": st.history.clear()}

    @app.post("/api/settings/privacy")
    def set_privacy(enabled: bool):
        st.history.privacy = bool(enabled)
        return {"privacy_mode": st.history.privacy}

    # ------------------------------------------------------------------ benchmarks / samples
    @app.get("/api/benchmarks")
    def benchmarks():
        out = []
        for p in sorted(RESULTS.glob("*.json")):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            out.append({"file": p.name, "environment": d.get("environment", "untagged (training run)"),
                        "data": d})
        return {"items": out, "timing_notes": env.TIMING_NOTES}

    @app.get("/api/samples")
    def samples():
        def read(name):
            p = SAMPLES / name
            if not p.exists():
                return []
            return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
                    if ln.strip() and not ln.startswith("#")]
        return {"note": "handwritten demo samples, not benchmark data",
                "urls": read("demo_urls.txt"), "sms": read("demo_sms.txt")}

    # ------------------------------------------------------------------ frontend
    if WEB.is_dir():
        app.mount("/static", StaticFiles(directory=WEB), name="static")

        @app.get("/")
        def index():
            return FileResponse(WEB / "index.html")

    @app.exception_handler(HTTPException)
    def http_err(request: Request, exc: HTTPException):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    return app


def serve(host: str = "127.0.0.1", port: int = 8080, use_llm: bool = True) -> None:
    import uvicorn
    uvicorn.run(create_app(use_llm=use_llm), host=host, port=port, log_level="info")
