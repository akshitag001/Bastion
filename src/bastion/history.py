"""Local scan history in SQLite (default ~/.bastion/history.db; override with BASTION_HISTORY_DB).

Privacy mode stores only a SHA-256 of the scanned text (plus verdict/risk/indicators), never the text itself.
Nothing here ever leaves the device.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
  id          TEXT PRIMARY KEY,
  ts          REAL NOT NULL,
  channel     TEXT NOT NULL,
  input       TEXT,              -- NULL in privacy mode
  input_sha256 TEXT NOT NULL,
  verdict     TEXT NOT NULL,
  risk        INTEGER NOT NULL,
  confidence  REAL,
  indicators  TEXT,              -- JSON list
  explainer   TEXT,
  explanation TEXT,
  environment TEXT
);
CREATE INDEX IF NOT EXISTS scans_ts ON scans(ts DESC);
"""


def default_path() -> Path:
    return Path(os.environ.get("BASTION_HISTORY_DB", Path.home() / ".bastion" / "history.db"))


class History:
    def __init__(self, path: str | Path | None = None, privacy: bool = False):
        self.path = Path(path) if path else default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.privacy = privacy
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)

    def add(self, scan_id: str, report: dict, environment: str) -> None:
        ev = report["evidence"]
        text = ev["input"]
        row = (scan_id, time.time(), ev["channel"], None if self.privacy else text,
               hashlib.sha256(text.encode("utf-8")).hexdigest(), report["verdict"], report["risk"],
               report["confidence"], json.dumps(ev["indicators"]), report["explainer"],
               None if self.privacy else report["explanation"], environment)
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", row)
            self._db.commit()

    def update_explanation(self, scan_id: str, explainer: str, explanation: str) -> None:
        with self._lock:
            self._db.execute("UPDATE scans SET explainer=?, explanation=? WHERE id=?",
                             (explainer, None if self.privacy else explanation, scan_id))
            self._db.commit()

    def list(self, verdict: str | None = None, limit: int = 200) -> list[dict]:
        q, args = "SELECT * FROM scans", []
        if verdict:
            q += " WHERE verdict = ?"
            args.append(verdict.upper())
        q += " ORDER BY ts DESC LIMIT ?"
        args.append(max(1, min(int(limit), 5000)))
        with self._lock:
            rows = self._db.execute(q, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["indicators"] = json.loads(d["indicators"] or "[]")
            out.append(d)
        return out

    def clear(self) -> int:
        with self._lock:
            n = self._db.execute("DELETE FROM scans").rowcount
            self._db.commit()
            self._db.execute("VACUUM")
        return n

    def to_csv(self, verdict: str | None = None) -> str:
        buf = io.StringIO()
        cols = ["id", "ts", "channel", "input", "input_sha256", "verdict", "risk", "confidence",
                "indicators", "explainer", "explanation", "environment"]
        w = csv.DictWriter(buf, fieldnames=cols)
        w.writeheader()
        for r in self.list(verdict, limit=5000):
            r["indicators"] = "; ".join(r["indicators"])
            r["ts"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["ts"]))
            w.writerow(r)
        return buf.getvalue()
