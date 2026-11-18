"""Where Bastion finds models/, data/ and benchmarks/.

BASTION_ROOT (set in the twin image to /app) wins; otherwise the repo checkout this file
lives in (editable install / running from source); otherwise the current directory.
"""

from __future__ import annotations

import os
from pathlib import Path


def root() -> Path:
    env = os.environ.get("BASTION_ROOT")
    if env:
        return Path(env)
    repo = Path(__file__).resolve().parents[2]       # <repo>/src/bastion/paths.py
    if (repo / "pyproject.toml").is_file():
        return repo
    return Path.cwd()


ROOT = root()
MODELS = ROOT / "models"
SAMPLES = ROOT / "data" / "samples"
RESULTS = ROOT / "benchmarks" / "results"
