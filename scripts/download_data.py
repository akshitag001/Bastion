"""Download public datasets into data/raw/ (run once, needs internet; inference never does).

  python scripts/download_data.py sms      # UCI SMS Spam Collection
  python scripts/download_data.py url      # PhiUSIIL Phishing URL dataset (UCI)

!! Verify each URL, licence and label convention before use and record it in docs/DATASETS.md.
   PhiUSIIL: check its documentation for which label value means phishing; pass it to
   training/train_url.py --phish-value accordingly.
"""

from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

SOURCES = {
    "sms": ("https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip", "sms_spam.zip"),
    "url": ("https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip", "phiusiil.zip"),
}


def fetch(key: str) -> None:
    url, name = SOURCES[key]
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / name
    if not dest.exists():
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, dest)
    with zipfile.ZipFile(dest) as z:
        for member in z.namelist():
            target = (RAW / member).resolve()
            if RAW.resolve() not in target.parents and target != RAW.resolve():
                raise RuntimeError(f"unsafe path in zip: {member}")
        z.extractall(RAW)
        print(f"extracted: {z.namelist()}")


if __name__ == "__main__":
    keys = sys.argv[1:] or list(SOURCES)
    for k in keys:
        fetch(k)
