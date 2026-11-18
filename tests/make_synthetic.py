"""SYNTHETIC data generator - ONLY for smoke-testing that the training code runs.
Metrics on this data are meaningless. Never report them; never ship models trained on it."""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

random.seed(0)
BENIGN_DOMAINS = ["example.com", "wikipedia.org", "github.com", "python.org", "nptel.ac.in",
                  "iitd.ac.in", "irctc.co.in", "mozilla.org", "kernel.org", "arm.com"]
WORDS = ["docs", "news", "about", "blog", "help", "learn", "product", "team", "events", "search"]
PHISH_WORDS = ["login", "verify", "secure", "kyc", "update", "account", "otp", "unlock"]
BRANDS = ["paypal", "sbi", "hdfc", "apple", "amazon", "paytm"]
TLDS = ["xyz", "top", "tk", "icu", "com", "in"]


def benign_url(i: int) -> str:
    d = random.choice(BENIGN_DOMAINS)
    sub = random.choice(["www.", "", "docs."])
    path = "/".join(random.sample(WORDS, random.randint(0, 3)))
    return f"https://{sub}{d}/{path}"


def phish_url(i: int) -> str:
    style = random.randint(0, 2)
    b, w = random.choice(BRANDS), random.choice(PHISH_WORDS)
    if style == 0:
        return f"http://{random.randint(10,250)}.{random.randint(0,255)}.{random.randint(0,255)}.{i % 255}/{b}/{w}.php"
    if style == 1:
        return f"http://{b}-{w}-{i}.{random.choice(TLDS)}/{w}?id={random.randint(1000,9999)}"
    return f"https://{b}.com.{w}-portal{i}.{random.choice(TLDS)}/signin"


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "synthetic_urls.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["url", "label"])
        for i in range(600):
            w.writerow([benign_url(i), 0])
            w.writerow([phish_url(i), 1])
    ham = ["Are we meeting at {t}?", "Call me when you reach home", "Happy birthday {n}!",
           "Class moved to room {t}", "Pls bring the charger tomorrow", "Dinner at {t} ok?"]
    spam = ["URGENT your account {n} will be blocked, verify KYC at http://x{n}.xyz",
            "Congratulations you won Rs {n}! Call 98765{t} to claim",
            "Your parcel is on hold pay Rs {t} at bit.ly/p{n}",
            "Share OTP {n} to get cashback reward now"]
    filler = ["ok", "pls", "bro", "today", "sir", "dear", "now", "asap", "thanks", "hi", "kindly",
              "team", "note", "info", "msg", "alert", "update", "fyi", "re", "imp"]
    with open(out / "synthetic_sms.tsv", "w") as f:
        for i in range(400):
            # random filler words make rows unique even after digit-normalised de-duplication
            pad = lambda: " ".join(random.sample(filler, 3))  # noqa: E731
            f.write(f"ham\t{pad()} {random.choice(ham).format(t=random.randint(1,99999), n='Ravi')}\n")
            f.write(f"spam\t{pad()} {random.choice(spam).format(t=random.randint(10000,99999), n=i)}\n")
    print(f"wrote synthetic data to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/edgeguard_synth")
