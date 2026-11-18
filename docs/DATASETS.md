# Datasets

| Channel | Dataset | Source | Licence | Label convention | Status |
|---|---|---|---|---|---|
| URL (eval only) | Tranco top-1M popular domains | tranco-list.eu | free; components CC BY 3.0 / BY-SA 4.0 / BY-NC 4.0 | all treated as legitimate (approximate) | downloaded, false-positive check |
| SMS | UCI SMS Spam Collection | archive.ics.uci.edu/dataset/228 | **CC BY 4.0** (UCI page, checked 18 Nov 2026) | `spam` / `ham` | downloaded, trained |
| URL | PhiUSIIL Phishing URL Dataset | archive.ics.uci.edu/dataset/967 | **CC BY 4.0** (UCI page, checked 18 Nov 2026) | **`label=1` legitimate, `label=0` phishing** (verified, see below) | downloaded, trained |

Download: `python scripts/download_data.py sms url` (files land in `data/raw/`, git-ignored).

## PhiUSIIL label convention (verified empirically)
- 235,795 rows: label 1 = 134,850, label 0 = 100,945.
- `label=1` examples: `uni-mainz.de`, `southbankmosaics.com`, `voicefmradio.co.uk` (ordinary sites).
- `label=0` examples: `f0519141.xsph.ru`, `shprakserf.gq`, `service-mitld.firebaseapp.com` (phishing lures).
- So **0 = phishing**: train with `--phish-value 0`.
- 787 `*.google.com` rows are labelled phishing: they are lures **hosted on** Google services
  (`docs.google.com/forms/...` 526, `sites.google.com` 179, `storage.cloud.google.com` 51), which is correct labelling.

## URL collection artefacts: why PhiUSIIL's 0.99 ROC-AUC is not real-world performance (important)
PhiUSIIL's benign URLs were collected very differently from its phishing URLs:

| Property | benign | phishing |
|---|---|---|
| has any path after the host | **0.0 %** | 28.4 % |
| starts with `www.` | **100.0 %** | 41.4 % |
| uses `https` | **100.0 %** | 48.7 % |

A lexical model learns these collection habits instead of phishing. Our own test split carries the same artefacts, so
it cannot reveal this, which is why we added an outside check: the false-positive rate of the **real detector** on popular
legitimate domains from the **Tranco** top-sites list (`https://{domain}/`), ranks 1-2000 used to compare options and
ranks 2001-12000 reported.

| ML input ("view") | PhiUSIIL test FPR (end-to-end) | Tranco popular domains flagged | Result file (`benchmarks/results/archive/`) |
|---|---|---|---|
| full URL | leak probe alone reaches AUC 0.899 | (not deployed) | `url_train_20261118_185756.json` |
| `host` = scheme://host/ | 0.025 | **99.5 %** | `url_train_20261119_000332_view-host.json` |
| `domain` = registered domain only | 0.994 | 100 % | `url_train_20261119_000141_view-domain.json` |
| **`canon` = host without scheme / `www.`** at a recall-0.95 threshold | 0.774 | 84.8 % | `url_train_20261119_000334_view-canon.json` |
| **`canon` at a 2 % validation-FPR threshold (deployed)** | **0.016** | **2.6 %** | `benchmarks/results/url_train_20261119_000943.json` |

**Decision (deployed):** `canon` view, operating threshold chosen on validation for <= 2 % FPR. The two independent benign
sources (PhiUSIIL long-tail sites, Tranco popular sites) now agree on the false-positive rate (1.6 % vs 2.6 %), which the
`host` view did not (2.5 % vs 99.5 %). End-to-end recall on PhiUSIIL phishing is **0.737** (precision 0.983).

**Consequence for the recall >= 0.95 target:** with lexical, offline-only URL features this dataset cannot support 0.95
recall at a usable false-positive rate (the canon model at 0.95 recall flags ~77-85 % of legitimate hosts). The rule engine
still catches the structural tricks (raw IPs, brand-in-subdomain, look-alike domains, `@`, shorteners, abused TLDs), and
SMS keeps its 0.95 recall target. Phase 2 options: an on-device phishing blocklist snapshot (Safe-Browsing-style local list),
page-content features when the user opts in, and a training set whose benign URLs have realistic paths.

Tranco list: downloaded 18 Nov 2026 (daily list dated 17 Nov 2026), `data/raw/top-1m.csv`, free of charge; its component
lists are under CC BY 3.0 / CC BY-SA 4.0 / CC BY-NC 4.0 (tranco-list.eu). Used for evaluation only. A few popular domains do
host abuse, so the Tranco number is an approximate false-positive rate.

## SMS operating point
The first calibrated SMS run chose its 0.95-recall threshold from a 37-scam validation half: threshold 0.033, FPR 0.163
(`archive/sms_train_20261118_195416.json`). The deployed model takes the threshold from 5-fold out-of-fold predictions on the
whole train split (422 scams): end-to-end recall 0.944, precision 0.951, FPR 0.0066 (`sms_train_20261119_001008.json`).

## Leakage controls
- **SMS duplicates:** `train_sms.py` dedups (lower-case, whitespace + digit normalisation): 5,574 -> 5,137 messages.
- **Same-site leakage:** URL split grouped by registered domain; train/test domain overlap = 0.

## Known limitations
- **SMS domain gap:** UCI is older, English, mostly UK spam. Indian UPI/KYC/courier scams are under-represented.
  The handwritten demo set (`data/samples/demo_sms.txt`) is NOT a test set; never report metrics on it.
- **Network channel (stretch):** NSL-KDD / UNSW-NB15 are dated lab traffic; note this limitation if used.
