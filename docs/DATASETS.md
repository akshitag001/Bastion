# Datasets

| Channel | Dataset | Source | Licence | Label convention | Status |
|---|---|---|---|---|---|
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

## URL collection artefact: benign URLs are all bare homepages (important)
- **0.0 % of benign URLs have any path**, against 27.2 % of phishing URLs (in the 200k training sample; 28.4 % in the full set).
  Mean URL length: benign 26.2 chars, phishing 45.7.
- A model trained on full URLs learns "has a path = phishing" and would flag ordinary deep links
  (e.g. a bank's `/personal-banking/...` page). The full-URL leak probe (3 trivial features) reaches AUC 0.899.
- **Mitigation (implemented):** the URL ML model sees only `scheme://host/` (`--view host`, stored in the model bundle and
  applied identically at inference). Path signals (keywords, `@`, `//` redirects, encoding) are handled by the rule engine.
  The host-view leak probe drops to AUC 0.782. Some host-level artefact remains (benign hosts are mostly short,
  well-known domains), so treat headline URL metrics as optimistic for real-world traffic.
- Source: `benchmarks/results/url_train_20261118_185756.json`.

## Leakage controls
- **SMS duplicates:** `train_sms.py` dedups (lower-case, whitespace + digit normalisation): 5,574 -> 5,137 messages.
- **Same-site leakage:** URL split grouped by registered domain; train/test domain overlap = 0.

## Known limitations
- **SMS domain gap:** UCI is older, English, mostly UK spam. Indian UPI/KYC/courier scams are under-represented.
  The handwritten demo set (`data/samples/demo_sms.txt`) is NOT a test set; never report metrics on it.
- **Network channel (stretch):** NSL-KDD / UNSW-NB15 are dated lab traffic; note this limitation if used.
