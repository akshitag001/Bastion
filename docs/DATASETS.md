# Datasets

| Channel | Dataset | Source | Licence | Label convention | Status |
|---|---|---|---|---|---|
| SMS | UCI SMS Spam Collection | archive.ics.uci.edu (id 228) | verify | `spam` / `ham` | not downloaded |
| URL | PhiUSIIL Phishing URL Dataset | archive.ics.uci.edu (id 967) | verify | **check docs: which value = phishing** | not downloaded |
| URL (alt.) | PhishTank / OpenPhish feeds (phishing) + Tranco top sites (benign) | organiser-suggested | verify terms | build yourself | optional |

## Known risks (write findings here after training)
- **SMS duplicates:** UCI has many exact/near duplicates. `train_sms.py` dedups (lower-case, whitespace + digit normalisation) before splitting.
- **SMS domain gap:** UCI is older, English, mostly UK spam. Indian UPI/KYC/courier scams are under-represented.
  The handwritten demo set (`data/samples/demo_sms.txt`) is NOT a test set; never report metrics on it.
- **URL collection artefacts:** in many URL datasets benign URLs are bare homepages while phishing URLs have long paths,
  so a model can "cheat" on length. `train_url.py` reports a **leak probe** (only url_len, path_len, n_dots_host). If its
  AUC is near 1.0, say so in the report and treat headline metrics with caution.
- **Same-site leakage:** URL split is grouped by registered domain.
- **Network (stretch):** NSL-KDD / UNSW-NB15 are dated lab traffic; note this limitation.
