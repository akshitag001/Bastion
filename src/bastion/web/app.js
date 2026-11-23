/* Bastion web UI. Vanilla JS, no external requests.
   Security: scanned text is attacker-controlled, so it is only ever rendered with textContent, never innerHTML. */
"use strict";

const $ = (s, root = document) => root.querySelector(s);
const el = (tag, props = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") n.className = v; else if (k === "text") n.textContent = v; else n.setAttribute(k, v);
  }
  for (const k of kids) n.append(k);
  return n;
};
const api = async (path, opts = {}) => {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  const body = r.headers.get("content-type")?.includes("json") ? await r.json() : await r.text();
  if (!r.ok) throw new Error(body?.error || r.statusText);
  return body;
};
const fmt = (v, d = 2) => (v === null || v === undefined) ? "–" : (typeof v === "number" ? +v.toFixed(d) : v);
// round half up on the 3-decimal value the JSON reports (0.9435 -> 94.4%), avoiding binary-float artefacts
const pct = (v, d = 1) => (v === null || v === undefined) ? "–" : (Math.round(v * 10 ** (d + 2) + 1e-9) / 10 ** d).toFixed(d) + "%";
const titleCase = s => s.charAt(0) + s.slice(1).toLowerCase();
const STAGES = [["detector", "Features + detectors"], ["fusion", "Risk fusion (verdict)"],
  ["llama", "Llama explanation"], ["advice", "Advice"], ["idle", "Idle"]];
window.bastion = { status: null, live: null, bench: null };

/* ---------------- router ---------------- */
const pages = ["scan", "history", "twin", "bench", "about"];
function route() {
  const name = (location.hash.match(/^#\/(\w+)/) || [])[1] || "scan";
  const page = pages.includes(name) ? name : "scan";
  for (const p of pages) $(`#page-${p}`).hidden = p !== page;
  document.querySelectorAll("nav a").forEach(a => {
    if (a.dataset.page === page) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
  if (page === "scan") loadRecent();
  if (page === "history") loadHistory();
  if (page === "bench") loadBench();
  if (page === "about") loadAbout();
  window.dispatchEvent(new CustomEvent("bastion:page", { detail: page }));
}
window.addEventListener("hashchange", () => { if (location.hash.startsWith("#/")) route(); });

/* ---------------- status: badges, device card ---------------- */
async function loadStatus() {
  try {
    const s = await api("/api/status");
    window.bastion.status = s;
    const env = s.environment.environment, off = s.network.offline;
    const eb = $("#envBadge"); eb.textContent = env; eb.title = s.environment.timing_note;
    eb.className = "chip " + (env === "raspberry-pi-5" ? "ok" : "lav");
    const nb = $("#netBadge"); nb.textContent = off ? "● OFFLINE" : "● online";
    nb.className = "chip " + (off ? "ok" : "warn");
    $("#privacyToggle").checked = !!s.privacy_mode;
    const e = s.environment, mem = e.memory_limit_bytes;
    $("#devArch").textContent = e.machine.toLowerCase();
    $("#devCores").textContent = `${fmt(e.cpu_limit, 0)} cores`;
    $("#devMem").textContent = mem ? `${(mem / 2 ** 30).toFixed(mem > 2 ** 34 ? 0 : 1)} GiB` : "";
    const dn = $("#devNet"); dn.textContent = off ? "OFFLINE" : "ONLINE"; dn.className = "net " + (off ? "off" : "on");
    $("#devEnv").textContent = env;
    const llm = s.llm.enabled ? (s.llm.available ? "Llama ready" : "Llama not loaded") : "Llama off";
    $("#devModels").textContent = `${s.models.url} · ${s.models.sms} · ${llm}`;
    $("#footEnv").textContent = `${env} · v${s.version}`;
  } catch (e) { $("#envBadge").textContent = "server unreachable"; }
}

/* ---------------- hero stats (from result JSONs) ---------------- */
async function loadHeroStats() {
  try {
    const d = await api("/api/benchmarks");
    window.bastion.bench = d;
    const latest = {}, edge = [];
    d.items.forEach(i => { if (i.data.task) latest[i.data.task] = i; else if (i.data.url_detect) edge.push(i); });
    const sms = latest.sms?.data.end_to_end_test?.warned_suspicious_or_malicious;
    const url = latest.url?.data.end_to_end_test?.warned_suspicious_or_malicious;
    if (sms) { $("#statRecall").textContent = pct(sms.recall); $("#statRecallSrc").textContent = `recall · ${sms.n} held-out SMS`; }
    if (url) { $("#statFpr").textContent = pct(url.fpr); $("#statFprSrc").textContent = `false-positive rate · ${url.n.toLocaleString()} URLs`; }
    const here = window.bastion.status?.environment?.environment;
    const pick = edge.find(i => i.environment === here) || edge[edge.length - 1];
    if (pick) { $("#statLat").textContent = `${fmt(pick.data.sms_detect.p50_ms, 1)} ms`; $("#statLatSrc").textContent = `SMS p50 · ${pick.environment}`; }
  } catch {}
}

/* ---------------- scan ---------------- */
const input = $("#scanInput");
const looksUrl = t => /^\s*(https?:\/\/|www\.)?\S+\.[a-z]{2,}(\/\S*)?\s*$/i.test(t) && !/\s/.test(t.trim());
input.addEventListener("input", () => {
  const t = input.value.trim();
  $("#detected").textContent = t ? `Detected: ${looksUrl(t) ? "link (URL)" : "message (SMS)"} · ${t.length} chars` : "Type or paste to scan · Ctrl+Enter";
});
input.addEventListener("keydown", e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) $("#scanForm").requestSubmit(); });

function scanText(text) {
  if (!location.hash.startsWith("#/scan") && location.hash !== "") { location.hash = "#/scan"; }
  input.value = text; input.dispatchEvent(new Event("input"));
  $("#scanner").scrollIntoView({ behavior: "smooth", block: "start" });
  $("#scanForm").requestSubmit();
}
$("#topScan").addEventListener("submit", e => {
  e.preventDefault();
  const t = $("#topInput").value.trim();
  if (t) { scanText(t); $("#topInput").value = ""; }
});
$("#ctaScan").addEventListener("click", e => { e.preventDefault(); $("#scanner").scrollIntoView({ behavior: "smooth" }); input.focus({ preventScroll: true }); });

let stream = null;
$("#scanForm").addEventListener("submit", async e => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  if (stream) { stream.close(); stream = null; }
  const btn = $("#scanBtn");
  btn.disabled = true; btn.textContent = "Scanning…";
  try {
    const r = await api("/api/scan", { method: "POST", body: JSON.stringify({ text }) });
    renderResult(r);
    if (r.explain_stream) startStream(r);
    loadRecent();
  } catch (err) {
    $("#detected").textContent = "Error: " + err.message;
  } finally { btn.disabled = false; btn.textContent = "Scan"; }
});

function renderResult(r) {
  $("#result").hidden = false;
  $("#verdictCard").className = "rcard verdict " + r.verdict;
  const pill = $("#verdictLabel"); pill.textContent = titleCase(r.verdict); pill.className = "pill " + r.verdict;
  $("#riskNum").textContent = r.risk;
  $("#riskFill").style.width = `${Math.max(2, r.risk)}%`;
  $("#confNum").textContent = r.confidence;
  $("#channelLabel").textContent = r.channel === "url" ? "Link" : "SMS";
  $("#modelLabel").textContent = r.evidence.model;
  $("#detectMs").textContent = r.timings_ms.detect;
  $("#resEnv").textContent = r.environment;
  const chips = $("#chips"); chips.replaceChildren();
  const inds = r.evidence.indicators.length ? r.evidence.indicators : ["No specific warning signs found"];
  inds.forEach(i => chips.append(el("li", { text: i })));
  renderActions(r.actions);
  const ex = $("#explanation");
  ex.textContent = r.explanation; ex.classList.remove("streaming");
  $("#explainNote").textContent = r.explain_stream ? "Generating the AI explanation on this device…" : `Template explanation (${r.explainer_note}).`;
  setExplainTag(r.explain_stream ? "llama" : r.explainer);
  $("#rawJson").textContent = JSON.stringify(r, null, 2);
}
function setExplainTag(kind) {
  // say what actually produced the text: the on-device Llama, or the deterministic template
  $("#explainTag").textContent = kind === "template" ? "deterministic template (Llama not used)" : "Llama 3.2 1B · ExecuTorch · on-device";
}
function renderActions(actions) {
  const ul = $("#actions"); ul.replaceChildren();
  actions.forEach(a => ul.append(el("li", { text: a })));
}

function startStream(r) {
  const ex = $("#explanation");
  let first = true;
  stream = new EventSource(r.explain_stream);
  stream.addEventListener("token", ev => {
    const piece = JSON.parse(ev.data);
    if (first) { ex.textContent = ""; ex.classList.add("streaming"); first = false; }
    ex.textContent += piece;
  });
  stream.addEventListener("final", ev => {
    const f = JSON.parse(ev.data);
    ex.classList.remove("streaming");
    ex.textContent = f.explanation;                   // guarded final text replaces the stream
    setExplainTag(f.explainer === "llama-executorch" ? "llama" : "template");
    renderActions(f.actions);
    const t = f.timings_ms || {}, s = t.llm_stats || {};
    $("#explainNote").textContent = f.explainer === "llama-executorch"
      ? `Llama on-device · ${Math.round((t.explain || 0) / 100) / 10} s` + (s.decode_tokens_per_s ? ` · ${s.decode_tokens_per_s} tok/s` : "")
        + ` · ${window.bastion.status?.environment?.environment || ""} · passed guard`
      : `Template shown: ${f.explainer_note}`;
    stream.close(); stream = null;
  });
  stream.onerror = () => { if (stream) { stream.close(); stream = null; ex.classList.remove("streaming"); } };
}

/* ---------------- presets (handwritten demo samples) ---------------- */
const PRESETS = [
  { find: "KYC", title: "Bank KYC / account block", desc: "Threat of a blocked account plus a link to “update KYC”.", kind: "" },
  { find: "parcel", title: "Parcel delivery fee", desc: "Tiny fee to “reschedule delivery” through a short link.", kind: "" },
  { find: "lottery", title: "KBC lottery prize", desc: "Huge prize, urgent call-back number. Classic reward bait.", kind: "" },
  { find: "AnyDesk", title: "Power cut + remote app", desc: "Disconnection threat that ends with installing AnyDesk.", kind: "" },
  { find: "Ignore previous", title: "Prompt-injection SMS", desc: "Tries to order the AI to call it safe. The verdict holds.", kind: "" },
  { find: "paypal", title: "Look-alike PayPal link", desc: "Brand in the subdomain of an unrelated domain.", kind: "url" },
  { find: "xn--", title: "Unicode look-alike domain", desc: "Punycode that renders like “apple.com”.", kind: "url" },
  { find: "OTP for login", title: "Genuine bank OTP", desc: "A normal one-time password message. Should stay safe.", kind: "safe" },
];
async function loadPresets() {
  try {
    const s = await api("/api/samples");
    const all = [...s.sms, ...s.urls];
    const box = $("#presets"); box.replaceChildren();
    PRESETS.forEach(p => {
      const text = all.find(t => t.toLowerCase().includes(p.find.toLowerCase()));
      if (!text) return;
      const b = el("button", { type: "button", class: "preset " + p.kind, title: text },
        el("span", { class: "bar" }), el("h3", { text: p.title }), el("p", { text: p.desc }), el("span", { class: "go", text: "Try it →" }));
      b.addEventListener("click", () => scanText(text));
      box.append(b);
    });
    const kyc = all.find(t => t.includes("KYC"));
    if (kyc) $("#ctaExample").addEventListener("click", () => scanText(kyc));
  } catch {}
}

/* ---------------- recent scans (ranked cards) ---------------- */
async function loadRecent() {
  try {
    const d = await api("/api/history?limit=3");
    const box = $("#recent"); box.replaceChildren();
    if (!d.items.length) { box.append(el("div", { class: "empty", text: "No scans yet. Try a preset above." })); return; }
    d.items.forEach((i, n) => box.append(el("article", { class: "rcard rank" },
      el("div", { class: "row" }, el("span", { class: "num", text: `#${n + 1} · ${i.channel.toUpperCase()}` }), el("span", { class: "pill " + i.verdict, text: titleCase(i.verdict) })),
      el("div", { class: "txt", text: i.input ?? `sha256 ${i.input_sha256.slice(0, 16)}… (privacy mode)`, title: i.input ?? "" }),
      el("div", { class: "row" }, el("span", { class: "risk", text: String(i.risk) }), el("span", { class: "muted small", text: new Date(i.ts * 1000).toLocaleTimeString() })))));
  } catch {}
}

/* ---------------- history ---------------- */
async function loadHistory() {
  const v = $("#histFilter").value;
  $("#csvBtn").href = "/api/history.csv" + (v ? `?verdict=${encodeURIComponent(v)}` : "");
  const d = await api("/api/history" + (v ? `?verdict=${encodeURIComponent(v)}` : ""));
  const body = $("#histBody"); body.replaceChildren();
  if (!d.items.length) body.append(el("tr", {}, el("td", { colspan: "6", class: "muted", text: "No scans yet." })));
  d.items.forEach(i => body.append(el("tr", {},
    el("td", { text: new Date(i.ts * 1000).toLocaleString() }),
    el("td", { text: i.channel.toUpperCase() }),
    el("td", { class: "input", text: i.input ?? `sha256 ${i.input_sha256.slice(0, 16)}… (privacy mode)` }),
    el("td", {}, el("span", { class: "pill " + i.verdict, text: titleCase(i.verdict) })),
    el("td", { class: "mono", text: i.risk }),
    el("td", { class: "muted", text: i.explainer }))));
}
$("#histFilter").addEventListener("change", loadHistory);
$("#clearBtn").addEventListener("click", async () => {
  if (!confirm("Delete all scan history on this device?")) return;
  await api("/api/history", { method: "DELETE" }); loadHistory();
});
$("#privacyToggle").addEventListener("change", async e => {
  await api(`/api/settings/privacy?enabled=${e.target.checked}`, { method: "POST" });
});

/* ---------------- benchmarks ---------------- */
async function loadBench() {
  const d = await api("/api/benchmarks");
  const box = $("#benchBody"); box.replaceChildren();
  if (!d.items.length) { box.append(el("p", { class: "muted", text: "No results yet. Run `bastion bench` or the trainers." })); return; }
  const groups = {};
  d.items.forEach(i => (groups[i.environment] ||= []).push(i));
  const order = Object.keys(groups).sort((p, q) => (p.startsWith("accuracy") ? -1 : 0) - (q.startsWith("accuracy") ? -1 : 0));
  for (const envName of order) {
    const g = el("div", { class: "rcard envgroup" }, el("h3", { text: envName }));
    if (d.timing_notes[envName]) g.append(el("p", { class: "muted small", text: d.timing_notes[envName] }));
    for (const it of groups[envName]) {
      const x = it.data;
      g.append(el("div", { class: "fileref", text: it.file }));
      const m = el("div", { class: "metrics" });
      const add = (label, val) => m.append(el("div", { class: "metric" }, el("b", { text: String(val) }), el("span", { text: label })));
      if (x.task === "url" || x.task === "sms") {
        const e2e = x.end_to_end_test;
        if (e2e) {
          const w = e2e.warned_suspicious_or_malicious;
          g.append(el("p", { class: "small", text: `Deployed ${x.task.toUpperCase()} detector, end-to-end on held-out test data (${w.n.toLocaleString()} items; warned = SUSPICIOUS or MALICIOUS)` }));
          add("recall", fmt(w.recall, 3)); add("precision", fmt(w.precision, 3)); add("false-positive rate", fmt(w.fpr, 4));
          add("F1", fmt(w.f1, 3)); add("ROC-AUC (risk)", fmt(w.roc_auc, 3));
          const fp = x.popular_domain_fp_check;
          if (fp) add("popular sites flagged (Tranco)", pct(fp.report_ranks_2001_12000.warned_rate));
          if (x.calibration) add("operating threshold", fmt(x.calibration.operating_threshold, 3));
          add("model", x.selected + (x.view ? ` (${x.view} view)` : ""));
          g.append(m);
        }
        g.append(el("p", { class: "small muted", text: "Candidate classifiers (validation ROC-AUC, used for model selection):" }));
        const chart = el("div", { class: "barchart" });
        for (const [name, r] of Object.entries(x.models)) {
          const auc = r.val_roc_auc ?? r["test_at_0.5"].roc_auc;
          chart.append(el("div", { class: "br" }, el("span", { text: name + (name === x.selected ? " ✓" : "") }),
            el("span", { class: "bt" }, Object.assign(el("i"), { style: `width:${auc * 100}%` })), el("span", { class: "mono small", text: `AUC ${fmt(auc, 3)}` })));
        }
        g.append(chart);
      } else {
        if (x.url_detect) { add("URL detect p50 ms", fmt(x.url_detect.p50_ms, 2)); add("URL detect p95 ms", fmt(x.url_detect.p95_ms, 2)); }
        if (x.sms_detect) { add("SMS detect p50 ms", fmt(x.sms_detect.p50_ms, 2)); add("SMS detect p95 ms", fmt(x.sms_detect.p95_ms, 2)); }
        if (x.rss_mb) add("RSS after load MB", fmt(x.rss_mb.after_load, 1));
        if (x.llm && x.llm.available) { add("explain p50 ms", fmt(x.llm.explain_wall_ms.p50_ms, 0)); add("LLM fallbacks", x.llm.fallbacks); }
        g.append(m);
      }
    }
    box.append(g);
  }
}

/* ---------------- about ---------------- */
async function loadAbout() {
  const d = await api("/api/benchmarks");
  const box = $("#modelCards"); box.replaceChildren();
  const latest = {};
  d.items.filter(i => i.data.task).forEach(i => { latest[i.data.task] = i; });
  if (!Object.keys(latest).length) { box.textContent = "No trained models yet (rules-only mode)."; return; }
  for (const [task, it] of Object.entries(latest)) {
    const x = it.data, w = x.end_to_end_test ? x.end_to_end_test.warned_suspicious_or_malicious : x.models[x.selected]["test_at_0.5"];
    const fp = x.popular_domain_fp_check ? ` Flags ${pct(x.popular_domain_fp_check.report_ranks_2001_12000.warned_rate)} of popular legitimate sites (Tranco).` : "";
    box.append(el("p", {}, el("b", { text: `${task.toUpperCase()}: ${x.selected}` }),
      document.createTextNode(` trained on ${x.dataset} (${(x.rows || x.rows_after_dedup).toLocaleString()} rows${x.view ? ", " + x.view + " view" : ""}). `
        + `Deployed detector on held-out test data: recall ${w.recall}, precision ${w.precision}, false-positive rate ${w.fpr}.${fp} Source: ${it.file}`)));
  }
}

/* ---------------- live metrics (shared with twin.js) ---------------- */
async function pollLive() {
  try {
    const m = await api("/api/metrics/live");
    window.bastion.live = m;
    window.dispatchEvent(new CustomEvent("bastion:live", { detail: m }));
    renderTwinSide(m);
  } catch {}
}
function renderTwinSide(m) {
  if ($("#page-twin").hidden) return;
  const s = window.bastion.status;
  const eb = $("#twinEnv"); eb.textContent = m.environment; eb.className = "envbig " + m.environment;
  $("#twinNote").textContent = s?.environment?.timing_note || "";
  const ol = $("#stages"); ol.replaceChildren();
  STAGES.forEach(([k, label]) => ol.append(el("li", { class: m.stage.stage === k ? "on" : "", text: label })));
  const cores = $("#cores"); cores.replaceChildren();
  m.cpu_percent_per_core.forEach((p, i) => cores.append(el("div", { class: "core", title: `core ${i}: ${p}%` },
    Object.assign(el("i"), { style: `height:${p}%` }), el("b", { text: Math.round(p) }))));
  $("#cpuLimit").textContent = `${m.cpu_percent_per_core.length} visible · limit ${m.cpu_limit}`;
  const used = m.mem.used, lim = m.mem.limit;
  $("#memBar").style.width = Math.min(100, used / lim * 100) + "%";
  $("#memText").textContent = `${(used / 2 ** 30).toFixed(2)} / ${(lim / 2 ** 30).toFixed(2)} GiB · Bastion process ${(m.mem.process_rss / 2 ** 20).toFixed(0)} MiB`;
  const net = $("#twinNet"); net.textContent = m.network.offline ? "OFFLINE" : "ONLINE"; net.className = "netbig " + (m.network.offline ? "off" : "on");
  const ms = $("#modelState"); ms.replaceChildren();
  if (s) {
    const kv = (k, v) => ms.append(el("li", {}, el("span", { text: k }), el("span", { text: v })));
    kv("URL model", s.models.url); kv("SMS model", s.models.sms);
    kv("Llama .pte", s.llm.enabled ? (s.llm.available ? `${s.llm.model} (${s.llm.backend})` : "not available") : "disabled");
    kv("Llama resident", m.llm_loaded ? "loaded" : "not loaded");
  }
}

/* ---------------- boot ---------------- */
loadStatus().then(loadHeroStats); loadPresets(); route();
setInterval(pollLive, 1500); pollLive();
setInterval(loadStatus, 15000);
