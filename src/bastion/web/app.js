/* Bastion web UI. Vanilla JS, no external requests.
   Security: scanned text is attacker-controlled, so it is only ever rendered with textContent, never innerHTML. */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
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
const STAGES = [["detector", "Features + detectors"], ["fusion", "Risk fusion (verdict)"],
  ["llama", "Llama explanation"], ["advice", "Advice"], ["idle", "Idle"]];
window.bastion = { status: null, live: null };

/* ---------------- router ---------------- */
const pages = ["scan", "history", "twin", "bench", "about"];
function route() {
  const name = (location.hash.match(/^#\/(\w+)/) || [])[1] || "scan";
  const page = pages.includes(name) ? name : "scan";
  for (const p of pages) $(`#page-${p}`).hidden = p !== page;
  document.querySelectorAll("nav a").forEach(a => {
    if (a.dataset.page === page) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
  if (page === "history") loadHistory();
  if (page === "bench") loadBench();
  if (page === "about") loadAbout();
  window.dispatchEvent(new CustomEvent("bastion:page", { detail: page }));
}
window.addEventListener("hashchange", route);

/* ---------------- status badges ---------------- */
async function loadStatus() {
  try {
    const s = await api("/api/status");
    window.bastion.status = s;
    const env = s.environment.environment;
    const eb = $("#envBadge");
    eb.textContent = env; eb.title = s.environment.timing_note;
    eb.className = "badge " + (env === "raspberry-pi-5" ? "ok" : "warn");
    const nb = $("#netBadge");
    nb.textContent = s.network.offline ? "OFFLINE" : "online";
    nb.className = "badge " + (s.network.offline ? "ok" : "warn");
    $("#privacyToggle").checked = !!s.privacy_mode;
  } catch (e) { $("#envBadge").textContent = "server unreachable"; }
}

/* ---------------- scan ---------------- */
const input = $("#scanInput");
const looksUrl = t => /^\s*(https?:\/\/|www\.)?\S+\.[a-z]{2,}(\/\S*)?\s*$/i.test(t) && !/\s/.test(t.trim());
input.addEventListener("input", () => {
  const t = input.value.trim();
  $("#detected").textContent = t ? `Detected: ${looksUrl(t) ? "link (URL)" : "message (SMS)"} · ${t.length} chars` : "Type or paste to scan";
});
input.addEventListener("keydown", e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) $("#scanForm").requestSubmit(); });

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
  } catch (err) {
    $("#detected").textContent = "Error: " + err.message;
  } finally { btn.disabled = false; btn.textContent = "Scan"; }
});

function renderResult(r) {
  $("#result").hidden = false;
  const card = $("#verdictCard");
  card.className = "verdict " + r.verdict;
  $("#verdictLabel").textContent = r.verdict;
  $("#riskNum").textContent = r.risk;
  $("#gaugeArc").style.strokeDasharray = `${r.risk} 100`;
  $("#confNum").textContent = r.confidence;
  $("#channelLabel").textContent = r.channel === "url" ? "Link" : "SMS";
  $("#modelLabel").textContent = r.evidence.model;
  $("#detectMs").textContent = r.timings_ms.detect;
  $("#resEnv").textContent = r.environment;
  const chips = $("#chips"); chips.replaceChildren();
  const inds = r.evidence.indicators.length ? r.evidence.indicators : ["No specific warning signs found"];
  inds.forEach(i => chips.append(el("li", { text: i })));
  (r.evidence.sub_evidence || []).forEach(s => chips.append(el("li", { text: `Link inside: ${s.label} (${s.risk}/100)` })));
  renderActions(r.actions);
  const ex = $("#explanation");
  ex.textContent = r.explanation; ex.classList.remove("streaming");
  $("#explainNote").textContent = r.explain_stream ? "Generating the AI explanation on this device…"
    : `Template explanation (${r.explainer_note}).`;
  $("#rawJson").textContent = JSON.stringify(r, null, 2);
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
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
    renderActions(f.actions);
    const t = f.timings_ms || {};
    const s = t.llm_stats || {};
    $("#explainNote").textContent = f.explainer === "llama-executorch"
      ? `Llama on-device · ${Math.round((t.explain || 0) / 100) / 10} s` + (s.decode_tokens_per_s ? ` · ${s.decode_tokens_per_s} tok/s` : "")
        + ` · ${window.bastion.status?.environment?.environment || ""} (passed guard)`
      : `Template shown: ${f.explainer_note}`;
    stream.close(); stream = null;
  });
  stream.onerror = () => { if (stream) { stream.close(); stream = null; ex.classList.remove("streaming"); } };
}

async function loadExamples() {
  try {
    const s = await api("/api/samples");
    const box = $("#exampleBtns");
    const picks = [...s.urls.slice(0, 3), ...s.sms.slice(0, 4)];
    picks.forEach(t => {
      const b = el("button", { type: "button", text: t.length > 34 ? t.slice(0, 32) + "…" : t, title: t });
      b.addEventListener("click", () => { input.value = t; input.dispatchEvent(new Event("input")); input.focus(); });
      box.append(b);
    });
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
    el("td", {}, el("span", { class: "pill " + i.verdict, text: i.verdict })),
    el("td", { text: i.risk }),
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
const fmt = (v, d = 2) => (v === null || v === undefined) ? "–" : (typeof v === "number" ? +v.toFixed(d) : v);
async function loadBench() {
  const d = await api("/api/benchmarks");
  const box = $("#benchBody"); box.replaceChildren();
  if (!d.items.length) { box.append(el("p", { class: "muted", text: "No results yet. Run `bastion bench` or the trainers." })); return; }
  const groups = {};
  d.items.forEach(i => (groups[i.environment] ||= []).push(i));
  for (const [envName, items] of Object.entries(groups)) {
    const g = el("div", { class: "card envgroup" }, el("h2", { text: envName }));
    if (d.timing_notes[envName]) g.append(el("p", { class: "muted small", text: d.timing_notes[envName] }));
    for (const it of items) {
      const x = it.data;
      g.append(el("h3", { class: "small muted", text: it.file }));
      const m = el("div", { class: "metrics" });
      const add = (label, val) => m.append(el("div", { class: "metric" }, el("b", { text: String(val) }), el("span", { text: label })));
      if (x.task === "url" || x.task === "sms") {
        const best = x.models[x.selected];
        const t = best["test_at_0.5"];
        add("selected model", x.selected); add("F1 @0.5", fmt(t.f1, 4)); add("recall @0.5", fmt(t.recall, 4));
        add("FPR @0.5", fmt(t.fpr, 4)); add("ROC-AUC", fmt(t.roc_auc, 4)); add("size KB", Math.round(best.size_bytes / 1024));
        if (x.leak_probe_trivial_features) add("leak probe AUC", fmt(x.leak_probe_trivial_features.roc_auc, 3));
        g.append(m);
        const chart = el("div", { class: "barchart" });
        for (const [name, r] of Object.entries(x.models)) {
          const f1 = r["test_at_0.5"].f1;
          chart.append(el("div", { class: "br" }, el("span", { text: name }),
            el("span", { class: "bt" }, Object.assign(el("i"), { style: `width:${f1 * 100}%` })), el("span", { text: `F1 ${fmt(f1, 3)}` })));
        }
        g.append(chart);
      } else {
        if (x.url_detect) { add("URL detect p50 ms", fmt(x.url_detect.p50_ms, 3)); add("URL detect p95 ms", fmt(x.url_detect.p95_ms, 3)); }
        if (x.sms_detect) { add("SMS detect p50 ms", fmt(x.sms_detect.p50_ms, 3)); add("SMS detect p95 ms", fmt(x.sms_detect.p95_ms, 3)); }
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
    const x = it.data, t = x.models[x.selected]["test_at_0.5"];
    box.append(el("p", {}, el("b", { text: `${task.toUpperCase()}: ${x.selected}` }),
      document.createTextNode(` trained on ${x.dataset} (${x.rows} rows${x.view ? ", " + x.view + " view" : ""}). `
        + `Test F1 ${t.f1}, recall ${t.recall}, FPR ${t.fpr}, ROC-AUC ${t.roc_auc}. Source: ${it.file}`)));
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
  const envName = m.environment;
  const eb = $("#twinEnv"); eb.textContent = envName; eb.className = "envbig " + envName;
  $("#twinNote").textContent = s?.environment?.timing_note || "";
  const ol = $("#stages"); ol.replaceChildren();
  STAGES.forEach(([k, label]) => ol.append(el("li", { class: m.stage.stage === k ? "on" : "", text: label })));
  const cores = $("#cores"); cores.replaceChildren();
  m.cpu_percent_per_core.forEach((p, i) => {
    const c = el("div", { class: "core", title: `core ${i}: ${p}%` }, Object.assign(el("i"), { style: `height:${p}%` }), el("b", { text: Math.round(p) }));
    cores.append(c);
  });
  $("#cpuLimit").textContent = `${m.cpu_percent_per_core.length} cores visible · container CPU limit ${m.cpu_limit}`;
  const used = m.mem.used, lim = m.mem.limit;
  $("#memBar").style.width = Math.min(100, used / lim * 100) + "%";
  $("#memText").textContent = `${(used / 2 ** 30).toFixed(2)} / ${(lim / 2 ** 30).toFixed(2)} GiB · Bastion process ${(m.mem.process_rss / 2 ** 20).toFixed(0)} MiB`;
  const net = $("#twinNet");
  net.textContent = m.network.offline ? "OFFLINE" : "ONLINE";
  net.className = "netbig " + (m.network.offline ? "off" : "on");
  const ms = $("#modelState"); ms.replaceChildren();
  if (s) {
    const kv = (k, v) => ms.append(el("li", {}, el("span", { text: k }), el("span", { text: v })));
    kv("URL model", s.models.url); kv("SMS model", s.models.sms);
    kv("Llama .pte", s.llm.enabled ? (s.llm.available ? `${s.llm.model} (${s.llm.backend})` : "not available") : "disabled");
    kv("Llama resident", m.llm_loaded ? "loaded" : "not loaded");
  }
}

/* ---------------- boot ---------------- */
loadStatus(); loadExamples(); route();
setInterval(pollLive, 1500); pollLive();
setInterval(loadStatus, 15000);
