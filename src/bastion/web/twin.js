/* Device Twin: a real-geometry board model (glTF, vendored locally) inside a soft halo with dotted orbits.
   Hover (or tap) a part to see what it is and what Bastion uses it for. While the pipeline runs, the SoC glows
   (indigo = detectors, orange = Llama), a pulse travels input -> SoC -> memory -> status LED, and the LED takes the
   verdict colour.
   Note: the mesh is a Raspberry Pi 4 Model B assembly used as a stand-in for the Pi 5 (same footprint and most
   connectors); the labels say where the Pi 5 differs.
   One renderer, moved between the home hero and the Device page. It renders only while one of them is visible,
   so it never competes with scans for CPU. Everything is served locally and works offline. */
import * as THREE from "/static/vendor/three.module.min.js";
import { GLTFLoader } from "/static/vendor/addons/GLTFLoader.js";

const HOSTS = { scan: "heroCanvas", twin: "twinCanvas" };
const MODEL_URL = "/static/vendor/models/pi-board.glb";
let host = null, tip = null;
let renderer, scene, camera, rig, board, ledMat, ledGlow, hotspot, pulse, pathCurve = null, visible = false;
let stage = "idle", verdict = null, pulseT = 0, dragging = false, moved = false, lastX = 0, rotY = -0.6, raf = 0;
let hovered = -1, pointer = null, mouseOverPart = false;
const orbits = [], parts = [], pickables = [];
// camera: orbit-style view (tilt, zoom, look-at target); explore mode unlocks tilt/zoom/focus
const DEFAULT_ELEV = Math.atan2(5.6, 7.8), DEFAULT_LOOK = new THREE.Vector3(0, -0.2, 0);
const view = { dist: 12, elev: DEFAULT_ELEV, tElev: DEFAULT_ELEV, zoom: 1, tZoom: 1, look: DEFAULT_LOOK.clone(), tLook: DEFAULT_LOOK.clone() };
let exploring = false, prevHost = null, focusIdx = -1, interacted = false, lastY = 0, pinch = null, ready = false;
const touches = new Map();
const raycaster = new THREE.Raycaster(), ndc = new THREE.Vector2();

const COLORS = { SAFE: 0x2fb35f, SUSPICIOUS: 0xf0a020, MALICIOUS: 0xe23b22, idle: 0x6d6ae8 };
const CAT = { Compute: 0xf06a2a, Memory: 0xf06a2a, Storage: 0x2fb35f, Power: 0xe23b22, "I/O": 0x6d6ae8,
  Network: 0x2b9be0, Video: 0x9b59d0, Expansion: 0x6d6ae8 };
// STAGE positions along the path: input (ports) -> SoC (detectors, fusion) -> memory (Llama) -> LED (advice)
const STAGE_T = { detector: 0.3, fusion: 0.45, llama: 0.72, advice: 0.95, idle: 0, "loading-llama": 0.72 };

// node name in the model -> label. `role` = what Bastion does with it (kept factual).
const PARTS = [
  [["CPU", "BROADCOM"], "Arm SoC", "Compute", "Pi 5: BCM2712, 4x Cortex-A76 @ 2.4 GHz. Runs the URL and SMS detectors, the fusion step and Llama 3.2 1B (ExecuTorch, XNNPACK, 4 threads)."],
  ["RAM", "LPDDR4X memory", "Memory", "Holds the 802 MB Llama .pte, loaded once and kept resident so each explanation starts without reloading."],
  ["Micro SD slot", "microSD slot", "Storage", "OS, detector models, the Llama program and the local SQLite scan history (hash-only in privacy mode)."],
  ["USB-C port", "USB-C power", "Power", "Pi 5: 5 V / 5 A. Bastion runs CPU-only; the optional NPU is out of scope for Phase 1."],
  ["Ethernet port", "Gigabit Ethernet", "Network", "Not needed to scan: inference is fully offline. The twin runs with no route out."],
  ["Wireless/Bluetooth", "Wi-Fi / Bluetooth", "Network", "Not needed to scan: nothing leaves the board."],
  ["USB 3.0", "USB 3.0 ports", "I/O", "Two 5 Gbps Type-A ports."],
  ["USB 2.0", "USB 2.0 ports", "I/O", "Two Type-A host ports."],
  ["Micro HDMI port", "Micro-HDMI 0", "Video", "Display output for the local web UI on a kiosk screen."],
  ["Micro HDMI port.001", "Micro-HDMI 1", "Video", "Second 4K display output."],
  ["MIPI port 1", "Display connector", "Expansion", "MIPI DSI ribbon connector."],
  ["MIPI port 1.001", "Camera connector", "Expansion", "MIPI CSI ribbon connector."],
  ["Audio jack", "3.5 mm A/V jack", "I/O", "On this Pi 4B model only; the Pi 5 has no A/V jack."],
];

/* ---------- helpers ---------- */
function glowTexture(inner = "rgba(255,255,255,1)", outer = "rgba(255,255,255,0)") {
  const c = document.createElement("canvas"); c.width = c.height = 256;
  const g = c.getContext("2d"), grd = g.createRadialGradient(128, 128, 0, 128, 128, 128);
  grd.addColorStop(0, inner); grd.addColorStop(1, outer);
  g.fillStyle = grd; g.fillRect(0, 0, 256, 256);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}
let GLOW_TEX = null;
function glow(color, size, opacity = 1) {
  GLOW_TEX ??= glowTexture();
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: GLOW_TEX, color, transparent: true, opacity,
    depthWrite: false, blending: THREE.AdditiveBlending }));
  s.scale.set(size, size, 1); return s;
}
function dottedEllipse(rx, rz, color, opacity) {
  const pts = new THREE.EllipseCurve(0, 0, rx, rz, 0, Math.PI * 2).getPoints(160).map(p => new THREE.Vector3(p.x, 0, p.y));
  const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineDashedMaterial({ color, dashSize: 0.09, gapSize: 0.13, transparent: true, opacity, depthWrite: false }));
  line.computeLineDistances(); return line;
}
// per-part materials so highlighting one part never lights up another that shared the material
function ownMaterials(node) {
  const mats = [];
  node.traverse(o => {
    if (!o.isMesh) return;
    o.material = Array.isArray(o.material) ? o.material.map(m => m.clone()) : o.material.clone();
    (Array.isArray(o.material) ? o.material : [o.material]).forEach(m => { if (m.emissive) { m.emissive.setHex(0); mats.push(m); } });
  });
  return mats;
}
function setEmissive(mats, hex, intensity) { mats.forEach(m => { m.emissive.setHex(hex); m.emissiveIntensity = intensity; }); }

/* ---------- board ---------- */
function loadBoard() {
  board = new THREE.Group(); rig.add(board);
  new GLTFLoader().load(MODEL_URL, gltf => {
    const model = gltf.scene;
    model.scale.setScalar(100);                       // model is in metres: 85 mm board -> 8.5 scene units
    model.traverse(o => {
      if (!o.isMesh) return;
      (Array.isArray(o.material) ? o.material : [o.material]).forEach(m => {
        const n = (m.name || "").toLowerCase();
        if (/metal|alumin|silver|steel|stainless/.test(n)) { m.metalness = 0.6; m.roughness = 0.35; }
        else if (n.includes("gold")) { m.metalness = 0.55; m.roughness = 0.4; }
      });
    });
    const rot = board.rotation.y, bob = board.position.y;            // measure in board-local space
    board.rotation.y = 0; board.position.y = 0; board.updateMatrixWorld(true);
    let box = new THREE.Box3().setFromObject(model), c = box.getCenter(new THREE.Vector3());
    model.position.set(-c.x, -box.min.y - 0.1, -c.z);
    board.add(model);
    board.updateMatrixWorld(true);
    box = new THREE.Box3().setFromObject(model);

    const byName = {};
    model.traverse(o => { const d = o.userData?.designator || o.name; if (d && !byName[d]) byName[d] = o; });
    const register = (nodes, name, cat, role) => {
      nodes = nodes.filter(Boolean); if (!nodes.length) return null;
      const idx = parts.length, mats = [];
      nodes.forEach(n => { n.userData.partIdx = idx; mats.push(...ownMaterials(n)); n.traverse(o => { if (o.isMesh) pickables.push(o); }); });
      const pbox = new THREE.Box3(); nodes.forEach(n => pbox.expandByObject(n));
      parts.push({ name, cat, role, mats, box: pbox, local: board.worldToLocal(pbox.getCenter(new THREE.Vector3())) });
      return parts[idx];
    };
    PARTS.forEach(([node, name, cat, role]) => register([node].flat().map(n => byName[n]), name, cat, role));
    const gpio = []; for (let i = 0; i < 20; i++) gpio.push(byName[i ? `Header2x13.${String(i).padStart(3, "0")}` : "Header2x13"]);
    register(gpio, "40-pin GPIO header", "Expansion", "General-purpose I/O; unused by Bastion.");
    register([byName["Header2x13.020"], byName["Header2x13.021"]], "PoE header", "Power", "Power-over-Ethernet pins for a PoE HAT.");

    // live path + status LED, placed from the real part positions (board-local coordinates; board is unrotated here)
    const socBox = parts[0]?.box, surface = socBox ? socBox.min.y + 0.04 : box.min.y + 0.3;   // PCB top (SoC sits on it)
    const top = p => p ? new THREE.Vector3((p.box.min.x + p.box.max.x) / 2, surface, (p.box.min.z + p.box.max.z) / 2) : null;
    const P = n => { const v = top(parts.find(p => p.name === n)); return v && board.worldToLocal(v); };
    const soc = P("Arm SoC"), ram = P("LPDDR4X memory"), eth = P("Gigabit Ethernet"), usb = P("USB 2.0 ports");
    // status LED where the Pi's PWR/ACT LEDs sit: beside the GPIO header, at the end away from the Ethernet/USB block
    const g = parts.find(p => p.name === "40-pin GPIO header")?.box, e = parts.find(p => p.name === "Gigabit Ethernet")?.box;
    let ledPos;
    if (g && e) {
      const gc = g.getCenter(new THREE.Vector3()), ec = e.getCenter(new THREE.Vector3()), bc = box.getCenter(new THREE.Vector3());
      const alongX = (g.max.x - g.min.x) > (g.max.z - g.min.z);
      const far = alongX ? (Math.abs(g.min.x - ec.x) > Math.abs(g.max.x - ec.x) ? g.min.x : g.max.x)
                         : (Math.abs(g.min.z - ec.z) > Math.abs(g.max.z - ec.z) ? g.min.z : g.max.z);
      const inward = alongX ? Math.sign(bc.z - gc.z) * 0.6 : Math.sign(bc.x - gc.x) * 0.6;
      const v = alongX ? new THREE.Vector3(far + Math.sign(bc.x - far) * 0.15, surface + 0.08, gc.z + inward)
                       : new THREE.Vector3(gc.x + inward, surface + 0.08, far + Math.sign(bc.z - far) * 0.15);
      ledPos = board.worldToLocal(v);
    } else ledPos = board.worldToLocal(new THREE.Vector3(box.min.x + 0.45, surface + 0.08, box.max.z - 0.5));
    if (soc && ram && eth) {
      const ledOnPcb = ledPos.clone(); ledOnPcb.y = board.worldToLocal(new THREE.Vector3(0, surface, 0)).y;
      pathCurve = new THREE.CatmullRomCurve3([eth, usb ?? eth, soc, ram, ledOnPcb].filter(Boolean), false, "centripetal");
      board.add(new THREE.Mesh(new THREE.TubeGeometry(pathCurve, 120, 0.035, 6, false),     // glowing trace on the PCB
        new THREE.MeshBasicMaterial({ color: 0xa9a6ff, transparent: true, opacity: 0.85, depthWrite: false })));
      hotspot.position.copy(soc).add(new THREE.Vector3(0, 0.5, 0));
    }
    ledMat = new THREE.MeshStandardMaterial({ color: COLORS.idle, emissive: COLORS.idle, emissiveIntensity: 0.9 });
    const led = new THREE.Mesh(new THREE.SphereGeometry(0.12, 18, 18), ledMat); led.position.copy(ledPos); board.add(led);
    ledGlow = glow(COLORS.idle, 1.2, 0.75); ledGlow.position.copy(ledPos); board.add(ledGlow);
    board.rotation.y = rot; board.position.y = bob;
    if (exploring) buildPartChips();
  }, undefined, err => {
    console.warn("twin: board model failed to load", err);
    host?.append(Object.assign(document.createElement("div"), { className: "fallback", textContent: "Board model unavailable." }));
  });
  hotspot = glow(0xff6a2a, 2.8, 0); board.add(hotspot);
  pulse = new THREE.Group();
  pulse.add(new THREE.Mesh(new THREE.SphereGeometry(0.12, 16, 16), new THREE.MeshBasicMaterial({ color: 0xff8a1f, depthTest: false })));
  pulse.add(glow(0xff7a1a, 1.0, 0.9)); pulse.visible = false;
  board.add(pulse);
}

function buildOrbits() {
  // three tilted dotted orbits; satellites = URL detector (indigo), SMS detector (green), Llama explainer (orange)
  [{ rx: 5.6, rz: 4.0, tilt: [0.32, 0, 0.18], color: 0x6d6ae8, speed: 0.12, phase: 0.2 },
   { rx: 6.1, rz: 3.4, tilt: [-0.28, 0, -0.32], color: 0x2fb35f, speed: -0.09, phase: 2.4 },
   { rx: 5.1, rz: 4.7, tilt: [0.55, 0, -0.08], color: 0xf08c00, speed: 0.07, phase: 4.1 }]
    .forEach(o => {
      const g = new THREE.Group(); g.rotation.set(...o.tilt); rig.add(g);
      g.add(dottedEllipse(o.rx, o.rz, 0x8f8cf0, 0.5));
      const sat = new THREE.Group();
      sat.add(new THREE.Mesh(new THREE.SphereGeometry(0.11, 16, 16), new THREE.MeshBasicMaterial({ color: 0xffffff })));
      sat.add(new THREE.Mesh(new THREE.TorusGeometry(0.13, 0.035, 8, 24), new THREE.MeshBasicMaterial({ color: o.color })));
      sat.add(glow(o.color, 0.8, 0.55));
      g.add(sat);
      orbits.push({ ...o, sat, g, t: o.phase });
    });
}

/* ---------- hover / tap labels ---------- */
function pick(e) {
  const r = renderer.domElement.getBoundingClientRect();
  ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  const hit = raycaster.intersectObjects(pickables, false)[0];
  let o = hit?.object;
  while (o && o.userData.partIdx === undefined) o = o.parent;
  return o ? o.userData.partIdx : -1;
}
function showTip(idx, e) {
  if (idx !== hovered) {
    if (hovered >= 0) setEmissive(parts[hovered].mats, 0, 0);
    hovered = idx;
    if (idx >= 0) setEmissive(parts[idx].mats, CAT[parts[idx].cat] ?? 0x6d6ae8, 0.55);
  }
  if (!tip) {
    tip = document.createElement("div"); tip.className = "twin-tip"; tip.setAttribute("role", "status");
    tip.append(document.createElement("small"), document.createElement("b"), document.createElement("span"));
  }
  if (tip.parentElement !== host) host.append(tip);
  if (idx < 0) { tip.classList.remove("on"); renderer.domElement.style.cursor = "grab"; return; }
  const p = parts[idx], [cat, name, role] = tip.children;
  cat.textContent = p.cat; name.textContent = p.name; role.textContent = p.role;     // textContent only
  tip.style.setProperty("--cc", "#" + (CAT[p.cat] ?? 0x6d6ae8).toString(16).padStart(6, "0"));
  const r = host.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  // keep the label inside the card: open above the pointer if it fits there, else below, else clamp
  const tw = Math.min(250, r.width - 16), th = tip.offsetHeight || 110, gap = 16;
  tip.style.width = tw + "px";
  tip.style.left = Math.min(Math.max(x, tw / 2 + 8), r.width - tw / 2 - 8) + "px";
  const above = y - th - gap >= 4, below = !above && y + gap + th <= r.height - 4;
  tip.classList.toggle("below", !above);
  tip.style.top = (above || below ? y : Math.max(4, r.height - th - 4) - gap) + "px";
  tip.classList.add("on");
  renderer.domElement.style.cursor = "pointer";
}

function init() {
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "low-power", preserveDrawingBuffer: true });
  } catch (e) {
    host.append(Object.assign(document.createElement("div"), { className: "fallback", textContent: "3D view unavailable (no WebGL). Live metrics are shown on the right." }));
    return false;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  host.append(renderer.domElement);
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(34, 1, 0.1, 120);

  scene.add(new THREE.HemisphereLight(0xffffff, 0xb9b5f2, 1.6));
  const key = new THREE.DirectionalLight(0xffffff, 2.0); key.position.set(4, 9, 6); scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.6); fill.position.set(-5, 4, 3); scene.add(fill);
  const rim = new THREE.DirectionalLight(0x8f8cf0, 0.9); rim.position.set(-6, 3, -6); scene.add(rim);

  const halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture("rgba(160,156,245,0.55)", "rgba(160,156,245,0)"),
    transparent: true, depthWrite: false }));
  halo.scale.set(12.5, 12.5, 1); halo.position.set(0, -0.6, -2.5); halo.renderOrder = -1; scene.add(halo);
  const shadow = new THREE.Mesh(new THREE.PlaneGeometry(11, 8),
    new THREE.MeshBasicMaterial({ map: glowTexture("rgba(60,54,150,0.32)", "rgba(60,54,150,0)"), transparent: true, depthWrite: false }));
  shadow.rotation.x = -Math.PI / 2; shadow.position.y = -1.15; scene.add(shadow);

  rig = new THREE.Group(); scene.add(rig);
  loadBoard();
  buildOrbits();

  const el = renderer.domElement;
  el.style.touchAction = "pan-y"; el.style.cursor = "grab";
  el.addEventListener("pointerdown", e => {
    touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (exploring && touches.size === 2) {                      // two fingers: pinch zoom
      const [a, b] = [...touches.values()]; pinch = { d: Math.hypot(a.x - b.x, a.y - b.y), z: view.tZoom }; dragging = false; return;
    }
    dragging = true; moved = false; lastX = e.clientX; lastY = e.clientY; el.setPointerCapture(e.pointerId);
  });
  el.addEventListener("pointermove", e => {
    if (touches.has(e.pointerId)) touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pinch && touches.size === 2) {
      const [a, b] = [...touches.values()];
      view.tZoom = THREE.MathUtils.clamp(pinch.z * Math.hypot(a.x - b.x, a.y - b.y) / pinch.d, 0.8, 3); interacted = true; return;
    }
    if (dragging) {
      const dx = e.clientX - lastX, dy = e.clientY - lastY; if (Math.abs(dx) + Math.abs(dy) > 3) moved = true;
      rotY += dx * 0.01; lastX = e.clientX; lastY = e.clientY;
      if (exploring) { view.tElev = THREE.MathUtils.clamp(view.tElev + dy * 0.006, 0.3, 1.45); interacted = true; }
      if (moved && !exploring) showTip(-1, e);
    } else if (e.pointerType === "mouse") pointer = e;           // picked once per frame in frame()
  });
  const release = e => {
    touches.delete(e.pointerId); if (touches.size < 2) pinch = null;
    if (!dragging) return;
    dragging = false;
    if (moved || !parts.length) return;
    const idx = pick(e);
    if (exploring) focusPart(idx); else showTip(idx, e);       // tap/click: label (explore: fly to the part)
  };
  el.addEventListener("pointerup", release);
  el.addEventListener("pointercancel", e => { touches.delete(e.pointerId); pinch = null; dragging = false; });
  el.addEventListener("pointerleave", e => { pointer = null; mouseOverPart = false; if (!dragging && tip && !exploring) showTip(-1, e); });
  el.addEventListener("wheel", e => {                           // wheel zoom only in explore (never hijack page scroll)
    if (!exploring) return;
    e.preventDefault(); interacted = true;
    view.tZoom = THREE.MathUtils.clamp(view.tZoom * Math.exp(-e.deltaY * 0.001), 0.8, 3);
  }, { passive: false });
  const ro = new ResizeObserver(resize);
  [...Object.values(HOSTS), "exploreCanvas"].forEach(id => { const h = document.getElementById(id); if (h) ro.observe(h); });
  resize();
  return true;
}

function resize() {
  if (!renderer) return;
  const w = host.clientWidth || 600, h = host.clientHeight || 420;
  renderer.setSize(w, h, false); renderer.domElement.style.width = "100%"; renderer.domElement.style.height = "100%";
  camera.aspect = w / h; camera.updateProjectionMatrix();
  // fit board + orbits (explore: the board alone) for any aspect ratio
  const vfov = THREE.MathUtils.degToRad(camera.fov), hfov = 2 * Math.atan(Math.tan(vfov / 2) * camera.aspect);
  const [hw, hh] = exploring ? [5.6, 3.9] : [7.1, 5.0];
  view.dist = Math.max(hw / Math.tan(hfov / 2), hh / Math.tan(vfov / 2));
  placeCamera();
}
function placeCamera() {
  const d = view.dist / view.zoom;
  // never inside the board: when the camera is over the board footprint, keep it above the tallest parts (ports ~1.5)
  let elev = view.elev;
  if (Math.cos(elev) * d < 6.2) elev = Math.max(elev, Math.asin(Math.min(1, (2.4 - view.look.y) / d)));
  camera.position.set(view.look.x, view.look.y + Math.sin(elev) * d, view.look.z + Math.cos(elev) * d);
  camera.lookAt(view.look);
}

/* ---------- explore mode: full-screen, zoom / tilt / focus ---------- */
const $id = id => document.getElementById(id);
function buildPartChips() {
  const box = $id("exploreParts"); if (!box || box.childElementCount === parts.length) return;
  box.replaceChildren(...parts.map((p, i) => {
    const b = document.createElement("button"); b.type = "button"; b.setAttribute("aria-pressed", "false");
    b.style.setProperty("--cc", "#" + (CAT[p.cat] ?? 0x6d6ae8).toString(16).padStart(6, "0"));
    b.append(document.createElement("i"), document.createTextNode(p.name));
    b.addEventListener("click", () => focusPart(i === focusIdx ? -1 : i));
    return b;
  }));
}
function focusPart(idx) {
  focusIdx = idx; interacted = true;
  [...($id("exploreParts")?.children || [])].forEach((b, i) => b.setAttribute("aria-pressed", String(i === idx)));
  if (idx < 0) { view.tLook.copy(DEFAULT_LOOK); view.tZoom = Math.min(view.tZoom, 1.4); showTip(-1, {}); return; }
  const s = parts[idx].box.getSize(new THREE.Vector3()), size = Math.max(s.x, s.z);
  view.tZoom = THREE.MathUtils.clamp(2.4 - size * 0.25, 1.5, 2.2);
}
function resetView() {
  focusPart(-1); view.tZoom = 1; view.tElev = DEFAULT_ELEV; interacted = false;
}
function openExplore() {
  if (!ready || exploring) return;
  exploring = true; prevHost = host; host = $id("exploreCanvas");
  $id("explore").hidden = false; document.body.classList.add("exploring");
  host.append(renderer.domElement); renderer.domElement.style.touchAction = "none";
  orbits.forEach(o => { o.g.visible = false; });
  showTip(-1, {}); buildPartChips(); resetView(); resize();
  visible = true; if (!raf) raf = requestAnimationFrame(frame);
  host.focus({ preventScroll: true });
}
function closeExplore() {
  if (!exploring) return;
  exploring = false; resetView(); showTip(-1, {});
  view.zoom = 1; view.elev = DEFAULT_ELEV; view.look.copy(DEFAULT_LOOK);
  $id("explore").hidden = true; document.body.classList.remove("exploring");
  orbits.forEach(o => { o.g.visible = true; });
  renderer.domElement.style.touchAction = "pan-y";
  host = prevHost; host.append(renderer.domElement); resize();
  visible = !host.closest("[hidden]");
  if (visible && !raf) raf = requestAnimationFrame(frame);
}
document.addEventListener("click", e => { if (e.target.closest("[data-explore]")) openExplore(); });
$id("exploreClose")?.addEventListener("click", closeExplore);
$id("exploreReset")?.addEventListener("click", resetView);
document.addEventListener("keydown", e => {
  if (!exploring) return;
  const k = e.key;
  if (k === "Escape") closeExplore();
  else if (k === "ArrowLeft" || k === "ArrowRight") rotY += k === "ArrowLeft" ? -0.12 : 0.12;
  else if (k === "ArrowUp" || k === "ArrowDown") view.tElev = THREE.MathUtils.clamp(view.tElev + (k === "ArrowUp" ? 0.08 : -0.08), 0.3, 1.45);
  else if (k === "+" || k === "=") view.tZoom = Math.min(3, view.tZoom * 1.2);
  else if (k === "-" || k === "_") view.tZoom = Math.max(0.8, view.tZoom / 1.2);
  else return;
  interacted = true; e.preventDefault();
});

function frame() {
  raf = 0;
  if (!visible) return;
  const now = performance.now() / 1000, busy = stage !== "idle";
  if (pointer && parts.length) {
    const idx = pick(pointer); mouseOverPart = idx >= 0;
    if (idx >= 0 || !exploring || focusIdx < 0) showTip(idx, pointer);
    pointer = null;
  }
  const still = exploring ? interacted || focusIdx >= 0 : hovered >= 0;
  if (!dragging && !still && !busy) rotY += 0.0018;              // pause while a label is open / the user is exploring
  board.rotation.y = rotY;
  board.position.y = exploring ? 0 : Math.sin(now * 1.1) * 0.06;

  // camera easing; in explore, follow the focused part as the board turns
  if (exploring && focusIdx >= 0) view.tLook.copy(board.localToWorld(parts[focusIdx].local.clone()));
  const ease = 0.12;
  view.zoom += (view.tZoom - view.zoom) * ease; view.elev += (view.tElev - view.elev) * ease; view.look.lerp(view.tLook, ease);
  placeCamera();
  if (exploring && focusIdx >= 0 && !mouseOverPart) {
    const v = view.tLook.clone().project(camera), r = renderer.domElement.getBoundingClientRect();
    showTip(focusIdx, { clientX: r.left + (v.x + 1) / 2 * r.width, clientY: r.top + (1 - v.y) / 2 * r.height });
  }

  const llamaOn = stage === "llama" || stage === "loading-llama";
  orbits.forEach((o, i) => {
    const on = busy && (i === 2 ? llamaOn : !llamaOn);
    o.t += (on ? o.speed * 6 : o.speed) / 60;
    o.sat.position.set(Math.cos(o.t) * o.rx, 0, Math.sin(o.t) * o.rz);
    o.sat.scale.setScalar(on ? 1.35 + 0.15 * Math.sin(now * 8) : 1);
  });

  if (pathCurve) {
    const target = STAGE_T[stage] ?? 0;
    pulseT = busy ? (pulseT + 0.01) % Math.max(target, 0.05) : Math.max(0, pulseT - 0.03);
    pulse.visible = busy;
    pulse.position.copy(pathCurve.getPointAt(Math.min(pulseT, 0.999)));
  }
  const heat = llamaOn ? 0xff6a2a : busy ? 0x6d6ae8 : 0;
  if (parts[0] && hovered !== 0) setEmissive(parts[0].mats, heat, heat ? 0.45 + 0.25 * Math.sin(now * 5.5) : 0);
  hotspot.material.color.setHex(heat || 0xff6a2a);
  hotspot.material.opacity += ((busy ? 0.7 + 0.2 * Math.sin(now * 5.5) : 0) - hotspot.material.opacity) * 0.1;

  if (ledMat) {
    const c = COLORS[verdict] ?? COLORS.idle;
    ledMat.color.setHex(c); ledMat.emissive.setHex(c);
    ledGlow.material.color.setHex(c); ledGlow.material.opacity = 0.6 + 0.25 * Math.sin(now * 3);
  }
  renderer.render(scene, camera);
  raf = requestAnimationFrame(frame);
}

window.addEventListener("bastion:page", e => {
  if (exploring) closeExplore();
  const id = HOSTS[e.detail];
  visible = !!id;
  if (!visible) return;
  host = document.getElementById(id);
  if (!host) { visible = false; return; }
  if (!ready) ready = init();
  else if (renderer && renderer.domElement.parentElement !== host) { host.append(renderer.domElement); resize(); }
  if (ready && !raf) raf = requestAnimationFrame(frame);
});
window.addEventListener("bastion:live", e => {
  stage = e.detail.stage.stage;
  if (e.detail.stage.verdict) verdict = e.detail.stage.verdict;
});
// the page event may have fired before this module loaded
{
  const page = (location.hash.match(/^#\/(\w+)/) || [])[1] || "scan";
  window.dispatchEvent(new CustomEvent("bastion:page", { detail: page }));
}
// Snapshot hook for automated screenshots: headless browsers may not composite WebGL canvases, so this renders one
// frame, reads the pixels back with readPixels and overlays them as an <img> (exactly what the renderer drew).
window.__twinSnapshot = () => {
  if (!renderer) return false;
  renderer.render(scene, camera);
  const gl = renderer.getContext(), w = gl.drawingBufferWidth, h = gl.drawingBufferHeight;
  const px = new Uint8Array(w * h * 4);
  gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  const g = c.getContext("2d"), img = g.createImageData(w, h);
  for (let y = 0; y < h; y++) img.data.set(px.subarray((h - 1 - y) * w * 4, (h - y) * w * 4), y * w * 4);   // flip rows
  g.putImageData(img, 0, 0);
  const el = document.createElement("img");
  el.src = c.toDataURL("image/png");
  el.style.cssText = "position:absolute;inset:0;width:100%;height:100%;pointer-events:none";
  host.append(el);
  return true;
};
// screenshot/test hook: open the label of a named part at the centre of its on-screen box
window.__twinHover = name => {
  const i = parts.findIndex(p => p.name === name); if (i < 0) return false;
  board.updateMatrixWorld(true);
  const b = new THREE.Box3(); pickables.filter(m => { let o = m; while (o && o.userData.partIdx === undefined) o = o.parent; return o?.userData.partIdx === i; })
    .forEach(m => b.expandByObject(m));
  const v = b.getCenter(new THREE.Vector3()).project(camera), r = renderer.domElement.getBoundingClientRect();
  const at = { clientX: r.left + (v.x + 1) / 2 * r.width, clientY: r.top + (1 - v.y) / 2 * r.height };
  showTip(i, at);
  return { x: at.clientX, y: at.clientY };
};
// screenshot/test hook: current live state as the twin sees it
window.__twinState = () => ({ stage, verdict, led: ledMat ? ledMat.color.getHexString() : null, parts: parts.length,
  led_pos: ledGlow ? ledGlow.position.toArray().map(v => +v.toFixed(2)) : null });
