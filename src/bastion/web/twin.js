/* Device Twin: a 3D board drawn from simple primitives (no logos, no product images).
   Renders only while the Twin page is visible so it never competes with scans for CPU. */
import * as THREE from "/static/vendor/three.module.min.js";

const host = document.getElementById("twinCanvas");
let renderer, scene, camera, board, soc, socMat, led, ledMat, fan, pulse, pathCurve, visible = false;
let stage = "idle", verdict = null, pulseT = 0, dragging = false, lastX = 0, rotY = -0.5, raf = 0;

const COLORS = { SAFE: 0x3fb950, SUSPICIOUS: 0xe3b341, MALICIOUS: 0xf85149, idle: 0x2b6cb0 };
// path along the board: input (ports) -> features -> detector -> fusion (SoC) -> Llama (RAM) -> advice (LED)
const STAGE_T = { detector: 0.3, fusion: 0.5, llama: 0.72, advice: 0.92, idle: 0, "loading-llama": 0.72 };

function label(text, w = 512, h = 128, size = 64) {
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  const g = c.getContext("2d");
  g.fillStyle = "rgba(0,0,0,0)"; g.fillRect(0, 0, w, h);
  g.fillStyle = "#cfe8d6"; g.font = `700 ${size}px system-ui, sans-serif`; g.textAlign = "center"; g.textBaseline = "middle";
  g.fillText(text, w / 2, h / 2);
  const t = new THREE.CanvasTexture(c); t.anisotropy = 4;
  return new THREE.Mesh(new THREE.PlaneGeometry(w / 200, h / 200), new THREE.MeshBasicMaterial({ map: t, transparent: true }));
}

function box(w, h, d, color, x, y, z, opts = {}) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d),
    new THREE.MeshStandardMaterial({ color, roughness: opts.rough ?? 0.6, metalness: opts.metal ?? 0.1 }));
  m.position.set(x, y, z); board.add(m); return m;
}

function init() {
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "low-power", preserveDrawingBuffer: true });
  } catch (e) {
    host.append(Object.assign(document.createElement("div"), { className: "fallback", textContent: "3D view unavailable (no WebGL). Live metrics are shown on the right." }));
    return false;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  host.append(renderer.domElement);
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
  camera.position.set(0, 6.2, 7.4); camera.lookAt(0, 0, 0);
  scene.add(new THREE.HemisphereLight(0xbfd8ff, 0x0b0f14, 1.1));
  const key = new THREE.DirectionalLight(0xffffff, 1.4); key.position.set(4, 8, 5); scene.add(key);

  board = new THREE.Group(); scene.add(board);
  // PCB (85 x 56 mm board, scaled) with rounded feel via a thin edge
  box(8.5, 0.16, 5.6, 0x0f5132, 0, 0, 0, { rough: 0.8 });
  box(8.52, 0.02, 5.62, 0xc9a227, 0, -0.09, 0, { metal: 0.8, rough: 0.3 });
  // SoC with heat spreader
  soc = box(1.5, 0.22, 1.5, 0x9aa4ae, -0.6, 0.19, 0.1, { metal: 0.85, rough: 0.35 });
  socMat = soc.material; socMat.emissive = new THREE.Color(0x000000);
  // RAM
  box(1.1, 0.16, 1.4, 0x1b1f24, 1.15, 0.16, 0.1, { rough: 0.5 });
  // GPIO header (2 x 20 pins)
  for (let i = 0; i < 20; i++) for (let j = 0; j < 2; j++)
    box(0.12, 0.32, 0.12, 0xd4af37, -3.6 + i * 0.36, 0.24, -2.45 + j * 0.24, { metal: 0.9, rough: 0.3 });
  // Ports: 2x USB stacks, Ethernet (right edge); 2x micro-HDMI, USB-C power (front edge)
  box(1.3, 1.0, 1.15, 0xb8c0c8, 3.75, 0.58, -1.9, { metal: 0.8, rough: 0.35 });
  box(1.3, 1.0, 1.15, 0xb8c0c8, 3.75, 0.58, -0.45, { metal: 0.8, rough: 0.35 });
  box(1.3, 0.95, 1.25, 0xb8c0c8, 3.75, 0.55, 1.4, { metal: 0.8, rough: 0.35 });
  [-2.6, -1.6].forEach(x => box(0.55, 0.22, 0.45, 0x8e979f, x, 0.19, 2.65, { metal: 0.8 }));
  box(0.7, 0.26, 0.5, 0x8e979f, -3.5, 0.21, 2.6, { metal: 0.8 });
  // Fan (active cooler) above SoC
  fan = new THREE.Group();
  const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 0.06, 24), new THREE.MeshStandardMaterial({ color: 0x222831 }));
  fan.add(hub);
  for (let i = 0; i < 7; i++) {
    const blade = new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.02, 0.16), new THREE.MeshStandardMaterial({ color: 0x2d333b, transparent: true, opacity: 0.85 }));
    blade.position.x = 0.4; const arm = new THREE.Group(); arm.add(blade); arm.rotation.y = (i / 7) * Math.PI * 2; arm.rotation.z = 0.2; fan.add(arm);
  }
  fan.position.set(-0.6, 0.62, 0.1); board.add(fan);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.85, 0.04, 8, 48), new THREE.MeshStandardMaterial({ color: 0x30363d }));
  ring.rotation.x = Math.PI / 2; ring.position.copy(fan.position); board.add(ring);
  // Status LED
  ledMat = new THREE.MeshStandardMaterial({ color: COLORS.idle, emissive: COLORS.idle, emissiveIntensity: 0.6 });
  led = new THREE.Mesh(new THREE.SphereGeometry(0.12, 16, 16), ledMat); led.position.set(-3.9, 0.18, 1.8); board.add(led);
  // Board label (our own text, no logos)
  const lbl = label("Pi 5 twin"); lbl.rotation.x = -Math.PI / 2; lbl.position.set(1.9, 0.09, 2.05); board.add(lbl);

  // Data path + pulse
  pathCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(3.6, 0.3, 1.4), new THREE.Vector3(2.4, 0.3, 1.2), new THREE.Vector3(0.8, 0.3, 1.3),
    new THREE.Vector3(-0.6, 0.35, 0.9), new THREE.Vector3(1.15, 0.3, -0.7), new THREE.Vector3(-1.5, 0.3, 1.6),
    new THREE.Vector3(-3.9, 0.3, 1.8)]);
  const tube = new THREE.Mesh(new THREE.TubeGeometry(pathCurve, 80, 0.025, 6, false),
    new THREE.MeshBasicMaterial({ color: 0x4cc2ff, transparent: true, opacity: 0.35 }));
  board.add(tube);
  pulse = new THREE.Mesh(new THREE.SphereGeometry(0.1, 16, 16), new THREE.MeshBasicMaterial({ color: 0x4cc2ff }));
  board.add(pulse);

  const el = renderer.domElement;
  el.addEventListener("pointerdown", e => { dragging = true; lastX = e.clientX; el.setPointerCapture(e.pointerId); });
  el.addEventListener("pointermove", e => { if (dragging) { rotY += (e.clientX - lastX) * 0.01; lastX = e.clientX; } });
  el.addEventListener("pointerup", () => { dragging = false; });
  new ResizeObserver(resize).observe(host);
  resize();
  return true;
}

function resize() {
  if (!renderer) return;
  const w = host.clientWidth || 600, h = host.clientHeight || 420;
  renderer.setSize(w, h, false); renderer.domElement.style.width = "100%"; renderer.domElement.style.height = "100%";
  camera.aspect = w / h; camera.updateProjectionMatrix();
  // fit the whole board (about 10 x 7.5 units incl. ports) for any aspect ratio
  const vfov = THREE.MathUtils.degToRad(camera.fov), hfov = 2 * Math.atan(Math.tan(vfov / 2) * camera.aspect);
  const dist = Math.max(5.0 / Math.tan(hfov / 2), 3.8 / Math.tan(vfov / 2));
  camera.position.copy(new THREE.Vector3(0, 6.2, 7.4).normalize().multiplyScalar(dist));
  camera.lookAt(0, 0, 0);
}

function frame() {
  raf = 0;
  if (!visible) return;
  const busy = stage !== "idle";
  if (!dragging) rotY += busy ? 0.0 : 0.0015;
  board.rotation.y = rotY;
  fan.rotation.y += busy ? 0.45 : 0.08;
  // pulse runs from the input towards the current stage
  const target = STAGE_T[stage] ?? 0;
  pulseT = busy ? (pulseT + 0.012) % Math.max(target, 0.05) : Math.max(0, pulseT - 0.03);
  pulse.visible = busy;
  pulse.position.copy(pathCurve.getPointAt(Math.min(pulseT, 0.999)));
  const glow = stage === "fusion" || stage === "detector" ? 0x1f6feb : stage === "llama" || stage === "loading-llama" ? 0xe3b341 : 0x000000;
  socMat.emissive.setHex(glow); socMat.emissiveIntensity = 0.5 + 0.3 * Math.sin(performance.now() / 180);
  const c = COLORS[verdict] ?? COLORS.idle;
  ledMat.color.setHex(c); ledMat.emissive.setHex(c);
  renderer.render(scene, camera);
  raf = requestAnimationFrame(frame);
}

let ready = false;
window.addEventListener("bastion:page", e => {
  visible = e.detail === "twin";
  if (visible && !ready) ready = init();
  if (visible && ready && !raf) raf = requestAnimationFrame(frame);
});
window.addEventListener("bastion:live", e => {
  stage = e.detail.stage.stage;
  if (e.detail.stage.verdict) verdict = e.detail.stage.verdict;
});
// the page event may have fired before this module loaded
if (location.hash.startsWith("#/twin")) window.dispatchEvent(new CustomEvent("bastion:page", { detail: "twin" }));
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
