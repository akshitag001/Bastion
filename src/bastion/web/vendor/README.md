# Vendored frontend assets (served locally; the UI makes no external requests)

| Path | What | Source / licence |
|---|---|---|
| `three.module.min.js` | three.js r170 | npm `three@0.170.0`, MIT (`THREE_LICENSE.txt`) |
| `addons/GLTFLoader.js`, `addons/BufferGeometryUtils.js` | glTF loader + helper | `three@0.170.0/examples/jsm`, MIT (`THREE_LICENSE.txt`). Only change: the `'three'` import is rewritten to `/static/vendor/three.module.min.js` so no import map or CDN is needed. |
| `models/pi-board.glb` | Raspberry Pi 4 Model B board assembly (glTF 2.0, 1.26 MB, part names kept for hover labels) | Supplied by the project team (SketchUp assembly converted to glTF with Blender). Used as a stand-in for the Pi 5 board in the Device Twin view. |
| `fonts/*.woff2` | Space Grotesk, Inter, JetBrains Mono | SIL Open Font License 1.1 (`fonts/OFL-LICENSE.txt`) |
