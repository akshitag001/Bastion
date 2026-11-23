"""Export Llama 3.2 1B Instruct to an ExecuTorch .pte WITHOUT Docker (Windows / Linux / macOS).

Uses the same versioned config as the container export (twin/export/llama3_2_1b_8da4w.yaml) with local paths, in an
environment that has torch 2.15.0+cpu, executorch 1.6.0 (and torchtune 0.6.1 for --hf-dir conversion).

  # from Meta's original checkpoint (consolidated.00.pth + params.json):
  python twin/export/export_local.py --meta-dir models/llama/hf/original
  # from a Hugging Face snapshot (model.safetensors + config.json): converted to Meta format here with the same
  # key mapping and RoPE un-permutation as torchtune's hf_to_tune + tune_to_meta (which ExecuTorch's
  # examples/models/llama/convert_weights.py calls). torchtune 0.6.1 itself cannot be imported next to the
  # torchao 0.18 that executorch 1.6.0 requires, so the ~30-line mapping is reimplemented below.
  python twin/export/export_local.py --hf-dir models/llama/hf-unsloth

Writes models/llama/llama3_2_1b_8da4w.pte and models/llama/export_info.json (source, versions, sha256, timings).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "twin" / "export" / "llama3_2_1b_8da4w.yaml"
OUT_DIR = ROOT / "models" / "llama"
OUT_NAME = "llama3_2_1b_8da4w.pte"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hf_to_meta(hf_dir: Path, ckpt_out: Path, params_out: Path) -> None:
    import torch
    from safetensors.torch import load_file

    cfg = json.loads((hf_dir / "config.json").read_text())
    n_heads, n_kv, dim = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["hidden_size"]
    head_dim = cfg.get("head_dim") or dim // n_heads
    sd = {}
    for f in sorted(hf_dir.glob("*.safetensors")):
        sd.update(load_file(str(f)))

    def unpermute(w: torch.Tensor, heads: int) -> torch.Tensor:
        # HF stores q/k rows in "rotate_half" order; Meta interleaves pairs (torchtune hf_to_tune _permute)
        return w.view(heads, 2, head_dim // 2, dim).transpose(1, 2).reshape(heads * head_dim, dim)

    out = {"tok_embeddings.weight": sd["model.embed_tokens.weight"], "norm.weight": sd["model.norm.weight"]}
    out["output.weight"] = sd["lm_head.weight"] if "lm_head.weight" in sd else out["tok_embeddings.weight"]
    for i in range(cfg["num_hidden_layers"]):
        h, m = f"model.layers.{i}.", f"layers.{i}."
        out[m + "attention.wq.weight"] = unpermute(sd[h + "self_attn.q_proj.weight"], n_heads)
        out[m + "attention.wk.weight"] = unpermute(sd[h + "self_attn.k_proj.weight"], n_kv)
        out[m + "attention.wv.weight"] = sd[h + "self_attn.v_proj.weight"]
        out[m + "attention.wo.weight"] = sd[h + "self_attn.o_proj.weight"]
        out[m + "feed_forward.w1.weight"] = sd[h + "mlp.gate_proj.weight"]
        out[m + "feed_forward.w2.weight"] = sd[h + "mlp.down_proj.weight"]
        out[m + "feed_forward.w3.weight"] = sd[h + "mlp.up_proj.weight"]
        out[m + "attention_norm.weight"] = sd[h + "input_layernorm.weight"]
        out[m + "ffn_norm.weight"] = sd[h + "post_attention_layernorm.weight"]
    torch.save({k: v.contiguous() for k, v in out.items()}, ckpt_out)
    rs = cfg.get("rope_scaling")
    params = {"dim": dim, "hidden_dim": cfg["intermediate_size"], "n_heads": n_heads, "n_kv_heads": n_kv,
              "n_layers": cfg["num_hidden_layers"], "norm_eps": cfg["rms_norm_eps"], "rope_theta": cfg["rope_theta"],
              "use_scaled_rope": bool(rs), "rope_scale_factor": rs["factor"] if rs else 8, "vocab_size": cfg["vocab_size"]}
    params_out.write_text(json.dumps(params, indent=2) + chr(10))
    print(f"converted {len(out)} tensors -> {ckpt_out.name}; params: {params}")


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$ " + " ".join(cmd).replace(str(ROOT), "."), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)        # list args, no shell


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--meta-dir", type=Path, help="folder with consolidated.00.pth + params.json")
    src.add_argument("--hf-dir", type=Path, help="Hugging Face snapshot folder with model.safetensors + config.json")
    ap.add_argument("--source-note", default="", help="where the weights came from (recorded in export_info.json)")
    a = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    work = OUT_DIR / "meta-converted"
    if a.hf_dir:
        work.mkdir(exist_ok=True)
        ckpt, params = work / "consolidated.00.pth", work / "params.json"
        if not ckpt.exists():
            t0 = time.time()
            hf_to_meta(a.hf_dir, ckpt, params)
            print(f"converted HF -> Meta in {time.time() - t0:.0f}s")
    else:
        ckpt, params = a.meta_dir / "consolidated.00.pth", a.meta_dir / "params.json"

    cfg = CONFIG.read_text(encoding="utf-8")
    cfg = (cfg.replace("/work/models/llama/hf/original/consolidated.00.pth", ckpt.resolve().as_posix())
              .replace("/work/models/llama/hf/original/params.json", params.resolve().as_posix())
              .replace("output_dir: /work/models/llama", f"output_dir: {OUT_DIR.resolve().as_posix()}"))
    with tempfile.TemporaryDirectory() as td:
        local_cfg = Path(td) / "llama3_2_1b_8da4w_local.yaml"
        local_cfg.write_text(cfg, encoding="utf-8")
        t0 = time.time()
        # export_llm writes output_name relative to the working directory (seen with 1.6.0), so run it in OUT_DIR
        run([sys.executable, "-m", "executorch.extension.llm.export.export_llm", "--config", str(local_cfg)], cwd=OUT_DIR)
        export_s = round(time.time() - t0, 1)

    write_info(a, ckpt, export_s)
    return 0


def write_info(a, ckpt: Path, export_s) -> None:
    pte = OUT_DIR / OUT_NAME
    import executorch  # noqa: F401  (version below)
    from importlib.metadata import version
    info = {
        "pte": OUT_NAME, "size_bytes": pte.stat().st_size, "sha256": sha256(pte),
        "config": "twin/export/llama3_2_1b_8da4w.yaml",
        "source": "meta original checkpoint" if a.meta_dir else "hugging face snapshot converted to meta format "
                  "(torchtune hf_to_tune/tune_to_meta mapping, twin/export/export_local.py)",
        "source_note": a.source_note, "checkpoint_sha256": sha256(ckpt),
        "versions": {p: version(p) for p in ("executorch", "torch", "torchao")},
        "export_seconds": export_s, "machine": platform.platform(), "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (OUT_DIR / "export_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    sys.exit(main())
