"""Run complete supervised and open inference for one scene, with provenance."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/pointnext_lod"))
from artifact_cache import file_hash, matches, record, stage_identity


def environment(python: Path):
    """Avoid inheriting base Conda's incompatible NumPy/MKL DLLs on Windows."""
    prefix = python.resolve().parent
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONNOUSERSITE="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", KMP_DUPLICATE_LIB_OK="TRUE",
               CONDA_PREFIX=str(prefix), CONDA_DEFAULT_ENV=prefix.name)
    additions = [prefix, prefix / "Library/mingw-w64/bin", prefix / "Library/usr/bin",
                 prefix / "Library/bin", prefix / "Scripts", prefix / "bin"]
    env["PATH"] = os.pathsep.join([*(str(p) for p in additions), env.get("PATH", "")])
    return env


def find_clip_python():
    conda = os.environ.get("CONDA_EXE", "conda")
    info = json.loads(subprocess.check_output([conda, "env", "list", "--json"]))
    for prefix in info["envs"]:
        if Path(prefix).name.lower() == "pointclipv2_lod":
            return Path(prefix) / ("python.exe" if os.name == "nt" else "bin/python")
    raise RuntimeError("PointCLIPV2_LOD environment not found; use --pointclip-python")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pointnext-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--pointclip-python", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/pointnext/industrial-part-14class-best.pth")
    args = parser.parse_args()
    glb = args.glb.resolve(strict=True)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    stem = glb.stem.replace(" ", "_")
    clip_python = args.pointclip_python or find_clip_python()
    point_root = ROOT / "tools/pointnext_lod/PointNeXt"
    cfg = point_root / "cfgs/industrial_part/pointnext-s-14class-encoder-compare-inference.yaml"
    classes = ROOT / "tools/pointnext_lod/industrial_part_14classes.txt"
    prompts = ROOT / "tools/pointclipv2/industrial_open_vocab_prompts.json"
    clip_root = ROOT / "tools/pointclipv2/PointCLIP_V2"
    clip_script = ROOT / "tools/pointclipv2/run_pointclipv2_zeroshot_glb.py"
    status = {"schema": 1, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "glb": str(glb), "glb_sha256": file_hash(glb),
              "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": file_hash(args.checkpoint),
              "prompt_sha256": file_hash(prompts), "stages": [], "status": "running"}

    def save():
        (out / "inference_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    def execute(name, python, command, stamp=None, identity=None, outputs=None):
        started = time.perf_counter()
        reuse = stamp is not None and matches(stamp, identity, outputs)
        stage = {"name": name, "command": [str(python), *map(str, command)], "reused": reuse}
        status["current_stage"] = name
        save()
        if not reuse:
            log = out / "logs" / f"{name}.log"
            print(f"START {name}; log={log}", flush=True)
            with log.open("wb") as stream:
                result = subprocess.run([str(python), "-u", "-X", "faulthandler", *map(str, command)],
                                        cwd=ROOT, env=environment(python), stdout=stream, stderr=subprocess.STDOUT)
            stage["returncode"] = result.returncode
            if result.returncode:
                status.update(status="failed", error=log.read_text(encoding="utf-8", errors="replace")[-8000:])
                status["stages"].append(stage)
                save()
                raise RuntimeError(f"{name} failed ({result.returncode}); see {log}")
            if stamp is not None:
                record(stamp, identity, outputs)
        stage["wall_seconds"] = time.perf_counter() - started
        status["stages"].append(stage)
        save()
        print(f"DONE {name}: {stage['wall_seconds']:.1f}s; reused={reuse}", flush=True)

    execute("01_pointnext", args.pointnext_python, [ROOT / "tools/pointnext_lod/batch_analyze_glb_lod.py",
            "--resources-dir", glb.parent, "--models", glb.name, "--output-dir", out,
            "--pointnext-root", point_root, "--cfg", cfg, "--ckpt", args.checkpoint.resolve(),
            "--classes-file", classes, "--num-points", "8192", "--batch-size", "16",
            "--pointnext-only", "--skip-existing"])
    pn = out / f"{stem}_pointnext_analysis.csv"
    for name, suffix, candidate, excluded in [("02_pointclip_full", "full", False, ""),
                                               ("03_pointclip_candidates", "candidates", True, "micro_uncertain")]:
        outputs = [out / f"{stem}_pointclipv2_{suffix}.csv", out / f"{stem}_pointclipv2_{suffix}.json",
                   out / f"{stem}_pointclipv2_{suffix}_summary.csv"]
        inputs = [glb, prompts, clip_script, ROOT / "tools/pointnext_lod/gltf_hierarchy.py", clip_root / "zeroshot_cls/trainers",
                  clip_root / "zeroshot_cls/clip", Path(__file__)]
        if candidate:
            inputs.append(pn)
        weight = Path.home() / ".cache/clip/ViT-B-16.pt"
        if weight.exists():
            inputs.append(weight)
        identity = stage_identity(inputs, {"points": 8192, "batch": 8, "top_k": 5, "seed": 2026,
                                         "backbone": "ViT-B/16", "exclude_roles": excluded,
                                         "candidate_gate": [0.70, 0.08] if candidate else None,
                                         "python": str(clip_python)})
        command = [clip_script, "--glb", glb, "--pointclip-root", clip_root, "--prompt-json", prompts,
                   "--num-points", "8192", "--batch-size", "8", "--top-k", "5", "--seed", "2026",
                   "--exclude-roles=" + excluded, "--output-csv", outputs[0],
                   "--output-json", outputs[1], "--summary-csv", outputs[2]]
        if candidate:
            command += ["--candidate-csv", pn, "--candidate-high-confidence", "0.70", "--candidate-low-margin", "0.08"]
        execute(name, clip_python, command, out / f"{stem}_{suffix}.provenance.json", identity, outputs)
    status["status"] = "inference_complete_analysis_pending"
    status["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    save()


if __name__ == "__main__":
    main()
