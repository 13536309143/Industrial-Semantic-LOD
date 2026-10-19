"""Batch PointNeXt + feature LOD analysis for multiple GLB models.

For each GLB, this runs:
1) PointNeXt per-mesh classification and geometry extraction.
2) Feature/LOD post-processing from the PointNeXt analysis CSV.

The output keeps one row per original mesh, so downstream LOD systems can still
address the original mesh/node identifiers.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from artifact_cache import matches, record, stage_identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch analyze GLB files for LOD constraints.")
    parser.add_argument(
        "--resources-dir",
        type=Path,
        required=True,
        help="Directory containing GLB files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lod_analysis_outputs"),
        help="Directory for generated CSV/JSON files.",
    )
    parser.add_argument("--pointnext-root", type=Path, required=True)
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--classes-file", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional explicit GLB file list. If omitted, all *.glb files under --resources-dir are processed.",
    )
    parser.add_argument("--num-points", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--pointnext-only",
        action="store_true",
        help="Only run PointNeXt analysis, skip LOD feature post-processing.",
    )
    return parser.parse_args()


def run(cmd: list[str], cwd: Path) -> None:
    print("\n> " + " ".join(f'"{item}"' if " " in item else item for item in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def model_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")


def main() -> None:
    args = parse_args()
    resources_dir = args.resources_dir.resolve()
    output_dir = args.output_dir.resolve()
    script_dir = Path(__file__).resolve().parent
    analyze_script = script_dir / "analyze_large_glb_parts_pointnext.py"
    lod_script = script_dir / "build_lod_constraints_from_analysis.py"
    models = args.models
    if models is None:
        models = sorted(path.name for path in resources_dir.glob("*.glb") if path.is_file())
        if not models:
            raise FileNotFoundError(f"No .glb files found in {resources_dir}")

    missing = []
    for name in models:
        path = resources_dir / name
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing GLB files:\n" + "\n".join(missing))

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Models:")
    for name in models:
        print(f"  {name}")

    for name in models:
        glb_path = resources_dir / name
        stem = model_stem(glb_path)
        analysis_csv = output_dir / f"{stem}_pointnext_analysis.csv"
        analysis_json = output_dir / f"{stem}_pointnext_analysis.json"
        summary_csv = output_dir / f"{stem}_pointnext_class_summary.csv"
        review_csv = output_dir / f"{stem}_pointnext_review_candidates.csv"
        lod_csv = output_dir / f"{stem}_lod_constraints.csv"
        lod_summary = output_dir / f"{stem}_lod_constraints_summary.json"

        analysis_outputs = [analysis_csv, analysis_json, summary_csv, review_csv]
        provenance = output_dir / f"{stem}_pointnext_provenance.json"
        identity = stage_identity(
            [glb_path, args.ckpt, args.cfg, args.classes_file, analyze_script,
             script_dir / "analyze_gltf_parts_pointnext.py",
             script_dir / "checkpoint_loading.py", script_dir / "artifact_cache.py",
             args.pointnext_root / "openpoints"],
            {"num_points": args.num_points, "batch_size": args.batch_size,
             "device": args.device, "seed": 2026, "inference_mode": "mesh"},
        )
        if not (args.skip_existing and matches(provenance, identity, analysis_outputs)):
            run(
                [
                    sys.executable,
                    str(analyze_script),
                    "--glb",
                    str(glb_path),
                    "--pointnext-root",
                    str(args.pointnext_root.resolve()),
                    "--cfg",
                    str(args.cfg.resolve()),
                    "--ckpt",
                    str(args.ckpt.resolve()),
                    "--classes-file",
                    str(args.classes_file.resolve()),
                    "--num-points",
                    str(args.num_points),
                    "--batch-size",
                    str(args.batch_size),
                    "--device",
                    args.device,
                    "--inference-mode",
                    "mesh",
                    "--output-csv",
                    str(analysis_csv),
                    "--output-json",
                    str(analysis_json),
                    "--summary-csv",
                    str(summary_csv),
                    "--review-csv",
                    str(review_csv),
                ],
                cwd=script_dir,
            )
            record(provenance, identity, analysis_outputs)
        else:
            print(f"Reuse verified PointNeXt output: {analysis_csv}")

        if args.pointnext_only:
            continue

        lod_provenance = output_dir / f"{stem}_lod_provenance.json"
        lod_identity = stage_identity([analysis_csv, lod_script], {})
        if not (args.skip_existing and matches(lod_provenance, lod_identity, [lod_csv, lod_summary])):
            run(
                [
                    sys.executable,
                    str(lod_script),
                    "--input-csv",
                    str(analysis_csv),
                    "--output-csv",
                    str(lod_csv),
                    "--summary-json",
                    str(lod_summary),
                ],
                cwd=script_dir,
            )
            record(lod_provenance, lod_identity, [lod_csv, lod_summary])
        else:
            print(f"Skip existing LOD output: {lod_csv}")

    print(f"\nDone. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
