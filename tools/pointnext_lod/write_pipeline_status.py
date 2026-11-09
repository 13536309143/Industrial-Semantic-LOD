"""Record completed scene artifacts after all three batch stages succeed."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from artifact_cache import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--cfg", type=Path, required=True)
    args = parser.parse_args()
    scenes = []
    for glb in sorted(args.resources_dir.glob("*.glb")):
        stem = glb.stem.replace(" ", "_")
        artifacts = [args.output_dir / (stem + suffix) for suffix in (
            "_pointnext_analysis.csv", "_pointnext_provenance.json",
            "_pointclipv2_candidates.csv", "_pointclipv2_candidates.provenance.json",
            "_pointnext_pointclip_merged.csv", "_lod_constraints_fused.csv",
            "_lod_constraints_fused_summary.json")]
        scenes.append({"scene": glb.name, "artifacts": {p.name: file_hash(p) for p in artifacts}})
    if not scenes:
        raise RuntimeError("No scene outputs to finalize")
    result = {"schema": 1, "status": "scene_inference_and_fusion_complete",
              "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "checkpoint": str(args.ckpt.resolve()), "checkpoint_sha256": file_hash(args.ckpt),
              "config_sha256": file_hash(args.cfg), "scenes": scenes,
              "lod_renderer_cache_status": "must_rebuild_with_current_policies",
              "scope": "Inference artifacts only; not labeled accuracy or LOD quality evaluation."}
    path = args.output_dir / "deployment_status.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    main()
