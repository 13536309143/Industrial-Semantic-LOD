"""Non-destructive semantic-input audit and opt-in v2 routing preview.

This does not remove meshes, overwrite policies, infer occlusion, or claim
classification accuracy. Routing thresholds are engineering preview settings.
"""

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/pointnext_lod"))
from artifact_cache import file_hash

CRITICAL_CANDIDATES = {"bearings_bushings_guides", "gears_pulleys_chains", "motors_gearmotors",
                       "wheels_castors", "rotating_fluid_machinery", "springs",
                       "joints_clamps_structural_connectors"}


def scene_diagonal(rows):
    low = [min(float(row[f"bbox_min_{axis}"]) for row in rows) for axis in "xyz"]
    high = [max(float(row[f"bbox_max_{axis}"]) for row in rows) for axis in "xyz"]
    return math.dist(low, high)


def route(row, diagonal, small_ratio=.005, large_ratio=.10, complex_faces=10000):
    size = float(row["bbox_diagonal"])
    faces = int(row["face_count"])
    confidence = float(row.get("confidence") or 0)
    margin = confidence-float(row.get("second_confidence") or 0)
    known = confidence >= .70 and margin >= .08
    relative_size = size/diagonal
    critical = known and row.get("predicted_class") in CRITICAL_CANDIDATES
    if size <= 0 or faces <= 0:
        action = "inspect_invalid_geometry"
    elif relative_size >= large_ratio or faces >= complex_faces:
        action = "global_plus_local_multiscale"
    elif relative_size <= small_ratio:
        action = "small_with_critical_evidence" if critical else "small_with_context"
    elif known:
        action = "known_part_evidence"
    else:
        action = "open_coarse_then_fine_or_abstain"
    return {**{k: row[k] for k in ["order", "node_index", "node_name", "mesh_index"]},
            "relative_bbox_diagonal": relative_size, "source_faces": faces,
            "source_pointnext_class": row.get("predicted_class", ""),
            "source_pointnext_confidence": confidence,
            "semantic_status": "known_candidate" if known else "unknown_or_ambiguous",
            "importance_status": "candidate_functional_preserve" if critical else "not_assessed",
            "routing_action": action, "visibility_status": "unmeasured",
            "allow_geometry_drop": False, "skip_all_semantics": False,
            "reason": "Preview only. Smallness/uncertainty does not establish irrelevance or invisibility."}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointnext-csv", type=Path, required=True)
    parser.add_argument("--pointclip-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    read = lambda path: list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    rows, clip = read(args.pointnext_csv), read(args.pointclip_csv)
    diagonal = scene_diagonal(rows)
    total_faces = sum(int(row["face_count"]) for row in rows)
    clip_by_key = {(row["order"], row["node_index"], row["mesh_index"]): row for row in clip}
    assert len(clip_by_key) == len(clip) == len(rows)
    sys.path.insert(0, str(ROOT / "tools/pointclipv2"))
    from build_lod_ablation_variants import load_lod_builder
    builder = load_lod_builder(ROOT)
    bins = []
    for lower, upper in [(0,.005),(.005,.01),(.01,.02),(.02,.10),(.10,float("inf"))]:
        subset = [row for row in rows if lower<float(row["bbox_diagonal"])/diagonal<=upper]
        if not subset:
            continue
        bins.append({"relative_size_min_exclusive": lower, "relative_size_max_inclusive": upper if math.isfinite(upper) else "inf",
                     "count":len(subset), "record_percent":100*len(subset)/len(rows),
                     "triangle_percent":100*sum(int(row["face_count"]) for row in subset)/total_faces,
                     "mean_pointnext_confidence":statistics.mean(float(row["confidence"]) for row in subset),
                     "accepted_percent":100*sum(float(row["confidence"])>=.7 for row in subset)/len(subset),
                     "usable_open_percent":100*sum(bool(builder.pointclip_role_scores(clip_by_key[(row['order'],row['node_index'],row['mesh_index'])],True)) for row in subset)/len(subset)})
    preview = [route(row, diagonal) for row in rows]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    write_csv(args.output_dir/"size_bin_statistics.csv",bins)
    write_csv(args.output_dir/"semantic_routing_v2_preview.csv",preview)
    summary = {"schema":1,"scene_diagonal":diagonal,"rows":len(rows),"source_triangles":total_faces,
               "bins":bins,"routing_counts":dict(Counter(row["routing_action"] for row in preview)),
               "visibility_unmeasured":True,"geometry_removed":0,"production_policies_modified":False,
               "accuracy_measured":False,"settings":{"small_ratio":.005,"large_ratio":.10,"complex_faces":10000},
               "threshold_scope":"Scheduling preview only, not calibrated semantic or deletion criteria.",
               "inputs_sha256":{str(path.resolve()):file_hash(path) for path in [args.pointnext_csv,args.pointclip_csv]},
               "source_sha256":file_hash(Path(__file__))}
    (args.output_dir/"routing_audit_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding="utf-8")
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__ == "__main__":
    main()
