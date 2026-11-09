"""Generate six LOD policy ablation variants from PointNeXt and PointCLIP outputs.

The variants isolate which evidence source is allowed to choose the final
P1-P10 policy:

1. PointNeXt
2. PointCLIP
3. PointNeXt + structure
4. PointCLIP + structure
5. PointNeXt + PointCLIP
6. PointNeXt + PointCLIP + structure
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple


JOIN_KEYS = ("order", "node_index", "mesh_index")

ABLATION_MODES = [
    ("pointnext", "1_pointnext"),
    ("pointclip", "2_pointclip"),
    ("pointnext_structure", "3_pointnext_structure"),
    ("pointclip_structure", "4_pointclip_structure"),
    ("pointnext_pointclip", "5_pointnext_pointclip"),
    ("pointnext_pointclip_structure", "6_pointnext_pointclip_structure"),
]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build six semantic LOD ablation variants.")
    parser.add_argument("--pointnext-csv", type=Path, required=True)
    parser.add_argument(
        "--pointclip-csv",
        type=Path,
        default=None,
        help="Legacy PointCLIP CSV input. Used for both full and candidate roles if the explicit inputs below are omitted.",
    )
    parser.add_argument(
        "--pointclip-full-csv",
        type=Path,
        default=None,
        help="Full PointCLIP CSV used by PointCLIP-only ablations.",
    )
    parser.add_argument(
        "--pointclip-candidates-csv",
        type=Path,
        default=None,
        help="Candidate-only PointCLIP CSV used by PointNeXt+PointCLIP ablations.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default=None, help="Output filename prefix. Defaults to PointNeXt CSV stem.")
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--high-confidence", type=float, default=0.70)
    parser.add_argument("--low-confidence", type=float, default=0.40)
    parser.add_argument("--low-margin", type=float, default=0.08)
    parser.add_argument("--write-json", action="store_true", help="Also write per-row JSON files.")
    return parser.parse_args()


def load_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def key_for(row: dict) -> Tuple[str, str, str]:
    return tuple(str(row.get(key, "")) for key in JOIN_KEYS)


def pointclip_fields(row: dict | None) -> dict:
    keys = [
        "pointclip_top1_id",
        "pointclip_top1_name",
        "pointclip_top1_role",
        "pointclip_top1_score",
        "pointclip_top2_id",
        "pointclip_top2_name",
        "pointclip_top2_role",
        "pointclip_top2_score",
        "pointclip_margin",
        "pointclip_topk_json",
    ]
    if row is None:
        return {key: "" for key in keys}
    return {key: row.get(key, "") for key in keys}


def merge_rows(pointnext_rows: List[dict], pointclip_rows: List[dict], label: str) -> List[dict]:
    pointclip_by_key = {key_for(row): row for row in pointclip_rows}
    merged = []
    matched = 0
    for row in pointnext_rows:
        clip_row = pointclip_by_key.get(key_for(row))
        out = dict(row)
        out.update(pointclip_fields(clip_row))
        out["pointclip_matched"] = "1" if clip_row else "0"
        matched += int(clip_row is not None)
        merged.append(out)
    print(f"Matched PointCLIP {label} rows: {matched}/{len(pointnext_rows)}")
    return merged


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_lod_builder(project_root: Path):
    module_path = project_root / "tools" / "pointnext_lod" / "build_lod_constraints_from_analysis.py"
    spec = importlib.util.spec_from_file_location("lod_builder", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load LOD builder: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["lod_builder"] = module
    spec.loader.exec_module(module)
    return module


def priority_number(lod_priority: str) -> int:
    if lod_priority.startswith("P"):
        head = lod_priority.split("_", 1)[0][1:]
        if head.isdigit():
            return int(head)
    return 0


def summarize_mode(builder, rows: List[dict], thresholds: dict, mode: str) -> dict:
    summary = builder.counter_to_dict(builder.summarize(rows, thresholds))
    counts = Counter(row["lod_priority"] for row in rows)
    matched = sum(1 for row in rows if str(row.get("pointclip_matched", "0")) == "1")
    cullable = sum(1 for row in rows if str(row.get("allow_cull", "")).lower() in {"true", "1", "yes"})
    preserve = sum(1 for row in rows if priority_number(row["lod_priority"]) >= 8)
    aggressive = sum(1 for row in rows if 1 <= priority_number(row["lod_priority"]) <= 3)
    total = max(1, len(rows))
    summary.update(
        {
            "evidence_mode": mode,
            "pointclip_matched_rows": matched,
            "pointclip_matched_share": matched / total,
            "cullable_count": cullable,
            "cullable_share": cullable / total,
            "preserve_p8_p10_count": preserve,
            "preserve_p8_p10_share": preserve / total,
            "aggressive_p1_p3_count": aggressive,
            "aggressive_p1_p3_share": aggressive / total,
            "lod_priority_counts_ordered": {f"P{i}": sum(v for k, v in counts.items() if priority_number(k) == i) for i in range(1, 11)},
        }
    )
    return summary


def compare_row(summary: dict, output_csv: Path) -> dict:
    counts = summary["lod_priority_counts_ordered"]
    total = max(1, int(summary["num_rows"]))
    row = {
        "mode": summary["evidence_mode"],
        "output_csv": str(output_csv),
        "num_rows": summary["num_rows"],
        "pointclip_matched_rows": summary["pointclip_matched_rows"],
        "pointclip_matched_share": f"{summary['pointclip_matched_share']:.6f}",
        "cullable_count": summary["cullable_count"],
        "cullable_share": f"{summary['cullable_share']:.6f}",
        "aggressive_p1_p3_count": summary["aggressive_p1_p3_count"],
        "aggressive_p1_p3_share": f"{summary['aggressive_p1_p3_share']:.6f}",
        "preserve_p8_p10_count": summary["preserve_p8_p10_count"],
        "preserve_p8_p10_share": f"{summary['preserve_p8_p10_share']:.6f}",
    }
    for i in range(1, 11):
        count = int(counts[f"P{i}"])
        row[f"P{i}_count"] = count
        row[f"P{i}_share"] = f"{count / total:.6f}"
    return row


def main() -> None:
    args = parse_args()
    builder = load_lod_builder(args.project_root)

    pointnext_rows = load_rows(args.pointnext_csv)
    if not pointnext_rows:
        raise ValueError(f"No rows in {args.pointnext_csv}")

    pointclip_full_csv = args.pointclip_full_csv or args.pointclip_csv
    pointclip_candidates_csv = args.pointclip_candidates_csv or args.pointclip_csv
    if pointclip_full_csv is None:
        raise ValueError("Need --pointclip-full-csv or legacy --pointclip-csv")
    if pointclip_candidates_csv is None:
        raise ValueError("Need --pointclip-candidates-csv or legacy --pointclip-csv")

    full_rows = load_rows(pointclip_full_csv)
    candidate_rows = load_rows(pointclip_candidates_csv)
    merged_full_rows = merge_rows(pointnext_rows, full_rows, "full")
    merged_candidate_rows = merge_rows(pointnext_rows, candidate_rows, "candidate")
    thresholds = builder.make_thresholds(pointnext_rows)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or args.pointnext_csv.stem.replace("_pointnext_analysis", "")

    compare_rows = []
    for evidence_mode, prefix in ABLATION_MODES:
        mode_args = argparse.Namespace(
            high_confidence=args.high_confidence,
            low_confidence=args.low_confidence,
            low_margin=args.low_margin,
            evidence_mode=evidence_mode,
        )
        if evidence_mode in {"pointclip", "pointclip_structure"}:
            input_rows = merged_full_rows
        elif evidence_mode in {"pointnext_pointclip", "pointnext_pointclip_structure"}:
            input_rows = merged_candidate_rows
        else:
            input_rows = pointnext_rows

        output_rows = [builder.classify_row(row, thresholds, mode_args) for row in input_rows]
        output_csv = output_dir / f"{stem}_lod_constraints_{prefix}.csv"
        output_summary = output_dir / f"{stem}_lod_constraints_{prefix}_summary.json"
        write_csv(output_csv, output_rows)

        summary = summarize_mode(builder, output_rows, thresholds, evidence_mode)
        output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.write_json:
            output_json = output_dir / f"{stem}_lod_constraints_{prefix}.json"
            output_json.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding="utf-8")

        compare_rows.append(compare_row(summary, output_csv))
        print(f"Wrote {evidence_mode}: {output_csv}")

    compare_csv = output_dir / f"{stem}_lod_ablation_compare.csv"
    write_csv(compare_csv, compare_rows)
    print(f"Wrote comparison: {compare_csv}")

    if any(row["mode"] in {"pointclip", "pointclip_structure"} and float(row["pointclip_matched_share"]) < 0.999 for row in compare_rows):
        print(
            "Warning: PointCLIP-only variants are incomplete because the PointCLIP CSV does not cover every PointNeXt row. "
            "Run PointCLIP without --candidate-csv for a strict PointCLIP-only ablation."
        )


if __name__ == "__main__":
    main()
