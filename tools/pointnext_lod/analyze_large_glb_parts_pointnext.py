"""Stream-analyze very large binary GLB files with a trained PointNeXt model.

This is designed for GLB files with tens of thousands of mesh nodes. It reads
the GLB JSON and BIN chunks directly, decodes one mesh node at a time, batches
sampled point clouds for PointNeXt inference, and streams CSV/JSON output.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import mmap
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from checkpoint_loading import load_classifier_checkpoint
from gltf_hierarchy import attach_world_matrices

from analyze_gltf_parts_pointnext import (
    load_cfg,
    load_classes,
    normalize_unit_sphere,
    resolve_classes_file,
    setup_pointnext,
    shape_hint,
)


COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
TYPE_COUNTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT4": 16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream classify every mesh node in a large GLB.")
    parser.add_argument("--glb", type=Path, default=Path("c.glb"))
    parser.add_argument("--pointnext-root", type=Path, default=Path("PointNeXt"))
    parser.add_argument(
        "--cfg",
        type=Path,
        default=Path("PointNeXt/cfgs/industrial_part/pointnext-s-14class-encoder-compare-inference.yaml"),
    )
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--classes-file", type=Path, default=Path("industrial_part_14classes.txt"))
    parser.add_argument("--num-points", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N mesh nodes for smoke tests.")
    parser.add_argument("--output-csv", type=Path, default=Path("glb_part_analysis_c.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("glb_part_analysis_c.json"))
    parser.add_argument("--summary-csv", type=Path, default=Path("glb_part_class_summary_c.csv"))
    parser.add_argument("--review-csv", type=Path, default=Path("glb_part_review_candidates_c.csv"))
    parser.add_argument(
        "--inference-mode",
        choices=["mesh", "grid-group"],
        default="mesh",
        help="mesh: classify each mesh independently. grid-group: classify centroid grid groups and copy group predictions to each mesh.",
    )
    parser.add_argument(
        "--group-cell-size",
        type=float,
        default=1.0,
        help="Spatial grid size in GLB world units for --inference-mode grid-group.",
    )
    parser.add_argument(
        "--min-group-members",
        type=int,
        default=2,
        help="Groups smaller than this are classified as individual meshes.",
    )
    return parser.parse_args()


def read_glb_chunks(path: Path) -> tuple[dict, int, int]:
    with path.open("rb") as file:
        magic, version, total_length = struct.unpack("<III", file.read(12))
        if magic != 0x46546C67 or version != 2:
            raise ValueError(f"Not a GLB v2 file: {path}")

        json_len, json_type = struct.unpack("<II", file.read(8))
        if json_type != int.from_bytes(b"JSON", "little"):
            raise ValueError("First GLB chunk is not JSON")
        gltf = json.loads(file.read(json_len).decode("utf-8").rstrip("\x00 \t\r\n"))

        bin_header_offset = 12 + 8 + json_len
        file.seek(bin_header_offset)
        bin_len, bin_type = struct.unpack("<II", file.read(8))
        if bin_type != int.from_bytes(b"BIN\x00", "little"):
            raise ValueError("Second GLB chunk is not BIN")
        bin_offset = bin_header_offset + 8
        attach_world_matrices(gltf, node_matrix)
        return gltf, bin_offset, bin_len


def accessor_array(gltf: dict, mm: mmap.mmap, bin_offset: int, accessor_index: int) -> np.ndarray:
    accessor = gltf["accessors"][accessor_index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    dtype = COMPONENT_DTYPES[accessor["componentType"]]
    item_count = TYPE_COUNTS[accessor["type"]]
    count = int(accessor["count"])
    accessor_offset = int(accessor.get("byteOffset", 0))
    view_offset = int(view.get("byteOffset", 0))
    stride = view.get("byteStride")
    start = bin_offset + view_offset + accessor_offset

    if stride is None:
        total = count * item_count
        arr = np.frombuffer(mm, dtype=dtype, count=total, offset=start)
        if item_count > 1:
            arr = arr.reshape(count, item_count)
        return np.asarray(arr)

    item_size = np.dtype(dtype).itemsize * item_count
    rows = []
    for row in range(count):
        offset = start + row * int(stride)
        rows.append(np.frombuffer(mm, dtype=dtype, count=item_count, offset=offset).copy())
    arr = np.stack(rows, axis=0)
    return arr.reshape(count, item_count) if item_count > 1 else arr.reshape(count)


def quaternion_to_matrix(q: Iterable[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T
    translation = np.asarray(node.get("translation", [0, 0, 0]), dtype=np.float64)
    scale = np.asarray(node.get("scale", [1, 1, 1]), dtype=np.float64)
    rotation = quaternion_to_matrix(node.get("rotation", [0, 0, 0, 1]))
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation @ np.diag(scale)
    matrix[:3, 3] = translation
    return matrix


def apply_transform(vertices: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    vertices64 = vertices.astype(np.float64, copy=False)
    return (vertices64 @ matrix[:3, :3].T + matrix[:3, 3]).astype(np.float32)


def mesh_for_node(gltf: dict, mm: mmap.mmap, bin_offset: int, node: dict) -> tuple[np.ndarray, np.ndarray]:
    mesh = gltf["meshes"][node["mesh"]]
    transform = node.get("_world_matrix")
    if transform is None:
        attach_world_matrices(gltf, node_matrix)
        transform = node["_world_matrix"]
    vertices_list = []
    faces_list = []
    vertex_offset = 0

    for primitive in mesh.get("primitives", []):
        position_accessor = primitive.get("attributes", {}).get("POSITION")
        if position_accessor is None:
            continue
        vertices = accessor_array(gltf, mm, bin_offset, position_accessor).astype(np.float32, copy=False)
        vertices = apply_transform(vertices, transform)

        indices_accessor = primitive.get("indices")
        if indices_accessor is not None:
            indices = accessor_array(gltf, mm, bin_offset, indices_accessor).astype(np.int64, copy=False)
            faces = indices.reshape(-1, 3) if len(indices) >= 3 else np.empty((0, 3), dtype=np.int64)
        else:
            faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)

        vertices_list.append(vertices)
        if len(faces) > 0:
            faces_list.append(faces + vertex_offset)
        vertex_offset += len(vertices)

    if not vertices_list:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int64)
    vertices = np.concatenate(vertices_list, axis=0)
    faces = np.concatenate(faces_list, axis=0) if faces_list else np.empty((0, 3), dtype=np.int64)
    return vertices, faces


def sample_points(vertices: np.ndarray, faces: np.ndarray, num_points: int, rng: np.random.Generator) -> np.ndarray:
    if len(vertices) == 0:
        raise ValueError("mesh has no vertices")
    if len(faces) == 0:
        replace = len(vertices) < num_points
        return vertices[rng.choice(len(vertices), size=num_points, replace=replace)].astype(np.float32)

    tri = vertices[faces]
    edge1 = tri[:, 1] - tri[:, 0]
    edge2 = tri[:, 2] - tri[:, 0]
    areas = np.linalg.norm(np.cross(edge1, edge2), axis=1) * 0.5
    valid = np.isfinite(areas) & (areas > 1e-12)
    if not np.any(valid):
        replace = len(vertices) < num_points
        return vertices[rng.choice(len(vertices), size=num_points, replace=replace)].astype(np.float32)
    tri = tri[valid]
    areas = areas[valid]
    probs = areas / areas.sum()
    chosen = tri[rng.choice(len(tri), size=num_points, replace=True, p=probs)]
    r1 = np.sqrt(rng.random((num_points, 1), dtype=np.float32))
    r2 = rng.random((num_points, 1), dtype=np.float32)
    points = (1 - r1) * chosen[:, 0] + r1 * (1 - r2) * chosen[:, 1] + r1 * r2 * chosen[:, 2]
    return points.astype(np.float32)


def basic_stats(vertices: np.ndarray, faces: np.ndarray) -> dict:
    bounds_min = vertices.min(axis=0).astype(float)
    bounds_max = vertices.max(axis=0).astype(float)
    size = bounds_max - bounds_min
    diag = float(np.linalg.norm(size))
    bbox_volume = float(np.prod(np.maximum(size, 0.0)))
    ordered = sorted((float(v) for v in size), reverse=True)
    long_side, mid_side, short_side = ordered
    elongation = long_side / max(mid_side, 1e-12)
    flatness = short_side / max(long_side, 1e-12)
    compactness = bbox_volume / max(diag**3, 1e-12)
    centroid = vertices.mean(axis=0).astype(float)

    surface_area = 0.0
    if len(faces) > 0:
        tri = vertices[faces]
        surface_area = float((np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * 0.5).sum())

    return {
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "surface_area": surface_area,
        "mesh_volume": "",
        "bbox_min_x": float(bounds_min[0]),
        "bbox_min_y": float(bounds_min[1]),
        "bbox_min_z": float(bounds_min[2]),
        "bbox_max_x": float(bounds_max[0]),
        "bbox_max_y": float(bounds_max[1]),
        "bbox_max_z": float(bounds_max[2]),
        "bbox_size_x": float(size[0]),
        "bbox_size_y": float(size[1]),
        "bbox_size_z": float(size[2]),
        "bbox_diagonal": diag,
        "bbox_volume": bbox_volume,
        "obb_size_long": long_side,
        "obb_size_mid": mid_side,
        "obb_size_short": short_side,
        "elongation": float(elongation),
        "flatness": float(flatness),
        "compactness": float(compactness),
        "shape_hint": shape_hint(long_side, mid_side, short_side, flatness, elongation),
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_z": float(centroid[2]),
    }


def build_model(cfg, build_model_from_cfg, load_checkpoint, ckpt_path: Path, device: torch.device):
    model = build_model_from_cfg(cfg.model).to(device)
    load_classifier_checkpoint(model, ckpt_path)
    model.eval()
    return model


def predict_batch(model, points_batch: list[np.ndarray], device: torch.device, use_amp: bool):
    points = torch.from_numpy(np.stack(points_batch, axis=0)).to(device=device, non_blocking=True)
    data = {"pos": points.contiguous(), "x": points.transpose(1, 2).contiguous()}
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
        probs = torch.softmax(model(data).float(), dim=1)
        top2 = torch.topk(probs, k=min(2, probs.shape[1]), dim=1)
    return top2.indices.cpu().numpy(), top2.values.cpu().numpy()


FIELDNAMES = [
    "order", "node_index", "node_name", "mesh_index", "mesh_name",
    "inference_mode", "group_id", "group_size",
    "predicted_label", "predicted_class", "confidence",
    "second_label", "second_class", "second_confidence",
    "vertex_count", "face_count", "surface_area", "mesh_volume",
    "bbox_min_x", "bbox_min_y", "bbox_min_z", "bbox_max_x", "bbox_max_y", "bbox_max_z",
    "bbox_size_x", "bbox_size_y", "bbox_size_z", "bbox_diagonal", "bbox_volume",
    "obb_size_long", "obb_size_mid", "obb_size_short", "elongation", "flatness",
    "compactness", "shape_hint", "centroid_x", "centroid_y", "centroid_z",
]


def update_stats(class_stats, total_conf, shape_counter, row: dict) -> None:
    confidence = float(row["confidence"])
    stats = class_stats[row["predicted_class"]]
    stats["count"] += 1
    stats["conf_sum"] += confidence
    stats["min_confidence"] = min(stats["min_confidence"], confidence)
    stats["max_confidence"] = max(stats["max_confidence"], confidence)
    stats["low_conf_lt_0_4"] += int(confidence < 0.4)
    stats["shape_counter"][row["shape_hint"]] += 1
    total_conf.append(confidence)
    shape_counter[row["shape_hint"]] += 1


def write_review_if_needed(review_writer, row: dict) -> bool:
    confidence = float(row["confidence"])
    second_conf = float(row["second_confidence"])
    margin = confidence - second_conf
    if confidence >= 0.4 and margin >= 0.08:
        return False
    review_row = {
        key: row[key]
        for key in [
            "order", "node_name", "predicted_class", "confidence",
            "second_class", "second_confidence", "shape_hint",
            "bbox_size_x", "bbox_size_y", "bbox_size_z", "surface_area",
        ]
    }
    review_row["margin"] = margin
    review_row["reason"] = "low_confidence" if confidence < 0.4 else "low_margin"
    review_writer.writerow(review_row)
    return True


def write_summary(summary_csv: Path, class_stats: dict) -> list[dict]:
    summary_rows = []
    for cls, stats in sorted(class_stats.items(), key=lambda item: (-item[1]["count"], item[0])):
        count = stats["count"]
        summary_rows.append({
            "predicted_class": cls,
            "count": count,
            "avg_confidence": stats["conf_sum"] / max(count, 1),
            "min_confidence": stats["min_confidence"],
            "max_confidence": stats["max_confidence"],
            "low_conf_lt_0_4": stats["low_conf_lt_0_4"],
            "shape_counts": "; ".join(f"{name}:{num}" for name, num in stats["shape_counter"].most_common()),
        })
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)
    return summary_rows


def group_key_for_row(row: dict, cell_size: float) -> str:
    if cell_size <= 0:
        raise ValueError("--group-cell-size must be > 0")
    ix = math.floor(float(row["centroid_x"]) / cell_size)
    iy = math.floor(float(row["centroid_y"]) / cell_size)
    iz = math.floor(float(row["centroid_z"]) / cell_size)
    return f"grid_{ix}_{iy}_{iz}"


def sample_group_points(
    gltf: dict,
    mm: mmap.mmap,
    bin_offset: int,
    group_items: list[tuple[int, dict, dict]],
    num_points: int,
    seed: int,
) -> np.ndarray:
    areas = [max(float(row["surface_area"]), 1e-12) for _, _, row in group_items]
    total_area = sum(areas)
    counts = [max(1, int(round(num_points * area / total_area))) for area in areas]
    diff = num_points - sum(counts)
    if diff != 0:
        order = sorted(range(len(counts)), key=lambda idx: areas[idx], reverse=True)
        step = 1 if diff > 0 else -1
        remaining = abs(diff)
        cursor = 0
        while remaining > 0 and order:
            idx = order[cursor % len(order)]
            if step > 0 or counts[idx] > 1:
                counts[idx] += step
                remaining -= 1
            cursor += 1

    sampled_parts = []
    for (node_index, node, _), count in zip(group_items, counts):
        vertices, faces = mesh_for_node(gltf, mm, bin_offset, node)
        rng = np.random.default_rng(seed + node_index)
        sampled_parts.append(sample_points(vertices, faces, count, rng))
    points = np.concatenate(sampled_parts, axis=0)
    rng = np.random.default_rng(seed)
    if len(points) > num_points:
        points = points[rng.choice(len(points), size=num_points, replace=False)]
    elif len(points) < num_points:
        points = points[rng.choice(len(points), size=num_points, replace=True)]
    return normalize_unit_sphere(points.astype(np.float32, copy=False))


def run_grid_grouped(args, gltf, bin_offset, mesh_nodes, model, class_names, device, use_amp) -> None:
    print("Pass 1/2: computing per-mesh geometry and grid groups")
    rows: list[dict] = []
    groups: dict[str, list[tuple[int, dict, dict]]] = defaultdict(list)

    with args.glb.open("rb") as glb_file, mmap.mmap(glb_file.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for order, (node_index, node) in enumerate(mesh_nodes, start=1):
            vertices, faces = mesh_for_node(gltf, mm, bin_offset, node)
            if len(vertices) == 0:
                continue
            mesh = gltf["meshes"][node["mesh"]]
            row = {
                "order": order,
                "node_index": node_index,
                "node_name": node.get("name", f"node_{node_index}"),
                "mesh_index": int(node["mesh"]),
                "mesh_name": mesh.get("name", f"mesh_{node['mesh']}"),
                "inference_mode": "grid-group",
                "group_id": "",
                "group_size": "",
                "predicted_label": "",
                "predicted_class": "",
                "confidence": "",
                "second_label": "",
                "second_class": "",
                "second_confidence": "",
                **basic_stats(vertices, faces),
            }
            key = group_key_for_row(row, args.group_cell_size)
            rows.append(row)
            groups[key].append((node_index, node, row))
            if order % 5000 == 0:
                print(f"  geometry {order}/{len(mesh_nodes)}")

    expanded_groups: dict[str, list[tuple[int, dict, dict]]] = {}
    for key, items in groups.items():
        if len(items) < args.min_group_members:
            for node_index, node, row in items:
                expanded_groups[f"mesh_{row['order']}"] = [(node_index, node, row)]
        else:
            expanded_groups[key] = items
    groups = expanded_groups
    print(f"Groups for inference: {len(groups)}")

    group_predictions: dict[str, tuple[int, float, int, float]] = {}
    pending_ids: list[str] = []
    pending_points: list[np.ndarray] = []

    def flush_group_batch() -> None:
        if not pending_points:
            return
        labels, scores = predict_batch(model, pending_points, device, use_amp)
        for group_id, row_labels, row_scores in zip(pending_ids, labels, scores):
            label = int(row_labels[0])
            confidence = float(row_scores[0])
            second_label = int(row_labels[1]) if len(row_labels) > 1 else -1
            second_conf = float(row_scores[1]) if len(row_scores) > 1 else 0.0
            group_predictions[group_id] = (label, confidence, second_label, second_conf)
        pending_ids.clear()
        pending_points.clear()

    print("Pass 2/2: sampling grouped geometry and running inference")
    with args.glb.open("rb") as glb_file, mmap.mmap(glb_file.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for index, (group_id, items) in enumerate(groups.items(), start=1):
            pending_ids.append(group_id)
            pending_points.append(sample_group_points(gltf, mm, bin_offset, items, args.num_points, args.seed + index))
            if len(pending_points) >= args.batch_size:
                flush_group_batch()
            if index % 1000 == 0:
                print(f"  groups {index}/{len(groups)}")
        flush_group_batch()

    group_sizes = {group_id: len(items) for group_id, items in groups.items()}
    row_group_ids = {}
    for group_id, items in groups.items():
        for _, _, row in items:
            row_group_ids[int(row["order"])] = group_id

    class_stats: dict[str, dict] = defaultdict(lambda: {
        "count": 0,
        "conf_sum": 0.0,
        "min_confidence": 1.0,
        "max_confidence": 0.0,
        "low_conf_lt_0_4": 0,
        "shape_counter": Counter(),
    })
    total_conf = []
    shape_counter = Counter()
    review_count = 0

    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as csv_file, \
            args.review_csv.open("w", encoding="utf-8-sig", newline="") as review_file, \
            args.output_json.open("w", encoding="utf-8") as json_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        csv_writer.writeheader()
        review_fields = [
            "order", "node_name", "predicted_class", "confidence",
            "second_class", "second_confidence", "shape_hint",
            "bbox_size_x", "bbox_size_y", "bbox_size_z", "surface_area",
            "margin", "reason",
        ]
        review_writer = csv.DictWriter(review_file, fieldnames=review_fields)
        review_writer.writeheader()
        json_file.write("[\n")
        first_json = True
        for row in sorted(rows, key=lambda item: int(item["order"])):
            group_id = row_group_ids[int(row["order"])]
            label, confidence, second_label, second_conf = group_predictions[group_id]
            row.update({
                "group_id": group_id,
                "group_size": group_sizes[group_id],
                "predicted_label": label,
                "predicted_class": class_names[label] if 0 <= label < len(class_names) else str(label),
                "confidence": confidence,
                "second_label": second_label,
                "second_class": class_names[second_label] if 0 <= second_label < len(class_names) else "",
                "second_confidence": second_conf,
            })
            csv_writer.writerow(row)
            if not first_json:
                json_file.write(",\n")
            json.dump(row, json_file, ensure_ascii=False)
            first_json = False
            update_stats(class_stats, total_conf, shape_counter, row)
            review_count += int(write_review_if_needed(review_writer, row))
        json_file.write("\n]\n")

    summary_rows = write_summary(args.summary_csv, class_stats)
    print(f"Wrote: {args.output_csv.resolve()}")
    print(f"Wrote: {args.output_json.resolve()}")
    print(f"Wrote: {args.summary_csv.resolve()}")
    print(f"Wrote: {args.review_csv.resolve()}")
    print(f"Rows: {len(total_conf)}, avg_conf={float(np.mean(total_conf)):.3f}, median_conf={float(np.median(total_conf)):.3f}")
    print(f"Review candidates: {review_count}")
    print("Top classes:")
    for row in summary_rows[:10]:
        print(f"  {row['predicted_class']}: {row['count']} avg_conf={row['avg_confidence']:.3f}")


def main() -> None:
    args = parse_args()
    pointnext_root, EasyConfig, build_model_from_cfg, load_checkpoint = setup_pointnext(args.pointnext_root)
    cfg = load_cfg(args.cfg, EasyConfig)
    classes_file = resolve_classes_file(args, cfg, pointnext_root)
    class_names = load_classes(classes_file)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(args.device)
    use_amp = bool(cfg.get("use_amp", False)) and not args.no_amp

    print(f"Reading GLB structure: {args.glb.resolve()}")
    gltf, bin_offset, bin_len = read_glb_chunks(args.glb.resolve())
    mesh_nodes = [(idx, node) for idx, node in enumerate(gltf.get("nodes", [])) if "mesh" in node]
    if args.limit is not None:
        mesh_nodes = mesh_nodes[: args.limit]
    print(f"Mesh nodes to process: {len(mesh_nodes)}")
    print(f"Points per part: {args.num_points}, batch size: {args.batch_size}")
    print(f"Inference mode: {args.inference_mode}")

    model = build_model(cfg, build_model_from_cfg, load_checkpoint, args.ckpt.resolve(), device)

    if args.inference_mode == "grid-group":
        run_grid_grouped(args, gltf, bin_offset, mesh_nodes, model, class_names, device, use_amp)
        return

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.review_csv.parent.mkdir(parents=True, exist_ok=True)

    class_stats: dict[str, dict] = defaultdict(lambda: {
        "count": 0,
        "conf_sum": 0.0,
        "min_confidence": 1.0,
        "max_confidence": 0.0,
        "low_conf_lt_0_4": 0,
        "shape_counter": Counter(),
    })
    total_conf = []
    shape_counter = Counter()
    review_count = 0

    pending_points: list[np.ndarray] = []
    pending_rows: list[dict] = []

    def flush_batch(csv_writer, review_writer, json_file, first_json: bool) -> bool:
        nonlocal review_count
        if not pending_points:
            return first_json
        labels, scores = predict_batch(model, pending_points, device, use_amp)
        for row, row_labels, row_scores in zip(pending_rows, labels, scores):
            label = int(row_labels[0])
            confidence = float(row_scores[0])
            second_label = int(row_labels[1]) if len(row_labels) > 1 else -1
            second_conf = float(row_scores[1]) if len(row_scores) > 1 else 0.0
            row.update({
                "predicted_label": label,
                "predicted_class": class_names[label] if 0 <= label < len(class_names) else str(label),
                "confidence": confidence,
                "second_label": second_label,
                "second_class": class_names[second_label] if 0 <= second_label < len(class_names) else "",
                "second_confidence": second_conf,
            })
            csv_writer.writerow(row)
            if not first_json:
                json_file.write(",\n")
            json.dump(row, json_file, ensure_ascii=False)
            first_json = False

            stats = class_stats[row["predicted_class"]]
            stats["count"] += 1
            stats["conf_sum"] += confidence
            stats["min_confidence"] = min(stats["min_confidence"], confidence)
            stats["max_confidence"] = max(stats["max_confidence"], confidence)
            stats["low_conf_lt_0_4"] += int(confidence < 0.4)
            stats["shape_counter"][row["shape_hint"]] += 1
            total_conf.append(confidence)
            shape_counter[row["shape_hint"]] += 1

            margin = confidence - second_conf
            if confidence < 0.4 or margin < 0.08:
                review_row = {
                    key: row[key]
                    for key in [
                        "order", "node_name", "predicted_class", "confidence",
                        "second_class", "second_confidence", "shape_hint",
                        "bbox_size_x", "bbox_size_y", "bbox_size_z", "surface_area",
                    ]
                }
                review_row["margin"] = margin
                review_row["reason"] = "low_confidence" if confidence < 0.4 else "low_margin"
                review_writer.writerow(review_row)
                review_count += 1
        pending_points.clear()
        pending_rows.clear()
        return first_json

    with args.glb.open("rb") as glb_file, mmap.mmap(glb_file.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        with args.output_csv.open("w", encoding="utf-8-sig", newline="") as csv_file, \
                args.review_csv.open("w", encoding="utf-8-sig", newline="") as review_file, \
                args.output_json.open("w", encoding="utf-8") as json_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
            csv_writer.writeheader()
            review_fields = [
                "order", "node_name", "predicted_class", "confidence",
                "second_class", "second_confidence", "shape_hint",
                "bbox_size_x", "bbox_size_y", "bbox_size_z", "surface_area",
                "margin", "reason",
            ]
            review_writer = csv.DictWriter(review_file, fieldnames=review_fields)
            review_writer.writeheader()
            json_file.write("[\n")
            first_json = True

            for order, (node_index, node) in enumerate(mesh_nodes, start=1):
                vertices, faces = mesh_for_node(gltf, mm, bin_offset, node)
                if len(vertices) == 0:
                    continue
                rng = np.random.default_rng(args.seed + order)
                sampled = normalize_unit_sphere(sample_points(vertices, faces, args.num_points, rng))
                mesh = gltf["meshes"][node["mesh"]]
                row = {
                    "order": order,
                    "node_index": node_index,
                    "node_name": node.get("name", f"node_{node_index}"),
                    "mesh_index": int(node["mesh"]),
                    "mesh_name": mesh.get("name", f"mesh_{node['mesh']}"),
                    "inference_mode": "mesh",
                    "group_id": f"mesh_{order}",
                    "group_size": 1,
                    "predicted_label": "",
                    "predicted_class": "",
                    "confidence": "",
                    "second_label": "",
                    "second_class": "",
                    "second_confidence": "",
                    **basic_stats(vertices, faces),
                }
                pending_points.append(sampled)
                pending_rows.append(row)
                if len(pending_points) >= args.batch_size:
                    first_json = flush_batch(csv_writer, review_writer, json_file, first_json)
                if order % 1000 == 0:
                    print(f"Processed {order}/{len(mesh_nodes)}")

            first_json = flush_batch(csv_writer, review_writer, json_file, first_json)
            json_file.write("\n]\n")

    summary_rows = []
    for cls, stats in sorted(class_stats.items(), key=lambda item: (-item[1]["count"], item[0])):
        count = stats["count"]
        summary_rows.append({
            "predicted_class": cls,
            "count": count,
            "avg_confidence": stats["conf_sum"] / max(count, 1),
            "min_confidence": stats["min_confidence"],
            "max_confidence": stats["max_confidence"],
            "low_conf_lt_0_4": stats["low_conf_lt_0_4"],
            "shape_counts": "; ".join(f"{name}:{num}" for name, num in stats["shape_counter"].most_common()),
        })
    with args.summary_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"Wrote: {args.output_csv.resolve()}")
    print(f"Wrote: {args.output_json.resolve()}")
    print(f"Wrote: {args.summary_csv.resolve()}")
    print(f"Wrote: {args.review_csv.resolve()}")
    print(f"Rows: {len(total_conf)}, avg_conf={float(np.mean(total_conf)):.3f}, median_conf={float(np.median(total_conf)):.3f}")
    print(f"Review candidates: {review_count}")
    print("Top classes:")
    for row in summary_rows[:10]:
        print(f"  {row['predicted_class']}: {row['count']} avg_conf={row['avg_confidence']:.3f}")


if __name__ == "__main__":
    main()
