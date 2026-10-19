"""Run PointCLIP V2 style zero-shot open-vocabulary inference on GLB mesh nodes.

This wrapper keeps PointCLIP V2 isolated from the PointNeXt environment. It
reuses PointCLIP V2's realistic point-cloud projection and CLIP image/text
encoders, but reads the same GLB mesh-node order as the PointNeXt analyzer so
the resulting CSV can be joined by order, node_index, and mesh_index.
"""

from __future__ import annotations

import argparse
import csv
import json
import mmap
import os
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

# PyTorch, NumPy/MKL, and CLIP can load different OpenMP runtimes on Windows.
# This script is an offline inference tool, so allowing duplicate OpenMP
# runtimes is the pragmatic way to keep PointCLIP V2 usable across conda setups.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pointnext_lod"))
from gltf_hierarchy import attach_world_matrices

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
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="PointCLIP V2 zero-shot GLB mesh-node inference.")
    parser.add_argument("--glb", type=Path, required=True)
    parser.add_argument("--pointclip-root", type=Path, default=Path(__file__).resolve().parent / "PointCLIP_V2")
    parser.add_argument("--prompt-json", type=Path, default=Path(__file__).resolve().parent / "industrial_open_vocab_prompts.json")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=None,
        help="Optional PointNeXt analysis CSV. If set, only non-high-confidence rows are sent to PointCLIP.",
    )
    parser.add_argument(
        "--candidate-high-confidence",
        type=float,
        default=0.70,
        help="PointNeXt confidence threshold for skipping PointCLIP.",
    )
    parser.add_argument(
        "--candidate-low-margin",
        type=float,
        default=0.08,
        help="PointNeXt top1-top2 margin threshold for skipping PointCLIP.",
    )
    parser.add_argument(
        "--exclude-roles",
        default="micro_uncertain",
        help="Comma-separated prompt roles excluded from CLIP matching. Use an empty string to disable.",
    )
    parser.add_argument("--num-points", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--backbone", default="ViT-B/16", choices=["RN50", "RN101", "ViT-B/32", "ViT-B/16"])
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N mesh nodes for smoke tests.")
    parser.add_argument("--project-root", type=Path, default=root)
    return parser.parse_args()


def add_import_paths(args: argparse.Namespace) -> None:
    zeroshot = args.pointclip_root / "zeroshot_cls"
    for path in [zeroshot, zeroshot / "Dassl3D"]:
        text = str(path.resolve())
        if text not in sys.path:
            sys.path.insert(0, text)


def parse_role_filter(text: str) -> Set[str]:
    return {item.strip() for item in text.split(",") if item.strip()}


def load_prompt_library(path: Path, exclude_roles: Set[str]) -> Tuple[List[Dict], List[str]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    templates = data.get("templates") or ["a 3D object: {name}."]
    labels = [
        label for label in (data.get("labels") or [])
        if label.get("role", "") not in exclude_roles
    ]
    if not labels:
        raise ValueError(f"No labels found in prompt library: {path}")
    return labels, templates


def candidate_keys_from_pointnext(
    path: Path,
    high_confidence: float,
    low_margin: float,
) -> Tuple[Set[Tuple[str, str, str]], Counter]:
    keys: Set[Tuple[str, str, str]] = set()
    counts = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            confidence = float(row.get("confidence") or 0.0)
            second = float(row.get("second_confidence") or 0.0)
            margin = confidence - second
            high_reliable = confidence >= high_confidence and margin >= low_margin
            if high_reliable:
                counts["skipped_high_confidence"] += 1
                continue
            keys.add((str(row.get("order", "")), str(row.get("node_index", "")), str(row.get("mesh_index", ""))))
            counts["pointclip_candidates"] += 1
    return keys, counts


def prompts_for_label(label: Dict, templates: List[str]) -> List[str]:
    names = [label["name"], *label.get("aliases", [])]
    prompts = []
    for name in names:
        for template in templates:
            prompts.append(template.format(name=name))
    return prompts


def build_text_features(clip_module, model, labels: List[Dict], templates: List[str], device: torch.device) -> torch.Tensor:
    features = []
    for label in labels:
        prompts = prompts_for_label(label, templates)
        tokens = clip_module.tokenize(prompts).to(device)
        with torch.no_grad():
            text = model.encode_text(tokens)
            text = text / text.norm(dim=-1, keepdim=True)
            text = text.mean(dim=0)
            text = text / text.norm()
        features.append(text)
    return torch.stack(features, dim=0)


def normalize_points(points: np.ndarray) -> np.ndarray:
    center = (points.max(axis=0) + points.min(axis=0)) * 0.5
    points = points - center
    scale = np.linalg.norm(points, axis=1).max()
    if scale > 1e-12:
        points = points / scale
    return points.astype(np.float32, copy=False)


def read_glb_chunks(path: Path) -> Tuple[dict, int, int]:
    with path.open("rb") as file:
        magic, version, _ = struct.unpack("<III", file.read(12))
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
        attach_world_matrices(gltf, node_matrix)
        return gltf, bin_header_offset + 8, bin_len


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
        return np.asarray(arr.reshape(count, item_count) if item_count > 1 else arr)

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


def mesh_for_node(gltf: dict, mm: mmap.mmap, bin_offset: int, node: dict) -> Tuple[np.ndarray, np.ndarray]:
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
        vertices = (vertices.astype(np.float64, copy=False) @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)

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


def infer_batch(model, projection, text_features, points_batch: List[np.ndarray], top_k: int, device: torch.device):
    points = torch.from_numpy(np.stack(points_batch, axis=0)).to(device=device, non_blocking=True)
    with torch.no_grad():
        images = projection.get_img(points)
        images = torch.nn.functional.interpolate(images, size=(224, 224), mode="bilinear", align_corners=True)
        images = images.type(model.dtype)
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        num_views = projection.num_views
        image_features = image_features.reshape(len(points_batch), num_views, -1).mean(dim=1)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ text_features.t()
        probs = torch.softmax(logits.float(), dim=1)
        values, indices = torch.topk(probs, k=min(top_k, probs.shape[1]), dim=1)
    return indices.cpu().numpy(), values.cpu().numpy()


def row_from_prediction(base: Dict, labels: List[Dict], idxs: np.ndarray, scores: np.ndarray) -> Dict:
    top = []
    for rank, (label_index, score) in enumerate(zip(idxs.tolist(), scores.tolist()), start=1):
        label = labels[int(label_index)]
        top.append({
            "rank": rank,
            "id": label["id"],
            "name": label["name"],
            "role": label["role"],
            "score": float(score),
        })

    top1 = top[0]
    top2 = top[1] if len(top) > 1 else {"id": "", "name": "", "role": "", "score": 0.0}
    return {
        **base,
        "pointclip_top1_id": top1["id"],
        "pointclip_top1_name": top1["name"],
        "pointclip_top1_role": top1["role"],
        "pointclip_top1_score": top1["score"],
        "pointclip_top2_id": top2["id"],
        "pointclip_top2_name": top2["name"],
        "pointclip_top2_role": top2["role"],
        "pointclip_top2_score": top2["score"],
        "pointclip_margin": float(top1["score"] - top2["score"]),
        "pointclip_topk_json": json.dumps(top, ensure_ascii=False),
    }


def write_summary(path: Path, rows: List[Dict]) -> None:
    role_counter = Counter(row["pointclip_top1_role"] for row in rows)
    label_counter = Counter(row["pointclip_top1_id"] for row in rows)
    summary_rows = []
    for label_id, count in label_counter.most_common():
        role = next(row["pointclip_top1_role"] for row in rows if row["pointclip_top1_id"] == label_id)
        avg_score = sum(float(row["pointclip_top1_score"]) for row in rows if row["pointclip_top1_id"] == label_id) / count
        summary_rows.append({
            "pointclip_top1_id": label_id,
            "pointclip_top1_role": role,
            "count": count,
            "avg_score": avg_score,
            "role_total": role_counter[role],
        })
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        fieldnames = ["pointclip_top1_id", "pointclip_top1_role", "count", "avg_score", "role_total"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    args = parse_args()
    global torch
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PointCLIP V2 projection in this wrapper requires CUDA.")
    add_import_paths(args)

    from clip import clip
    from trainers.mv_utils_zs import Realistic_Projection

    excluded_roles = parse_role_filter(args.exclude_roles)
    labels, templates = load_prompt_library(args.prompt_json, excluded_roles)
    if excluded_roles:
        print(f"PointCLIP prompt role filter: excluded {sorted(excluded_roles)}, using {len(labels)} labels")
    candidate_keys: Optional[Set[Tuple[str, str, str]]] = None
    if args.candidate_csv:
        candidate_keys, candidate_counts = candidate_keys_from_pointnext(
            args.candidate_csv,
            args.candidate_high_confidence,
            args.candidate_low_margin,
        )
        print(
            "PointCLIP candidate gate: "
            f"{candidate_counts['pointclip_candidates']} candidates, "
            f"{candidate_counts['skipped_high_confidence']} high-confidence rows skipped"
        )

    device = torch.device(args.device)
    model, _ = clip.load(args.backbone, device=device)
    model.eval()
    text_features = build_text_features(clip, model, labels, templates, device)
    projection = Realistic_Projection()

    gltf, bin_offset, _ = read_glb_chunks(args.glb)
    mesh_nodes = [(idx, node) for idx, node in enumerate(gltf.get("nodes", [])) if "mesh" in node]
    if args.limit is not None:
        mesh_nodes = mesh_nodes[:args.limit]

    rows: List[Dict] = []
    pending_base: List[Dict] = []
    pending_points: List[np.ndarray] = []

    def flush() -> None:
        if not pending_points:
            return
        indices, scores = infer_batch(model, projection, text_features, pending_points, args.top_k, device)
        for base, idxs, vals in zip(pending_base, indices, scores):
            rows.append(row_from_prediction(base, labels, idxs, vals))
        pending_base.clear()
        pending_points.clear()

    with args.glb.open("rb") as glb_file, mmap.mmap(glb_file.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for order, (node_index, node) in enumerate(mesh_nodes, start=1):
            mesh_index = int(node["mesh"])
            if candidate_keys is not None and (str(order), str(node_index), str(mesh_index)) not in candidate_keys:
                continue
            vertices, faces = mesh_for_node(gltf, mm, bin_offset, node)
            if len(vertices) == 0:
                continue
            mesh = gltf["meshes"][mesh_index]
            rng = np.random.default_rng(args.seed + int(node_index))
            points = sample_points(vertices, faces, args.num_points, rng)
            pending_points.append(normalize_points(points))
            pending_base.append({
                "order": order,
                "node_index": node_index,
                "node_name": node.get("name", f"node_{node_index}"),
                "mesh_index": mesh_index,
                "mesh_name": mesh.get("name", f"mesh_{mesh_index}"),
            })
            if len(pending_points) >= args.batch_size:
                flush()
            if order % 500 == 0:
                print(f"PointCLIPV2 processed {order}/{len(mesh_nodes)} mesh nodes")
        flush()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "order", "node_index", "node_name", "mesh_index", "mesh_name",
        "pointclip_top1_id", "pointclip_top1_name", "pointclip_top1_role", "pointclip_top1_score",
        "pointclip_top2_id", "pointclip_top2_name", "pointclip_top2_role", "pointclip_top2_score",
        "pointclip_margin", "pointclip_topk_json",
    ]
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote: {args.output_csv}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as file:
            json.dump(rows, file, ensure_ascii=False, indent=2)
        print(f"Wrote: {args.output_json}")

    if args.summary_csv:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        write_summary(args.summary_csv, rows)
        print(f"Wrote: {args.summary_csv}")


if __name__ == "__main__":
    main()
