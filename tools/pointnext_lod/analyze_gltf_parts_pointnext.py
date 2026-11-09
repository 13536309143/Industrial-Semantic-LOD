"""Analyze GLTF sub-parts with a trained PointNeXt classification checkpoint.

The script keeps GLTF node order, groups multiple primitives that belong to the
same node, computes geometric statistics in model units, and predicts a class
for each sub-part.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import trimesh
from pygltflib import GLTF2

from checkpoint_loading import load_classifier_checkpoint


HEX_SUFFIX_RE = re.compile(r"^(?P<base>.+)_[0-9a-fA-F]{6}$")


@dataclass
class PartMesh:
    order: int
    node_index: int
    node_name: str
    mesh_index: int
    mesh_name: str
    mesh: trimesh.Trimesh


@dataclass
class PartResult:
    order: int
    node_index: int
    node_name: str
    mesh_index: int
    mesh_name: str
    predicted_label: int
    predicted_class: str
    confidence: float
    second_label: int
    second_class: str
    second_confidence: float
    vertex_count: int
    face_count: int
    surface_area: float
    mesh_volume: float | None
    bbox_min_x: float
    bbox_min_y: float
    bbox_min_z: float
    bbox_max_x: float
    bbox_max_y: float
    bbox_max_z: float
    bbox_size_x: float
    bbox_size_y: float
    bbox_size_z: float
    bbox_diagonal: float
    bbox_volume: float
    obb_size_long: float
    obb_size_mid: float
    obb_size_short: float
    elongation: float
    flatness: float
    compactness: float
    shape_hint: str
    centroid_x: float
    centroid_y: float
    centroid_z: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify and measure every GLTF sub-part with PointNeXt."
    )
    parser.add_argument("--gltf", type=Path, default=Path("1.gltf"))
    parser.add_argument(
        "--pointnext-root",
        type=Path,
        default=Path("PointNeXt"),
        help="Path to the PointNeXt repository.",
    )
    parser.add_argument(
        "--cfg",
        type=Path,
        default=Path("PointNeXt/cfgs/industrial_part/pointnext-s-14class-encoder-compare-inference.yaml"),
    )
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to *_ckpt_best.pth.")
    parser.add_argument(
        "--classes-file",
        type=Path,
        default=None,
        help="Optional classes.txt. Defaults to cfg dataset data_dir/classes.txt.",
    )
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-csv", type=Path, default=Path("gltf_part_analysis.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("gltf_part_analysis.json"))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def setup_pointnext(pointnext_root: Path):
    root = pointnext_root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"PointNeXt root not found: {root}")
    sys.path.insert(0, str(root))

    from openpoints.models import build_model_from_cfg
    from openpoints.utils import EasyConfig, load_checkpoint

    return root, EasyConfig, build_model_from_cfg, load_checkpoint


def load_cfg(cfg_path: Path, EasyConfig):
    cfg = EasyConfig()
    cfg.load(str(cfg_path), recursive=True)
    if not cfg.model.get("criterion_args", False):
        cfg.model.criterion_args = cfg.criterion_args
    if cfg.model.get("in_channels", None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels
    return cfg


def resolve_classes_file(args: argparse.Namespace, cfg, pointnext_root: Path) -> Path:
    if args.classes_file is not None:
        return args.classes_file.resolve()

    data_dir = Path(cfg.dataset.common.data_dir)
    if not data_dir.is_absolute():
        data_dir = (pointnext_root / data_dir).resolve()
    classes_file = data_dir / "classes.txt"
    if classes_file.exists():
        return classes_file

    fallback = Path(__file__).resolve().parent / "industrial_part_14classes.txt"
    if fallback.exists():
        return fallback
    return classes_file


def load_classes(classes_file: Path) -> list[str]:
    if not classes_file.exists():
        raise FileNotFoundError(
            f"classes.txt not found: {classes_file}. Pass --classes-file explicitly."
        )

    classes: list[str] = []
    for line in classes_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        classes.append(parts[1] if len(parts) == 2 else parts[0])
    return classes


def base_node_name(name: str) -> str:
    match = HEX_SUFFIX_RE.match(name)
    return match.group("base") if match else name


def load_ordered_part_meshes(gltf_path: Path) -> list[PartMesh]:
    gltf = GLTF2().load(str(gltf_path))
    scene = trimesh.load(str(gltf_path), force="scene", process=False)
    if not isinstance(scene, trimesh.Scene):
        raise ValueError(f"Expected GLTF scene, got {type(scene)}")

    by_base_name: dict[str, list[tuple[str, str, np.ndarray]]] = {}
    for scene_node in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[scene_node]
        by_base_name.setdefault(base_node_name(scene_node), []).append(
            (scene_node, geom_name, np.asarray(transform, dtype=np.float64))
        )

    parts: list[PartMesh] = []
    for node_index, node in enumerate(gltf.nodes or []):
        if node.mesh is None:
            continue

        node_name = node.name or f"node_{node_index}"
        mesh_info = gltf.meshes[node.mesh] if gltf.meshes else None
        mesh_name = (mesh_info.name if mesh_info and mesh_info.name else None) or f"mesh_{node.mesh}"
        candidates = by_base_name.get(node_name, [])

        if not candidates:
            # Some exporters alter node names. Fall back to mesh order naming.
            candidates = by_base_name.get(f"Mesh.{node.mesh:03d}", [])
        if not candidates and node.mesh == 0:
            candidates = by_base_name.get("Mesh", [])
        if not candidates:
            raise ValueError(f"Could not find geometry for GLTF node {node_index}: {node_name}")

        meshes = []
        for _, geom_name, transform in candidates:
            geom = scene.geometry[geom_name].copy()
            geom.apply_transform(transform)
            if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0:
                meshes.append(geom)
        if not meshes:
            continue

        merged = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        parts.append(
            PartMesh(
                order=len(parts) + 1,
                node_index=node_index,
                node_name=node_name,
                mesh_index=int(node.mesh),
                mesh_name=mesh_name,
                mesh=merged,
            )
        )
    return parts


def sample_mesh_points(mesh: trimesh.Trimesh, num_points: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(vertices) == 0:
        raise ValueError("mesh has no vertices")

    if len(faces) == 0:
        replace = len(vertices) < num_points
        indices = rng.choice(len(vertices), size=num_points, replace=replace)
        return vertices[indices].astype(np.float32, copy=False)

    tri = vertices[faces]
    edge1 = tri[:, 1] - tri[:, 0]
    edge2 = tri[:, 2] - tri[:, 0]
    areas = np.linalg.norm(np.cross(edge1, edge2), axis=1) * 0.5
    valid = np.isfinite(areas) & (areas > 1e-12)
    if not np.any(valid):
        replace = len(vertices) < num_points
        indices = rng.choice(len(vertices), size=num_points, replace=replace)
        return vertices[indices].astype(np.float32, copy=False)

    tri = tri[valid]
    areas = areas[valid]
    probabilities = areas / areas.sum()
    face_indices = rng.choice(len(tri), size=num_points, replace=True, p=probabilities)
    chosen = tri[face_indices]
    r1 = np.sqrt(rng.random((num_points, 1), dtype=np.float32))
    r2 = rng.random((num_points, 1), dtype=np.float32)
    points = (
        (1.0 - r1) * chosen[:, 0]
        + r1 * (1.0 - r2) * chosen[:, 1]
        + r1 * r2 * chosen[:, 2]
    )
    return points.astype(np.float32, copy=False)


def normalize_unit_sphere(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32, copy=True)
    points -= points.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(points, axis=1).max()
    if scale > 0:
        points /= scale
    return points


def build_model(cfg, build_model_from_cfg, load_checkpoint, ckpt_path: Path, device: torch.device):
    model = build_model_from_cfg(cfg.model).to(device)
    load_classifier_checkpoint(model, ckpt_path)
    model.eval()
    return model


def predict_batches(
    model,
    normalized_points: list[np.ndarray],
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> list[tuple[int, float, int, float]]:
    predictions: list[tuple[int, float, int, float]] = []
    for start in range(0, len(normalized_points), batch_size):
        batch_np = np.stack(normalized_points[start : start + batch_size], axis=0)
        points = torch.from_numpy(batch_np).to(device=device, non_blocking=True)
        data = {
            "pos": points.contiguous(),
            "x": points.transpose(1, 2).contiguous(),
        }
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
            logits = model(data)
            probs = torch.softmax(logits.float(), dim=1)
            top2 = torch.topk(probs, k=min(2, probs.shape[1]), dim=1)
        labels = top2.indices.detach().cpu().numpy()
        scores = top2.values.detach().cpu().numpy()
        for row_labels, row_scores in zip(labels, scores):
            first_label = int(row_labels[0])
            first_score = float(row_scores[0])
            second_label = int(row_labels[1]) if len(row_labels) > 1 else -1
            second_score = float(row_scores[1]) if len(row_scores) > 1 else 0.0
            predictions.append((first_label, first_score, second_label, second_score))
    return predictions


def safe_float(value) -> float | None:
    try:
        value = float(value)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def shape_hint(long_side: float, mid_side: float, short_side: float, flatness: float, elongation: float) -> str:
    if long_side <= 0:
        return "empty"
    if flatness < 0.12:
        return "flat_plate"
    if elongation > 3.0:
        return "long_slender"
    if flatness > 0.65 and elongation < 1.6:
        return "compact_block"
    if mid_side / max(long_side, 1e-12) > 0.75 and flatness > 0.35:
        return "round_or_cubic"
    return "irregular"


def mesh_stats(part: PartMesh) -> dict:
    mesh = part.mesh
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    bbox_min = bounds[0]
    bbox_max = bounds[1]
    bbox_size = bbox_max - bbox_min
    bbox_diagonal = float(np.linalg.norm(bbox_size))
    bbox_volume = float(np.prod(np.maximum(bbox_size, 0.0)))

    try:
        obb_extents = np.asarray(mesh.bounding_box_oriented.primitive.extents, dtype=np.float64)
    except Exception:
        obb_extents = bbox_size
    ordered = sorted((float(x) for x in obb_extents), reverse=True)
    while len(ordered) < 3:
        ordered.append(0.0)
    long_side, mid_side, short_side = ordered[:3]
    elongation = long_side / max(mid_side, 1e-12)
    flatness = short_side / max(long_side, 1e-12)
    compactness = bbox_volume / max(bbox_diagonal**3, 1e-12)

    volume = safe_float(mesh.volume) if mesh.is_watertight else None
    centroid = np.asarray(mesh.centroid, dtype=np.float64)

    return {
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "surface_area": float(mesh.area),
        "mesh_volume": volume,
        "bbox_min_x": float(bbox_min[0]),
        "bbox_min_y": float(bbox_min[1]),
        "bbox_min_z": float(bbox_min[2]),
        "bbox_max_x": float(bbox_max[0]),
        "bbox_max_y": float(bbox_max[1]),
        "bbox_max_z": float(bbox_max[2]),
        "bbox_size_x": float(bbox_size[0]),
        "bbox_size_y": float(bbox_size[1]),
        "bbox_size_z": float(bbox_size[2]),
        "bbox_diagonal": bbox_diagonal,
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


def make_results(
    parts: list[PartMesh],
    predictions: list[tuple[int, float, int, float]],
    class_names: list[str],
) -> list[PartResult]:
    results: list[PartResult] = []
    for part, pred in zip(parts, predictions):
        label, confidence, second_label, second_confidence = pred
        stats = mesh_stats(part)
        results.append(
            PartResult(
                order=part.order,
                node_index=part.node_index,
                node_name=part.node_name,
                mesh_index=part.mesh_index,
                mesh_name=part.mesh_name,
                predicted_label=label,
                predicted_class=class_names[label] if 0 <= label < len(class_names) else str(label),
                confidence=confidence,
                second_label=second_label,
                second_class=class_names[second_label] if 0 <= second_label < len(class_names) else "",
                second_confidence=second_confidence,
                **stats,
            )
        )
    return results


def write_outputs(results: Iterable[PartResult], csv_path: Path, json_path: Path) -> None:
    rows = [asdict(item) for item in results]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    pointnext_root, EasyConfig, build_model_from_cfg, load_checkpoint = setup_pointnext(args.pointnext_root)
    cfg = load_cfg(args.cfg, EasyConfig)
    num_points = int(args.num_points or cfg.get("num_points", 8192))

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)
    use_amp = bool(cfg.get("use_amp", False)) and not args.no_amp

    classes_file = resolve_classes_file(args, cfg, pointnext_root)
    class_names = load_classes(classes_file)
    if len(class_names) != int(cfg.num_classes):
        raise ValueError(f"classes.txt has {len(class_names)} classes, cfg.num_classes={cfg.num_classes}")

    print(f"GLTF: {args.gltf.resolve()}")
    print(f"Checkpoint: {args.ckpt.resolve()}")
    print(f"Classes: {classes_file} ({len(class_names)})")
    print(f"Points per part: {num_points}")
    print(f"Device: {device}, AMP: {use_amp}")

    parts = load_ordered_part_meshes(args.gltf.resolve())
    if not parts:
        raise ValueError(f"No mesh nodes found in {args.gltf}")
    print(f"Sub-parts: {len(parts)}")

    normalized_points = []
    for part in parts:
        points = sample_mesh_points(part.mesh, num_points, seed=args.seed + part.order)
        normalized_points.append(normalize_unit_sphere(points))

    model = build_model(cfg, build_model_from_cfg, load_checkpoint, args.ckpt.resolve(), device)
    predictions = predict_batches(model, normalized_points, args.batch_size, device, use_amp)
    results = make_results(parts, predictions, class_names)
    write_outputs(results, args.output_csv.resolve(), args.output_json.resolve())

    print(f"Wrote: {args.output_csv.resolve()}")
    print(f"Wrote: {args.output_json.resolve()}")
    for item in results[:10]:
        print(
            f"{item.order:03d} {item.node_name}: {item.predicted_class} "
            f"({item.confidence:.3f}), size=({item.bbox_size_x:.3g}, "
            f"{item.bbox_size_y:.3g}, {item.bbox_size_z:.3g}), shape={item.shape_hint}"
        )
    if len(results) > 10:
        print(f"... {len(results) - 10} more rows in the output files")


if __name__ == "__main__":
    main()
