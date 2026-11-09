"""Build per-mesh LOD constraint classes from GLB/GLTF part analysis CSV.

The input is produced by analyze_large_glb_parts_pointnext.py or
analyze_gltf_parts_pointnext.py. Output keeps one row per original mesh and adds
size-first, shape-second, semantic/detail-aware LOD policy fields. The final
policy is a P1-P10 semantic-structural class used by the Vulkan LOD frontend and
builder.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


FASTENER_TYPES = {
    "screws_bolts_studs",
    "nuts",
    "washers_rings_spacers",
    "pins_rivets_keys",
}
MOTION_CRITICAL_TYPES = {
    "bearings_bushings_guides",
    "gears_pulleys_chains",
    "motors_gearmotors",
    "wheels_castors",
    "rotating_fluid_machinery",
    "springs",
}
INTERFACE_TYPES = {
    "pipe_fittings_valves_nozzles",
}
STRUCTURAL_KEY_TYPES = {
    "joints_clamps_structural_connectors",
}
BULK_STATIC_TYPES = {
    "plates_discs_shapes",
}
CONTROL_TYPES = {
    "handles_controls",
}

P10_POLICY_TABLE = {
    # name, near-ratio, mid-ratio, far-ratio, allow-cull, screen-error-weight
    # P1/P2 are the only default cullable classes. P3/P4 can be simplified
    # aggressively, but they often carry visible silhouette or occlusion mass.
    1: ("P1_micro_uncertain", 0.30, 0.10, 0.03, True, 0.40),
    2: ("P2_repeated_fastener", 0.42, 0.18, 0.06, True, 0.55),
    3: ("P3_large_static_bulk", 0.55, 0.30, 0.12, False, 0.70),
    4: ("P4_ordinary_low_detail", 0.60, 0.34, 0.15, False, 0.82),
    5: ("P5_balanced_visible", 0.68, 0.42, 0.20, False, 0.98),
    6: ("P6_high_detail_shape", 0.76, 0.50, 0.28, False, 1.12),
    7: ("P7_interface_fluid", 0.82, 0.58, 0.36, False, 1.25),
    8: ("P8_structural_control", 0.86, 0.64, 0.42, False, 1.38),
    9: ("P9_motion_precision", 0.92, 0.74, 0.52, False, 1.55),
    10: ("P10_critical_preserve", 0.96, 0.84, 0.65, False, 1.80),
}

ROLE_PROTOTYPES = {
    "micro_uncertain": {
        "priority": 1,
        "function_role": "low_confidence_or_micro_detail",
        "strategy": "micro_uncertain_cull_far",
        "protected_features": ["placement", "coarse_silhouette"],
        "expected_features": {"micro_score": 0.90, "uncertainty_score": 0.75, "detail_score": 0.15},
    },
    "repeated_fastener": {
        "priority": 2,
        "function_role": "fastening_or_repeated_standard_part",
        "strategy": "repeated_fastener_aggressive",
        "protected_features": ["placement", "head_silhouette", "axis_if_visible"],
        "expected_features": {"micro_score": 0.45, "slender_score": 0.45, "ring_disk_score": 0.45, "detail_score": 0.35},
    },
    "large_static_bulk": {
        "priority": 3,
        "function_role": "large_static_support_or_cover",
        "strategy": "large_static_bulk_fast_simplify",
        "protected_features": ["outer_silhouette", "major_openings"],
        "expected_features": {"bulk_score": 0.90, "size_score": 0.85, "detail_score": 0.25},
    },
    "ordinary_low_detail": {
        "priority": 4,
        "function_role": "ordinary_low_detail_part",
        "strategy": "ordinary_low_detail_fast_simplify",
        "protected_features": ["coarse_silhouette"],
        "expected_features": {"simple_score": 0.80, "detail_score": 0.25, "uncertainty_score": 0.25},
    },
    "balanced_visible": {
        "priority": 5,
        "function_role": "balanced_visible_structure",
        "strategy": "balanced_visible_part",
        "protected_features": ["silhouette", "visible_feature_edges"],
        "expected_features": {"medium_size_score": 0.70, "detail_score": 0.50, "simple_score": 0.45},
    },
    "high_detail_shape": {
        "priority": 6,
        "function_role": "high_detail_visible_shape",
        "strategy": "high_detail_shape",
        "protected_features": ["silhouette", "high_curvature_regions", "feature_edges"],
        "expected_features": {"high_detail_score": 0.90, "density_score": 0.75, "plate_score": 0.10},
    },
    "interface_fluid": {
        "priority": 7,
        "function_role": "interface_or_fluid_connection",
        "strategy": "interface_or_fluid_connection",
        "protected_features": ["interface_boundary", "port_silhouette", "mounting_surfaces"],
        "expected_features": {"compact_score": 0.70, "ring_disk_score": 0.55, "detail_score": 0.55},
    },
    "structural_control": {
        "priority": 8,
        "function_role": "structural_connector_or_control_surface",
        "strategy": "structural_control_protect",
        "protected_features": ["mounting_boundaries", "contact_surfaces", "control_silhouette"],
        "expected_features": {"plate_score": 0.55, "compact_score": 0.55, "detail_score": 0.55, "size_score": 0.55},
    },
    "motion_precision": {
        "priority": 9,
        "function_role": "motion_or_precision_guidance",
        "strategy": "motion_precision_protect",
        "protected_features": ["axis", "guide_surfaces", "contact_surfaces", "silhouette"],
        "expected_features": {"slender_score": 0.65, "ring_disk_score": 0.55, "compact_score": 0.50, "detail_score": 0.65},
    },
    "critical_preserve": {
        "priority": 10,
        "function_role": "critical_motion_or_transmission_part",
        "strategy": "critical_preserve",
        "protected_features": ["periodic_profile", "center_axis", "contact_surfaces", "silhouette"],
        "expected_features": {"high_detail_score": 0.80, "ring_disk_score": 0.65, "density_score": 0.70, "compact_score": 0.55},
    },
}

CLASS_ROLE_PRIORS = {
    "screws_bolts_studs": {"repeated_fastener": 0.95, "motion_precision": 0.15},
    "nuts": {"repeated_fastener": 0.90, "interface_fluid": 0.20},
    "washers_rings_spacers": {"repeated_fastener": 0.75, "interface_fluid": 0.35, "large_static_bulk": 0.10},
    "pins_rivets_keys": {"repeated_fastener": 0.70, "motion_precision": 0.35},
    "bearings_bushings_guides": {"critical_preserve": 0.85, "motion_precision": 0.75},
    "gears_pulleys_chains": {"critical_preserve": 0.95, "motion_precision": 0.70},
    "motors_gearmotors": {"critical_preserve": 0.90, "motion_precision": 0.45},
    "wheels_castors": {"motion_precision": 0.80, "critical_preserve": 0.45},
    "rotating_fluid_machinery": {"critical_preserve": 0.75, "interface_fluid": 0.45},
    "springs": {"critical_preserve": 0.85, "motion_precision": 0.55},
    "pipe_fittings_valves_nozzles": {"interface_fluid": 0.95, "critical_preserve": 0.25},
    "joints_clamps_structural_connectors": {"structural_control": 0.80, "interface_fluid": 0.55},
    "plates_discs_shapes": {"large_static_bulk": 0.55, "ordinary_low_detail": 0.45, "balanced_visible": 0.35},
    "handles_controls": {"structural_control": 0.70, "high_detail_shape": 0.45, "balanced_visible": 0.35},
}

GENERIC_POINTCLIP_LABELS = {
    "plain_rectangular_block",
    "plain_disc",
    "generic_mounting_pad",
    "simple_cover_plate",
    "simple_bracket",
    "simple_plug",
    "plain_cap",
}

EVIDENCE_MODES = {
    "pointnext": "PointNeXt only",
    "pointclip": "PointCLIP only",
    "pointnext_structure": "PointNeXt + structure",
    "pointclip_structure": "PointCLIP + structure",
    "pointnext_pointclip": "PointNeXt + PointCLIP",
    "pointnext_pointclip_structure": "PointNeXt + PointCLIP + structure",
}


def make_p10_policy(priority: int, strategy: str) -> dict:
    name, near, mid, far, allow_cull, screen_error_weight = P10_POLICY_TABLE[priority]
    return {
        "lod_priority": name,
        "lod_strategy": strategy,
        "target_ratio_near": near,
        "target_ratio_mid": mid,
        "target_ratio_far": far,
        "allow_cull": allow_cull,
        "screen_error_weight": screen_error_weight,
    }


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge0 == edge1:
        return 1.0 if value >= edge1 else 0.0
    t = clamp01((value - edge0) / (edge1 - edge0))
    return t * t * (3.0 - 2.0 * t)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create LOD constraint classes from part analysis CSV."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument(
        "--high-confidence",
        type=float,
        default=0.70,
        help="Prediction confidence at or above this is treated as reliable.",
    )
    parser.add_argument(
        "--low-confidence",
        type=float,
        default=0.40,
        help="Below this, semantic type is downgraded and geometry fallback is used more strongly.",
    )
    parser.add_argument(
        "--low-margin",
        type=float,
        default=0.08,
        help="If top1-top2 confidence is below this, mark semantic prediction as ambiguous.",
    )
    parser.add_argument(
        "--evidence-mode",
        choices=sorted(EVIDENCE_MODES),
        default="pointnext_pointclip_structure",
        help="Ablation mode controlling which evidence sources may affect P1-P10 policy selection.",
    )
    return parser.parse_args()


def to_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value not in {"", None} else default
    except Exception:
        return default


def to_int(row: dict, key: str, default: int = 0) -> int:
    try:
        value = row.get(key, "")
        return int(float(value)) if value not in {"", None} else default
    except Exception:
        return default


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, round((len(sorted_values) - 1) * p)))
    return sorted_values[index]


def make_thresholds(rows: list[dict]) -> dict:
    diagonals = sorted(to_float(row, "bbox_diagonal") for row in rows)
    faces = sorted(to_int(row, "face_count") for row in rows)
    areas = sorted(to_float(row, "surface_area") for row in rows)
    return {
        "diag_p10": percentile(diagonals, 0.10),
        "diag_p25": percentile(diagonals, 0.25),
        "diag_p50": percentile(diagonals, 0.50),
        "diag_p75": percentile(diagonals, 0.75),
        "diag_p90": percentile(diagonals, 0.90),
        "diag_p97": percentile(diagonals, 0.97),
        "face_p50": percentile(faces, 0.50),
        "face_p75": percentile(faces, 0.75),
        "face_p90": percentile(faces, 0.90),
        "face_p97": percentile(faces, 0.97),
        "area_p50": percentile(areas, 0.50),
        "area_p75": percentile(areas, 0.75),
        "area_p90": percentile(areas, 0.90),
    }


def classify_size(row: dict, thresholds: dict) -> tuple[str, int]:
    diag = to_float(row, "bbox_diagonal")
    if diag <= thresholds["diag_p10"]:
        return "S0_micro", 0
    if diag <= thresholds["diag_p25"]:
        return "S1_tiny", 1
    if diag <= thresholds["diag_p50"]:
        return "S2_small", 2
    if diag <= thresholds["diag_p75"]:
        return "S3_medium", 3
    if diag <= thresholds["diag_p90"]:
        return "S4_large", 4
    if diag <= thresholds["diag_p97"]:
        return "S5_xlarge", 5
    return "S6_huge", 6


def semantic_group(predicted_class: str, reliable: bool) -> str:
    if not reliable:
        return "semantic_uncertain"
    if predicted_class in FASTENER_TYPES:
        return "fastener_repeated"
    if predicted_class in MOTION_CRITICAL_TYPES:
        return "motion_or_precision_part"
    if predicted_class in INTERFACE_TYPES:
        return "fluid_or_interface_part"
    if predicted_class in STRUCTURAL_KEY_TYPES:
        return "structural_key_part"
    if predicted_class in BULK_STATIC_TYPES:
        return "bulk_static_part"
    if predicted_class in CONTROL_TYPES:
        return "control_or_handle"
    return "other_part"


def geometry_fallback(row: dict) -> str:
    long_side = to_float(row, "obb_size_long")
    mid_side = to_float(row, "obb_size_mid")
    short_side = to_float(row, "obb_size_short")
    elongation = to_float(row, "elongation")
    flatness = to_float(row, "flatness")
    compactness = to_float(row, "compactness")
    face_count = to_int(row, "face_count")

    if flatness < 0.015:
        return "geom_ultra_thin_sheet"
    if flatness < 0.08:
        return "geom_thin_plate"
    if elongation >= 8:
        return "geom_wire_or_rod"
    if elongation >= 3:
        return "geom_slender_bar"
    if compactness > 0.12 and flatness > 0.35:
        return "geom_compact_block"
    if abs(long_side - mid_side) / max(long_side, 1e-12) < 0.15 and flatness < 0.25:
        return "geom_ring_or_disk"
    if face_count >= 2000:
        return "geom_high_detail_irregular"
    return "geom_simple_irregular"


def detail_level(row: dict, thresholds: dict) -> tuple[str, int]:
    face_count = to_int(row, "face_count")
    area = to_float(row, "surface_area")
    diag = to_float(row, "bbox_diagonal")
    density = face_count / max(diag * diag, 1e-12)

    score = 0
    if face_count >= thresholds["face_p97"]:
        score += 3
    elif face_count >= thresholds["face_p90"]:
        score += 2
    elif face_count >= thresholds["face_p75"]:
        score += 1

    if area >= thresholds["area_p90"]:
        score += 1
    if density > 5000:
        score += 1

    if score >= 4:
        return "D4_very_high", 4
    if score == 3:
        return "D3_high", 3
    if score == 2:
        return "D2_medium", 2
    if score == 1:
        return "D1_low", 1
    return "D0_very_low", 0


def confidence_state(row: dict, high: float, low: float, margin_threshold: float) -> tuple[str, bool, float]:
    confidence = to_float(row, "confidence")
    second = to_float(row, "second_confidence")
    margin = confidence - second
    if confidence >= high and margin >= margin_threshold:
        return "C2_high", True, margin
    if confidence < low or margin < margin_threshold:
        return "C0_low_or_ambiguous", False, margin
    return "C1_medium", True, margin


def structural_feature_vector(
    row: dict,
    size_rank: int,
    detail_rank: int,
    reliable: bool,
    margin: float,
    fallback: str,
) -> dict[str, float]:
    diag = to_float(row, "bbox_diagonal")
    long_side = to_float(row, "obb_size_long")
    mid_side = to_float(row, "obb_size_mid")
    short_side = to_float(row, "obb_size_short")
    elongation = to_float(row, "elongation")
    flatness = to_float(row, "flatness")
    compactness = to_float(row, "compactness")
    face_count = max(0, to_int(row, "face_count"))
    area = max(0.0, to_float(row, "surface_area"))
    shape = row.get("shape_hint", "")

    aspect_xy = abs(long_side - mid_side) / max(long_side, 1e-9)
    thickness_ratio = short_side / max(long_side, 1e-9)
    area_density = area / max(diag * diag, 1e-12)
    face_density = face_count / max(diag * diag, 1e-12)
    density_score = clamp01((math.log10(max(face_density, 1.0)) - 2.0) / 4.0)

    plate_score = max(
        0.90 if shape == "flat_plate" else 0.0,
        0.85 if fallback in {"geom_ultra_thin_sheet", "geom_thin_plate"} else 0.0,
        clamp01((0.16 - flatness) / 0.16),
        clamp01((0.08 - thickness_ratio) / 0.08),
    )
    slender_score = max(
        0.90 if shape == "long_slender" else 0.0,
        0.85 if fallback in {"geom_wire_or_rod", "geom_slender_bar"} else 0.0,
        smoothstep(2.6, 8.0, elongation),
    )
    compact_score = max(
        0.85 if shape == "compact_block" else 0.0,
        0.70 if fallback in {"geom_compact_block", "geom_simple_irregular"} else 0.0,
        smoothstep(0.05, 0.18, compactness),
    )
    ring_disk_score = max(
        0.75 if fallback == "geom_round_or_cubic" and plate_score > 0.25 else 0.0,
        clamp01((0.22 - aspect_xy) / 0.22) * clamp01((0.45 - flatness) / 0.45),
    )
    simple_score = clamp01(1.0 - (detail_rank / 4.0) * 0.85)
    high_detail_score = max(smoothstep(2.0, 4.0, detail_rank), density_score)
    size_score = clamp01(size_rank / 6.0)
    micro_score = clamp01((2.0 - size_rank) / 2.0)
    medium_size_score = clamp01(1.0 - abs(size_rank - 3.0) / 3.0)
    bulk_score = max(
        0.0,
        smoothstep(4.0, 6.0, size_rank) * max(plate_score, compact_score, simple_score),
        smoothstep(5.0, 6.0, size_rank) * clamp01(area_density / 12.0),
    )
    uncertainty_score = 1.0 - clamp01(margin / 0.30)
    if reliable:
        uncertainty_score *= 0.45

    return {
        "micro_score": micro_score,
        "size_score": size_score,
        "medium_size_score": medium_size_score,
        "detail_score": clamp01(detail_rank / 4.0),
        "high_detail_score": high_detail_score,
        "density_score": density_score,
        "slender_score": slender_score,
        "plate_score": plate_score,
        "compact_score": compact_score,
        "ring_disk_score": ring_disk_score,
        "bulk_score": bulk_score,
        "simple_score": simple_score,
        "uncertainty_score": uncertainty_score,
    }


def prototype_match_score(features: dict[str, float], role: str) -> float:
    expected = ROLE_PROTOTYPES[role]["expected_features"]
    score_sum = 0.0
    weight_sum = 0.0
    for name, target in expected.items():
        actual = features.get(name, 0.0)
        weight = 0.75 + target
        score_sum += max(0.0, 1.0 - abs(actual - target)) * weight
        weight_sum += weight
    return clamp01(score_sum / max(weight_sum, 1e-9))


def semantic_role_scores(predicted_class: str, confidence: float, margin: float, reliable: bool) -> dict[str, float]:
    priors = CLASS_ROLE_PRIORS.get(predicted_class, {})
    evidence_quality = clamp01(confidence) * (0.70 + 0.30 * clamp01(margin / 0.30))
    if not reliable:
        evidence_quality *= 0.55
    return {role: clamp01(prior * evidence_quality) for role, prior in priors.items()}


def pointclip_role_scores(row: dict, include_micro_uncertain: bool = False) -> dict[str, float]:
    role = row.get("pointclip_top1_role", "")
    if role == "micro_uncertain" and not include_micro_uncertain:
        return {}
    if role not in ROLE_PROTOTYPES:
        return {}
    label_id = row.get("pointclip_top1_id", "")
    top1 = to_float(row, "pointclip_top1_score")
    top2 = to_float(row, "pointclip_top2_score")
    margin = to_float(row, "pointclip_margin", top1 - top2)

    # CLIP probabilities are spread over an open vocabulary. Treat top-1 as
    # supporting evidence only when either the absolute score or the margin is
    # meaningful. This prevents broad labels from steering low-confidence parts.
    if top1 < 0.055 and margin < 0.025:
        return {}

    top2_role = row.get("pointclip_top2_role", "")
    score_quality = smoothstep(0.035, 0.14, top1)
    margin_quality = smoothstep(0.006, 0.08, max(margin, 0.0))
    quality = clamp01(0.55 * score_quality + 0.45 * margin_quality)
    if top2_role == role:
        quality = clamp01(quality + 0.10)
    if label_id in GENERIC_POINTCLIP_LABELS:
        quality *= 0.72
    if quality < 0.25:
        return {}
    scores = {role: quality}

    second_role = top2_role
    if second_role in ROLE_PROTOTYPES and second_role != role:
        scores[second_role] = clamp01(smoothstep(0.035, 0.14, top2) * 0.35)
    return scores


def best_role(scores: dict[str, float]) -> tuple[str, float]:
    if not scores:
        return "ordinary_low_detail", 0.0
    return max(scores.items(), key=lambda item: (item[1], ROLE_PROTOTYPES[item[0]]["priority"]))


def inference_reasons(
    role: str,
    features: dict[str, float],
    neural_role: str,
    neural_score: float,
    structural_role: str,
    structural_score: float,
    pointclip_role: str,
    pointclip_score: float,
    semantic: str,
    reliable: bool,
) -> list[str]:
    reasons = []
    if neural_score >= 0.35:
        reasons.append(f"pointnext_prior={neural_role}:{neural_score:.2f}")
    if structural_score >= 0.55:
        reasons.append(f"structural_match={structural_role}:{structural_score:.2f}")
    if pointclip_score >= 0.35:
        reasons.append(f"pointclip_open_vocab={pointclip_role}:{pointclip_score:.2f}")
    if not reliable:
        reasons.append("low_or_ambiguous_pointnext_confidence")
    if features["bulk_score"] >= 0.65:
        reasons.append("large_static_bulk_evidence")
    if features["micro_score"] >= 0.55:
        reasons.append("micro_part")
    if features["slender_score"] >= 0.65:
        reasons.append("slender_axis_shape")
    if features["plate_score"] >= 0.65:
        reasons.append("thin_plate_or_sheet_shape")
    if features["ring_disk_score"] >= 0.65:
        reasons.append("round_disk_or_ring_like_shape")
    if features["high_detail_score"] >= 0.65:
        reasons.append("high_mesh_detail")
    if not reasons:
        reasons.append(f"semantic_group={semantic}")
    if role in {"critical_preserve", "motion_precision"}:
        reasons.append("functional_contact_or_motion_preservation")
    return reasons


def infer_structural_semantics(
    row: dict,
    size_rank: int,
    detail_rank: int,
    reliable: bool,
    semantic: str,
    fallback: str,
    margin: float,
) -> dict:
    predicted_class = row.get("predicted_class", "")
    confidence = to_float(row, "confidence")
    features = structural_feature_vector(row, size_rank, detail_rank, reliable, margin, fallback)
    structural_scores = {role: prototype_match_score(features, role) for role in ROLE_PROTOTYPES}
    neural_scores = semantic_role_scores(predicted_class, confidence, margin, reliable)
    pointclip_scores = pointclip_role_scores(row)

    neural_weight = clamp01(0.18 + 0.58 * confidence + 0.14 * clamp01(margin / 0.30))
    if not reliable:
        neural_weight *= 0.55
    pointclip_role, pointclip_score = best_role(pointclip_scores) if pointclip_scores else ("", 0.0)
    pointclip_weight = 0.18 * pointclip_score
    structural_weight = max(0.20, 1.0 - neural_weight - pointclip_weight)
    total_weight = neural_weight + structural_weight + pointclip_weight

    fused_scores = {}
    for role in ROLE_PROTOTYPES:
        fused_scores[role] = (
            neural_weight * neural_scores.get(role, 0.0)
            + structural_weight * structural_scores.get(role, 0.0)
            + pointclip_weight * pointclip_scores.get(role, 0.0)
        ) / max(total_weight, 1e-9)

    if features["bulk_score"] >= 0.70 and semantic in {"bulk_static_part", "other_part", "semantic_uncertain"}:
        fused_scores["large_static_bulk"] += 0.18
    if features["micro_score"] >= 0.55 and (not reliable or semantic == "fastener_repeated"):
        fused_scores["micro_uncertain"] += 0.22
    if semantic == "fastener_repeated" and reliable and features["micro_score"] < 0.55:
        fused_scores["repeated_fastener"] += 0.16
    if semantic == "motion_or_precision_part" and reliable:
        target_role = "critical_preserve" if predicted_class in {
            "bearings_bushings_guides",
            "gears_pulleys_chains",
            "motors_gearmotors",
            "springs",
            "rotating_fluid_machinery",
        } else "motion_precision"
        fused_scores[target_role] += 0.20
    if semantic == "fluid_or_interface_part" and reliable:
        fused_scores["interface_fluid"] += 0.18
    if semantic == "structural_key_part" and reliable:
        fused_scores["structural_control"] += 0.16
    if features["high_detail_score"] >= 0.78 and features["bulk_score"] < 0.55:
        fused_scores["high_detail_shape"] += 0.12

    role, fused_score = best_role({key: clamp01(value) for key, value in fused_scores.items()})
    structural_role, structural_score = best_role(structural_scores)
    neural_role, neural_score = best_role(neural_scores)

    prototype = ROLE_PROTOTYPES[role]
    policy = make_p10_policy(prototype["priority"], prototype["strategy"])
    reasons = inference_reasons(
        role,
        features,
        neural_role,
        neural_score,
        structural_role,
        structural_score,
        pointclip_role,
        pointclip_score,
        semantic,
        reliable,
    )
    rounded_features = {key: round(value, 4) for key, value in features.items()}

    return {
        "policy": policy,
        "inferred_role": role,
        "function_role": prototype["function_role"],
        "semantic_structural_score": round(fused_score, 4),
        "neural_role": neural_role,
        "neural_role_score": round(neural_score, 4),
        "pointclip_role": pointclip_role,
        "pointclip_role_score": round(pointclip_score, 4),
        "structural_match_role": structural_role,
        "structural_match_score": round(structural_score, 4),
        "protected_features": json.dumps(prototype["protected_features"], ensure_ascii=False),
        "inference_reason": ";".join(reasons),
        "structural_features": json.dumps(rounded_features, ensure_ascii=False),
    }


def add_weighted_scores(dst: dict[str, float], scores: dict[str, float], weight: float) -> float:
    if not scores or weight <= 0.0:
        return 0.0
    for role, score in scores.items():
        dst[role] = dst.get(role, 0.0) + clamp01(score) * weight
    return weight


def infer_ablation_semantics(
    row: dict,
    size_rank: int,
    detail_rank: int,
    reliable: bool,
    semantic: str,
    fallback: str,
    margin: float,
    evidence_mode: str,
) -> dict:
    predicted_class = row.get("predicted_class", "")
    confidence = to_float(row, "confidence")
    use_pointnext = "pointnext" in evidence_mode
    use_pointclip = "pointclip" in evidence_mode
    use_structure = "structure" in evidence_mode

    features = structural_feature_vector(row, size_rank, detail_rank, reliable, margin, fallback)
    structural_scores = {role: prototype_match_score(features, role) for role in ROLE_PROTOTYPES}
    neural_scores = semantic_role_scores(predicted_class, confidence, margin, reliable)
    pointclip_scores = pointclip_role_scores(row, include_micro_uncertain=use_pointclip and not use_pointnext)

    fused_scores: dict[str, float] = {}
    total_weight = 0.0
    evidence_notes = []

    if use_pointnext:
        pointnext_weight = clamp01(0.25 + 0.75 * confidence) * (0.70 + 0.30 * clamp01(margin / 0.30))
        if not reliable:
            pointnext_weight *= 0.55
            neural_scores = dict(neural_scores)
            neural_scores["micro_uncertain"] = max(neural_scores.get("micro_uncertain", 0.0), 0.45)
        if neural_scores:
            total_weight += add_weighted_scores(fused_scores, neural_scores, pointnext_weight)
            evidence_notes.append("pointnext")
        else:
            fused_scores["micro_uncertain"] = max(fused_scores.get("micro_uncertain", 0.0), 0.25)
            evidence_notes.append("pointnext_no_prior")

    if use_pointclip:
        if pointclip_scores:
            _, pointclip_score = best_role(pointclip_scores)
            pointclip_weight = 0.35 + 0.65 * pointclip_score
            total_weight += add_weighted_scores(fused_scores, pointclip_scores, pointclip_weight)
            evidence_notes.append("pointclip")
        else:
            evidence_notes.append("pointclip_missing_or_weak")
            if not use_pointnext and not use_structure:
                fused_scores["micro_uncertain"] = max(fused_scores.get("micro_uncertain", 0.0), 1.0)

    if use_structure:
        structure_weight = 1.0 if not (use_pointnext or use_pointclip) else 0.70
        total_weight += add_weighted_scores(fused_scores, structural_scores, structure_weight)
        evidence_notes.append("structure")

    if total_weight > 0.0:
        fused_scores = {role: score / total_weight for role, score in fused_scores.items()}

    if use_structure:
        if features["bulk_score"] >= 0.70 and (not use_pointnext or semantic in {"bulk_static_part", "other_part", "semantic_uncertain"}):
            fused_scores["large_static_bulk"] = fused_scores.get("large_static_bulk", 0.0) + 0.18
        if features["micro_score"] >= 0.55 and (not use_pointnext or not reliable or semantic == "fastener_repeated"):
            fused_scores["micro_uncertain"] = fused_scores.get("micro_uncertain", 0.0) + 0.22
        if features["high_detail_score"] >= 0.78 and features["bulk_score"] < 0.55:
            fused_scores["high_detail_shape"] = fused_scores.get("high_detail_shape", 0.0) + 0.12

    if use_pointnext:
        if semantic == "fastener_repeated" and reliable and (not use_structure or features["micro_score"] < 0.55):
            fused_scores["repeated_fastener"] = fused_scores.get("repeated_fastener", 0.0) + 0.16
        if semantic == "motion_or_precision_part" and reliable:
            target_role = "critical_preserve" if predicted_class in {
                "bearings_bushings_guides",
                "gears_pulleys_chains",
                "motors_gearmotors",
                "springs",
                "rotating_fluid_machinery",
            } else "motion_precision"
            fused_scores[target_role] = fused_scores.get(target_role, 0.0) + 0.20
        if semantic == "fluid_or_interface_part" and reliable:
            fused_scores["interface_fluid"] = fused_scores.get("interface_fluid", 0.0) + 0.18
        if semantic == "structural_key_part" and reliable:
            fused_scores["structural_control"] = fused_scores.get("structural_control", 0.0) + 0.16

    role, fused_score = best_role({key: clamp01(value) for key, value in fused_scores.items()})
    structural_role, structural_score = best_role(structural_scores)
    neural_role, neural_score = best_role(neural_scores)
    pointclip_role, pointclip_score = best_role(pointclip_scores) if pointclip_scores else ("", 0.0)

    prototype = ROLE_PROTOTYPES[role]
    policy = make_p10_policy(prototype["priority"], prototype["strategy"])
    reasons = inference_reasons(
        role,
        features,
        neural_role if use_pointnext else "",
        neural_score if use_pointnext else 0.0,
        structural_role if use_structure else "",
        structural_score if use_structure else 0.0,
        pointclip_role if use_pointclip else "",
        pointclip_score if use_pointclip else 0.0,
        semantic,
        reliable,
    )
    reasons.append(f"evidence_mode={evidence_mode}")
    rounded_features = {key: round(value, 4) for key, value in features.items()}

    return {
        "policy": policy,
        "evidence_mode": evidence_mode,
        "evidence_sources": ";".join(evidence_notes),
        "inferred_role": role,
        "function_role": prototype["function_role"],
        "semantic_structural_score": round(fused_score, 4),
        "neural_role": neural_role if use_pointnext else "",
        "neural_role_score": round(neural_score, 4) if use_pointnext else 0.0,
        "pointclip_role": pointclip_role if use_pointclip else "",
        "pointclip_role_score": round(pointclip_score, 4) if use_pointclip else 0.0,
        "structural_match_role": structural_role if use_structure else "",
        "structural_match_score": round(structural_score, 4) if use_structure else 0.0,
        "protected_features": json.dumps(prototype["protected_features"], ensure_ascii=False),
        "inference_reason": ";".join(reasons),
        "structural_features": json.dumps(rounded_features, ensure_ascii=False),
    }


def classify_row(row: dict, thresholds: dict, args: argparse.Namespace) -> dict:
    size_class, size_rank = classify_size(row, thresholds)
    confidence_class, reliable, margin = confidence_state(
        row, args.high_confidence, args.low_confidence, args.low_margin
    )
    semantic = semantic_group(row.get("predicted_class", ""), reliable)
    fallback = geometry_fallback(row)
    detail_class, detail_rank = detail_level(row, thresholds)
    evidence_mode = getattr(args, "evidence_mode", "pointnext_pointclip_structure")
    if evidence_mode == "pointnext_pointclip_structure":
        inference = infer_structural_semantics(row, size_rank, detail_rank, reliable, semantic, fallback, margin)
        inference.setdefault("evidence_mode", evidence_mode)
        inference.setdefault("evidence_sources", "pointnext;pointclip;structure")
    else:
        inference = infer_ablation_semantics(row, size_rank, detail_rank, reliable, semantic, fallback, margin, evidence_mode)
    policy = inference.pop("policy")

    small_low_conf_refined = size_rank <= 2 and not reliable
    if small_low_conf_refined:
        semantic = f"{semantic}:{fallback}"

    return {
        **row,
        "size_class": size_class,
        "detail_class": detail_class,
        "confidence_class": confidence_class,
        "confidence_margin": margin,
        "semantic_group": semantic,
        "geometry_fallback_class": fallback,
        "small_low_conf_refined": small_low_conf_refined,
        **inference,
        **policy,
    }


def summarize(rows: list[dict], thresholds: dict) -> dict:
    return {
        "num_rows": len(rows),
        "thresholds": thresholds,
        "size_class_counts": Counter(row["size_class"] for row in rows),
        "detail_class_counts": Counter(row["detail_class"] for row in rows),
        "confidence_class_counts": Counter(row["confidence_class"] for row in rows),
        "semantic_group_counts": Counter(row["semantic_group"] for row in rows),
        "geometry_fallback_counts": Counter(row["geometry_fallback_class"] for row in rows),
        "inferred_role_counts": Counter(row["inferred_role"] for row in rows),
        "function_role_counts": Counter(row["function_role"] for row in rows),
        "neural_role_counts": Counter(row["neural_role"] for row in rows),
        "pointclip_role_counts": Counter(row["pointclip_role"] for row in rows),
        "structural_match_role_counts": Counter(row["structural_match_role"] for row in rows),
        "lod_priority_counts": Counter(row["lod_priority"] for row in rows),
        "lod_strategy_counts": Counter(row["lod_strategy"] for row in rows),
        "predicted_class_counts": Counter(row["predicted_class"] for row in rows),
    }


def counter_to_dict(obj):
    if isinstance(obj, Counter):
        return dict(obj)
    if isinstance(obj, dict):
        return {key: counter_to_dict(value) for key, value in obj.items()}
    return obj


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(args.input_csv.open(encoding="utf-8-sig", newline="")))
    if not rows:
        raise ValueError(f"No rows in {args.input_csv}")

    thresholds = make_thresholds(rows)
    output_rows = [classify_row(row, thresholds, args) for row in rows]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    summary = summarize(output_rows, thresholds)
    if args.output_json:
        args.output_json.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.summary_json:
        args.summary_json.write_text(json.dumps(counter_to_dict(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {args.output_csv.resolve()}")
    if args.output_json:
        print(f"Wrote: {args.output_json.resolve()}")
    if args.summary_json:
        print(f"Wrote: {args.summary_json.resolve()}")
    print("LOD priority counts:")
    for key, value in summary["lod_priority_counts"].most_common():
        print(f"  {key}: {value}")
    print("LOD strategy counts:")
    for key, value in summary["lod_strategy_counts"].most_common():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
