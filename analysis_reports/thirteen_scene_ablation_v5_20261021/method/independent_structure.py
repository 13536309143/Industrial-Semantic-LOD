"""Supplementary geometry-only / open+geometry baselines, without PointNeXt evidence.

The deployed fusion is unchanged. These diagnostics remove the supervised
uncertainty feature from geometric prototype matching and renormalize weights.
"""

import json


IDENTITY_GEOMETRY = {
    "order", "node_index", "node_name", "mesh_index", "mesh_name", "inference_mode",
    "group_id", "group_size", "vertex_count", "face_count", "surface_area", "mesh_volume",
    "bbox_diagonal", "bbox_volume", "obb_size_long", "obb_size_mid", "obb_size_short",
    "elongation", "flatness", "compactness", "shape_hint",
}


def classify_independent(builder, row, thresholds, use_pointclip=False):
    size_class, size_rank = builder.classify_size(row, thresholds)
    detail_class, detail_rank = builder.detail_level(row, thresholds)
    fallback = builder.geometry_fallback(row)
    features = builder.structural_feature_vector(row, size_rank, detail_rank, False, 0.0, fallback)
    features.pop("uncertainty_score")
    geometry = {}
    for role, prototype in builder.ROLE_PROTOTYPES.items():
        expected = {k: v for k, v in prototype["expected_features"].items() if k != "uncertainty_score"}
        weights = {k: 0.75 + target for k, target in expected.items()}
        geometry[role] = sum(max(0.0, 1.0-abs(features[k]-target))*weights[k]
                             for k, target in expected.items()) / sum(weights.values())
    if features["bulk_score"] >= 0.70:
        geometry["large_static_bulk"] += 0.18
    if features["micro_score"] >= 0.55:
        geometry["micro_uncertain"] += 0.22
    if features["high_detail_score"] >= 0.78 and features["bulk_score"] < 0.55:
        geometry["high_detail_shape"] += 0.12
    geometry = {k: builder.clamp01(v) for k, v in geometry.items()}
    open_scores = builder.pointclip_role_scores(row, include_micro_uncertain=True) if use_pointclip else {}
    clip_weight = 0.35 + 0.65*max(open_scores.values()) if open_scores else 0.0
    fused = {role: (0.70*score+clip_weight*open_scores.get(role, 0))/(0.70+clip_weight)
             for role, score in geometry.items()}
    role, score = builder.best_role(fused)
    geometric_role, geometric_score = builder.best_role(geometry)
    keep = {key: value for key, value in row.items()
            if key in IDENTITY_GEOMETRY or key.startswith(("bbox_min_", "bbox_max_", "bbox_size_", "centroid_"))
            or (use_pointclip and key.startswith("pointclip_"))}
    return {**keep, "size_class": size_class, "detail_class": detail_class,
            "evidence_mode": "pointclip_structure_independent" if use_pointclip else "structure_independent",
            "evidence_sources": "pointclip;structure" if open_scores else "structure",
            "inferred_role": role, "function_role": builder.ROLE_PROTOTYPES[role]["function_role"],
            "structural_match_role": geometric_role, "structural_match_score": geometric_score,
            "pointclip_role": builder.best_role(open_scores)[0] if open_scores else "",
            "pointclip_role_score": max(open_scores.values()) if open_scores else 0.0,
            "semantic_structural_score": score,
            "structural_features": json.dumps(features, ensure_ascii=False),
            "inference_reason": "supplementary_baseline;supervised_uncertainty_feature_removed",
            **builder.make_p10_policy(builder.ROLE_PROTOTYPES[role]["priority"], builder.ROLE_PROTOTYPES[role]["strategy"])}
