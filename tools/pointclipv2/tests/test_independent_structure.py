"""Prove the supplementary G/C+G baselines do not consume supervised evidence."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_lod_ablation_variants import load_lod_builder
from independent_structure import classify_independent


class IndependentStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = load_lod_builder(Path(__file__).resolve().parents[3])

    def row(self):
        return {"order": "1", "node_index": "0", "mesh_index": "0", "face_count": "800",
                "vertex_count": "400", "surface_area": "2", "bbox_diagonal": "1.5",
                "bbox_volume": "0.5", "obb_size_long": "1", "obb_size_mid": "0.8",
                "obb_size_short": "0.6", "elongation": "1.25", "flatness": "0.6",
                "compactness": "0.14", "shape_hint": "compact_block",
                "predicted_class": "nuts", "confidence": "0.2", "second_confidence": "0.19",
                "pointclip_top1_id": "transmission", "pointclip_top1_role": "critical_preserve",
                "pointclip_top1_score": "0.8", "pointclip_top2_role": "ordinary_low_detail",
                "pointclip_top2_score": "0.01", "pointclip_margin": "0.79"}

    def test_supervised_label_and_confidence_changes_do_not_change_either_baseline(self):
        row = self.row()
        thresholds = self.builder.make_thresholds([row])
        for use_clip in [False, True]:
            original = classify_independent(self.builder, row, thresholds, use_clip)
            for label, confidence in [("springs", "0.999"), ("unknown", "0.001")]:
                modified = dict(row, predicted_class=label, confidence=confidence, second_confidence="0")
                self.assertEqual(original, classify_independent(self.builder, modified, thresholds, use_clip))
            self.assertNotIn("confidence", original)
            self.assertNotIn("predicted_class", original)
            self.assertNotIn("uncertainty_score", json.loads(original["structural_features"]))

    def test_open_plus_geometry_responds_to_open_evidence(self):
        row = self.row()
        thresholds = self.builder.make_thresholds([row])
        first = classify_independent(self.builder, row, thresholds, True)
        changed = dict(row, pointclip_top1_role="repeated_fastener")
        second = classify_independent(self.builder, changed, thresholds, True)
        self.assertNotEqual(first["inferred_role"], second["inferred_role"])
        self.assertEqual(classify_independent(self.builder, row, thresholds, False),
                         classify_independent(self.builder, changed, thresholds, False))


if __name__ == "__main__":
    unittest.main()
