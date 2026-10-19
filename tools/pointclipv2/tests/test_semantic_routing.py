import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit_semantic_routing import route


class SemanticRoutingTests(unittest.TestCase):
    def row(self, **changes):
        return dict(order="1",node_index="0",node_name="part",mesh_index="0",bbox_diagonal=".002",
                    face_count="200",predicted_class="springs",confidence=".1",second_confidence=".09",**changes)

    def test_small_unknown_is_not_invisible_or_disposable(self):
        result=route(self.row(),1)
        self.assertEqual(result["semantic_status"],"unknown_or_ambiguous")
        self.assertEqual(result["importance_status"],"not_assessed")
        self.assertEqual(result["visibility_status"],"unmeasured")
        self.assertFalse(result["allow_geometry_drop"])
        self.assertFalse(result["skip_all_semantics"])

    def test_small_known_critical_part_has_preservation_request(self):
        row=self.row();row.update(confidence=".95",second_confidence=".01")
        result=route(row,1)
        self.assertEqual(result["routing_action"],"small_with_critical_evidence")
        self.assertFalse(result["allow_geometry_drop"])

    def test_large_extent_and_high_tessellation_both_request_multiscale(self):
        for changes in [{"bbox_diagonal":".2"},{"face_count":"581563"}]:
            row=self.row();row.update(changes)
            self.assertEqual(route(row,1)["routing_action"],"global_plus_local_multiscale")


if __name__ == "__main__":
    unittest.main()
