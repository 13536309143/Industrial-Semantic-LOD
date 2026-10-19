import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_named_scene import name_hints
from summarize_named_scenes import expected_known_classes


class NamedEvidenceTests(unittest.TestCase):
    def test_nut_feeder_is_not_a_nut_label(self):
        self.assertEqual(expected_known_classes({'name_candidate_role': 'fastener', 'leaf_name': '螺母送料盘-1'}), set())
        self.assertEqual(expected_known_classes({'name_candidate_role': 'fastener', 'leaf_name': 'K31-02螺母-17'}), {'nuts'})

    def test_assembly_prefix_does_not_label_leaf(self):
        leaf, hits = name_hints('CCD装配体/JX22-018-A18-5')
        self.assertEqual(leaf, 'JX22-018-A18-5')
        self.assertEqual(hits, [])

    def test_mounting_plate_retains_camera_context_but_is_not_camera(self):
        leaf, hits = name_hints('光源安装板-1')
        self.assertEqual(hits[0][0], 'mount_or_support')
        self.assertIn('sensing_or_inspection', [role for role, term in hits])

    def test_workpiece_fastener_preserves_both_hints(self):
        leaf, hits = name_hints('镜向产品（螺母）-35')
        self.assertEqual({role for role, term in hits}, {'fastener', 'product_or_workpiece'})


if __name__ == '__main__':
    unittest.main()
