import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adaptive_p10_v5_policy import reprofile, infer, PROFILES
import test_assembly_context as fixtures
from fuse_open_semantic_v2 import builder


class BudgetV5Tests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.AssemblyContextTests()
        self.row = self.fixture.row()
        self.desc = dict(self.fixture.descriptor(), surface_area_normalized=1.,
                         largest_component_area_fraction=1., boundary_length_normalized=0.)
        self.context = self.fixture.context()

    def profile(self, role='unknown', **kw):
        base = dict(function_role=role, confidence=.9, resolved_object_class='unchanged')
        base.update(kw.get('base', {}))
        return reprofile(base, kw.get('row', self.row), kw.get('desc', self.desc),
                         kw.get('context', self.context), kw.get('repeats', 1),
                         kw.get('enabled', True))

    def priority(self, result):
        return int(result['lod_priority'].split('_')[0][1:])

    def test_unresolved_function_is_split_by_geometry_without_claiming_identity(self):
        cases = [
            self.profile(context=dict(self.context, scene_size_ratio=.001, local_group_index=None)),
            self.profile(context=dict(self.context, scene_size_ratio=.03, local_group_index=None)),
            self.profile(context=dict(self.context, scene_size_ratio=.2, local_group_index=None)),
            self.profile(row=dict(self.row, obb_size_mid=.03, obb_size_short=.03),
                         context=dict(self.context, scene_size_ratio=.2)),
            self.profile(row=dict(self.row, obb_size_mid=.03, obb_size_short=.0001),
                         desc=dict(self.desc, sharp_length_normalized=120., curved_length_normalized=256.,
                                   significant_components=16, largest_component_area_fraction=.2),
                         context=dict(self.context, scene_size_ratio=.2)),
        ]
        self.assertGreaterEqual(len(set(map(self.priority, cases))), 4)
        for result in cases:
            self.assertEqual(result['function_role'], 'unknown')
            self.assertEqual(result['resolved_object_class'], 'unchanged')
            self.assertFalse(result['allow_geometry_drop'])

    def test_arbitrarily_dense_curves_alone_do_not_force_P6(self):
        outputs = [self.profile(desc=dict(self.desc, curved_length_normalized=v),
                               context=dict(self.context, scene_size_ratio=.01, local_group_index=None))
                   for v in (33., 256., 1e9)]
        self.assertTrue(all(self.priority(r) < 6 for r in outputs))
        self.assertEqual(outputs[1]['geometry_budget_score'], outputs[2]['geometry_budget_score'])

    def test_thin_plate_has_local_constraints_without_global_P6_floor(self):
        result = self.profile(row=dict(self.row, obb_size_short=.001))
        solid = self.profile()
        self.assertLess(self.priority(result), 6)
        self.assertEqual(result['budget_morphology'], 'plate')
        self.assertLess(result['max_axis_shrink'], solid['max_axis_shrink'])
        self.assertGreater(result['feature_lock_budget'], solid['feature_lock_budget'])

    def test_threaded_micro_fastener_can_get_P1_with_geometry_guards(self):
        result = self.profile('standard_fastener', repeats=8,
            desc=dict(self.desc, curved_length_normalized=500.),
            context=dict(self.context, scene_size_ratio=.001))
        self.assertEqual(self.priority(result), 1)
        self.assertTrue(result['preserve_components'])
        self.assertGreater(result['simplify_error_relative'], 0.)
        self.assertTrue(result['allow_cull'])
        self.assertTrue(result['runtime_geometry_omission'])
        self.assertFalse(result['allow_geometry_drop'])

    def test_large_or_conflicting_fastener_receives_no_routine_discount(self):
        for kwargs in [dict(context=dict(self.context, scene_size_ratio=.2)),
                       dict(base=dict(type_shape_conflict=True))]:
            result = self.profile('standard_fastener', **kwargs)
            self.assertFalse(result['ordinary_fastener_discount'])
            self.assertEqual(self.priority(result), result['geometry_budget_priority'])

    def test_precision_mount_function_changes_budget_not_object_identity(self):
        ordinary = self.profile('standard_fastener')
        precision = self.profile('precision_fastener')
        self.assertLess(self.priority(ordinary), self.priority(precision))
        self.assertEqual(ordinary['resolved_object_class'], precision['resolved_object_class'])

    def test_component_area_distribution_matters_beyond_component_count(self):
        dominant = self.profile(desc=dict(self.desc, significant_components=8,
                                         largest_component_area_fraction=.999))
        distributed = self.profile(desc=dict(self.desc, significant_components=8,
                                            largest_component_area_fraction=.3))
        self.assertGreater(distributed['geometry_budget_score'], dominant['geometry_budget_score'])
        self.assertLess(distributed['simplify_error_relative'], dominant['simplify_error_relative'])

    def test_geometry_disabled_is_independent_of_all_shape_size_and_repeat_inputs(self):
        a = self.profile(enabled=False)
        b = self.profile(enabled=False, row=dict(self.row, obb_size_mid=.001),
            desc=dict(self.desc, significant_components=30, curved_length_normalized=1e6),
            context=dict(self.context, scene_size_ratio=1.), repeats=100)
        self.assertEqual(a, b)

    def test_H0_global_context_has_no_implicit_local_size_influence(self):
        context = dict(self.context, local_group_index=None, group_size_ratio=.01)
        a = self.profile(context=context)
        b = self.profile(context=dict(context, group_size_ratio=1.))
        self.assertEqual(a, b)

    def test_invalid_geometry_never_uses_aggressive_base_cleanup(self):
        result = self.profile(desc=dict(self.desc, geometry_valid=False))
        self.assertGreaterEqual(self.priority(result), 5)
        self.assertEqual(result['base_simplify_ratio'], 1.)
        self.assertEqual(result['base_error_relative'], 0.)
        self.assertEqual(result['geometry_budget_family'], 'invalid_geometry')

    def test_reference_name_without_geometry_is_not_low_impact_evidence(self):
        result = self.profile('scene_reference_candidate', enabled=False)
        self.assertGreaterEqual(self.priority(result), 5)

    def test_face_count_confidence_and_node_name_are_not_budget_shortcuts(self):
        a = self.profile()
        b = self.profile(row=dict(self.row, face_count=1000000, vertex_count=3000000,
                                 node_name='CCD关键零件'), base=dict(confidence=.001))
        for field in ('lod_priority', 'geometry_budget_score', 'feature_lock_budget', 'target_ratio_far'):
            self.assertEqual(a[field], b[field])

    def test_previous_measured_quality_claim_is_removed(self):
        result = self.profile(base=dict(measured_proxy_faces=10, quality_fallback='none', quality_search='old'))
        self.assertNotIn('measured_proxy_faces', result)
        self.assertNotIn('quality_search', result)

    def test_function_and_section_both_required_for_P10(self):
        rod = dict(self.row, obb_size_mid=.03, obb_size_short=.03)
        self.assertEqual(self.priority(self.profile('process_tool', row=rod)), 10)
        self.assertLess(self.priority(self.profile('unknown', row=rod)), 10)
        self.assertEqual(self.priority(self.profile('process_tool', desc=dict(self.desc,
            curved_length_normalized=1e8))), 9)

    def test_all_ten_tiers_have_nonquota_evidence_cases(self):
        outputs = [self.profile('standard_fastener', repeats=8,
                       context=dict(self.context, scene_size_ratio=.001)),
                   self.profile('standard_fastener', context=dict(self.context, scene_size_ratio=.005)), self.profile(),
                   self.profile('enclosure_panel'), self.profile(enabled=False),
                   self.profile('motion_mount'), self.profile('fluid_connection'),
                   self.profile('actuation'), self.profile('precision_guidance'),
                   self.profile('process_tool', row=dict(self.row, obb_size_mid=.03, obb_size_short=.03))]
        self.assertEqual(list(map(self.priority, outputs)), list(range(1, 11)))
        for result in outputs:
            near, mid, far = [result['target_ratio_' + level] for level in ('near', 'mid', 'far')]
            self.assertTrue(0 < far <= mid <= near <= 1)
            for key in ('screen_error_weight', 'feature_lock_budget', 'simplify_error_relative'):
                self.assertTrue(math.isfinite(result[key]))

    def test_disabled_neural_branches_and_names_remain_isolated(self):
        thresholds = builder.make_thresholds([self.row])
        def output(mode, row=None, opened=None):
            return infer(row or self.row, opened or {}, fixtures.TAXONOMY, thresholds,
                         self.desc, self.context, fixtures.CONFIG, mode, 'off', False, 8)
        for mode in ('C', 'CG', 'G'):
            a = output(mode)
            b = output(mode, row=dict(self.row, predicted_class='springs', confidence=.001))
            for field in ('lod_priority', 'geometry_budget_score', 'function_role', 'target_ratio_far'):
                self.assertEqual(a[field], b[field])
        for mode in ('N', 'NG'):
            self.assertEqual(output(mode), output(mode, opened=dict(open_status='accepted', open_object_id='camera')))
        for mode in ('N', 'C', 'NG', 'CG', 'NC', 'NCG', 'G'):
            a = output(mode)
            b = output(mode, row=dict(self.row, node_name='CCD折弯刀工件'))
            for field in ('lod_priority', 'function_role', 'geometry_budget_score'):
                self.assertEqual(a[field], b[field])

    def test_native_low_budget_plate_keeps_extent_and_reduces_planar_redundancy(self):
        import numpy as np
        from probe_assembly_geometry import load_probe, prepare_geometry
        from run_adaptive_p10 import bounded_function, bounded_metrics
        dll_path = Path(__file__).resolve().parents[3] / '_bin/Release/semantic_geometry_probe.dll'
        if not dll_path.exists():
            self.skipTest('Native probe not built')
        dll, _ = load_probe(dll_path)
        size = 16
        vertices = np.array([[x / (size - 1), y / (size - 1), 0.]
                             for x in range(size) for y in range(size)], dtype=np.float32)
        faces = []
        for x in range(size - 1):
            for y in range(size - 1):
                a = x * size + y
                faces.extend([[a, a + 1, a + size], [a + 1, a + size + 1, a + size]])
        faces = np.array(faces, dtype=np.uint32)
        policy = self.profile(row=dict(self.row, obb_size_short=.001),
                              context=dict(self.context, scene_size_ratio=.001, local_group_index=None))
        self.assertLessEqual(self.priority(policy), 3)
        geometry = prepare_geometry(vertices, faces)
        metric = bounded_metrics(bounded_function(dll), geometry, policy['target_ratio_far'], policy)
        self.assertLess(metric['actual_faces'], len(faces))
        self.assertEqual(metric['missing_significant_components'], 0)
        self.assertLessEqual(metric['max_axis_shrink'], policy['max_axis_shrink'] + 1e-6)


if __name__ == '__main__':
    unittest.main()
