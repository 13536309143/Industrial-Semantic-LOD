import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from factorial_ablation_policy import classify,global_contexts
import test_assembly_context as fixtures
from fuse_open_semantic_v2 import builder


class FactorialAblationTests(unittest.TestCase):
    def setUp(self):self.fixture=fixtures.AssemblyContextTests();self.row=self.fixture.row()

    def output(self,mode,row=None,opened=None,tree='on'):
        row=row or self.row
        return classify(row,opened or {},fixtures.TAXONOMY,builder.make_thresholds([row]),self.fixture.descriptor(),
            self.fixture.context('feeding','螺母上料轨道'),fixtures.CONFIG,mode,tree)

    def test_NH_cannot_read_disabled_open_predictions(self):
        a=self.output('N')
        b=self.output('N',opened=dict(open_group_known_class='nuts',open_status='accepted',open_reliability=.99))
        self.assertEqual(a,b)

    def test_CH_and_CGH_cannot_read_disabled_supervised_predictions(self):
        opened=dict(open_status='class_only',open_class_group='known:nuts',open_group_known_class='nuts',
            open_family='fastening',open_reliability=.9,open_role_scores_json=json.dumps({'repeated_fastener':1}))
        for mode in ('C','CG'):
            a=self.output(mode,opened=opened)
            b=self.output(mode,row=dict(self.row,predicted_class='springs',confidence=.01,second_confidence=.009),opened=opened)
            self.assertEqual(a,b)
            self.assertNotIn('source_pointnext_class',a)

    def test_global_context_has_no_implicit_parent_scale(self):
        context=global_contexts([self.row])['1']
        self.assertEqual(context['group_size_ratio'],context['scene_size_ratio'])
        self.assertEqual(context['local_group_members'],0)
        self.assertIsNone(context['parent_index'])

    def test_leaf_name_is_never_a_main_matrix_decision_input(self):
        for mode in ('N','C','NG','CG','NC','NCG','G'):
            a=self.output(mode)
            b=self.output(mode,row=dict(self.row,node_name='工件产品CCD折弯头'))
            for field in ('function_role','lod_priority','resolved_object_class','target_ratio_far'):
                self.assertEqual(a[field],b[field])


if __name__=='__main__':unittest.main()
