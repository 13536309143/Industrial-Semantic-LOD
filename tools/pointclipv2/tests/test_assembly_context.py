import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from assembly_context_features import surface_descriptor,build_contexts,geometry_risk
from assembly_context_policy import apply_policy,leaf_function
from fuse_open_semantic_v2 import builder
from measured_geometry_guard import regression,constrain_schedule

DIRECTORY=Path(__file__).resolve().parents[1]
TAXONOMY=json.loads((DIRECTORY/'industrial_open_vocab_v2.json').read_text(encoding='utf-8'))
CONFIG=json.loads((DIRECTORY/'assembly_context_config.json').read_text(encoding='utf-8'))


class AssemblyContextTests(unittest.TestCase):
    def row(self,**changes):
        return dict(dict(order=1,node_index=1,mesh_index=0,node_name='part',mesh_name='part',
            vertex_count=20,face_count=100,surface_area=2.,bbox_diagonal=1.7,bbox_volume=1.,
            obb_size_long=1.,obb_size_mid=1.,obb_size_short=1.,elongation=1.,flatness=1.,compactness=.1,
            predicted_class='nuts',confidence=.95,second_confidence=.02,
            bbox_min_x=0.,bbox_min_y=0.,bbox_min_z=0.,bbox_max_x=1.,bbox_max_y=1.,bbox_max_z=1.),**changes)

    def descriptor(self):
        return dict(geometry_valid=True,sharp_length_normalized=7.,curved_length_normalized=0.,significant_components=1)

    def context(self,category=None,name='fixture'):
        return dict(scene_size_ratio=.01,group_size_ratio=.1,local_group_members=8,local_group_index=0,
            context=dict(categories=[category],distance=1,descendant_count=8,name=name,node_index=0)if category else None)

    def run_policy(self,row=None,context=None,mode='NCG',tree=False,leaf=False):
        row=row or self.row()
        return apply_policy(row,{},TAXONOMY,builder.make_thresholds([row]),self.descriptor(),
                            context or self.context(),CONFIG,mode,tree,leaf)

    def test_same_identity_different_assembly_changes_function_not_type(self):
        frame=self.run_policy(context=self.context('frame'),tree=True)
        fixture=self.run_policy(context=self.context('fixture'),tree=True)
        self.assertEqual(frame['resolved_object_class'],fixture['resolved_object_class'])
        self.assertEqual(frame['function_role'],'standard_fastener')
        self.assertEqual(fixture['function_role'],'precision_fastener')
        self.assertNotEqual(frame['lod_priority'],fixture['lod_priority'])

    def test_parent_label_is_not_inherited_as_child_identity(self):
        result=self.run_policy(context=self.context('inspection','CCD检测机构'),tree=True)
        self.assertIn('nuts',result['resolved_object_class'])
        self.assertNotIn('camera',result['resolved_object_class'])

    def test_tree_only_does_not_read_leaf_name(self):
        first=self.run_policy(tree=True)
        second=self.run_policy(row=self.row(node_name='产品螺母'),tree=True)
        for k in ('resolved_object_class','lod_priority','function_role','assembly_context_applied'):
            self.assertEqual(first[k],second[k])

    def test_geometry_only_is_independent_of_semantic_predictions(self):
        first=self.run_policy(mode='G')
        second=self.run_policy(row=self.row(predicted_class='springs',confidence=.1,second_confidence=.08),mode='G')
        self.assertEqual(first,second)
        self.assertNotIn('source_pointnext_class',first)

    def test_open_only_is_independent_of_supervised_predictions(self):
        first=self.run_policy(mode='C')
        second=self.run_policy(row=self.row(predicted_class='springs',confidence=.1,second_confidence=.08),mode='C')
        self.assertEqual(first,second)

    def test_scoped_tree_does_not_raise_every_fastener_in_large_workcell(self):
        context=self.context('tooling');context['context']['descendant_count']=100
        result=self.run_policy(context=context,tree=True)
        self.assertEqual(result['function_role'],'standard_fastener')

    def test_feeding_does_not_convert_arbitrary_bolt_into_workpiece(self):
        result=self.run_policy(row=self.row(predicted_class='screws_bolts_studs'),
                              context=self.context('feeding','螺母上料轨道'),tree=True)
        self.assertEqual(result['function_role'],'standard_fastener')

    def test_large_slender_standard_part_is_flagged_and_preserved(self):
        row=self.row(obb_size_long=100.,obb_size_mid=2.,obb_size_short=2.)
        context=self.context();context['scene_size_ratio']=.3
        result=self.run_policy(row=row,context=context)
        self.assertTrue(result['type_shape_conflict'])
        self.assertFalse(result['allow_cull'])
        self.assertGreaterEqual(result['feature_lock_floor'],.16)
        self.assertGreaterEqual(result['target_ratio_near'],.84)

    def test_geometry_descriptor_is_unit_and_rigid_rotation_invariant(self):
        vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=float)
        faces=np.array([[0,1,2],[0,3,1],[0,2,3],[1,3,2]])
        base=surface_descriptor(vertices,faces)
        theta=.37;rotation=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
        changed=surface_descriptor(vertices@rotation*1000+20000,faces)
        for k in ('sharp_length_normalized','curved_length_normalized','surface_area_normalized'):
            self.assertAlmostEqual(base[k],changed[k],places=5)

    def test_descriptor_welds_duplicated_seam_vertices(self):
        vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,0],[0,1,0],[0,0,1]],dtype=float)
        result=surface_descriptor(vertices,np.array([[0,1,2],[3,4,5]]))
        self.assertEqual(result['welded_vertices'],4)
        self.assertEqual(result['significant_components'],1)

    def test_compound_mount_names_do_not_become_functional_device(self):
        for text in ('CCD安装板','电机支架','同步带压板','轴承座'):
            self.assertEqual(leaf_function(text),'mounting_support')

    def test_bending_head_is_a_tool_hint_without_inventing_an_exact_object_type(self):
        result=self.run_policy(row=self.row(node_name='折弯头-1'),leaf=True)
        self.assertEqual(result['function_role'],'process_tool')
        self.assertEqual(result['resolved_status'],'name_identity_conflict')
        self.assertEqual(result['resolved_object_class'],'')
        self.assertTrue(result['lod_priority'].startswith('P9_'))

    def test_inspection_mount_has_relational_role(self):
        result=self.run_policy(row=self.row(node_name='CCD安装板'),context=self.context('inspection'),tree=True,leaf=True)
        self.assertEqual(result['function_role'],'inspection_mount')
        self.assertFalse(result['allow_geometry_drop'])

    def test_root_equipment_context_is_excluded(self):
        nodes=[dict(name='检测线',_parent_index=None,children=[1,2]),
               dict(name='part',_parent_index=0,mesh=0),dict(name='part2',_parent_index=0,mesh=1)]
        rows=[self.row(),self.row(node_index=2,mesh_index=1)]
        config=dict(CONFIG,max_context_fraction=.55)
        result=build_contexts(nodes,rows,config)
        # With two nodes a root is the whole equipment, regardless of rounding the size limit.
        self.assertFalse(result['1']['context_available'])

    def test_geometry_guard_detects_component_loss_even_with_small_error(self):
        old=dict(relative_surface_error=.05,max_axis_shrink=0.,missing_significant_components=0)
        new=dict(old,relative_surface_error=.01,missing_significant_components=1)
        self.assertIn('surface_component_loss_regression',regression(new,old,CONFIG))

    def test_geometry_guard_changes_retention_without_changing_object_identity(self):
        row=self.run_policy()
        result=constrain_schedule(row,.80,['relative_surface_error_regression'])
        self.assertEqual(row['resolved_object_class'],result['resolved_object_class'])
        self.assertGreaterEqual(result['target_ratio_near'],result['target_ratio_mid'])
        self.assertGreaterEqual(result['target_ratio_mid'],result['target_ratio_far'])
        self.assertFalse(result['allow_cull'])

    def test_named_planar_reference_has_separate_role_but_is_not_deleted(self):
        row=self.row(node_name='地面-2',obb_size_long=10.,obb_size_mid=8.,obb_size_short=.001)
        context=self.context();context['scene_size_ratio']=.5
        result=self.run_policy(row=row,context=context,leaf=True)
        self.assertTrue(result['scene_reference_geometry_supported'])
        self.assertTrue(result['lod_priority'].startswith('P4_'))
        self.assertFalse(result['allow_geometry_drop'])


if __name__=='__main__':unittest.main()
