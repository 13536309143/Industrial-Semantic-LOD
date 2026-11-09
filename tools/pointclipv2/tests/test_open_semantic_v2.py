import json
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from open_semantic_geometry import canonical_points, proposals
from open_semantic_evidence import infer_evidence, label_distribution, metadata_evidence
from fuse_open_semantic_v2 import fuse, builder
from combine_open_renderings import combine
from open_semantic_shading import sample_oriented_surface,shaded_views
from compare_open_semantic_upgrade import old_known

DIRECTORY=Path(__file__).resolve().parents[1]
TAXONOMY=json.loads((DIRECTORY/'industrial_open_vocab_v2.json').read_text(encoding='utf-8'))
CONFIG=json.loads((DIRECTORY/'open_semantic_v2_config.json').read_text(encoding='utf-8'))


class OpenSemanticTests(unittest.TestCase):
    def test_coarse_noncontradiction_retains_strong_object_group(self):
        depth=dict(status='accepted',family='fastening',object_id='cap_nut',class_group='known:nuts',
            class_group_known_class='nuts',class_group_margin=.02,class_group_agreement=.8,
            roles={'repeated_fastener':1},reliability=.9,family_margin=.03)
        surface=dict(status='family_only',family='fastening',object_id='',class_group='',roles={'repeated_fastener':1},reliability=.8)
        result=combine(depth,surface)
        self.assertEqual(result['status'],'class_only')
        self.assertEqual(result['class_group_known_class'],'nuts')
        self.assertEqual(result['object_id'],'')
    def test_nut_group_consensus_does_not_require_identical_nut_style(self):
        depth=dict(status='accepted',family='fastening',object_id='cap_nut',class_group='known:nuts',
                   class_group_known_class='nuts',roles={'repeated_fastener':1},reliability=.9,family_margin=.03)
        surface=dict(depth,object_id='hex_nut')
        result=combine(depth,surface)
        self.assertEqual(result['status'],'class_only')
        self.assertEqual(result['object_id'],'')
        self.assertEqual(result['class_group_known_class'],'nuts')

    def test_object_group_is_fused_and_can_flag_closed_class_conflict(self):
        row=self.row();opened=self.opened()
        opened.update(open_status='class_only',open_object_id='',open_class_group='known:nuts',open_group_known_class='nuts')
        result=fuse(row,opened,TAXONOMY,builder.make_thresholds([row]),'C')
        self.assertEqual(result['resolved_object_class'],'known:nuts')
        self.assertEqual(result['resolved_source'],'open_class_group')
        joint=fuse(row,opened,TAXONOMY,builder.make_thresholds([row]),'NC')
        self.assertTrue(joint['class_conflict'])

    def test_comparison_uses_consistent_legacy_and_new_family_mapping(self):
        self.assertEqual(old_known('seal_gasket_interface'),next(label['known_class']for label in TAXONOMY['labels']if label['id']=='seal_ring'))
        self.assertEqual(old_known('spring_washer'),'washers_rings_spacers')
        self.assertEqual(old_known('precision_locating_pin'),'pins_rivets_keys')

    def test_product_context_does_not_keep_false_closed_identity_as_resolved(self):
        row=self.row()
        hint=metadata_evidence('产品01-1',['设备','折弯机构'],TAXONOMY)
        result=fuse(row,self.opened('abstain'),TAXONOMY,builder.make_thresholds([row]),'NCG',hint)
        self.assertEqual(result['resolved_object_class'],'workpiece:unidentified')
        self.assertEqual(result['resolved_status'],'workpiece_candidate')
        self.assertTrue(result['needs_review'])

    def test_render_conflict_abstains_and_single_stream_cannot_claim_fine_identity(self):
        depth=dict(status='accepted',family='fastening',object_id='cap_nut',roles={'repeated_fastener':1},reliability=.9,family_margin=.03)
        surface=dict(status='accepted',family='guidance',object_id='rolling_bearing',roles={'motion_precision':1},reliability=.9,family_margin=.03)
        conflict=combine(depth,surface)
        self.assertEqual(conflict['status'],'abstain')
        self.assertEqual(conflict['roles'],{})
        fallback=combine(depth,dict(status='abstain',family='shape',roles={},reliability=.1))
        self.assertEqual(fallback['status'],'family_only')
        self.assertEqual(fallback['object_id'],'')

    def test_shading_samples_unit_surface_normals_and_nearest_depth(self):
        import torch
        vertices=np.array([[0,0,0],[1,0,0],[0,1,0]],dtype=np.float32)
        points,normals=sample_oriented_surface(vertices,np.array([[0,1,2]]),128,7)
        np.testing.assert_allclose(np.linalg.norm(normals,axis=1),1,atol=1e-6)
        self.assertTrue(np.isfinite(points).all())
        rotation=torch.eye(3)[None]
        near=torch.tensor([[[0.,0.,-.5]]]);far=torch.tensor([[[0.,0.,.5]]])
        nnear=torch.tensor([[[0.,0.,-1.]]]);nfar=torch.tensor([[[1.,0.,0.]]])
        alone=shaded_views(near,nnear,rotation,32)
        together=shaded_views(torch.cat([near,far],1),torch.cat([nnear,nfar],1),rotation,32)
        torch.testing.assert_close(alone,together)

    def test_family_only_fastener_does_not_trigger_aggressive_culling(self):
        row=self.row();opened=self.opened();opened['open_status']='family_only';opened['open_object_id']=''
        result=fuse(row,opened,TAXONOMY,builder.make_thresholds([row]),'C')
        self.assertGreaterEqual(int(result['lod_priority'].split('_')[0][1:]),5)
        self.assertFalse(result['allow_cull'])

    def row(self):
        return dict(order='1',node_index='0',mesh_index='0',node_name='part',predicted_class='bearings_bushings_guides',
            confidence='.95',second_confidence='.01',face_count='100',surface_area='.5',bbox_diagonal='1',bbox_volume='.1',
            obb_size_long='1',obb_size_mid='.5',obb_size_short='.2',elongation='2',flatness='.2',compactness='.1',shape_hint='compact_block')

    def opened(self,status='accepted'):
        return dict(open_status=status,open_object_id='cap_nut',open_family='fastening',open_reliability='.8',
                    open_role_scores_json=json.dumps({'repeated_fastener':.85,'balanced_visible':.15}) if status!='abstain' else '{}')

    def test_unknown_and_smallness_are_not_object_classes(self):
        self.assertEqual(len({label['id'] for label in TAXONOMY['labels']}),len(TAXONOMY['labels']))
        for label in TAXONOMY['labels']:
            self.assertNotIn('micro_uncertain',label['role_priors'])
            self.assertNotIn('tiny',label['name'])

    def test_metadata_compounds_and_workpiece_context(self):
        self.assertEqual(metadata_evidence('螺母送料盘-1',[],TAXONOMY)['object_id'],'feeder_bowl')
        self.assertEqual(metadata_evidence('CCD安装板-1',[],TAXONOMY)['object_id'],'mounting_plate')
        hint=metadata_evidence('螺母-1',['设备','螺母送料组'],TAXONOMY)
        self.assertTrue(hint['workpiece_candidate'])
        self.assertEqual(hint['object_id'],'hex_nut')
        self.assertNotEqual(metadata_evidence('电缸-1',[],TAXONOMY)['object_id'],'pneumatic_cylinder')

    def test_pca_orientation_preserves_rotated_geometry(self):
        points=np.random.default_rng(12).normal(size=(100,3))*[3,1,.3]
        rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        np.testing.assert_allclose(canonical_points(points),canonical_points(points@rotation),atol=1e-6)
        self.assertEqual(len(proposals(points,20000)[0]),4)

    def test_background_and_local_only_cannot_be_accepted(self):
        text={'labels':np.ones((66,3)),'families':np.ones((12,3)),'negative':np.ones((3,3))}
        evidence=infer_evidence(np.zeros((10,3)),np.ones(10),np.zeros(10),text,TAXONOMY,CONFIG)
        self.assertEqual(evidence['status'],'abstain')
        self.assertEqual(evidence['roles'],{})
        local=infer_evidence(np.ones((10,3)),np.ones(10),np.full(10,2),text,TAXONOMY,CONFIG)
        self.assertEqual(local['status'],'abstain')

    def test_broad_family_text_does_not_erase_fine_visual_support(self):
        taxonomy={'families':[{'id':'a'},{'id':'b'}], 'labels':[{'family':'a'},{'family':'a'},{'family':'b'},{'family':'b'}]}
        _,family,_=label_distribution(np.array([.4,.38,.2,.18]),np.array([.1,.9]),taxonomy,CONFIG)
        self.assertEqual(int(np.argmax(family)),0)

    def test_cg_is_independent_of_closed_predictions_and_confidence(self):
        row=self.row();thresholds=builder.make_thresholds([row])
        before=fuse(row,self.opened(),TAXONOMY,thresholds,'CG')
        after=fuse(dict(row,predicted_class='springs',confidence='.01',second_confidence='.009'),self.opened(),TAXONOMY,thresholds,'CG')
        self.assertEqual(before,after)
        self.assertNotIn('source_pointnext_class',before)

    def test_unknown_never_becomes_low_importance_or_cullable(self):
        row=dict(self.row(),confidence='.1',second_confidence='.09')
        result=fuse(row,self.opened('abstain'),TAXONOMY,builder.make_thresholds([row]),'NC')
        self.assertGreaterEqual(int(result['lod_priority'].split('_')[0][1:]),5)
        self.assertFalse(result['allow_cull'])
        self.assertEqual(result['resolved_status'],'unknown')

    def test_class_conflicts_are_explicit_not_silent_overwrites(self):
        row=self.row();thresholds=builder.make_thresholds([row])
        result=fuse(row,self.opened(),TAXONOMY,thresholds,'NCG')
        self.assertTrue(result['class_conflict'])
        self.assertEqual(result['resolved_status'],'conflict')
        self.assertEqual(result['resolved_object_class'],'')
        hint=metadata_evidence('螺母-1',['设备','螺母送料组'],TAXONOMY)
        candidate=fuse(row,self.opened(),TAXONOMY,thresholds,'NCG',hint)
        self.assertEqual(candidate['resolved_source'],'cad_name_weak_hint')
        self.assertGreaterEqual(int(candidate['lod_priority'].split('_')[0][1:]),7)
        self.assertFalse(candidate['allow_cull'])

    def test_same_family_different_known_classes_still_conflict(self):
        row=dict(self.row(),predicted_class='washers_rings_spacers')
        result=fuse(row,self.opened(),TAXONOMY,builder.make_thresholds([row]),'NC')
        self.assertTrue(result['class_conflict'])
        self.assertEqual(result['resolved_status'],'conflict')


if __name__=='__main__':
    unittest.main()
