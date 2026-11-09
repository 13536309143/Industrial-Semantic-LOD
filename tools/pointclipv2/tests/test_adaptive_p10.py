import ctypes
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from adaptive_p10_policy import reprofile, infer, PROFILES
from run_adaptive_p10 import bounded_function, bounded_metrics
from probe_assembly_geometry import load_probe, prepare_geometry
import test_assembly_context as fixtures
from fuse_open_semantic_v2 import builder


class AdaptivePolicyTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.AssemblyContextTests()
        self.row=self.fixture.row();self.context=self.fixture.context()
        self.desc=dict(self.fixture.descriptor(),surface_area_normalized=1.)

    def profile(self,role='unknown',**kw):
        return reprofile(dict(function_role=role,confidence=.9),kw.get('row',self.row),
                         kw.get('desc',self.desc),kw.get('context',self.context),kw.get('repeats',1))

    def test_all_ten_profiles_have_evidence_based_reachable_entries(self):
        cases=[self.profile('standard_fastener',context=dict(self.context,scene_size_ratio=.001),repeats=8),
            self.profile('standard_fastener'),self.profile('structural_frame'),self.profile('mounting_support'),
            self.profile('unknown',desc=dict(self.desc,sharp_length_normalized=20)),
            self.profile('unknown',row=dict(self.row,obb_size_mid=.1)),self.profile('fluid_connection'),
            self.profile('actuation'),self.profile('precision_guidance'),
            self.profile('process_tool',row=dict(self.row,obb_size_mid=.1))]
        self.assertEqual([int(r['lod_priority'].split('_')[0][1:])for r in cases],list(range(1,11)))

    def test_unknown_and_low_confidence_are_never_removal_evidence(self):
        for role in ('unknown','standard_fastener','process_tool'):
            r=self.profile(role)
            self.assertFalse(r['allow_cull']);self.assertFalse(r['allow_geometry_drop'])
            self.assertEqual(r['visibility_status'],'unmeasured')

    def test_thin_cross_section_floor_and_critical_gate(self):
        thin=dict(self.row,obb_size_short=.005)
        self.assertTrue(self.profile('mounting_support',row=thin)['lod_priority'].startswith('P6_'))
        self.assertTrue(self.profile('process_tool',row=thin)['lod_priority'].startswith('P10_'))
        self.assertTrue(self.profile('process_tool')['lod_priority'].startswith('P9_'))

    def test_invalid_descriptor_cannot_downrank_unknown_geometry(self):
        r=self.profile('unknown',desc=dict(self.desc,geometry_valid=False))
        self.assertTrue(r['lod_priority'].startswith('P5_'))

    def test_reference_name_needs_independent_geometry_support(self):
        r=self.profile('scene_reference_candidate')
        self.assertTrue(r['lod_priority'].startswith('P5_'))

    def test_disabled_branches_and_leaf_name_remain_independent(self):
        thresholds=builder.make_thresholds([self.row])
        def output(mode,row=None,opened=None):
            return infer(row or self.row,opened or {},fixtures.TAXONOMY,thresholds,self.desc,self.context,
                         fixtures.CONFIG,mode,'off',False,8)
        a=output('CG')
        b=output('CG',row=dict(self.row,predicted_class='springs',confidence=.001,second_confidence=.0009))
        for field in ('lod_priority','resolved_object_class','profile_reason','target_ratio_far'):
            self.assertEqual(a[field],b[field])
        for mode in ('N','C','NG','CG','NC','NCG','G'):
            a=output(mode);b=output(mode,row=dict(self.row,node_name='CCD折弯刀工件'))
            for field in ('lod_priority','function_role','resolved_object_class','target_ratio_far'):
                self.assertEqual(a[field],b[field])

    def test_ratio_schedules_and_error_limits_ordered(self):
        old=0.
        for p,(_,near,mid,far,weight,lock,error)in PROFILES.items():
            self.assertTrue(0<far<=mid<=near<=1)
            self.assertGreaterEqual(far,old);old=far
            self.assertTrue(0<lock<.35 and 0<error<.1)


class NativeGuardsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[3]/'_bin/Release/semantic_geometry_probe.dll'
        if not path.exists():raise unittest.SkipTest('Native probe not built')
        cls.dll,_=load_probe(path);cls.func=bounded_function(cls.dll)

    def test_separated_tetrahedra_survive_extreme_request(self):
        v=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],
                    [3,0,0],[3.03,0,0],[3,.03,0],[3,0,.03]],dtype=float)
        f=np.array([[0,1,2],[0,3,1],[1,3,2],[0,2,3],[4,5,6],[4,7,5],[5,7,6],[4,6,7]],dtype=np.uint32)
        g=prepare_geometry(v,f)
        m=bounded_metrics(self.func,g,.03,dict(simplify_error_relative=.02,max_axis_shrink=.025))
        self.assertEqual(m['missing_significant_components'],0)
        self.assertLessEqual(m['max_axis_shrink'],.025)
        self.assertGreater(m['actual_faces'],0)

    def test_real_hierarchy_build_with_transient_component_ids(self):
        # Disconnected triangulated cubes exercise union-find and the clustered build.
        v=np.array([[x,y,z]for x in (0.,1.)for y in (0.,1.)for z in (0.,1.)],dtype=np.float32)
        f=np.array([[0,2,3],[0,3,1],[4,5,7],[4,7,6],[0,1,5],[0,5,4],
                    [2,6,7],[2,7,3],[0,4,6],[0,6,2],[1,3,7],[1,7,5]],dtype=np.uint32)
        vertices=np.ascontiguousarray(np.concatenate([v+[i*2,0,0]for i in range(80)]),dtype=np.float32)
        faces=np.ascontiguousarray(np.concatenate([f+i*8 for i in range(80)]).ravel(),dtype=np.uint32)
        func=self.dll.semantic_probe_hierarchy
        u32=np.ctypeslib.ndpointer(dtype=np.uint32,flags='C_CONTIGUOUS');f32=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
        u64=np.ctypeslib.ndpointer(dtype=np.uint64,flags='C_CONTIGUOUS')
        func.argtypes=[u32,ctypes.c_size_t,f32,ctypes.c_size_t,*([ctypes.c_float]*5),u64];func.restype=ctypes.c_int
        counters=np.zeros(5,dtype=np.uint64)
        self.assertEqual(func(faces,len(faces),vertices,len(vertices),.58,.22,.035,.014,.06,counters),1)
        self.assertGreater(counters[0],0);self.assertGreaterEqual(counters[2],len(faces))

    def test_bounded_finest_cleanup_removes_planar_redundancy_without_mutating_input(self):
        size=16
        v=np.ascontiguousarray([[x,y,0.]for x in range(size)for y in range(size)],dtype=np.float32)
        faces=[]
        for x in range(size-1):
            for y in range(size-1):
                a=x*size+y;faces.extend([[a,a+1,a+size],[a+1,a+size+1,a+size]])
        f=np.ascontiguousarray(faces,dtype=np.uint32).ravel();original=f.copy()
        func=self.dll.semantic_probe_base_hierarchy
        u32=np.ctypeslib.ndpointer(dtype=np.uint32,flags='C_CONTIGUOUS');f32=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
        u64=np.ctypeslib.ndpointer(dtype=np.uint64,flags='C_CONTIGUOUS')
        func.argtypes=[u32,ctypes.c_size_t,f32,ctypes.c_size_t,ctypes.c_float,ctypes.c_float,u64];func.restype=ctypes.c_int
        counters=np.zeros(6,dtype=np.uint64)
        self.assertEqual(func(f,len(f),v,len(v),.5,.0002,counters),1)
        self.assertGreater(counters[5],0);self.assertLess(counters[5],len(f))
        np.testing.assert_array_equal(f,original)


if __name__=='__main__':unittest.main()
