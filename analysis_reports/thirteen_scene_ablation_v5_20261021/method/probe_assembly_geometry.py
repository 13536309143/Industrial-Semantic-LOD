"""Full-node CPU simplification proxy; not the application's cluster LOD/FPS benchmark."""
import argparse
import ctypes
import json
import mmap
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from audit_named_scene import read_csv,write_csv
from run_pointclipv2_zeroshot_glb import read_glb_chunks,mesh_for_node
from artifact_cache import file_hash

ROOT=Path(__file__).resolve().parents[2]


def load_probe(path):
    dll=ctypes.CDLL(str(path.resolve()))
    func=dll.semantic_probe_simplify
    u32=np.ctypeslib.ndpointer(dtype=np.uint32,flags='C_CONTIGUOUS')
    f32=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
    func.argtypes=[u32,u32,ctypes.c_size_t,f32,ctypes.c_size_t,ctypes.c_size_t,ctypes.POINTER(ctypes.c_float)]
    func.restype=ctypes.c_size_t
    return dll,func


def prepare_geometry(vertices,faces):
    v=np.asarray(vertices,dtype=np.float64);f=np.asarray(faces,dtype=np.uint32)
    center=v.mean(0);v=v-center
    scale=max(float(2*np.sqrt(np.mean(np.sum(v*v,axis=-1)))),1e-12)
    v/=scale
    _,remap=np.unique(np.round(v,6),axis=0,return_inverse=True)
    welded=remap[f]
    edges=np.concatenate([welded[:,[0,1]],welded[:,[1,2]],welded[:,[2,0]]])
    graph=coo_matrix((np.ones(len(edges),dtype=np.int8),(edges[:,0],edges[:,1])),shape=(int(remap.max())+1,)*2).tocsr()
    _,components=connected_components(graph,directed=False)
    vertex_components=components[remap]
    cross=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
    areas=np.linalg.norm(cross,axis=-1)*.5
    component_areas=np.bincount(vertex_components[f[:,0]],weights=areas)
    significant=np.flatnonzero(component_areas>max(areas.sum()*.001,1e-12))
    _,axes=np.linalg.eigh(v.T@v/max(len(v),1))
    extent=np.ptp(v[np.unique(f)]@axes,axis=0)
    return dict(v=np.ascontiguousarray(v,dtype=np.float32),f=np.ascontiguousarray(f.ravel()),
        axes=axes,extent=extent,vertex_components=vertex_components,component_areas=component_areas,
        significant=significant,max_extent=float(np.ptp(v,axis=0).max()))


def simplify_metrics(func,geometry,ratio):
    g=geometry;f=g['f'];v=g['v']
    target=max(3,int(len(f)*ratio)//3*3)
    out=np.empty(len(f),dtype=np.uint32);error=ctypes.c_float(0)
    n=int(func(out,f,len(f),v,len(v),target,ctypes.byref(error)))
    if not n:
        return dict(actual_faces=0,relative_surface_error=1.,max_axis_shrink=1.,missing_component_area_fraction=1.,missing_significant_components=len(g['significant']))
    indices=out[:n].reshape(-1,3)
    used=np.unique(indices)
    new_extent=np.ptp(v[used]@g['axes'],axis=0)
    valid=g['extent']>1e-7
    shrink=np.maximum(0,1-new_extent[valid]/g['extent'][valid])
    present=np.unique(g['vertex_components'][indices[:,0]])
    missing=np.setdiff1d(g['significant'],present)
    lost_area=float(g['component_areas'][missing].sum()/max(g['component_areas'].sum(),1e-12))
    return dict(actual_faces=n//3,relative_surface_error=float(error.value*g['max_extent']),
        max_axis_shrink=float(shrink.max())if len(shrink)else 0.,missing_component_area_fraction=lost_area,
        missing_significant_components=len(missing))


def matching_scale(baseline,candidate):
    target=sum(int(a['face_count'])*float(a['target_ratio_far'])for a in baseline)
    def demand(scale):return sum(int(r['face_count'])*np.clip(float(r['target_ratio_far'])*scale,.03,.65)for r in candidate)
    lo,hi=.05,10.
    for _ in range(48):
        mid=(lo+hi)/2
        if demand(mid)<target:lo=mid
        else:hi=mid
    return (lo+hi)/2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--report-dir',type=Path,required=True);p.add_argument('--round',required=True)
    p.add_argument('--models',nargs='+',default=list('abcdef'))
    p.add_argument('--probe-dll',type=Path,default=ROOT/'_bin/Release/semantic_geometry_probe.dll')
    args=p.parse_args()
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    dll,func=load_probe(args.probe_dll)
    all_summaries=[];inputs=[args.probe_dll,Path(__file__),Path(__file__).parent/'native/semantic_geometry_probe.cpp']
    for name in args.models:
        source=manifest[name];folder=args.output_root/name/args.round
        variants=['v2_NCG','v3_NCG','v3_NCGH','v3_NCGHM']
        policies={v:read_csv(folder/(v+'.csv'))for v in variants}
        scale=matching_scale(policies['v2_NCG'],policies['v3_NCGHM'])
        specs=[(v,policies[v],1.)for v in variants]+[('v3_NCGHM_same_requested_budget',policies['v3_NCGHM'],scale)]
        output=folder/'geometry_proxy.csv';stamp=folder/'geometry_proxy.provenance.json'
        glb=Path(source['glb']);current_inputs={str(q.resolve()):file_hash(q)for q in [glb,*inputs,*[folder/(v+'.csv')for v in variants]]}
        cached=False
        if stamp.exists()and output.exists():
            old=json.loads(stamp.read_text(encoding='utf-8'))
            cached=old.get('inputs')==current_inputs and old.get('output_sha256')==file_hash(output)
        if cached:
            result=read_csv(output)
        else:
            data,offset,_=read_glb_chunks(glb);result=[];metrics_cache={}
            with glb.open('rb')as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ)as mm:
                for j,base in enumerate(policies['v2_NCG']):
                    i=int(base['node_index']);node=data['nodes'][i]
                    transform=node['_world_matrix'][:3,:3];gram=transform.T@transform;gram/=max(np.trace(gram),1e-12)
                    shape_key=(node['mesh'],tuple(np.round(gram.ravel(),7)))
                    geometry=None
                    for variant,rows,multiplier in specs:
                        r=rows[j]
                        if r['node_index']!=base['node_index']:raise ValueError('Policy order mismatch')
                        ratio=float(np.clip(float(r['target_ratio_far'])*multiplier,.03,.65))
                        cache_key=(shape_key,round(ratio,8))
                        if cache_key not in metrics_cache:
                            if geometry is None:
                                v,f=mesh_for_node(data,mm,offset,node)
                                geometry=prepare_geometry(v,f)
                            metrics_cache[cache_key]=simplify_metrics(func,geometry,ratio)
                        metrics=metrics_cache[cache_key]
                        risk=json.loads(r.get('geometry_risk_json')or '{}')
                        result.append(dict(scene=name,node_index=i,mesh_index=node['mesh'],variant=variant,
                            source_faces=int(base['face_count']),requested_ratio=ratio,**metrics,
                            structural_risk=bool(risk.get('priority_floor',0)>=6),
                            metric_scope='single-mesh QEM proxy; no cluster hierarchy, feature locks, visibility or FPS'))
                    if (j+1)%500==0:print(f'Geometry proxy {name}: {j+1}/{len(policies["v2_NCG"])}',flush=True)
            write_csv(output,result)
            stamp.write_text(json.dumps(dict(inputs=current_inputs,output_sha256=file_hash(output),
                requested_budget_scale=scale,all_nodes=True,feature_locks_evaluated=False),ensure_ascii=False,indent=2),encoding='utf-8')
        by_variant=defaultdict(list)
        for row in result:by_variant[row['variant']].append(row)
        for variant,rows in by_variant.items():
            weight=np.array([float(r['source_faces'])for r in rows]);errors=np.array([float(r['relative_surface_error'])for r in rows])
            all_summaries.append(dict(scene=name,variant=variant,nodes=len(rows),source_faces=int(weight.sum()),
                achieved_triangle_ratio=sum(int(r['actual_faces'])for r in rows)/max(weight.sum(),1),
                requested_triangle_ratio=sum(float(r['requested_ratio'])*w for r,w in zip(rows,weight))/max(weight.sum(),1),
                mean_relative_surface_error=float(errors.mean()),face_weighted_relative_error=float(np.average(errors,weights=weight)),
                p95_relative_surface_error=float(np.quantile(errors,.95)),
                axis_shrink_over_10pct=sum(float(r['max_axis_shrink'])>.1 for r in rows),
                component_loss_nodes=sum(int(r['missing_significant_components'])>0 for r in rows),
                mean_missing_component_area_fraction=float(np.mean([float(r['missing_component_area_fraction'])for r in rows])),
                actual_renderer_lod=False,feature_locks_evaluated=False))
        print(f'{name}: full-node geometry proxy complete',flush=True)
    args.report_dir.mkdir(parents=True,exist_ok=True)
    write_csv(args.report_dir/'geometry_proxy_summary.csv',all_summaries)
    (args.report_dir/'geometry_proxy_summary.json').write_text(json.dumps(all_summaries,indent=2),encoding='utf-8')


if __name__=='__main__':main()
