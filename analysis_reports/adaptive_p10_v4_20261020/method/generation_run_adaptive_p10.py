"""Rebuild ten preservation profiles and evaluate all six scenes, without retraining."""
import argparse
import ctypes
import json
import mmap
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from adaptive_p10_policy import infer, PROFILES, VERSION, yes
from assembly_context_features import build_contexts, geometry_descriptors
from factorial_ablation_policy import global_contexts, MAIN_MODES
from fuse_open_semantic_v2 import builder
from audit_named_scene import read_csv, write_csv
from experiment_assembly_context import key
from artifact_cache import file_hash
from run_pointclipv2_zeroshot_glb import read_glb_chunks, mesh_for_node
from probe_assembly_geometry import load_probe, prepare_geometry, simplify_metrics

ROOT=Path(__file__).resolve().parents[2]
DIRECTORY=Path(__file__).parent


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def table(path,rows):
    columns=list(dict.fromkeys(k for r in rows for k in r))
    write_csv(path,[{k:r.get(k,'')for k in columns}for r in rows])


def bounded_function(dll):
    func=dll.semantic_probe_simplify_v4
    u32=np.ctypeslib.ndpointer(dtype=np.uint32,flags='C_CONTIGUOUS')
    f32=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
    func.argtypes=[u32,u32,ctypes.c_size_t,f32,ctypes.c_size_t,ctypes.c_size_t,u32,
                   ctypes.c_float,ctypes.c_float,ctypes.POINTER(ctypes.c_float)]
    func.restype=ctypes.c_size_t
    return func


def bounded_metrics(func,g,ratio,policy):
    f,v=g['f'],g['v']
    output=np.empty(len(f),dtype=np.uint32);error=ctypes.c_float(0)
    components=np.ascontiguousarray(g['vertex_components'],dtype=np.uint32)
    count=int(func(output,f,len(f),v,len(v),max(3,int(len(f)*ratio)//3*3),components,
                   float(policy['simplify_error_relative']),float(policy['max_axis_shrink']),ctypes.byref(error)))
    if not count:raise ValueError('Bounded simplification returned empty geometry')
    indices=output[:count].reshape(-1,3);used=np.unique(indices)
    extent=np.ptp(v[used]@g['axes'],axis=0);valid=g['extent']>1e-7
    missing=np.setdiff1d(g['significant'],np.unique(g['vertex_components'][indices[:,0]]))
    return dict(actual_faces=count//3,relative_surface_error=float(error.value),
        max_axis_shrink=float(np.maximum(0,1-extent[valid]/g['extent'][valid]).max())if valid.any()else 0.,
        missing_significant_components=len(missing),
        missing_component_area_fraction=float(g['component_areas'][missing].sum()/max(g['component_areas'].sum(),1e-12)))


def optimize_schedule(row,geometry,func,cache,shape):
    """Select a measured feasible request. Never changes identity or fills a P histogram."""
    result=dict(row);far=float(row['target_ratio_far']);p=int(row['lod_priority'].split('_')[0][1:])
    ratios=sorted(set([max(.03,far*(.55 if p<=5 else .85)),far,min(.92,far*1.5),1.]))
    feasible=[]
    for ratio in ratios:
        k=(shape,round(ratio,8),float(row['simplify_error_relative']),float(row['max_axis_shrink']))
        if k not in cache:cache[k]=bounded_metrics(func,geometry,ratio,row)
        m=cache[k]
        if m['missing_significant_components']==0 and m['max_axis_shrink']<=float(row['max_axis_shrink'])+1e-6:
            feasible.append((m['actual_faces'],-ratio,ratio,m))
    if not feasible:raise ValueError('Source geometry did not pass preservation checks')
    _,_,ratio,metrics=min(feasible,key=lambda x:(x[0],x[1]))
    # A ratio-one witness means the source must be retained; a tiny error budget
    # permits only essentially exact collapses in the recursive renderer path.
    if ratio==1.:
        result['simplify_error_relative']=1e-8
        result['quality_fallback']='source_or_nearly_exact_only'
        ratio=.92
    else:result['quality_fallback']='none'
    result.update(target_ratio_far=ratio, target_ratio_mid=max(float(row['target_ratio_mid']),ratio),
        target_ratio_near=max(float(row['target_ratio_near']),min(.96,ratio+.04)),
        measured_proxy_faces=metrics['actual_faces'],measured_proxy_error=metrics['relative_surface_error'],
        measured_proxy_axis_shrink=metrics['max_axis_shrink'],measured_proxy_component_loss=metrics['missing_significant_components'],
        quality_search='minimum measured feasible faces over frozen ratio grid; single-mesh diagnostic',
        runtime_scope='recursive guards active; single-mesh selected ratio is not a final rendered fraction')
    return result,metrics


def summary(scene,variant,rows,metrics=None):
    count=Counter(int(r['lod_priority'].split('_')[0][1:])for r in rows)
    faces=Counter()
    for r in rows:faces[int(r['lod_priority'].split('_')[0][1:])]+=int(r['face_count'])
    total=sum(faces.values())
    out=dict(scene=scene,variant=variant,nodes=len(rows),source_faces=total,
        requested_far_fraction=sum(int(r['face_count'])*float(r['target_ratio_far'])for r in rows)/total,
        hard_lock_budget_mean=sum(float(r.get('feature_lock_budget',r.get('feature_lock_floor',0)))for r in rows)/len(rows))
    for p in range(1,11):out.update({f'P{p}_count':count[p],f'P{p}_node_fraction':count[p]/len(rows),f'P{p}_face_fraction':faces[p]/total})
    if metrics:
        out.update(actual_triangle_fraction=sum(r['actual_faces']for r in metrics)/total,
            mean_relative_surface_error=float(np.mean([r['relative_surface_error']for r in metrics])),
            face_weighted_error=sum(m['relative_surface_error']*int(r['face_count'])for m,r in zip(metrics,rows))/total,
            p95_relative_surface_error=float(np.quantile([r['relative_surface_error']for r in metrics],.95)),
            axis_shrink_over_10pct=sum(r['max_axis_shrink']>.1 for r in metrics),
            axis_shrink_over_2_5pct=sum(r['max_axis_shrink']>.025 for r in metrics),
            component_loss_nodes=sum(r['missing_significant_components']>0 for r in metrics),
            actual_faces=sum(r['actual_faces']for r in metrics))
    return out


def export(manifest,root,report,models):
    mapping={f'{mode}_H0':suffix for mode,suffix in zip(MAIN_MODES,
        ['1_pointnext','2_pointclip','3_pointnext_structure','4_pointclip_structure','5_pointnext_pointclip','6_pointnext_pointclip_structure'])}
    mapping.update(NCG_H1_M1_selected='7_pointnext_pointclip_structure_metadata',
                   NCG_H0='6_pointnext_pointclip_structure',NCG_H1='assembly_tree_only',G_H0='geometry_only')
    records=[]
    for scene in models:
        for variant,suffix in mapping.items():
            records.append((root/scene/(variant+'.csv'),ROOT/'lod_analysis_outputs'/f'{scene}_lod_constraints_{suffix}.csv'))
        records.extend([(root/scene/'NCG_H0.csv',ROOT/'lod_analysis_outputs'/f'{scene}_lod_constraints_fused.csv'),
                        (root/scene/'NCG_H1_M1_selected.csv',ROOT/'lod_analysis_outputs'/f'{scene}_lod_constraints_metadata.csv')])
    # Validate the entire batch before any deployment file is changed.
    for source,destination in records:
        rows=read_csv(source)
        if len({key(r)for r in rows})!=len(rows)or not rows:raise ValueError('Invalid policy')
        for r in rows:
            near,mid,far=[float(r['target_ratio_'+level])for level in ('near','mid','far')]
            if not 1>=near>=mid>=far>0 or yes(r['allow_geometry_drop']):raise ValueError('Unsafe policy')
    import datetime
    backup=ROOT/'.artifacts/migrations'/('adaptive_p10_v4_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    published=[]
    for source,destination in records:
        if destination.exists()and file_hash(destination)!=file_hash(source):
            backup.mkdir(parents=True,exist_ok=True);shutil.copy2(destination,backup/destination.name)
        temp=destination.with_suffix('.tmp');shutil.copyfile(source,temp);temp.replace(destination)
        published.append(dict(source=str(source),destination=str(destination),sha256=file_hash(destination)))
    save(ROOT/'lod_analysis_outputs/adaptive_p10_v4_deployment.json',dict(version=VERSION,exports=published,backup=str(backup),
        mode6='visual NCG, no leaf/ancestor names',mode7='NCGHM with measured schedule',cache_rebuild_required=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',default=list('abcdef'))
    parser.add_argument('--output-root',type=Path,default=ROOT/'lod_analysis_outputs/adaptive_p10_v4_20261020')
    parser.add_argument('--report-dir',type=Path,default=ROOT/'analysis_reports/adaptive_p10_v4_20261020')
    parser.add_argument('--no-export',action='store_true')
    args=parser.parse_args();args.report_dir.mkdir(parents=True,exist_ok=True)
    manifest_path=ROOT/'lod_analysis_outputs/assembly_context_v3_20261020/scene_sources.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    taxonomy=json.loads((DIRECTORY/'industrial_open_vocab_v2.json').read_text(encoding='utf-8'))
    config=json.loads((DIRECTORY/'assembly_context_config.json').read_text(encoding='utf-8'))
    dllpath=ROOT/'_bin/Release/semantic_geometry_probe.dll';dll,oldfunc=load_probe(dllpath);newfunc=bounded_function(dll)
    summaries=[];inputs=[manifest_path,DIRECTORY/'adaptive_p10_policy.py',Path(__file__),dllpath]
    all_metrics=[];status=dict(status='running',version=VERSION,models=args.models,completed=[])
    specs=[(f'{mode}_H{int(tree=="on")}',mode,tree,False)for mode in (*MAIN_MODES,'G')for tree in ('off','on')]
    specs.append(('NCG_H1_M1','NCG','on',True))
    for scene in args.models:
        source=manifest[scene];glb=Path(source['glb']);pnpath=Path(source['pointnext_csv']);cpath=Path(source['open_csv'])
        for path,expected in [(glb,source['glb_sha256']),(pnpath,source['pointnext_sha256']),(cpath,source['open_sha256'])]:
            if file_hash(path)!=expected:raise ValueError('Changed evidence: '+str(path))
        pn=read_csv(pnpath);opened={key(r):r for r in read_csv(cpath)}
        if len(opened)!=len(pn)or set(opened)!={key(r)for r in pn}:raise ValueError('Incomplete evidence')
        data,offset,_=read_glb_chunks(glb)
        if len(pn)!=sum('mesh'in n for n in data['nodes']):raise ValueError('Partial scene')
        descriptor_path=ROOT/'lod_analysis_outputs/assembly_context_v3_20261020'/scene/'geometry_descriptors.json'
        # Reuse only content-verified geometry; recomputation is local and deterministic.
        descriptors=geometry_descriptors(glb,pn,descriptor_path)
        contexts={'off':global_contexts(pn),'on':build_contexts(data['nodes'],pn,config)}
        repeats=Counter(r['mesh_index']for r in pn);thresholds=builder.make_thresholds(pn)
        folder=args.output_root/scene;folder.mkdir(parents=True,exist_ok=True)
        policies={}
        for variant,mode,tree,leaf in specs:
            rows=[infer(r,opened[key(r)],taxonomy,thresholds,descriptors[str(r['node_index'])],
                        contexts[tree][str(r['node_index'])],config,mode,tree,leaf,repeats[r['mesh_index']])for r in pn]
            write_csv(folder/(variant+'.csv'),rows);policies[variant]=rows
            summaries.append(summary(scene,variant,rows))
        print(f'{scene}: {len(pn)} nodes x {len(specs)} profiles generated',flush=True)
        oldpath=ROOT/'lod_analysis_outputs/assembly_context_v3_20261020'/scene/'final_guarded/v3_NCGHM.csv'
        old=read_csv(oldpath);oldindex={key(r):r for r in old}
        old=[oldindex[key(r)]for r in pn]
        before=[];initial=[];after=[];selected=[];cache={}
        with glb.open('rb')as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ)as mm:
            for i,row in enumerate(pn):
                node=data['nodes'][int(row['node_index'])]
                transform=node['_world_matrix'][:3,:3];gram=transform.T@transform;gram/=max(float(np.trace(gram)),1e-12)
                shape=(node['mesh'],tuple(np.round(gram.ravel(),7)))
                vertices,faces=mesh_for_node(data,mm,offset,node);geometry=prepare_geometry(vertices,faces)
                base=policies['NCG_H1_M1'][i]
                oldkey=('old',shape,float(old[i]['target_ratio_far']))
                if oldkey not in cache:cache[oldkey]=simplify_metrics(oldfunc,geometry,float(old[i]['target_ratio_far']))
                b=cache[oldkey]
                initialkey=(shape,round(float(base['target_ratio_far']),8),float(base['simplify_error_relative']),float(base['max_axis_shrink']))
                if initialkey not in cache:cache[initialkey]=bounded_metrics(newfunc,geometry,float(base['target_ratio_far']),base)
                first=cache[initialkey]
                tuned,m=optimize_schedule(base,geometry,newfunc,cache,shape)
                selected.append(tuned);before.append(b);initial.append(first);after.append(m)
                for variant,metric in [('v3_deployed',b),('v4_profile',first),('v4_selected',m)]:
                    all_metrics.append(dict(scene=scene,node_index=row['node_index'],mesh_index=row['mesh_index'],
                        variant=variant,source_faces=int(row['face_count']),**metric))
                if (i+1)%500==0:print(f'{scene}: measured schedules {i+1}/{len(pn)}',flush=True)
        write_csv(folder/'NCG_H1_M1_selected.csv',selected)
        summaries.extend([summary(scene,'v3_deployed',old,before),summary(scene,'v4_profile',policies['NCG_H1_M1'],initial),
                          summary(scene,'v4_selected',selected,after)])
        status['completed'].append(scene);save(args.report_dir/'run_status.json',status)
        table(args.report_dir/'all_profile_comparison.csv',summaries)
        table(args.report_dir/'single_mesh_geometry_nodes.csv',all_metrics)
        inputs.extend([glb,pnpath,cpath,descriptor_path,oldpath])
    policies=list(args.output_root.glob('*/*.csv'))
    inputs.extend([DIRECTORY/'assembly_context_policy.py',DIRECTORY/'assembly_context_features.py',DIRECTORY/'factorial_ablation_policy.py',
        DIRECTORY/'assembly_context_config.json',DIRECTORY/'industrial_open_vocab_v2.json',DIRECTORY/'fuse_open_semantic_v2.py',
        ROOT/'src/scene/scene_semantic_lod.cpp',ROOT/'src/meshlod/meshlod_simplify.h',ROOT/'src/meshlod/meshlod_build.h'])
    save(args.report_dir/'provenance.json',dict(inputs={str(p.resolve()):file_hash(p)for p in inputs},
        outputs={str(p.resolve()):file_hash(p)for p in policies},all_nodes=True,calibrated_accuracy=False))
    save(args.report_dir/'profile_definitions.json',{p:dict(zip(('name','near','mid','far','screen_weight','lock_budget','relative_QEM_limit'),v))for p,v in PROFILES.items()})
    if not args.no_export:export(manifest,args.output_root,args.report_dir,args.models)
    status.update(status='complete',exported=not args.no_export);save(args.report_dir/'run_status.json',status)
    print('Adaptive ten-profile policy evaluation complete',flush=True)


if __name__=='__main__':main()
