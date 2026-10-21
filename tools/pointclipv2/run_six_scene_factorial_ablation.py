"""Full paired six-mode x hierarchy ablation, with frozen neural inference and no deployment."""
import argparse
import csv
import json
import math
import mmap
import shutil
import time
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np

from audit_named_scene import read_csv,write_csv,name_hints
from summarize_named_scenes import expected_known_classes
from assembly_context_features import build_contexts,geometry_descriptors,number
from factorial_ablation_policy import classify,global_contexts,MAIN_MODES,TREE_LEVELS
from fuse_open_semantic_v2 import builder
from experiment_assembly_context import key,identity_class
from probe_assembly_geometry import load_probe,prepare_geometry,simplify_metrics
from run_pointclipv2_zeroshot_glb import read_glb_chunks,mesh_for_node
from artifact_cache import file_hash

ROOT=Path(__file__).resolve().parents[2]
DIRECTORY=Path(__file__).parent
TITLES={'a':'无可用名称/装配树','b':'K31自动铆压机','c':'汽车链接器植入-折弯设备',
        'd':'手机喇叭检测线','e':'透气阀产线','f':'圆形电容穿心焊接机'}
OBJECT_STATUSES={'closed_candidate','open_candidate','open_class_candidate'}


def yes(value):return str(value).lower()=='true'
def priority(row):return int(str(row['lod_priority']).split('_')[0][1:])
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def fraction(a,b):return a/b if b else None
def pct(a):return '—'if a is None else f'{a:.2%}'


def source_diagnostic(scene,row,opened,taxonomy):
    confidence=number(row,'confidence');margin=confidence-number(row,'second_confidence')
    status=opened.get('open_status','abstain')
    diag=json.loads(opened.get('open_diagnostics_json')or '{}')
    candidate=diag.get('candidate_id')or opened.get('pointclip_top1_id','')
    leaf,hints=name_hints(row.get('node_name',''))
    expected=expected_known_classes(dict(leaf_name=leaf,name_candidate_role=hints[0][0]if hints else ''))
    return dict(scene=scene,node_index=str(row['node_index']),mesh_index=str(row['mesh_index']),
        node_name=row.get('node_name',''),face_count=int(row['face_count']),surface_area=number(row,'surface_area'),
        N_candidate=row.get('predicted_class',''),N_confidence=confidence,N_margin=margin,
        N_very_low=confidence<.4,N_medium=.4<=confidence<.7,N_small_margin=margin<.08,
        N_not_reliable=not(confidence>=.7 and margin>=.08),
        C_candidate=candidate,C_family=opened.get('open_family',''),C_status=status,
        C_quality=number(opened,'open_reliability'),C_abstain=status=='abstain',C_coarse_only=status=='family_only',
        C_not_object=status not in {'accepted','class_only'},C_reason=opened.get('open_reason',''),
        name_reference_classes='|'.join(sorted(expected)),name_reference_role=hints[0][0]if hints else '',
        name_reference_group=__import__('re').sub(r'(?:[-_.]\d+)+$','',leaf))


def summarize(scene,variant,mode,tree,leaf,rows,diagnostics,taxonomy,kind):
    node_hist=Counter(priority(r)for r in rows);face_hist=Counter();area_hist=Counter()
    total_faces=sum(int(r['face_count'])for r in rows);total_area=sum(number(r,'surface_area')for r in rows)
    for r in rows:face_hist[priority(r)]+=int(r['face_count']);area_hist[priority(r)]+=number(r,'surface_area')
    dindex={r['node_index']:r for r in diagnostics}
    named_total=matched=covered=named_critical_low=0;groups=defaultdict(list)
    for r in rows:
        d=dindex[str(r['node_index'])]
        expected=set(filter(None,d['name_reference_classes'].split('|')))
        if expected:
            named_total+=1;c=identity_class(r,taxonomy);covered+=bool(c);matched+=c in expected
            groups[(d['name_reference_classes'],d['name_reference_group'])].append(c in expected)
        named_critical_low+=d['name_reference_role']in {'motor_or_reducer','bearing_or_linear_guide','transmission','spring','sensing_or_inspection','tooling_or_process'}and priority(r)<=3
    result=dict(scene=scene,variant=variant,mode=mode,tree=tree,leaf_names=leaf,kind=kind,nodes=len(rows),
        source_faces=total_faces,source_surface_area=total_area,
        object_unresolved_count=sum(r['resolved_status']not in (OBJECT_STATUSES|({'metadata_candidate'}if leaf else set()))for r in rows),
        unknown_function_count=sum(r['function_role']=='unknown'for r in rows),
        identity_conflict_count=sum(yes(r.get('class_conflict',False))for r in rows),
        shape_conflict_count=sum(yes(r.get('type_shape_conflict',False))for r in rows),
        review_count=sum(yes(r.get('needs_review',False))for r in rows),
        tree_relation_count=sum(yes(r.get('assembly_context_applied',False))for r in rows),
        cullable_count=sum(yes(r['allow_cull'])for r in rows),
        N_not_reliable_count=sum(d['N_not_reliable']for d in diagnostics)if 'N'in mode else None,
        C_not_object_count=sum(d['C_not_object']for d in diagnostics)if 'C'in mode else None,
        C_abstain_count=sum(d['C_abstain']for d in diagnostics)if 'C'in mode else None,
        C_coarse_count=sum(d['C_coarse_only']for d in diagnostics)if 'C'in mode else None,
        name_reference_count=named_total,name_object_coverage_count=covered,name_consistent_count=matched,
        name_groups=len(groups),name_group_consistency_sum=sum(float(np.mean(v))for v in groups.values()),
        named_critical_low_count=named_critical_low,
        requested_far_face_sum=sum(int(r['face_count'])*float(r['target_ratio_far'])for r in rows),
        requested_near_face_sum=sum(int(r['face_count'])*float(r['target_ratio_near'])for r in rows),
        priority_sum=sum(priority(r)for r in rows))
    for p in range(1,11):result.update({f'P{p}_count':node_hist[p],f'P{p}_faces':face_hist[p],f'P{p}_area':area_hist[p]})
    return derived(result)


def derived(r):
    r=dict(r)
    for p in range(1,11):
        r[f'P{p}_node_fraction']=fraction(r[f'P{p}_count'],r['nodes'])
        r[f'P{p}_face_fraction']=fraction(r[f'P{p}_faces'],r['source_faces'])
        r[f'P{p}_area_fraction']=fraction(r[f'P{p}_area'],r['source_surface_area'])
    for key in ('object_unresolved','unknown_function','identity_conflict','shape_conflict','review','tree_relation','cullable',
                'N_not_reliable','C_not_object','C_abstain','C_coarse'):
        value=r[key+'_count'];r[key+'_fraction']=fraction(value,r['nodes'])if value is not None else None
    r['name_consistency_all_reference']=fraction(r['name_consistent_count'],r['name_reference_count'])
    r['name_object_coverage_fraction']=fraction(r['name_object_coverage_count'],r['name_reference_count'])
    r['name_consistency_covered']=fraction(r['name_consistent_count'],r['name_object_coverage_count'])
    r['name_group_macro_consistency']=fraction(r['name_group_consistency_sum'],r['name_groups'])
    r['requested_far_triangle_fraction']=fraction(r['requested_far_face_sum'],r['source_faces'])
    r['requested_near_triangle_fraction']=fraction(r['requested_near_face_sum'],r['source_faces'])
    r['mean_priority']=fraction(r['priority_sum'],r['nodes'])
    r['low_P1_P3_fraction']=sum(r[f'P{p}_node_fraction']for p in (1,2,3))
    r['high_P8_P10_fraction']=sum(r[f'P{p}_node_fraction']for p in (8,9,10))
    return r


def pooled(summaries):
    groups=defaultdict(list)
    for r in summaries:groups[r['variant']].append(r)
    result=[]
    for variant,rows in groups.items():
        out={k:rows[0][k]for k in ('variant','mode','tree','leaf_names','kind')};out['scene']='ALL'
        additive=[k for k in rows[0]if k.endswith(('_count','_faces','_area','_sum','_groups'))or k in {'nodes','source_surface_area'}]
        for k in additive:
            out[k]=sum(r[k]for r in rows)if all(r[k]is not None for r in rows)else None
        result.append(derived(out))
    return result


def pairing(scene,mode,off,on,reference='off',comparison='on',kind='main'):
    if [key(r)for r in off]!=[key(r)for r in on]:raise ValueError('Pair key mismatch')
    matrix=Counter();faces=Counter();changes=[]
    for a,b in zip(off,on):
        pa,pb=priority(a),priority(b);matrix[(pa,pb)]+=1;faces[(pa,pb)]+=int(a['face_count'])
        action=any(str(a[k])!=str(b[k])for k in ('lod_priority','target_ratio_near','target_ratio_mid','target_ratio_far','allow_cull','feature_lock_floor','feature_soft_floor'))
        if action or a['function_role']!=b['function_role']or a['resolved_object_class']!=b['resolved_object_class']:
            changes.append(dict(scene=scene,mode=mode,reference=reference,comparison=comparison,node_index=a['node_index'],
                node_name=a['node_name'],before_priority=pa,after_priority=pb,delta=pb-pa,
                before_function=a['function_role'],after_function=b['function_role'],
                before_identity=a['resolved_object_class'],after_identity=b['resolved_object_class'],
                before_far=float(a['target_ratio_far']),after_far=float(b['target_ratio_far']),
                ancestor_context=b.get('assembly_context_json',''),action_changed=action))
    summary=dict(scene=scene,mode=mode,reference=reference,comparison=comparison,kind=kind,nodes=len(off),
        priority_changed=sum(v for (a,b),v in matrix.items()if a!=b),priority_raised=sum(v for (a,b),v in matrix.items()if b>a),
        priority_lowered=sum(v for (a,b),v in matrix.items()if b<a),
        function_changed=sum(a['function_role']!=b['function_role']for a,b in zip(off,on)),
        identity_changed=sum(a['resolved_object_class']!=b['resolved_object_class']for a,b in zip(off,on)),
        action_changed=sum(r['action_changed']for r in changes),
        mean_priority_delta=sum((b-a)*v for (a,b),v in matrix.items())/len(off))
    for p in range(1,11):summary[f'P{p}_delta_pp']=100*(sum(v for (a,b),v in matrix.items()if b==p)-sum(v for (a,b),v in matrix.items()if a==p))/len(off)
    transitions=[dict(scene=scene,mode=mode,reference=reference,comparison=comparison,from_P=a,to_P=b,count=matrix[(a,b)],
        node_fraction=matrix[(a,b)]/len(off),source_faces=faces[(a,b)])for a in range(1,11)for b in range(1,11)]
    return summary,transitions,changes


def uncertainty_tables(diagnostics,taxonomy):
    rows=[];profiles=[]
    for scene in [*sorted({r['scene']for r in diagnostics}),'ALL']:
        data=[r for r in diagnostics if scene=='ALL'or r['scene']==scene]
        n=len(data);faces=sum(r['face_count']for r in data)
        profiles.append(dict(scene=scene,nodes=n,N_not_reliable=sum(r['N_not_reliable']for r in data),
            N_not_reliable_fraction=sum(r['N_not_reliable']for r in data)/n,
            N_very_low=sum(r['N_very_low']for r in data),N_small_margin=sum(r['N_small_margin']for r in data),
            N_mean_confidence=float(np.mean([r['N_confidence']for r in data])),
            N_low_face_fraction=sum(r['face_count']for r in data if r['N_not_reliable'])/faces,
            C_object_usable=sum(not r['C_not_object']for r in data),C_coarse=sum(r['C_coarse_only']for r in data),
            C_abstain=sum(r['C_abstain']for r in data),C_object_fraction=sum(not r['C_not_object']for r in data)/n,
            C_abstain_fraction=sum(r['C_abstain']for r in data)/n,
            C_not_object_face_fraction=sum(r['face_count']for r in data if r['C_not_object'])/faces))
        for branch,candidate,flag in [('N','N_candidate','N_not_reliable'),('C','C_candidate','C_not_object')]:
            total_low=sum(r[flag]for r in data)
            labels=sorted({r[candidate]for r in data})
            for label in labels:
                members=[r for r in data if r[candidate]==label];low=[r for r in members if r[flag]]
                rows.append(dict(scene=scene,branch=branch,candidate_type=label,total_candidates=len(members),
                    low_confidence_candidates=len(low),within_type_low_fraction=len(low)/len(members),
                    share_of_low_pool=fraction(len(low),total_low),low_faces=sum(r['face_count']for r in low),
                    mean_source_score=float(np.mean([r['N_confidence'if branch=='N'else 'C_quality']for r in members])),
                    score_scope='supervised probability threshold/margin'if branch=='N'else 'uncalibrated quality; low means no object-level output'))
    return profiles,rows


def probe_scenes(manifest,root,models,status_callback):
    dll,func=load_probe(ROOT/'_bin/Release/semantic_geometry_probe.dll')
    summaries=[]
    for scene in models:
        status_callback(scene,'geometry_probe')
        folder=root/scene
        names=[f'{mode}_H{h}'for mode in MAIN_MODES for h in (0,1)]
        policies={v:read_csv(folder/(v+'.csv'))for v in names}
        glb=Path(manifest[scene]['glb']);data,offset,_=read_glb_chunks(glb)
        result=[];metric_cache={}
        with glb.open('rb')as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ)as mm:
            for j,source in enumerate(policies[names[0]]):
                node=data['nodes'][int(source['node_index'])]
                gram=node['_world_matrix'][:3,:3].T@node['_world_matrix'][:3,:3];gram/=max(np.trace(gram),1e-12)
                shape=(node['mesh'],tuple(np.round(gram.ravel(),7)));geometry=None
                for variant,rows in policies.items():
                    r=rows[j];ratio=float(r['target_ratio_far']);ck=(shape,round(ratio,8))
                    if ck not in metric_cache:
                        if geometry is None:
                            v,f=mesh_for_node(data,mm,offset,node);geometry=prepare_geometry(v,f)
                        metric_cache[ck]=simplify_metrics(func,geometry,ratio)
                    result.append(dict(scene=scene,variant=variant,node_index=source['node_index'],mesh_index=source['mesh_index'],
                        source_faces=int(source['face_count']),requested_ratio=ratio,**metric_cache[ck]))
                if (j+1)%500==0:print(f'{scene}: geometry {j+1}/{len(policies[names[0]])}',flush=True)
        write_csv(folder/'geometry_probe.csv',result)
        grouped=defaultdict(list)
        for r in result:grouped[r['variant']].append(r)
        for variant,rows in grouped.items():
            faces=sum(r['source_faces']for r in rows);errors=[r['relative_surface_error']for r in rows]
            summaries.append(dict(scene=scene,variant=variant,mode=variant.split('_')[0],tree='on'if variant.endswith('H1')else 'off',
                nodes=len(rows),source_faces=faces,actual_faces=sum(r['actual_faces']for r in rows),
                requested_faces=sum(r['source_faces']*r['requested_ratio']for r in rows),
                error_sum=sum(errors),face_weighted_error_sum=sum(r['source_faces']*r['relative_surface_error']for r in rows),
                axis_shrink_nodes=sum(r['max_axis_shrink']>.1 for r in rows),
                component_loss_nodes=sum(r['missing_significant_components']>0 for r in rows),
                empty_geometry_nodes=sum(r['actual_faces']==0 for r in rows),
                p95_relative_error=float(np.quantile(errors,.95))))
        print(scene+': full main-matrix geometry probe complete',flush=True)
    for variant in [f'{m}_H{h}'for m in MAIN_MODES for h in (0,1)]:
        group=[r for r in summaries if r['variant']==variant]
        r={k:group[0][k]for k in ('variant','mode','tree')};r['scene']='ALL'
        for k in ('nodes','source_faces','actual_faces','requested_faces','error_sum','face_weighted_error_sum','axis_shrink_nodes','component_loss_nodes','empty_geometry_nodes'):
            r[k]=sum(x[k]for x in group)
        r['p95_relative_error']=None;summaries.append(r)
    for r in summaries:
        r.update(actual_triangle_fraction=r['actual_faces']/r['source_faces'],requested_triangle_fraction=r['requested_faces']/r['source_faces'],
            mean_relative_error=r['error_sum']/r['nodes'],face_weighted_relative_error=r['face_weighted_error_sum']/r['source_faces'])
    return summaries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'lod_analysis_outputs/assembly_context_v3_20261020/scene_sources.json')
    parser.add_argument('--output-root',type=Path,default=ROOT/'lod_analysis_outputs/six_scene_factorial_ablation_20261020')
    parser.add_argument('--report-dir',type=Path,default=ROOT/'analysis_reports/six_scene_factorial_ablation_20261020')
    parser.add_argument('--models',nargs='+',default=list('abcdef'))
    parser.add_argument('--skip-geometry-probe',action='store_true')
    args=parser.parse_args();args.report_dir.mkdir(parents=True,exist_ok=True);args.output_root.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    taxonomy=json.loads((DIRECTORY/'industrial_open_vocab_v2.json').read_text(encoding='utf-8'))
    config=json.loads((DIRECTORY/'assembly_context_config.json').read_text(encoding='utf-8'))
    status=dict(status='running',models=args.models,main_modes=MAIN_MODES,main_conditions=len(args.models)*12,failed=[])
    def progress(scene,stage):
        status.update(scene=scene,stage=stage);save(args.report_dir/'run_status.json',status)
    summaries=[];diagnostics=[];pairs=[];transitions=[];changes=[];inputs=[args.manifest,DIRECTORY/'industrial_open_vocab_v2.json',DIRECTORY/'assembly_context_config.json']
    inventory=[]
    for scene in args.models:
        try:
            progress(scene,'policy_matrix');source=manifest[scene]
            glb,pnpath,cpath=map(Path,[source['glb'],source['pointnext_csv'],source['open_csv']])
            for path,digest in [(glb,source['glb_sha256']),(pnpath,source['pointnext_sha256']),(cpath,source['open_sha256'])]:
                if file_hash(path)!=digest:raise ValueError('Changed inference source '+str(path))
            pn,opened=read_csv(pnpath),read_csv(cpath);cindex={key(r):r for r in opened}
            data,_,_=read_glb_chunks(glb);expected={(str(i),str(n['mesh']))for i,n in enumerate(data['nodes'])if 'mesh'in n}
            if len(pn)!=len(expected)or len(cindex)!=len(expected)or set(cindex)!={key(r)for r in pn}:raise ValueError('Incomplete neural evidence')
            descriptor_path=Path(source['glb']).parents[1]/'lod_analysis_outputs/assembly_context_v3_20261020'/scene/'geometry_descriptors.json'
            stamp=descriptor_path.with_suffix('.provenance.json')
            if descriptor_path.exists()and stamp.exists():
                proof=json.loads(stamp.read_text(encoding='utf-8'))
                valid=proof.get('output_sha256')==file_hash(descriptor_path)and all(Path(k).is_file()and file_hash(Path(k))==v for k,v in proof['inputs'].items())
            else:valid=False
            if valid:descriptors=json.loads(descriptor_path.read_text(encoding='utf-8'))
            else:
                descriptor_path=args.output_root/scene/'geometry_descriptors.json';descriptors=geometry_descriptors(glb,pn,descriptor_path)
            contexts={'off':global_contexts(pn),'topology':build_contexts(data['nodes'],pn,config,use_names=False),
                      'on':build_contexts(data['nodes'],pn,config,use_names=True)}
            thresholds=builder.make_thresholds(pn);conditions={};scene_diag=[]
            for row in pn:scene_diag.append(source_diagnostic(scene,row,cindex[key(row)],taxonomy))
            diagnostics.extend(scene_diag)
            condition_specs=[(f'{mode}_H{0 if tree=="off"else "T"if tree=="topology"else 1}',mode,tree,False,
                'main'if mode in MAIN_MODES and tree!='topology'else 'topology_control'if tree=='topology'else 'G_supplement')
                for mode in (*MAIN_MODES,'G')for tree in TREE_LEVELS]
            condition_specs += [('NCG_H0_M1','NCG','off',True,'metadata_control'),('NCG_H1_M1','NCG','on',True,'metadata_control')]
            folder=args.output_root/scene;folder.mkdir(parents=True,exist_ok=True)
            for variant,mode,tree,leaf,kind in condition_specs:
                rows=[classify(row,cindex[key(row)],taxonomy,thresholds,descriptors[str(row['node_index'])],
                    contexts[tree][str(row['node_index'])],config,mode,tree,leaf)for row in pn]
                if {(str(r['node_index']),str(r['mesh_index']))for r in rows}!=expected:raise ValueError('Policy key mismatch')
                conditions[variant]=rows;write_csv(folder/(variant+'.csv'),rows)
                summaries.append(summarize(scene,variant,mode,tree,leaf,rows,scene_diag,taxonomy,kind))
            for mode in MAIN_MODES:
                for before,after,kind in [('H0','H1','main'),('H0','HT','topology_control'),('HT','H1','ancestor_name_increment')]:
                    pair,transition,changed=pairing(scene,mode,conditions[f'{mode}_{before}'],conditions[f'{mode}_{after}'],before,after,kind)
                    pairs.append(pair);transitions.extend(transition);changes.extend(changed)
            inventory.append(dict(scene=scene,title=TITLES[scene],nodes=len(pn),assembly_nodes=sum(bool(n.get('children'))for n in data['nodes']),
                source_faces=sum(int(r['face_count'])for r in pn),glb_sha256=source['glb_sha256']))
            inputs.extend([glb,pnpath,cpath,descriptor_path])
            print(f'{scene}: {len(pn)} nodes x {len(condition_specs)} conditions complete',flush=True)
        except Exception as exc:
            status['failed'].append(dict(scene=scene,error=str(exc)));save(args.report_dir/'run_status.json',status)
            print(f'SKIP {scene}: {exc}',flush=True)
    if status['failed']:
        status['status']='partial_failure';save(args.report_dir/'run_status.json',status);return 1
    summaries+=pooled(summaries)
    profiles,lowtypes=uncertainty_tables(diagnostics,taxonomy)
    write_csv(args.report_dir/'all_conditions_summary.csv',summaries)
    main_rows=[r for r in summaries if r['kind']=='main'];write_csv(args.report_dir/'main_72_comparison.csv',main_rows)
    write_csv(args.report_dir/'source_uncertainty_nodes.csv',diagnostics)
    write_csv(args.report_dir/'source_confidence_summary.csv',profiles)
    write_csv(args.report_dir/'low_confidence_candidate_types.csv',lowtypes)
    write_csv(args.report_dir/'tree_pairs.csv',pairs);write_csv(args.report_dir/'priority_transitions.csv',transitions)
    if changes:write_csv(args.report_dir/'changed_nodes.csv',changes)
    write_csv(args.report_dir/'scene_inventory.csv',inventory)
    geometry=[]
    if not args.skip_geometry_probe:
        geometry=probe_scenes(manifest,args.output_root,args.models,progress)
        write_csv(args.report_dir/'geometry_comparison.csv',geometry)
        inputs.extend([ROOT/'_bin/Release/semantic_geometry_probe.dll',DIRECTORY/'native/semantic_geometry_probe.cpp'])
    dataset=dict(inventory=inventory,summaries=summaries,main=main_rows,confidence=profiles,low_types=lowtypes,
        pairs=pairs,geometry=geometry,config=config,models=args.models,
        protocol=dict(main_modes=MAIN_MODES,main_tree_levels=['off','on'],leaf_names=False,
            incumbent_geometry_guard=False,full_tree_off_includes_parent_relative_geometry=True,
            disabled_branch_inputs_masked=True,neural_weights_frozen=True,classification_accuracy_available=False,
            actual_renderer_benchmark=False,confidence_N_rule='p>=0.70 and margin>=0.08',
            confidence_C_rule='accepted/class_only object output; family_only coarse; abstain rejected'))
    save(args.report_dir/'comparison_data.json',dataset)
    # A frozen source copy makes the ablation independent of future production-rule edits.
    code=[Path(__file__),DIRECTORY/'factorial_ablation_policy.py',DIRECTORY/'assembly_context_policy.py',
          DIRECTORY/'assembly_context_features.py',DIRECTORY/'fuse_open_semantic_v2.py',DIRECTORY/'probe_assembly_geometry.py']
    (args.report_dir/'method').mkdir(exist_ok=True)
    for path in code:shutil.copy2(path,args.report_dir/'method'/path.name)
    save(args.report_dir/'provenance.json',dict(inputs={str(p.resolve()):file_hash(p)for p in inputs+code},
        outputs={str(p.resolve()):file_hash(p)for p in args.report_dir.glob('*.csv')},
        policy_outputs={str(p.resolve()):file_hash(p)for scene in args.models for p in (args.output_root/scene).glob('*.csv')},
        total_nodes=sum(r['nodes']for r in inventory),main_scene_conditions=len(args.models)*12,
        total_scene_conditions=len(args.models)*23,production_csv_overwritten=False))
    status.update(status='complete',stage='policy_and_geometry_complete',total_nodes=sum(r['nodes']for r in inventory),
        total_policy_rows=sum(r['nodes']for r in inventory)*23)
    save(args.report_dir/'run_status.json',status)
    print('Full factorial outputs: '+str(args.report_dir),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
