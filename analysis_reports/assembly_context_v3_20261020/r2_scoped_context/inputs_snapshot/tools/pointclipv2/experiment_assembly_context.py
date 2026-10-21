"""Run policy ablations; CAD leaf-name checks remain weak proxies, not ground truth."""
import argparse
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from audit_named_scene import read_csv,write_csv,name_hints
from summarize_named_scenes import expected_known_classes
from run_pointclipv2_zeroshot_glb import read_glb_chunks
from fuse_open_semantic_v2 import fuse,builder,MODES
from assembly_context_features import geometry_descriptors,build_contexts,write_context_audit,geometry_risk
from assembly_context_policy import apply_policy,leaf_function,CLASS_FUNCTION
from artifact_cache import file_hash

ROOT=Path(__file__).resolve().parents[2]
DIRECTORY=Path(__file__).parent


def key(row):return tuple(str(row[k])for k in ('order','node_index','mesh_index'))
def priority(row):return int(row['lod_priority'].split('_')[0][1:])
def true(value):return str(value).lower()=='true'


def identity_class(row,taxonomy):
    value=row.get('resolved_object_class','')
    if value.startswith(('known:','closed:')):return value.split(':',1)[1]
    return next((x.get('known_class','')for x in taxonomy['labels']if x['id']==value),'')


def evaluate(name,variant,rows,pn,opened,descriptors,contexts,config,taxonomy):
    counts=Counter(priority(r)for r in rows)
    index={key(r):r for r in rows}
    named,examples=[],[]
    for source in pn:
        result=index[key(source)]
        leaf,hints=name_hints(source['node_name'])
        expected=expected_known_classes(dict(leaf_name=leaf,name_candidate_role=hints[0][0]if hints else ''))
        if expected:
            predicted=identity_class(result,taxonomy)
            group=re.sub(r'(?:[-_.]\d+)+$','',leaf)
            named.append(dict(scene=name,variant=variant,node_index=source['node_index'],mesh_index=source['mesh_index'],
                node_name=source['node_name'],reference_classes='|'.join(sorted(expected)),reference_group=group,
                resolved_class=predicted,covered=bool(predicted),consistent=predicted in expected,
                priority=priority(result),function=result['function_role'],
                includes_leaf_names=true(result.get('leaf_names_used',False)),
                scope='CAD-name weak reference; metadata-assisted rows excluded from visual accuracy comparisons'))
        if source.get('node_name') and (true(result.get('assembly_context_applied',False))or true(result.get('type_shape_conflict',False))):
            examples.append(dict(scene=name,variant=variant,node_index=source['node_index'],mesh_index=source['mesh_index'],
                node_name=source['node_name'],closed_type=source.get('predicted_class',''),
                resolved_identity=result['resolved_object_class'],function=result['function_role'],
                context=result.get('assembly_context_category',''),context_json=result.get('assembly_context_json',''),
                priority=priority(result),reason=result['inference_reason']))
    total_faces=sum(int(r['face_count'])for r in rows)
    mesh_groups=defaultdict(list)
    for r in rows:mesh_groups[r['mesh_index']].append(r)
    def budget(level,shared=False):
        if shared:
            return sum(int(v[0]['face_count'])*max(float(r['target_ratio_'+level])for r in v)*len(v)for v in mesh_groups.values())/max(total_faces,1)
        return sum(int(r['face_count'])*float(r['target_ratio_'+level])for r in rows)/max(total_faces,1)
    risky=underprotected=complexity_only=0
    role_proxy_low=frame_high=0
    for source,result in zip(pn,rows):
        risk=geometry_risk(source,descriptors[str(source['node_index'])],contexts[str(source['node_index'])],config)
        if risk['priority_floor']>=6:
            risky+=1;underprotected+=priority(result)<risk['priority_floor']
        leaf,hints=name_hints(source['node_name'])
        ref_role=hints[0][0]if hints else ''
        role_proxy_low+=ref_role in {'motor_or_reducer','bearing_or_linear_guide','transmission','spring','sensing_or_inspection','tooling_or_process'}and priority(result)<=3
        frame_high+=ref_role in {'frame_or_structure','enclosure'}and priority(result)>=9
    group_results=defaultdict(list)
    for r in named:group_results[(r['reference_classes'],r['reference_group'])].append(r)
    summary=dict(scene=name,variant=variant,nodes=len(rows),low_P1_P3=sum(counts[p]for p in (1,2,3)),
        high_P8_P10=sum(counts[p]for p in (8,9,10)),priority_counts=dict(counts),
        geometry_risk_nodes=risky,geometry_floor_violations=underprotected,
        named_critical_low_priority=role_proxy_low,named_frame_precision_priority=frame_high,
        tree_context_applied=sum(true(r.get('assembly_context_applied',False))for r in rows),
        shape_type_conflicts=sum(true(r.get('type_shape_conflict',False))for r in rows),
        auxiliary_candidates=sum(true(r.get('scene_auxiliary_candidate',False))for r in rows),
        unique_meshes=len(mesh_groups),shared_mesh_mixed_priorities=sum(len({priority(r)for r in v})>1 for v in mesh_groups.values()),
        requested_near_triangle_ratio=budget('near'),requested_far_triangle_ratio=budget('far'),
        shared_mesh_far_triangle_ratio=budget('far',True),
        named_proxy_nodes=len(named),named_object_coverage=sum(r['covered']for r in named),
        named_object_consistent=sum(r['consistent']for r in named),
        name_groups=len(group_results),group_macro_consistency=sum(sum(r['consistent']for r in v)/len(v)for v in group_results.values())/max(len(group_results),1),
        leaf_names_used=any(true(r.get('leaf_names_used',False))for r in rows),
        real_accuracy_measured=False,actual_lod_geometry_measured=False)
    return summary,named,examples


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--report-dir',type=Path,required=True)
    parser.add_argument('--models',nargs='+',default=list('abcdef'))
    parser.add_argument('--round',default='r1_geometry_context')
    parser.add_argument('--config',type=Path,default=DIRECTORY/'assembly_context_config.json')
    parser.add_argument('--refinement',choices=['broad','scoped'],default='scoped')
    parser.add_argument('--no-export',action='store_true')
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    taxonomy=json.loads((DIRECTORY/'industrial_open_vocab_v2.json').read_text(encoding='utf-8'))
    config=json.loads(args.config.read_text(encoding='utf-8'))
    report=args.report_dir/args.round;report.mkdir(parents=True,exist_ok=True)
    (report/'config.json').write_text(json.dumps(dict(config=config,refinement=args.refinement),indent=2),encoding='utf-8')
    summaries=[];all_named=[];all_cases=[];input_paths=[args.manifest,args.config,DIRECTORY/'industrial_open_vocab_v2.json']
    variants=[('v2_NCG',None,False,False),('v3_NCG','NCG',False,False),('v3_NCGH','NCG',True,False),('v3_NCGHM','NCG',True,True)]
    variants += [('v3_'+mode,mode,False,False)for mode in MODES if mode!='NCG']
    for name in args.models:
        source=manifest[name]
        glb,pnpath,openpath=map(Path,[source['glb'],source['pointnext_csv'],source['open_csv']])
        for path,expected in [(glb,source['glb_sha256']),(pnpath,source['pointnext_sha256']),(openpath,source['open_sha256'])]:
            if file_hash(path)!=expected:raise ValueError('Changed evidence: '+str(path))
        pn,opened=read_csv(pnpath),read_csv(openpath)
        cindex={key(r):r for r in opened}
        if len(cindex)!=len(pn)or set(cindex)!={key(r)for r in pn}:raise ValueError('Incomplete/duplicate scene keys: '+name)
        gltf,_,_=read_glb_chunks(glb)
        if len(pn)!=sum('mesh'in n for n in gltf['nodes']):raise ValueError('Partial scene: '+name)
        directory=args.output_root/name
        descriptors=geometry_descriptors(glb,pn,directory/'geometry_descriptors.json')
        contexts=build_contexts(gltf['nodes'],pn,config)
        write_context_audit(directory/'assembly_geometry_audit.csv',pn,contexts,descriptors)
        thresholds=builder.make_thresholds(pn)
        out=directory/args.round;out.mkdir(parents=True,exist_ok=True)
        for variant,mode,tree,leaf in variants:
            if variant=='v2_NCG':rows=[fuse(r,cindex[key(r)],taxonomy,thresholds,'NCG')for r in pn]
            else:rows=[apply_policy(r,cindex[key(r)],taxonomy,thresholds,descriptors[str(r['node_index'])],
                contexts[str(r['node_index'])],config,mode,tree,leaf,args.refinement)for r in pn]
            write_csv(out/(variant+'.csv'),rows)
            summary,named,cases=evaluate(name,variant,rows,pn,opened,descriptors,contexts,config,taxonomy)
            summaries.append(summary);all_named.extend(named);all_cases.extend(cases)
        input_paths += [glb,pnpath,openpath,directory/'geometry_descriptors.json']
        print(f'{name}: {len(pn)} nodes, {len(variants)} policy variants complete',flush=True)
    write_csv(report/'policy_comparison.csv',summaries)
    if all_named:write_csv(report/'name_weak_reference.csv',all_named)
    if all_cases:write_csv(report/'context_cases.csv',all_cases)
    (report/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')
    sources=[Path(__file__),DIRECTORY/'assembly_context_features.py',DIRECTORY/'assembly_context_policy.py',DIRECTORY/'fuse_open_semantic_v2.py']
    provenance=dict(inputs={str(p.resolve()):file_hash(p)for p in input_paths+sources},
        outputs={str(p.resolve()):file_hash(p)for p in report.iterdir()if p.is_file()and p.name!='provenance.json'},
        policy_outputs={str(p.resolve()):file_hash(p)for name in args.models for p in (args.output_root/name/args.round).glob('*.csv')},
        evaluation='weak leaf names; no leaf-assisted predictions reported as blind accuracy',
        no_nodes_deleted=True,actual_visibility_measured=False)
    # Each round keeps the exact source/config/descriptor inputs even after the next revision.
    relocations={}
    for original,digest in list(provenance['inputs'].items()):
        p=Path(original)
        if 'tools'in p.parts or p.name in {'scene_sources.json','geometry_descriptors.json'}:
            snapshot=report/'inputs_snapshot'/p.relative_to(ROOT)
            snapshot.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,snapshot)
            relocations[str(snapshot.resolve())]=original
            del provenance['inputs'][original]
            provenance['inputs'][str(snapshot.resolve())]=digest
    provenance['historical_input_locations']=relocations
    (report/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Policy experiment complete: '+str(report),flush=True)


if __name__=='__main__':main()
