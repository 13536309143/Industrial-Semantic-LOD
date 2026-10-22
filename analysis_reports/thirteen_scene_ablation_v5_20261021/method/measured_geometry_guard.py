"""Migration safeguard: keep new functional roles, test retention against the incumbent geometry."""
import argparse
import json
import mmap
import shutil
from pathlib import Path

from audit_named_scene import read_csv,write_csv
from assembly_context_features import geometry_descriptors,build_contexts
from assembly_context_policy import apply_policy
from fuse_open_semantic_v2 import builder
from experiment_assembly_context import evaluate,key
from artifact_cache import file_hash
from probe_assembly_geometry import load_probe,prepare_geometry,simplify_metrics
from run_pointclipv2_zeroshot_glb import read_glb_chunks,mesh_for_node

ROOT=Path(__file__).resolve().parents[2]
DIRECTORY=Path(__file__).parent


def regression(proposed,incumbent,config):
    reasons=[]
    if float(proposed['relative_surface_error'])>float(incumbent['relative_surface_error'])*config['guard_error_factor']+config['guard_error_slack']:
        reasons.append('relative_surface_error_regression')
    if float(proposed['max_axis_shrink'])>max(.1,float(incumbent['max_axis_shrink'])+config['guard_axis_slack']):
        reasons.append('silhouette_extent_regression')
    if int(proposed['missing_significant_components'])>int(incumbent['missing_significant_components']):
        reasons.append('surface_component_loss_regression')
    return reasons


def constrain_schedule(row,far,reason):
    row=dict(row)
    near=float(row['target_ratio_near']);mid=float(row['target_ratio_mid'])
    mid=max(mid,min(.94,far+.04));near=max(near,min(.96,mid+.04))
    row.update(target_ratio_far=far,target_ratio_mid=mid,target_ratio_near=near,
        feature_lock_floor=max(float(row.get('feature_lock_floor')or 0),.16),
        feature_soft_floor=max(float(row.get('feature_soft_floor')or 0),1.25),allow_cull=False,
        geometry_feedback_applied=True,geometry_feedback_reason=';'.join(reason),
        inference_reason=row['inference_reason']+';measured_geometry_retention_guard')
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--report-dir',type=Path,required=True);p.add_argument('--source-round',default='r4_final_policy')
    p.add_argument('--round',default='final_guarded');p.add_argument('--models',nargs='+',default=list('abcdef'))
    p.add_argument('--config',type=Path,default=DIRECTORY/'assembly_context_config.json')
    p.add_argument('--probe-dll',type=Path,default=ROOT/'_bin/Release/semantic_geometry_probe.dll')
    args=p.parse_args()
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    config=json.loads(args.config.read_text(encoding='utf-8'))
    taxonomy=json.loads((DIRECTORY/'industrial_open_vocab_v2.json').read_text(encoding='utf-8'))
    dll,func=load_probe(args.probe_dll)
    report=args.report_dir;report.mkdir(parents=True,exist_ok=True)
    summaries=[];named=[];cases=[];feedback=[];inputs=[args.manifest,args.config,args.probe_dll,DIRECTORY/'industrial_open_vocab_v2.json']
    output_files=[]
    for name in args.models:
        source=manifest[name];glb=Path(source['glb']);pnpath=Path(source['pointnext_csv']);openpath=Path(source['open_csv'])
        if file_hash(glb)!=source['glb_sha256']or file_hash(pnpath)!=source['pointnext_sha256']or file_hash(openpath)!=source['open_sha256']:
            raise ValueError('Changed scene/evidence '+name)
        pn,opened=read_csv(pnpath),read_csv(openpath)
        data,offset,_=read_glb_chunks(glb)
        directory=args.output_root/name
        descriptors=geometry_descriptors(glb,pn,directory/'geometry_descriptors.json')
        contexts=build_contexts(data['nodes'],pn,config)
        folder=directory/args.source_round
        raw_proxy=read_csv(folder/'geometry_proxy.csv')
        mindex={(r['variant'],r['node_index']):r for r in raw_proxy}
        out=directory/args.round;out.mkdir(parents=True,exist_ok=True)
        input_paths=list(folder.glob('v*.csv'))+[folder/'geometry_proxy.csv',directory/'geometry_descriptors.json',glb,pnpath,openpath]
        inputs.extend(input_paths)
        for path in folder.glob('v*.csv'):
            rows=read_csv(path);variant=path.stem
            guarded_variant=variant in {'v3_NCG','v3_NCGH','v3_NCGHM'}
            if guarded_variant:
                incumbent=read_csv(folder/'v2_NCG.csv')
                if [key(r)for r in rows]!=[key(r)for r in incumbent]:raise ValueError('Guard policy keys do not match')
                with glb.open('rb')as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ)as mm:
                    for r,old in zip(rows,incumbent):
                        node_index=r['node_index']
                        measured=mindex[(variant,node_index)];reference=mindex[('v2_NCG',node_index)]
                        reasons=regression(measured,reference,config)
                        initial=float(r['target_ratio_far']);selected=initial;chosen=measured
                        if reasons:
                            original=float(old['target_ratio_far'])
                            candidates=sorted(set([original,.36,.42,.52,.65,.80,config['guard_max_far_ratio']]))
                            geometry=None
                            for ratio in candidates:
                                if ratio<initial-1e-8:continue
                                if abs(ratio-original)<1e-8:metrics=reference
                                else:
                                    if geometry is None:
                                        v,f=mesh_for_node(data,mm,offset,data['nodes'][int(node_index)])
                                        geometry=prepare_geometry(v,f)
                                    metrics=simplify_metrics(func,geometry,ratio)
                                selected,chosen=ratio,metrics
                                if not regression(metrics,reference,config):break
                            updated=constrain_schedule(r,selected,reasons)
                            r.clear();r.update(updated)
                        else:r.update(geometry_feedback_applied=False,geometry_feedback_reason='within_incumbent_proxy_tolerance')
                        residual=regression(chosen,reference,config)
                        r['geometry_feedback_residual_review']=bool(residual)
                        if residual:r['needs_review']=True
                        feedback.append(dict(scene=name,variant=variant,node_index=node_index,node_name=r['node_name'],
                            original_far_ratio=initial,selected_far_ratio=selected,guard_applied=bool(reasons),
                            guard_reason=';'.join(reasons),residual_regression=';'.join(residual),
                            source_faces=r['face_count'],incumbent_actual_faces=reference['actual_faces'],
                            selected_actual_faces=chosen['actual_faces'],incumbent_error=reference['relative_surface_error'],
                            selected_error=chosen['relative_surface_error'],incumbent_axis_shrink=reference['max_axis_shrink'],
                            selected_axis_shrink=chosen['max_axis_shrink'],incumbent_component_loss=reference['missing_significant_components'],
                            selected_component_loss=chosen['missing_significant_components'],
                            scope='single-mesh QEM migration guard; not true recognition accuracy or renderer LOD validation'))
            destination=out/path.name;write_csv(destination,rows);output_files.append(destination)
            summary,checks,examples=evaluate(name,variant,rows,pn,opened,descriptors,contexts,config,taxonomy)
            summaries.append(summary);named.extend(checks);cases.extend(examples)
        # Leaf-only counterfactual isolates the extra functional effect of ancestors from name parsing.
        cindex={key(r):r for r in opened};thresholds=builder.make_thresholds(pn)
        leaf_only=[apply_policy(r,cindex[key(r)],taxonomy,thresholds,descriptors[str(r['node_index'])],
            contexts[str(r['node_index'])],config,'NCG',False,True)for r in pn]
        path=out/'v3_NCGM.csv';write_csv(path,leaf_only);output_files.append(path)
        summary,checks,examples=evaluate(name,'v3_NCGM',leaf_only,pn,opened,descriptors,contexts,config,taxonomy)
        summaries.append(summary);named.extend(checks);cases.extend(examples)
        print(name+': measured guard and final variants complete',flush=True)
    write_csv(report/'policy_comparison.csv',summaries)
    if named:write_csv(report/'name_weak_reference.csv',named)
    if cases:write_csv(report/'context_cases.csv',cases)
    write_csv(report/'geometry_feedback.csv',feedback)
    (report/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')
    code=[Path(__file__),DIRECTORY/'probe_assembly_geometry.py',DIRECTORY/'assembly_context_policy.py',DIRECTORY/'assembly_context_features.py',DIRECTORY/'experiment_assembly_context.py']
    proof=dict(inputs={str(q.resolve()):file_hash(q)for q in inputs+code},
        outputs={str(q.resolve()):file_hash(q)for q in report.iterdir()if q.is_file()and q.name!='provenance.json'},
        policy_outputs={str(q.resolve()):file_hash(q)for q in output_files},
        frozen_policy_config_sha256=file_hash(args.config),
        guard_scope='NCG/NCGH/NCGHM migration from incumbent NCG; pure C/G never consumes this incumbent',
        accuracy_measured=False,renderer_lod_measured=False)
    (report/'provenance.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
