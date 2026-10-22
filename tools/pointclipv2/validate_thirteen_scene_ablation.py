"""Independently stream every compressed policy to verify full paired coverage."""
import argparse
import csv
import gzip
import json
import math
import time
from collections import Counter
from pathlib import Path

from run_thirteen_scene_ablation import ROOT, DEFAULT_OUTPUT, DEFAULT_REPORT, digest, valid_stamp, save
from run_pointclipv2_zeroshot_glb import read_glb_chunks
from thirteen_scene_coverage import align_mesh_population


def load_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--report-dir',type=Path,default=DEFAULT_REPORT)
    p.add_argument('--wait',action='store_true')
    args=p.parse_args();result=dict(status='running',scenes=[],failed=[])
    target=args.report_dir/'full_validation.json'
    checkpoint=ROOT/'models/pointnext/industrial-part-14class-best.pth';checkpoint_hash=digest(checkpoint)
    inventory=json.loads((ROOT/'analysis_reports/thesis_ch5_setup_20261021/glb_inventory.json').read_text(encoding='utf-8'))
    expected_inventory={r['id']:r for r in inventory}
    classes={int(line.split()[0]):line.split()[1] for line in
        (ROOT/'tools/pointnext_lod/industrial_part_14classes.txt').read_text(encoding='utf-8').splitlines() if line.strip()}
    for i in range(1,14):
        scene=f'M{i}';folder=args.output_root/scene/'policies';done=folder/'completion.json'
        while not done.exists() and args.wait:
            status=json.loads((args.report_dir/'analysis_status.json').read_text(encoding='utf-8'))
            if status['status']!='running':break
            time.sleep(10)
        try:
            proof=json.loads(done.read_text(encoding='utf-8'))
            source=proof['source'];glb=Path(source['glb'])
            assert digest(glb)==source['glb_sha256']
            assert source['checkpoint_sha256']==checkpoint_hash
            pnproof=Path(source['pointnext_provenance'])
            assert valid_stamp(pnproof,source['glb_sha256'])
            parameters=json.loads(pnproof.read_text(encoding='utf-8'))['parameters']
            assert parameters['num_points']==8192 and parameters['inference_mode']=='mesh' and parameters['seed']==2026
            for name,h in proof['outputs'].items():assert digest(Path(name))==h,name
            pn=load_csv(Path(source['pointnext_csv']));opened=load_csv(Path(source['open_csv']))
            for r in pn:
                confidence,second=float(r['confidence']),float(r['second_confidence'])
                assert math.isfinite(confidence) and math.isfinite(second) and 0<=second<=confidence<=1
                assert classes[int(r['predicted_label'])]==r['predicted_class']
                assert classes[int(r['second_label'])]==r['second_class']
            for r in opened:
                quality=float(r['open_reliability'])
                assert math.isfinite(quality) and 0<=quality<=1
                assert r['open_status'] in {'accepted','class_only','family_only','abstain'}
            data,_,_=read_glb_chunks(glb)
            pn,opened,excluded=align_mesh_population(data,pn,opened)
            if excluded:
                audit=json.loads((folder/'coverage_audit.json').read_text(encoding='utf-8'))
                assert audit['excluded_empty_meshes']==excluded
                assert audit['adapter_sha256']==digest(Path(__file__).with_name('thirteen_scene_coverage.py'))
            nodekeys=lambda rows:{(str(r['node_index']),str(r['mesh_index'])) for r in rows}
            expected=nodekeys(pn)
            assert len(expected)==len(pn)==len(opened) and nodekeys(opened)==expected
            assert len(pn)+len(excluded)==expected_inventory[scene]['mesh_instances']
            faces=sum(int(r['face_count']) for r in pn)
            assert faces==expected_inventory[scene]['source_triangles_instances']
            summary={r['variant']:r for r in load_csv(folder/'summary.csv')}
            files=sorted(folder.glob('*_H*.csv.gz'))
            assert len(files)==23
            count=0
            for path in files:
                seen=set();hist=Counter();face_hist=Counter();n=0
                variant=path.name[:-7]
                with gzip.open(path,'rt',encoding='utf-8-sig',newline='') as stream:
                    for r in csv.DictReader(stream):
                        k=(r['node_index'],r['mesh_index']);assert k in expected and k not in seen
                        seen.add(k);n+=1
                        tier=int(r['lod_priority'].split('_')[0][1:]);assert 1<=tier<=10
                        hist[tier]+=1;face_hist[tier]+=int(r['face_count'])
                        near,mid,far=[float(r['target_ratio_'+v]) for v in ('near','mid','far')]
                        assert 0<far<=mid<=near<=1
                        assert r['allow_cull'].lower()=='false' and r['allow_geometry_drop'].lower()=='false'
                        assert math.isfinite(float(r['simplify_error_relative'])) and float(r['simplify_error_relative'])>0
                assert seen==expected and n==len(pn)
                for tier in range(1,11):
                    assert hist[tier]==int(summary[variant][f'P{tier}_count'])
                    assert face_hist[tier]==int(summary[variant][f'P{tier}_faces'])
                count+=n
            visual=Path(source['visual_directory'])
            for name in ['prediction.provenance.json','combined_prediction.provenance.json','cache/views.provenance.json','cache/shaded_views.provenance.json']:
                stamp=visual/name;assert valid_stamp(stamp,source['glb_sha256']),stamp
                assert json.loads(stamp.read_text(encoding='utf-8')).get('parameters',{}).get('limit') is None
            depth_errors=json.loads((visual/'cache/views.errors.json').read_text(encoding='utf-8'))
            shading=json.loads((visual/'shading_status.json').read_text(encoding='utf-8'))
            empty_ids={r['node_index'] for r in excluded}
            assert {int(r['node_index']) for r in depth_errors}<=empty_ids
            assert {int(r['node_index']) for r in shading.get('failed_geometry',[])}<=empty_ids
            case=dict(scene=scene,nodes=len(pn),source_faces=faces,conditions=len(files),policy_rows=count,
                depth_geometry_errors=len(depth_errors),shading_geometry_errors=len(shading.get('failed_geometry',[])),
                all_policy_keys_unique_and_complete=True,all_P_counts_and_face_counts_reconcile=True,
                all_ratios_ordered_and_finite=True,all_nodes_retained=True,source_hashes_verified=True)
            case.update(neural_scores_finite_and_in_range=True,supervised_label_map_consistent=True,
                open_statuses_valid=True,source_mesh_references=len(pn)+len(excluded),excluded_empty_meshes=len(excluded))
            case['rendering_errors_on_analyzable_geometry']=0
            result['scenes'].append(case);save(target,result)
            print(f'VERIFIED {scene}: {len(files)} conditions, {count:,} rows',flush=True)
        except Exception as exc:
            import traceback;traceback.print_exc()
            result['failed'].append(dict(scene=scene,error=str(exc)));save(target,result)
    result.update(status='passed' if len(result['scenes'])==13 and not result['failed'] else 'failed',
        nodes=sum(r['nodes'] for r in result['scenes']),policy_files=sum(r['conditions'] for r in result['scenes']),
        policy_rows=sum(r['policy_rows'] for r in result['scenes']))
    result.update(source_mesh_references=sum(r['source_mesh_references'] for r in result['scenes']),
        excluded_empty_meshes=sum(r['excluded_empty_meshes'] for r in result['scenes']))
    save(target,result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result['status']!='passed'


if __name__=='__main__':raise SystemExit(main())
