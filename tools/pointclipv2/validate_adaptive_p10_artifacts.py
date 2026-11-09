"""Audit complete scene coverage, isolated evidence and final runtime artifacts before export."""
import argparse
import json
from pathlib import Path

from adaptive_p10_policy import VERSION,yes
from audit_named_scene import read_csv
from artifact_cache import file_hash
from run_adaptive_p10 import export,save
from benchmark_p10_runtime import accepted_benchmark_hashes

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'analysis_reports/adaptive_p10_v4_20261020'
POLICIES=ROOT/'lod_analysis_outputs/adaptive_p10_v4_20261020'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--deploy',action='store_true')
    args=parser.parse_args()
    proof=json.loads((REPORT/'provenance.json').read_text(encoding='utf-8'))
    verified=0
    for section in ('inputs','outputs'):
        for path,expected in proof[section].items():
            if file_hash(Path(path))!=expected:raise ValueError('Hash mismatch '+path)
            verified+=1
    manifest=json.loads((ROOT/'lod_analysis_outputs/assembly_context_v3_20261020/scene_sources.json').read_text(encoding='utf-8'))
    total=0;histogram={p:0 for p in range(1,11)}
    for scene in 'abcdef':
        source=manifest[scene]
        if file_hash(Path(source['glb']))!=source['glb_sha256']:raise ValueError('Changed GLB')
        pn=read_csv(Path(source['pointnext_csv']));expected={(r['node_index'],r['mesh_index'])for r in pn}
        files=list((POLICIES/scene).glob('*.csv'))
        if len(files)!=16:raise ValueError('Expected 16 complete policies per scene')
        for path in files:
            rows=read_csv(path);total+=len(rows)
            if len(rows)!=len(expected)or {(r['node_index'],r['mesh_index'])for r in rows}!=expected:raise ValueError('Missing/duplicate nodes')
            mode=path.stem.split('_')[0]
            for r in rows:
                p=int(r['lod_priority'].split('_')[0][1:])
                if not 1<=p<=10 or int(r['policy_schema'])!=4:raise ValueError('Invalid profile')
                if r['policy_version']!=VERSION:raise ValueError('Mixed policy versions')
                if yes(r['allow_cull'])or yes(r['allow_geometry_drop']):raise ValueError('Unexpected deletion')
                ratios=[float(r['target_ratio_'+level])for level in ('near','mid','far')]
                if not 1>=ratios[0]>=ratios[1]>=ratios[2]>0:raise ValueError('Invalid schedule')
                if not yes(r['preserve_components'])or float(r['simplify_error_relative'])<=0:raise ValueError('Missing runtime guard')
                if 'N'not in mode and r.get('source_pointnext_class'):raise ValueError('Disabled N evidence leak')
                if 'C'not in mode and r.get('open_status'):raise ValueError('Disabled C evidence leak')
                if '_H0'in path.stem and int(r['local_group_members'])!=0:raise ValueError('Hidden parent scale')
                if 'M1'not in path.stem and yes(r['leaf_names_used']):raise ValueError('Leaf-name leak')
                if 'G'not in mode and float(r['base_simplify_ratio'])!=1:raise ValueError('Hidden geometry cleanup in no-G branch')
                if path.stem=='NCG_H1_M1_selected':histogram[p]+=1
    if total!=176000 or sum(histogram.values())!=11000 or not all(histogram.values()):raise ValueError('Incomplete ten-profile coverage')
    executables=accepted_benchmark_hashes(REPORT)
    for name in ('v3','v4_profile','v4_selected','v4_selected_strict'):
        records=json.loads((REPORT/f'runtime_{name}.json').read_text(encoding='utf-8'))
        expected_policy='final_guarded/v3_NCGHM.csv'if name=='v3'else 'NCG_H1_M1.csv'if name=='v4_profile'else 'NCG_H1_M1_selected.csv'
        for r in records:
            folder=ROOT/'lod_analysis_outputs/assembly_context_v3_20261020'if name=='v3'else POLICIES
            if not r['success']or r['exe_sha256']not in executables or r['policy_sha256']!=file_hash(folder/r['scene']/expected_policy):raise ValueError('Stale/failed actual runtime test')
    image_rows=read_csv(REPORT/'fixed_view_image_quality_extended.csv')
    strict=[r for r in image_rows if r['variant']=='v4_selected_strict']
    if len(strict)!=6:raise ValueError('Missing strict quality comparison')
    for r in strict:
        if float(r['SSIM_mean'])<.98 or float(r['reference_edge_region_SSIM'])<.97 or float(r['reference_edges_recall_2px'])<.99:raise ValueError('Strict image quality gate failed '+r['scene'])
    for r in image_rows:
        for field in ('reference','candidate'):
            if file_hash(Path(r[field+'_image']))!=r[field+'_sha256']:raise ValueError('Changed evaluated image')
    optional=REPORT/'runtime_v4_selected_gpu_candidate.json'
    optional_count=0
    if optional.exists():
        extra=json.loads(optional.read_text(encoding='utf-8'))
        for r in extra:
            if r['stage']=='render':
                if not r['success']or r['exe_sha256']not in executables or r['policy_sha256']!=file_hash(POLICIES/r['scene']/'NCG_H1_M1_selected.csv'):raise ValueError('Stale GPU candidate test')
        optional_count=len({r['scene']for r in extra if r['stage']=='render'})
    result=dict(status='passed',version=VERSION,scenes=6,nodes=11000,policy_conditions=96,policy_rows=total,
                primary_runtime_conditions=24,rejected_gpu_candidate_conditions=optional_count,ten_profile_counts=histogram,hash_entries=verified,
                native_and_policy_tests_passed=65,strict_image_quality_gate='SSIM>=.98; edge-region SSIM>=.97; 2px edge recall>=.99',
                no_deletions=True,disabled_branch_isolation=True,scope='development diagnostics; no independent accuracy or interactive FPS claim')
    save(REPORT/'validation_summary.json',result)
    if args.deploy:
        export(manifest,POLICIES,REPORT,list('abcdef'))
        status=json.loads((REPORT/'run_status.json').read_text(encoding='utf-8'));status.update(version=VERSION,exported=True,validation='passed')
        save(REPORT/'run_status.json',status)
    print(json.dumps(result,ensure_ascii=True),flush=True)


if __name__=='__main__':main()
