"""Only retune screen selection; preserve all measured geometry parameters verbatim."""
import json
import argparse
import shutil
import zipfile
from pathlib import Path

from adaptive_p10_policy import VERSION,runtime_weight
from audit_named_scene import read_csv,write_csv
from artifact_cache import file_hash

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'analysis_reports/adaptive_p10_v4_20261020'
POLICIES=ROOT/'lod_analysis_outputs/adaptive_p10_v4_20261020'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-name',default='development_v4_1_scene_ready')
    args=parser.parse_args()
    proof_path=REPORT/'provenance.json'
    proof=json.loads(proof_path.read_text(encoding='utf-8'))
    policies=sorted(POLICIES.glob('*/*.csv'))
    for path in policies:
        if file_hash(path)!=proof['outputs'][str(path.resolve())]:raise ValueError('Policy changed outside experiment')
    archive=REPORT/args.archive_name/'policies_before_visual_retune.zip'
    archive.parent.mkdir(parents=True,exist_ok=True)
    if archive.exists():raise ValueError('Visual retune snapshot already exists; do not silently retune twice')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3)as z:
        for path in policies:z.write(path,path.relative_to(POLICIES))
    records=[]
    allowed={'screen_error_weight','policy_version','strategy','runtime_visual_quality_rule'}
    for path in policies:
        rows=read_csv(path);updated=[]
        for old in rows:
            r=dict(old);p=int(r['lod_priority'].split('_')[0][1:]);e=json.loads(r['geometry_evidence_json'])
            r.update(screen_error_weight=runtime_weight(p,e),policy_version=VERSION,
                     strategy=VERSION+'_'+r['preservation_profile'],
                     runtime_visual_quality_rule='base profile * (1 + 0.1 salience) * (1 + 7 sensitivity salience + 3 salience^2)')
            assert all(str(old[k])==str(r[k])for k in old if k not in allowed)
            updated.append(r)
        before=file_hash(path);write_csv(path,updated)
        records.append(dict(path=str(path),before_sha256=before,after_sha256=file_hash(path)))
    proof['outputs']={str(p.resolve()):file_hash(p)for p in policies}
    for path in [ROOT/'tools/pointclipv2/adaptive_p10_policy.py',ROOT/'src/scene/scene_semantic_lod.cpp',Path(__file__)]:
        proof['inputs'][str(path.resolve())]=file_hash(path)
    if 'geometry_evaluation_reuse'in proof:proof.setdefault('geometry_evaluation_reuse_history',[]).append(proof['geometry_evaluation_reuse'])
    proof['geometry_evaluation_reuse']=dict(reason='Only runtime screen weight and version tags changed; all simplifier requests/error limits/geometry unchanged',
        archive=str(archive),archive_sha256=file_hash(archive),changed_columns=sorted(allowed),records=records)
    proof_path.write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
    status=json.loads((REPORT/'run_status.json').read_text());status['version']=VERSION
    (REPORT/'run_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Retuned {len(policies)} complete policies; geometry witnesses preserved')


if __name__=='__main__':main()
