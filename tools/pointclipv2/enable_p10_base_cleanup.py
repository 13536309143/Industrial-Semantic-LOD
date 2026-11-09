"""Add a bounded finest-level cleanup; preserve original-source diagnostic scope."""
import json
from pathlib import Path
from adaptive_p10_policy import base_controls
from audit_named_scene import read_csv,write_csv
from artifact_cache import file_hash

ROOT=Path(__file__).resolve().parents[2]


def main():
    report=ROOT/'analysis_reports/adaptive_p10_v4_20261020'
    proof_path=report/'provenance.json';proof=json.loads(proof_path.read_text(encoding='utf-8'))
    root=ROOT/'lod_analysis_outputs/adaptive_p10_v4_20261020'
    records=[]
    for path in sorted(root.glob('*/*.csv')):
        before=file_hash(path)
        if before!=proof['outputs'][str(path.resolve())]:raise ValueError('Untracked policy change')
        rows=read_csv(path)
        if any('base_simplify_ratio'in r for r in rows):raise ValueError('Base cleanup already present')
        for r in rows:
            p=int(r['lod_priority'].split('_')[0][1:]);r.update(base_controls(p,json.loads(r['geometry_evidence_json'])))
        write_csv(path,rows)
        records.append(dict(path=str(path),before_sha256=before,after_sha256=file_hash(path)))
    proof['outputs']={str(p.resolve()):file_hash(p)for p in root.glob('*/*.csv')}
    proof['base_cleanup_extension']=dict(records=records,
        diagnostic_scope='Existing single_mesh_geometry_nodes.csv evaluates original-source far simplification without feature locks or finest-level cleanup; real application tests include cleanup',
        quality_scope='incremental bounded error, original feature locks, component/extent checks; no deletion')
    for path in [Path(__file__),ROOT/'tools/pointclipv2/adaptive_p10_policy.py',ROOT/'src/scene/scene_semantic_lod.cpp',
                 ROOT/'src/meshlod/meshlod_build.h',ROOT/'src/meshlod/meshlod_types.h']:
        proof['inputs'][str(path.resolve())]=file_hash(path)
    proof_path.write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Added bounded source cleanup to {len(records)} complete policies',flush=True)


if __name__=='__main__':main()
