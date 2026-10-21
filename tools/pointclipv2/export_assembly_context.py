"""Validate all six scenes, then publish explicit visual and assembly/CAD policies."""
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from audit_named_scene import read_csv
from artifact_cache import file_hash
from run_pointclipv2_zeroshot_glb import read_glb_chunks

ROOT=Path(__file__).resolve().parents[2]
SUFFIXES={'v3_N':'1_pointnext','v3_C':'2_pointclip','v3_NG':'3_pointnext_structure',
    'v3_CG':'4_pointclip_structure','v3_NC':'5_pointnext_pointclip','v3_NCG':'6_pointnext_pointclip_structure',
    'v3_NCGHM':'7_pointnext_pointclip_structure_metadata'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--round',required=True);p.add_argument('--report-dir',type=Path,required=True)
    p.add_argument('--models',nargs='+',default=list('abcdef'))
    p.add_argument('--destination',type=Path,default=ROOT/'lod_analysis_outputs')
    args=p.parse_args()
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    proof=json.loads((args.report_dir/'provenance.json').read_text(encoding='utf-8'))
    for section in ('inputs','policy_outputs'):
        for path,expected in proof[section].items():
            if file_hash(Path(path))!=expected:raise ValueError('Stale experiment input/output: '+path)
    records=[]
    for name in args.models:
        source=manifest[name];glb=Path(source['glb'])
        if file_hash(glb)!=source['glb_sha256']:raise ValueError('Changed scene '+name)
        data,_,_=read_glb_chunks(glb)
        expected={(str(i),str(n['mesh']))for i,n in enumerate(data['nodes'])if 'mesh'in n}
        for variant,suffix in SUFFIXES.items():
            path=args.output_root/name/args.round/(variant+'.csv')
            rows=read_csv(path)
            if len(rows)!=len(expected)or {(r['node_index'],r['mesh_index'])for r in rows}!=expected:
                raise ValueError('Partial or duplicate policy: '+str(path))
            for r in rows:
                if r['allow_geometry_drop'].lower()!='false':raise ValueError('Unexpected geometry removal')
                ratios=[float(r['target_ratio_'+level])for level in ('near','mid','far')]
                if not 1>=ratios[0]>=ratios[1]>=ratios[2]>0:raise ValueError('Invalid ratio schedule')
                if not 0<=float(r['confidence'])<=1:raise ValueError('Invalid evidence reliability')
                if variant!='v3_NCGHM'and r['leaf_names_used'].lower()!='false':raise ValueError('Leaf-name leakage')
                if variant in {'v3_C','v3_CG'}and 'source_pointnext_class'in r:raise ValueError('Supervised evidence leakage')
            records.append((path,args.destination/f'{name}_lod_constraints_{suffix}.csv',name,variant))
        for variant,suffix in [('v3_NCG','fused'),('v3_NCGHM','metadata'),('v3_NCGH','assembly_tree_only'),('v3_G','geometry_only')]:
            records.append((args.output_root/name/args.round/(variant+'.csv'),args.destination/f'{name}_lod_constraints_{suffix}.csv',name,variant))
    backup=ROOT/'.artifacts/migrations'/('assembly_context_v3_export_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    args.destination.mkdir(parents=True,exist_ok=True)
    exports=[]
    for source,destination,name,variant in records:
        if destination.exists()and file_hash(destination)!=file_hash(source):
            backup.mkdir(parents=True,exist_ok=True);shutil.copy2(destination,backup/destination.name)
        temp=destination.with_suffix('.tmp');shutil.copyfile(source,temp);temp.replace(destination)
        exports.append(dict(scene=name,variant=variant,source=str(source.resolve()),destination=str(destination.resolve()),sha256=file_hash(destination)))
    status=dict(status='assembly_context_v3_policies_exported',round=args.round,exports=exports,
        original_files_backup=str(backup)if backup.exists()else None,
        mode_6='visual_only_NCG',mode_7='NCGHM_assembly_and_CAD_weak_evidence',
        renderer_cache_status='rebuild_required_not_executed',actual_accuracy_measured=False)
    (args.destination/'assembly_context_v3_deployment.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.destination/'deployment_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Exported {len(exports)} scene policies; select mode 7 and rebuild LOD for assembly/CAD strategy',flush=True)


if __name__=='__main__':main()
