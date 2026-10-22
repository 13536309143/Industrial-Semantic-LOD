"""Export verified upgraded policies using the renderer's existing CSV names."""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/pointnext_lod'))
from artifact_cache import file_hash,files_identity

SUFFIXES={'N':'1_pointnext','C':'2_pointclip','NG':'3_pointnext_structure','CG':'4_pointclip_structure',
          'NC':'5_pointnext_pointclip','NCG':'6_pointnext_pointclip_structure'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--models',nargs='+',required=True)
    parser.add_argument('--destination',type=Path,default=ROOT/'lod_analysis_outputs')
    args=parser.parse_args()
    args.destination.mkdir(parents=True,exist_ok=True)
    backup=ROOT/'.artifacts/migrations'/('open_semantic_v2_export_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    records=[]
    # Validate every scene before publishing any replacement.
    for name in args.models:
        folder=args.run_root/name
        provenance=json.loads((folder/'fusion/provenance.json').read_text(encoding='utf-8'))
        if files_identity(provenance['inputs'])!=provenance['inputs'] or files_identity(provenance['outputs'])!=provenance['outputs']:
            raise ValueError('Stale/incomplete policy source: '+name)
        glb=ROOT/'_downloaded_resources'/(name+'.glb')
        if provenance['inputs'][str(glb.resolve())]!=file_hash(glb):raise ValueError('Scene changed: '+name)
        for mode,suffix in SUFFIXES.items():
            records.append((folder/'fusion'/f'lod_constraints_{mode}.csv',args.destination/(name+'_lod_constraints_'+suffix+'.csv'),name,mode))
        records.append((folder/'fusion/lod_constraints_NCG.csv',args.destination/(name+'_lod_constraints_fused.csv'),name,'NCG'))
        records.append((folder/'fusion/lod_constraints_NCGM.csv',args.destination/(name+'_lod_constraints_metadata.csv'),name,'NCGM_explicit_auxiliary'))
        records.append((folder/'fusion/lod_constraints_NCGM.csv',args.destination/(name+'_lod_constraints_7_pointnext_pointclip_structure_metadata.csv'),name,'NCGM_explicit_auxiliary'))
    manifest=[]
    for source,destination,name,mode in records:
        if destination.exists() and file_hash(destination)!=file_hash(source):
            backup.mkdir(parents=True,exist_ok=True)
            shutil.copy2(destination,backup/destination.name)
        temporary=destination.with_suffix('.tmp')
        shutil.copyfile(source,temporary)
        temporary.replace(destination)
        manifest.append({'scene':name,'mode':mode,'source':str(source.resolve()),'destination':str(destination.resolve()),'sha256':file_hash(destination)})
    status={'status':'open_semantic_v2_policies_exported','exports':manifest,'backup_directory':str(backup)if backup.exists()else None,
            'default_renderer_mode_6':'NCG_visual_only','metadata_policy':'Explicit seventh UI strategy (zero-based index 6); not silently added to the six modes.',
            'renderer_lod_cache_status':'rebuild_required_not_executed','accuracy_validated':False}
    (args.destination/'open_semantic_v2_deployment.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.destination/'deployment_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Exported',len(manifest),'policy CSV files; reopen/rebuild scene LOD to consume them')


if __name__=='__main__':main()
