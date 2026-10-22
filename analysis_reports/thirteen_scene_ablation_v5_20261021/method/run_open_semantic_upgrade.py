"""One-command full inference, unified fusion and comparison for the upgraded branch."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from run_scene_semantic_comparison import ROOT, environment, find_clip_python
sys.path.insert(0, str(ROOT / 'tools/pointnext_lod'))
from artifact_cache import files_identity

MODELS = ['透气阀产线','汽车链接器植入-折弯设备','手机喇叭检测线','K31自动铆压机']


def verified(stamp):
    try:
        data = json.loads(stamp.read_text(encoding='utf-8'))
        return data['inputs'] == files_identity(data['inputs']) and data['outputs'] == files_identity(data['outputs'])
    except (OSError, ValueError, KeyError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',default=MODELS)
    parser.add_argument('--output-root',type=Path,default=ROOT/'lod_analysis_outputs/open_semantic_v2_20261019')
    parser.add_argument('--baseline-root',type=Path,default=ROOT/'lod_analysis_outputs/four_scenes_seed2027_20261019')
    parser.add_argument('--report-dir',type=Path,default=ROOT/'analysis_reports/open_semantic_v2_20261019')
    parser.add_argument('--pointclip-python',type=Path)
    parser.add_argument('--no-export',action='store_true',help='Keep results for analysis without exporting renderer CSV files.')
    args = parser.parse_args()
    args.output_root.mkdir(parents=True,exist_ok=True)
    args.report_dir.mkdir(parents=True,exist_ok=True)
    clip_python = args.pointclip_python or find_clip_python()
    status = {'status':'running','models':args.models,'stages':[]}
    def save():
        (args.output_root/'upgrade_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    def execute(scene,stage,python,command):
        start = time.perf_counter()
        log = args.output_root/f'{scene}_{stage}.log'
        print(f'{scene}: {stage}',flush=True)
        with log.open('wb') as stream:
            result = subprocess.run([str(python),'-u',*map(str,command)],cwd=ROOT,env=environment(python),stdout=stream,stderr=subprocess.STDOUT)
        status['stages'].append({'scene':scene,'stage':stage,'returncode':result.returncode,'wall_seconds':time.perf_counter()-start,'log':str(log)})
        save()
        if result.returncode:
            raise RuntimeError(f'{stage} failed, see {log}')
    failed=[]
    for name in args.models:
        try:
            glb=ROOT/'_downloaded_resources'/(name+'.glb')
            out=args.output_root/name
            pn=args.baseline_root/name/(name+'_pointnext_analysis.csv')
            if not verified(args.baseline_root/name/(name+'_pointnext_provenance.json')):
                closed=out/'closed'
                pointroot=ROOT/'tools/pointnext_lod/PointNeXt'
                execute(name,'pointnext',Path(sys.executable),[ROOT/'tools/pointnext_lod/batch_analyze_glb_lod.py',
                    '--resources-dir',glb.parent,'--models',glb.name,'--output-dir',closed,
                    '--pointnext-root',pointroot,'--cfg',pointroot/'cfgs/industrial_part/pointnext-s-14class-encoder-compare-inference.yaml',
                    '--ckpt',ROOT/'models/pointnext/industrial-part-14class-best.pth',
                    '--classes-file',ROOT/'tools/pointnext_lod/industrial_part_14classes.txt','--pointnext-only','--skip-existing'])
                pn=closed/(name+'_pointnext_analysis.csv')
            if not verified(out/'prediction.provenance.json'):
                execute(name,'open_inference',clip_python,[ROOT/'tools/pointclipv2/run_open_semantic_v2.py','--glb',glb,'--output-dir',out])
            else:
                print(name+': reuse verified open predictions',flush=True)
            execute(name,'surface_views',clip_python,[ROOT/'tools/pointclipv2/run_open_shading_v2.py','--glb',glb,'--output-dir',out])
            if not verified(out/'combined_prediction.provenance.json'):
                execute(name,'combine_views',Path(sys.executable),[ROOT/'tools/pointclipv2/combine_open_renderings.py','--glb',glb,'--output-dir',out])
            execute(name,'fusion',Path(sys.executable),[ROOT/'tools/pointclipv2/fuse_open_semantic_v2.py',
                '--glb',glb,'--pointnext-csv',pn,'--open-csv',out/'open_semantic_combined.csv','--output-dir',out/'fusion'])
        except Exception as exc:
            failed.append({'scene':name,'error':str(exc)})
            print('SKIP failed scene: '+str(exc),flush=True)
    status['failed']=failed
    status['status']='partial_failure' if failed else 'inference_and_fusion_complete'
    save()
    if not failed:
        try:
            execute('all','comparison',Path(sys.executable),[ROOT/'tools/pointclipv2/compare_open_semantic_upgrade.py',
                '--models',*args.models,'--run-root',args.output_root,'--baseline-root',args.baseline_root,'--report-dir',args.report_dir])
            if not args.no_export:
                execute('all','export',Path(sys.executable),[ROOT/'tools/pointclipv2/export_open_semantic_v2.py',
                    '--models',*args.models,'--run-root',args.output_root])
        except Exception as exc:
            status.update(status='postprocess_failed',error=str(exc))
            save()
            return 1
        status['status']='complete'
        status['report']=str(args.report_dir/'开放语义分支升级与对照分析.md')
        save()
    return 1 if failed else 0


if __name__=='__main__':
    raise SystemExit(main())
