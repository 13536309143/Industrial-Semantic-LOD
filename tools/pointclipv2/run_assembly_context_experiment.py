"""Six-scene inference and reproducible assembly-aware policy experiments."""
import argparse
import json
import subprocess
import shutil
import sys
import time
from pathlib import Path

from run_scene_semantic_comparison import ROOT, environment, find_clip_python
sys.path.insert(0, str(ROOT / 'tools/pointnext_lod'))
from artifact_cache import file_hash


def verified(stamp, scene_hash=None):
    try:
        value = json.loads(stamp.read_text(encoding='utf-8'))
        if scene_hash and scene_hash not in [v for k, v in value['inputs'].items() if k.lower().endswith('.glb')]:
            return False
        return all((section == 'inputs' and k.lower().endswith('.glb') and scene_hash == v)
                   or (Path(k).is_file() and file_hash(Path(k)) == v)
                   for section in ('inputs', 'outputs') for k, v in value[section].items())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', default=list('abcdef'))
    parser.add_argument('--output-root', type=Path, default=ROOT / 'lod_analysis_outputs/assembly_context_v3_20261020')
    parser.add_argument('--report-dir', type=Path, default=ROOT / 'analysis_reports/assembly_context_v3_20261020')
    parser.add_argument('--inference-only', action='store_true')
    parser.add_argument('--policies-only', action='store_true')
    parser.add_argument('--no-export', action='store_true')
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    status_path = args.output_root / 'experiment_status.json'
    manifest_path = args.output_root / 'scene_sources.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    status = {'status': 'running', 'models': args.models, 'stages': [], 'failed': []}

    def execute(scene, stage, python, command):
        log = args.output_root / f'{scene}_{stage}.log'
        status.update(current_scene=scene, current_stage=stage)
        write_json(status_path, status)
        print(f'{scene}: {stage}', flush=True)
        started = time.perf_counter()
        with log.open('wb') as stream:
            result = subprocess.run([str(python), '-u', *map(str, command)], cwd=ROOT,
                                    env=environment(python), stdout=stream, stderr=subprocess.STDOUT)
        status['stages'].append(dict(scene=scene, stage=stage, returncode=result.returncode,
                                     wall_seconds=time.perf_counter()-started, log=str(log)))
        write_json(status_path, status)
        if result.returncode:
            raise RuntimeError(f'{stage} failed; see {log}')

    if not args.policies_only:
        clip_python = find_clip_python()
        # A byte-identical renamed GLB has identical node/mesh keys. Keep original provenance intact.
        search_root = ROOT / 'lod_analysis_outputs'
        pn_stamps = list(search_root.rglob('*_pointnext_provenance.json'))
        visual_stamps = list((search_root / 'open_semantic_v2_20261019').glob('*/combined_prediction.provenance.json'))
        for name in args.models:
            try:
                glb = ROOT / '_downloaded_resources' / (name+'.glb')
                digest = file_hash(glb)
                pn = None
                for stamp in pn_stamps:
                    if verified(stamp, digest):
                        outputs = json.loads(stamp.read_text(encoding='utf-8'))['outputs']
                        pn = next((Path(k) for k in outputs if k.endswith('_pointnext_analysis.csv')), None)
                        if pn: break
                folder = args.output_root / name
                if pn is None:
                    pointroot = ROOT / 'tools/pointnext_lod/PointNeXt'
                    execute(name, 'pointnext', Path(sys.executable), [ROOT/'tools/pointnext_lod/batch_analyze_glb_lod.py',
                        '--resources-dir', glb.parent, '--models', glb.name, '--output-dir', folder/'closed',
                        '--pointnext-root', pointroot, '--cfg', pointroot/'cfgs/industrial_part/pointnext-s-14class-encoder-compare-inference.yaml',
                        '--ckpt', ROOT/'models/pointnext/industrial-part-14class-best.pth',
                        '--classes-file', ROOT/'tools/pointnext_lod/industrial_part_14classes.txt',
                        '--num-points', '8192', '--batch-size', '16', '--pointnext-only', '--skip-existing'])
                    pn = folder/'closed'/(name+'_pointnext_analysis.csv')
                visual = next((p.parent for p in visual_stamps if verified(p, digest)
                               and verified(p.parent/'prediction.provenance.json', digest)
                               and verified(p.parent/'cache/views.provenance.json',digest)
                               and verified(p.parent/'cache/shaded_views.provenance.json',digest)), None)
                if visual is None:
                    visual = folder/'visual'
                    if not verified(visual/'prediction.provenance.json', digest):
                        execute(name, 'open_depth', clip_python, [ROOT/'tools/pointclipv2/run_open_semantic_v2.py', '--glb', glb, '--output-dir', visual])
                    execute(name, 'open_surface', clip_python, [ROOT/'tools/pointclipv2/run_open_shading_v2.py', '--glb', glb, '--output-dir', visual])
                    if not verified(visual/'combined_prediction.provenance.json', digest):
                        execute(name, 'combine', Path(sys.executable), [ROOT/'tools/pointclipv2/combine_open_renderings.py', '--glb', glb, '--output-dir', visual])
                manifest[name] = dict(glb=str(glb), glb_sha256=digest, pointnext_csv=str(pn),
                    pointnext_sha256=file_hash(pn), open_csv=str(visual/'open_semantic_combined.csv'),
                    open_sha256=file_hash(visual/'open_semantic_combined.csv'), visual_directory=str(visual),
                    reused_identical_scene=not visual.is_relative_to(args.output_root),
                    evaluation_split='development_diagnostics')
                write_json(manifest_path, manifest)
                print(f'{name}: full evidence ready', flush=True)
            except Exception as exc:
                status['failed'].append(dict(scene=name, error=str(exc)))
                print(f'SKIP {name}: {exc}', flush=True)
    if not status['failed'] and not args.inference_only:
        try:
            for value in manifest.values():value['evaluation_split']='development_diagnostics'
            write_json(manifest_path,manifest)
            common=['--manifest',manifest_path,'--output-root',args.output_root,'--models',*args.models]
            execute('all', 'policy_experiments', Path(sys.executable), [ROOT/'tools/pointclipv2/experiment_assembly_context.py',
                *common,'--report-dir',args.report_dir,'--round','r4_final_policy','--refinement','scoped','--no-export'])
            cmake=shutil.which('cmake')or str(Path('C:/CMake/bin/cmake.exe'))
            # Building this CPU probe also detects stale native source; no GUI or renderer is launched.
            if not (ROOT/'build/CMakeCache.txt').exists():
                raise RuntimeError('Configure the project build directory (see README) before running geometry experiments')
            execute('all','build_geometry_probe',Path(sys.executable),['-c',
                'import subprocess,sys;sys.exit(subprocess.call(sys.argv[1:]))',cmake,'--build',ROOT/'build',
                '--config','Release','--target','semantic_geometry_probe','--parallel','2'])
            execute('all','geometry_proxy',Path(sys.executable),[ROOT/'tools/pointclipv2/probe_assembly_geometry.py',
                *common,'--report-dir',args.report_dir/'r4_final_policy','--round','r4_final_policy'])
            execute('all','measured_geometry_guard',Path(sys.executable),[ROOT/'tools/pointclipv2/measured_geometry_guard.py',
                *common,'--report-dir',args.report_dir/'final_guarded'])
            execute('all','summary',Path(sys.executable),[ROOT/'tools/pointclipv2/summarize_assembly_context.py',
                *common,'--report-dir',args.report_dir])
            if not args.no_export:
                execute('all','export',Path(sys.executable),[ROOT/'tools/pointclipv2/export_assembly_context.py',
                    *common,'--report-dir',args.report_dir/'final_guarded','--round','final_guarded'])
        except Exception as exc:
            status.update(status='postprocess_failed',error=str(exc));write_json(status_path,status)
            return 1
    status['status'] = 'partial_failure' if status['failed'] else 'inference_complete' if args.inference_only else 'complete'
    write_json(status_path, status)
    return bool(status['failed'])


if __name__ == '__main__':
    raise SystemExit(main())
