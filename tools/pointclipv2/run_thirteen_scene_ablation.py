"""All-node evidence acquisition for numbered GLBs and frozen V5 ablations.

Serial GPU jobs; hash-verified reuse survives byte-identical GLB renames.
Historical results and deployed viewer policies are never overwritten.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from run_scene_semantic_comparison import ROOT, environment, find_clip_python
sys.path.insert(0, str(ROOT / 'tools/pointnext_lod'))
from artifact_cache import file_hash

DEFAULT_OUTPUT = ROOT / 'lod_analysis_outputs/thirteen_scene_ablation_v5_20261021'
DEFAULT_REPORT = ROOT / 'analysis_reports/thirteen_scene_ablation_v5_20261021'
_hashes = {}


def digest(path):
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _hashes:
        _hashes[key] = file_hash(path)
    return _hashes[key]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    # Windows readers (including IDE previews) may briefly deny file deletion.
    # Retry the atomic publication rather than skipping a successful model.
    for attempt in range(100):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt==99:raise
            time.sleep(.05)


def valid_stamp(path, scene_hash):
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('parameters',{}).get('limit') is not None:
            return False
        scene_inputs = [v for k, v in data['inputs'].items() if k.lower().endswith('.glb')]
        if not scene_inputs or any(v != scene_hash for v in scene_inputs):
            return False
        for section in ('inputs', 'outputs'):
            for name, expected in data[section].items():
                if section == 'inputs' and name.lower().endswith('.glb'):
                    continue
                if digest(Path(name)) != expected:
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', nargs='+', type=int, default=list(range(1, 14)))
    p.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--report-dir', type=Path, default=DEFAULT_REPORT)
    p.add_argument('--pointclip-python', type=Path)
    p.add_argument('--inference-only', action='store_true')
    p.add_argument('--policies-only', action='store_true')
    args = p.parse_args()
    if len(set(args.models)) != len(args.models) or any(i not in range(1, 14) for i in args.models):
        p.error('models must be unique integers from 1 to 13')
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / 'scene_sources.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    status_path = args.report_dir / 'run_status.json'
    status = dict(status='running', started_at_utc=datetime.now(timezone.utc).isoformat(),
                  models=[f'M{i}' for i in args.models], stages=[], failed=[],
                  policy_version='adaptive_p10_v5.0', deployment_modified=False)
    scratch = ROOT / '.artifacts/thirteen_scene_ablation/tmp'
    scratch.mkdir(parents=True, exist_ok=True)

    def execute(scene, stage, python, command):
        log = args.output_root / 'logs' / f'{scene}_{stage}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        status.update(current_scene=scene, current_stage=stage)
        save(status_path, status)
        env = environment(python)
        env.update(TMP=str(scratch), TEMP=str(scratch), TMPDIR=str(scratch),
                   MPLCONFIGDIR=str(scratch / 'matplotlib'))
        print(f'{scene}: START {stage}; {log}', flush=True)
        start = time.perf_counter()
        with log.open('wb') as stream:
            result = subprocess.run([str(python), '-u', '-X', 'utf8', *map(str, command)],
                                    cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        status['stages'].append(dict(scene=scene, stage=stage, returncode=result.returncode,
            wall_seconds=time.perf_counter()-start, log=str(log), command=list(map(str, command))))
        save(status_path, status)
        if result.returncode:
            raise RuntimeError(f'{stage} failed ({result.returncode}): {log}')
        print(f'{scene}: DONE {stage}', flush=True)

    if not args.policies_only:
        clip_python = args.pointclip_python or find_clip_python()
        names = json.loads((ROOT / 'analysis_reports/thesis_ch5_setup_20261021/setup_sources.json').read_text(encoding='utf-8'))['model_names']
        checkpoint = ROOT / 'models/pointnext/industrial-part-14class-best.pth'
        weight_hash = digest(checkpoint)
        pn_stamps = list((ROOT / 'lod_analysis_outputs').rglob('*_pointnext_provenance.json'))
        visual_stamps = list((ROOT / 'lod_analysis_outputs').rglob('combined_prediction.provenance.json'))
        print(f'Indexed {len(pn_stamps)} supervised and {len(visual_stamps)} open-evidence manifests', flush=True)
        freeze = args.report_dir / 'method'
        freeze.mkdir(exist_ok=True)
        frozen = {}
        for path in [*Path(__file__).parent.glob('*.py'), *Path(__file__).parent.glob('*.json')]:
            frozen[str(path.resolve())] = digest(path)
            target = freeze / path.name
            if not target.exists():
                shutil.copy2(path, target)
            elif digest(target) != digest(path):
                orchestration={'run_thirteen_scene_ablation.py','analyze_thirteen_scene_ablation.py',
                    'visualize_thirteen_scene_ablation.py','validate_thirteen_scene_ablation.py',
                    'summarize_thirteen_scene_findings.py'}
                if path.name not in orchestration:
                    raise RuntimeError(f'Frozen method changed; use a new output directory: {path}')
                # Preserve the executed orchestration revision; neural inputs and
                # frozen V5 policy sources still require exact identity.
                revision=args.report_dir/'method_revisions'/f'{path.stem}_{digest(target)[:12]}{path.suffix}'
                revision.parent.mkdir(exist_ok=True)
                if not revision.exists():shutil.copy2(target,revision)
                shutil.copy2(path,target)
        save(args.report_dir / 'method_fingerprint.json', frozen)
        for number in args.models:
            scene = f'M{number}'
            try:
                glb = ROOT / '_downloaded_resources' / f'{number}.glb'
                scene_hash = digest(glb)
                folder = args.output_root / scene
                folder.mkdir(exist_ok=True)
                pn = None
                pn_source = None
                for stamp in pn_stamps:
                    proof = json.loads(stamp.read_text(encoding='utf-8'))
                    if proof.get('inputs', {}).get(str(checkpoint.resolve())) != weight_hash:
                        continue
                    expected_parameters={'num_points':8192,'seed':2026,'inference_mode':'mesh'}
                    if any(proof.get('parameters',{}).get(k)!=v for k,v in expected_parameters.items()):
                        continue
                    if valid_stamp(stamp, scene_hash):
                        pn = next((Path(k) for k in proof['outputs'] if k.endswith('_pointnext_analysis.csv')), None)
                        if pn:
                            pn_source = stamp
                            break
                if pn is None:
                    pointroot = ROOT / 'tools/pointnext_lod/PointNeXt'
                    execute(scene, 'supervised', Path(sys.executable), [ROOT/'tools/pointnext_lod/batch_analyze_glb_lod.py',
                        '--resources-dir', glb.parent, '--models', glb.name, '--output-dir', folder/'closed',
                        '--pointnext-root', pointroot, '--cfg', pointroot/'cfgs/industrial_part/pointnext-s-14class-encoder-compare-inference.yaml',
                        '--ckpt', checkpoint, '--classes-file', ROOT/'tools/pointnext_lod/industrial_part_14classes.txt',
                        '--num-points', '8192', '--batch-size', '16', '--pointnext-only', '--skip-existing'])
                    pn = folder/'closed'/f'{number}_pointnext_analysis.csv'
                    pn_source = pn.with_name(f'{number}_pointnext_provenance.json')
                visual = None
                for stamp in visual_stamps:
                    candidate = stamp.parent
                    if (valid_stamp(stamp, scene_hash)
                        and valid_stamp(candidate/'prediction.provenance.json', scene_hash)
                        and valid_stamp(candidate/'cache/views.provenance.json', scene_hash)
                        and valid_stamp(candidate/'cache/shaded_views.provenance.json', scene_hash)):
                        visual = candidate
                        break
                visual_reused = visual is not None
                if visual is None:
                    visual = folder/'visual'
                    if not valid_stamp(visual/'prediction.provenance.json', scene_hash):
                        execute(scene, 'open_depth', clip_python, [ROOT/'tools/pointclipv2/run_open_semantic_v2.py', '--glb', glb, '--output-dir', visual])
                    execute(scene, 'open_surface', clip_python, [ROOT/'tools/pointclipv2/run_open_shading_v2.py', '--glb', glb, '--output-dir', visual])
                    if not valid_stamp(visual/'combined_prediction.provenance.json', scene_hash):
                        execute(scene, 'combine', Path(sys.executable), [ROOT/'tools/pointclipv2/combine_open_renderings.py', '--glb', glb, '--output-dir', visual])
                manifest[scene] = dict(glb=str(glb), glb_sha256=scene_hash, title=names[scene],
                    pointnext_csv=str(pn), pointnext_sha256=digest(pn), pointnext_provenance=str(pn_source),
                    checkpoint_sha256=weight_hash, open_csv=str(visual/'open_semantic_combined.csv'),
                    open_sha256=digest(visual/'open_semantic_combined.csv'), visual_directory=str(visual),
                    open_evidence_reused=visual_reused, evaluation_split='scene_diagnostics_no_ground_truth')
                save(manifest_path, manifest)
                status['evidence_ready'] = list(manifest)
                save(status_path, status)
                print(f'{scene}: ALL-NODE EVIDENCE READY; open_reused={visual_reused}', flush=True)
            except Exception as exc:
                status['failed'].append(dict(scene=scene, error=str(exc)))
                save(status_path, status)
                print(f'SKIP {scene}: {exc}', flush=True)
    if not args.inference_only and not status['failed']:
        try:
            execute('ALL', 'v5_ablation', Path(sys.executable), [Path(__file__).with_name('analyze_thirteen_scene_ablation.py'),
                '--manifest', manifest_path, '--output-root', args.output_root, '--report-dir', args.report_dir,
                '--models', *[f'M{i}' for i in args.models]])
        except Exception as exc:
            status['failed'].append(dict(scene='ALL', error=str(exc)))
    status.update(status='partial_failure' if status['failed'] else 'inference_complete' if args.inference_only else 'complete',
                  completed_at_utc=datetime.now(timezone.utc).isoformat())
    save(status_path, status)
    return bool(status['failed'])


if __name__ == '__main__':
    raise SystemExit(main())
