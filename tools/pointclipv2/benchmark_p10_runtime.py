"""Isolated real application builds and fixed-view Vulkan benchmarks.

Hard links reuse input data; private CSVs/caches never replace a user's cache.
"""
import argparse
import ctypes
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from audit_named_scene import write_csv
from artifact_cache import file_hash

ROOT = Path(__file__).resolve().parents[2]
SUFFIX = '_lod_constraints_7_pointnext_pointclip_structure_metadata.csv'


def accepted_benchmark_hashes(report):
    """Preserve the exact measured binary when only final UI legend text changed."""
    hashes={file_hash(ROOT/'_bin/Release/t1.exe')}
    stamp=report/'benchmarked_binary.json'
    if stamp.exists():
        proof=json.loads(stamp.read_text(encoding='utf-8'))
        unchanged=all(Path(p).is_file()and file_hash(Path(p))==h for p,h in proof['backend_hashes'].items())
        ui_ok=file_hash(ROOT/'src/app/lodclusters_ui.cpp')==proof.get('approved_final_ui_sha256')
        if unchanged and ui_ok and file_hash(Path(proof['path']))==proof['sha256']:hashes.add(proof['sha256'])
    return hashes


class ProcessMemory(ctypes.Structure):
    _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(name,ctypes.c_size_t)for name in
        ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
         'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]


class MemoryStatus(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(name,ctypes.c_ulonglong)for name in
        ('totalPhysical','availablePhysical','totalPagefile','availablePagefile','totalVirtual','availableVirtual','extended')]


def run_process(command, folder, log, timeout=900):
    started = time.perf_counter()
    peak_rss = peak_private = 0
    reason = ''
    with log.open('wb') as out:
        p = subprocess.Popen(command, cwd=folder, stdout=out, stderr=subprocess.STDOUT,
                             creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        while p.poll() is None:
            mem=ProcessMemory();mem.cb=ctypes.sizeof(mem)
            if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(int(p._handle)),ctypes.byref(mem),mem.cb):
                peak_rss=max(peak_rss,mem.PeakWorkingSetSize)
                peak_private=max(peak_private,mem.PrivateUsage)
            available=MemoryStatus();available.length=ctypes.sizeof(available)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(available))
            if time.perf_counter()-started > timeout or available.availablePhysical < 2*1024**3:
                reason = 'timeout_or_low_system_memory'
                p.kill()
                p.wait()
                break
            time.sleep(.1)
    return dict(exit_code=p.returncode, seconds=time.perf_counter()-started,
                peak_working_set_bytes=peak_rss, peak_private_bytes=peak_private, failure=reason)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', required=True)
    p.add_argument('--models', nargs='+', default=list('abcdef'))
    p.add_argument('--policy-root', type=Path, required=True)
    p.add_argument('--policy-file', default='v3_NCGHM.csv')
    p.add_argument('--render', action='store_true')
    p.add_argument('--trials', type=int, default=1)
    p.add_argument('--capture-only', action='store_true', help='Reuse verified private caches and save fixed-view images')
    p.add_argument('--reference', action='store_true', help='Capture at a very small screen error for a source-level reference')
    p.add_argument('--render-only',action='store_true',help='Reuse private cache; do not rebuild geometry')
    p.add_argument('--lod-error',type=float,default=2.)
    p.add_argument('--label',default='')
    p.add_argument('--capture',action='store_true')
    p.add_argument('--compressed',type=int,choices=(0,1),default=0)
    p.add_argument('--ext-mesh',type=int,choices=(0,1),default=None)
    p.add_argument('--unified',action='store_true')
    p.add_argument('--report-dir', type=Path, default=ROOT/'analysis_reports/adaptive_p10_v4_20261020')
    args = p.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path=ROOT/'lod_analysis_outputs/assembly_context_v3_20261020/scene_sources.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    exe=ROOT/'_bin/Release/t1.exe'
    records=[]
    for scene in args.models:
        source=manifest[scene]; glb=Path(source['glb'])
        if file_hash(glb)!=source['glb_sha256']: raise ValueError('Changed input scene')
        policy=args.policy_root/scene/args.policy_file
        if not policy.is_file(): raise FileNotFoundError(policy)
        folder=ROOT/'.artifacts/p10_runtime'/args.variant/scene
        folder.mkdir(parents=True,exist_ok=True)
        link=folder/(scene+'.glb')
        if not link.exists(): os.link(glb,link)
        if file_hash(link)!=source['glb_sha256']: raise ValueError('Different hardlinked scene')
        local_policy=folder/(scene+SUFFIX)
        reuse=args.capture_only or args.render_only
        if reuse and (not local_policy.exists()or file_hash(local_policy)!=file_hash(policy)):
            raise ValueError('Capture policy differs from cache build input')
        if not reuse:shutil.copyfile(policy,local_policy)
        cache=folder/(scene+'.glb.p10bench')
        command=[str(exe),'--scene',str(link),'--semanticlodmode','6','--cachesuffix','.p10bench',
                 '--processingthreadpct','.15','--autoloadcache','0','--processingonly','1','--compressed',str(args.compressed)]
        record=dict(scene=scene,variant=args.variant,stage='preprocess',policy_sha256=file_hash(policy),
                    scene_sha256=source['glb_sha256'],exe_sha256=file_hash(exe),compressed=bool(args.compressed))
        print(f'{scene} {args.variant}: '+('reuse private cache'if reuse else 'real cluster hierarchy build'),flush=True)
        log=folder/'preprocess.log'
        if reuse:
            previous=json.loads((args.report_dir/f'runtime_{args.variant}.json').read_text(encoding='utf-8'))
            original=next(r for r in previous if r['scene']==scene and r['stage']=='preprocess')
            if original['policy_sha256']!=file_hash(policy)or original['exe_sha256']!=file_hash(exe)or not cache.is_file():
                raise ValueError('Stale benchmark cache')
            record.update(original)
        else:record.update(run_process(command,folder,log))
        text=log.read_text(encoding='utf-8',errors='replace')
        policy_loaded=str(folder/(scene+SUFFIX)).lower()in text.lower()
        record['success']=record['exit_code']==0 and 'endProcessOnlySave completed successfully'in text and cache.is_file()and policy_loaded
        record['expected_policy_loaded']=policy_loaded
        record['cache_bytes']=cache.stat().st_size if cache.is_file()else None
        record['log']=str(log)
        records.append(record)
        if (args.render or args.capture_only)and record['success']:
            for trial in range(args.trials):
                render_command=[str(exe),'--scene',str(link),'--semanticlodmode','6','--cachesuffix','.p10bench',
                    '--compressed',str(args.compressed),
                    '--autoloadcache','1','--autosavecache','0','--mappedcache','1','--headless',
                    '--headlessframes','100000','--headlesswidth','1920','--headlessheight','1080',
                    '--supersample','1','--sequencewaitforscene','1',
                    '--streaming','0','--renderer','0','--renderstats','1',
                    '--shadowray','0','--visualize','0','--facetshading','0',
                    '--sequencestring',f'SEQUENCE "warmup" --loderror {args.lod_error} --sequenceframes 256 --sequenceresetframes 8 --sequenceaverages 0 SEQUENCE "fixed_view{("_"+args.label)if args.label else ""}" --loderror {args.lod_error}']
                if args.capture_only:
                    error='.0001'if args.reference else str(args.lod_error)
                    label='source_reference'if args.reference else 'fixed_view_capture'
                    render_command[-1]=f'SEQUENCE "warmup" --loderror {error} --sequenceframes 64 --sequenceresetframes 8 --sequenceaverages 0 SEQUENCE "{label}" --loderror {error}'
                    render_command.extend(['--sequencescreenshot','2'])
                elif args.capture:render_command.extend(['--sequencescreenshot','2'])
                if args.ext_mesh is not None:render_command.extend(['--extmeshshader',str(args.ext_mesh)])
                if args.unified:render_command.extend(['--separategroups','0'])
                print(f'{scene} {args.variant}: fixed-view Vulkan trial {trial+1}',flush=True)
                rlog=folder/(f'capture_{"reference"if args.reference else "lod"}.log'if args.capture_only else f'render_{args.label+"_"if args.label else ""}{trial+1}.log')
                r=dict(scene=scene,variant=args.variant,stage='capture'if args.capture_only else 'render',trial=trial+1,
                       policy_sha256=file_hash(policy),scene_sha256=source['glb_sha256'],exe_sha256=file_hash(exe),compressed=bool(args.compressed))
                r.update(run_process(render_command,folder,rlog))
                contents=rlog.read_text(encoding='utf-8',errors='replace')
                r.update(log=str(rlog),success=r['exit_code']==0 and 'MemoryReport 1'in contents,
                         sequence_wait_for_scene=True,semantic_mode_index=6,screen_error=.0001 if args.reference else args.lod_error,streaming=False,
                         ext_mesh=args.ext_mesh,unified_traversal=args.unified)
                memory=re.findall(r'Geometry;\s*(\d+);\s*(\d+);',contents)
                if memory:r.update(geometry_gpu_actual_bytes=int(memory[-1][0]),geometry_gpu_reserved_bytes=int(memory[-1][1]))
                records.append(r)
        columns=list(dict.fromkeys(k for r in records for k in r))
        prefix=('reference_'if args.reference else 'capture_')if args.capture_only else 'runtime_'
        label=('_'+args.label)if args.label else ''
        write_csv(args.report_dir/f'{prefix}{args.variant}{label}.csv',[{k:r.get(k,'')for k in columns}for r in records])
        (args.report_dir/f'{prefix}{args.variant}{label}.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    if not all(r['success']for r in records):return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
