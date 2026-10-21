"""Capture unchanged fixed cameras and compare to a tiny-screen-error reference."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter,sobel,binary_dilation
from audit_named_scene import write_csv
from artifact_cache import file_hash

ROOT=Path(__file__).resolve().parents[2]


def compare(reference,candidate):
    a=np.asarray(Image.open(reference).convert('RGB'),dtype=np.float32)/255
    b=np.asarray(Image.open(candidate).convert('RGB'),dtype=np.float32)/255
    if a.shape!=b.shape:raise ValueError('Different viewports')
    height,width=a.shape[:2]
    # Sample every second pixel for the SSIM diagnostic; report exact evaluated size.
    a=a[::2,::2];b=b[::2,::2]
    x=a.mean(-1);y=b.mean(-1)
    ux=gaussian_filter(x,1.5);uy=gaussian_filter(y,1.5)
    vx=np.maximum(0,gaussian_filter(x*x,1.5)-ux*ux)
    vy=np.maximum(0,gaussian_filter(y*y,1.5)-uy*uy)
    cov=gaussian_filter(x*y,1.5)-ux*uy
    ssim=((2*ux*uy+.01**2)*(2*cov+.03**2))/((ux*ux+uy*uy+.01**2)*(vx+vy+.03**2))
    ea=np.hypot(sobel(x,0),sobel(x,1))>.12
    eb=np.hypot(sobel(y,0),sobel(y,1))>.12
    if np.count_nonzero(ea)<100:raise ValueError('Reference is blank or has too little scene content')
    region=binary_dilation(ea,iterations=3)
    mse=float(np.mean((a-b)**2))
    edge_mse=float(np.mean((a-b)[region]**2))if region.any()else mse
    return dict(width=width,height=height,evaluation_stride=2,
        PSNR_dB=float(-10*np.log10(max(mse,1e-12))),SSIM_mean=float(ssim.mean()),
        reference_edge_region_SSIM=float(ssim[region].mean())if region.any()else None,
        reference_edge_region_PSNR_dB=float(-10*np.log10(max(edge_mse,1e-12))),
        reference_edges_recall_2px=float((ea&binary_dilation(eb,iterations=1)).sum()/max(ea.sum(),1)),
        scope='single fixed view; JPEG captures; sampled image diagnostics, not visibility ground truth or topology proof')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',default=list('abcdef'))
    p.add_argument('--report-dir',type=Path,default=ROOT/'analysis_reports/adaptive_p10_v4_20261020')
    p.add_argument('--supplements-only',action='store_true')
    args=p.parse_args();records=[]
    bench=Path(__file__).with_name('benchmark_p10_runtime.py')
    jobs=[('v3','lod_analysis_outputs/assembly_context_v3_20261020','final_guarded/v3_NCGHM.csv',True),
          ('v3','lod_analysis_outputs/assembly_context_v3_20261020','final_guarded/v3_NCGHM.csv',False),
          ('v4_selected','lod_analysis_outputs/adaptive_p10_v4_20261020','NCG_H1_M1_selected.csv',False)]
    for variant,root,filename,reference in ([]if args.supplements_only else jobs):
        command=[sys.executable,'-u',str(bench),'--variant',variant,'--policy-root',str(ROOT/root),
                 '--policy-file',filename,'--capture-only','--report-dir',str(args.report_dir),'--models',*args.models]
        if reference:command.append('--reference')
        subprocess.run(command,check=True)
    images=args.report_dir/'images';images.mkdir(exist_ok=True)
    import shutil
    for scene in args.models:
        source=ROOT/'.artifacts/p10_runtime/v3'/scene/'screenshot_1_source_reference.jpg'
        destination=images/f'{scene}_source_reference.jpg';shutil.copyfile(source,destination)
        for variant in ('v3','v4_selected'):
            candidate=ROOT/'.artifacts/p10_runtime'/variant/scene/'screenshot_1_fixed_view_capture.jpg'
            copy=images/f'{scene}_{variant}.jpg';shutil.copyfile(candidate,copy)
            records.append(dict(scene=scene,variant=variant,**compare(destination,copy),
                reference_image=str(destination),candidate_image=str(copy),reference_sha256=file_hash(destination),candidate_sha256=file_hash(copy)))
    write_csv(args.report_dir/'fixed_view_image_quality.csv',records)
    (args.report_dir/'fixed_view_image_quality.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    supplements=[('v4_selected_strict','v4_selected','screenshot_1_fixed_view_strict.jpg'),
                 ('v4_selected_gpu_candidate','v4_selected','screenshot_1_fixed_view_gpu_candidate.jpg'),
                 ('v4_compact','v4_compact','screenshot_1_fixed_view.jpg')]
    extended=list(records)
    policy_root=ROOT/'lod_analysis_outputs/adaptive_p10_v4_20261020'
    for label,folder,filename in supplements:
        runtime=args.report_dir/f'runtime_{label}.json'
        if not runtime.exists():continue
        measured=json.loads(runtime.read_text(encoding='utf-8'))
        for scene in args.models:
            reference=images/f'{scene}_source_reference.jpg'
            source=ROOT/'.artifacts/p10_runtime'/folder/scene/filename
            rr=[r for r in measured if r['scene']==scene and r['stage']=='render']
            if not source.exists()or not rr or rr[-1]['policy_sha256']!=file_hash(policy_root/scene/'NCG_H1_M1_selected.csv'):continue
            copy=images/f'{scene}_{label}.jpg';shutil.copyfile(source,copy)
            extended.append(dict(scene=scene,variant=label,**compare(reference,copy),
                reference_image=str(reference),candidate_image=str(copy),reference_sha256=file_hash(reference),candidate_sha256=file_hash(copy)))
    write_csv(args.report_dir/'fixed_view_image_quality_extended.csv',extended)
    (args.report_dir/'fixed_view_image_quality_extended.json').write_text(json.dumps(extended,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Reference and policy image comparisons complete',flush=True)


if __name__=='__main__':main()
