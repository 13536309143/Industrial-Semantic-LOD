"""Run scene-ready, sequential GPU experiments; never overlap GPU trials."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',default=list('abcdef'))
    parser.add_argument('--trials',type=int,default=2)
    parser.add_argument('--skip-template',action='store_true')
    args=parser.parse_args();directory=Path(__file__).parent
    jobs=[('v3','lod_analysis_outputs/assembly_context_v3_20261020','final_guarded/v3_NCGHM.csv'),
          ('v4_profile','lod_analysis_outputs/adaptive_p10_v4_20261020','NCG_H1_M1.csv'),
          ('v4_selected','lod_analysis_outputs/adaptive_p10_v4_20261020','NCG_H1_M1_selected.csv')]
    for variant,root,filename in jobs:
        if args.skip_template and variant=='v4_profile':continue
        subprocess.run([sys.executable,'-u',str(directory/'benchmark_p10_runtime.py'),'--variant',variant,
                        '--policy-root',str(ROOT/root),'--policy-file',filename,'--render','--trials',str(args.trials),
                        '--models',*args.models],check=True)
    subprocess.run([sys.executable,'-u',str(directory/'evaluate_p10_images.py'),'--models',*args.models],check=True)


if __name__=='__main__':main()
