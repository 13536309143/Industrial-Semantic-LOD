@echo off
setlocal

cd /d "%~dp0\..\.."

set OUT=lod_analysis_outputs
set PCV2=tools\pointclipv2
set POINTCLIP_ROOT=%PCV2%\PointCLIP_V2
set PROMPTS=%PCV2%\industrial_open_vocab_prompts.json

for %%M in (a b c o1778 o3049) do (
  echo.
  echo ===== PointCLIPV2 candidate zero-shot %%M.glb =====
  python %PCV2%\run_pointclipv2_zeroshot_glb.py ^
    --glb _downloaded_resources\%%M.glb ^
    --pointclip-root %POINTCLIP_ROOT% ^
    --prompt-json %PROMPTS% ^
    --candidate-csv %OUT%\%%M_pointnext_analysis.csv ^
    --num-points 8192 ^
    --batch-size 8 ^
    --top-k 5 ^
    --output-csv %OUT%\%%M_pointclipv2_candidates.csv ^
    --output-json %OUT%\%%M_pointclipv2_candidates.json ^
    --summary-csv %OUT%\%%M_pointclipv2_candidates_summary.csv
)

endlocal
