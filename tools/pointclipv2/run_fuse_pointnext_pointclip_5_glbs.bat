@echo off
setlocal

cd /d "%~dp0\..\.."

set OUT=lod_analysis_outputs
set PCV2=tools\pointclipv2

for %%M in (a b c o1778 o3049) do (
  echo.
  echo ===== Fuse PointNeXt + PointCLIPV2 %%M =====
  python %PCV2%\fuse_pointnext_pointclip_lod.py ^
    --pointnext-csv %OUT%\%%M_pointnext_analysis.csv ^
    --pointclip-csv %OUT%\%%M_pointclipv2_candidates.csv ^
    --merged-csv %OUT%\%%M_pointnext_pointclip_merged.csv ^
    --output-csv %OUT%\%%M_lod_constraints_fused.csv ^
    --summary-json %OUT%\%%M_lod_constraints_fused_summary.json
)

endlocal
