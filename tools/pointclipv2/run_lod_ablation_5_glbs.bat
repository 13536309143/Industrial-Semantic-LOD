@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0\..\.."

set "OUT=lod_analysis_outputs"
set "PCV2=tools\pointclipv2"
set "ABLATION_OUT=%OUT%\lod_ablation"

if not exist "%ABLATION_OUT%" mkdir "%ABLATION_OUT%"

for %%M in (a b c o1778 o3049) do (
  echo.
  echo ===== LOD ablation %%M =====
  set "POINTNEXT_CSV=%OUT%\%%M_pointnext_analysis.csv"
  set "POINTCLIP_FULL_CSV=%OUT%\%%M_pointclipv2_full.csv"
  set "POINTCLIP_CANDIDATES_CSV=%OUT%\%%M_pointclipv2_candidates.csv"

  if not exist "!POINTNEXT_CSV!" (
    echo Missing PointNeXt CSV: !POINTNEXT_CSV!
    exit /b 1
  )

  if not exist "!POINTCLIP_FULL_CSV!" (
    echo Warning: full PointCLIP CSV not found. Falling back to candidate-only CSV:
    echo   !POINTCLIP_CANDIDATES_CSV!
    echo PointCLIP-only ablations will be incomplete unless you run run_pointclipv2_full_5_glbs.bat first.
    set "POINTCLIP_FULL_CSV=!POINTCLIP_CANDIDATES_CSV!"
  )

  if not exist "!POINTCLIP_CANDIDATES_CSV!" (
    echo Missing PointCLIP candidate CSV: !POINTCLIP_CANDIDATES_CSV!
    exit /b 1
  )

  if not exist "!POINTCLIP_FULL_CSV!" (
    echo Missing PointCLIP full CSV: !POINTCLIP_FULL_CSV!
    exit /b 1
  )

  python %PCV2%\build_lod_ablation_variants.py ^
    --pointnext-csv "!POINTNEXT_CSV!" ^
    --pointclip-full-csv "!POINTCLIP_FULL_CSV!" ^
    --pointclip-candidates-csv "!POINTCLIP_CANDIDATES_CSV!" ^
    --output-dir "%ABLATION_OUT%" ^
    --stem %%M
  if errorlevel 1 (
    echo LOD ablation failed for %%M
    exit /b 1
  )
)

echo.
echo ===== Done =====
echo Ablation outputs:
echo   %ABLATION_OUT%

endlocal
