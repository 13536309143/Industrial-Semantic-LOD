@echo off
setlocal

cd /d "%~dp0\..\.."

if "%PYTHON_EXE%"=="" set "PYTHON_EXE=python"

for %%M in (a b c o1778 o3049) do (
  echo.
  echo ===== Rebuild %%M LOD constraints from existing PointNeXt analysis =====
  "%PYTHON_EXE%" tools\pointnext_lod\build_lod_constraints_from_analysis.py ^
    --input-csv "%cd%\lod_analysis_outputs\%%M_pointnext_analysis.csv" ^
    --output-csv "%cd%\lod_analysis_outputs\%%M_lod_constraints.csv" ^
    --summary-json "%cd%\lod_analysis_outputs\%%M_lod_constraints_summary.json"
  if errorlevel 1 (
    echo Failed while rebuilding %%M_lod_constraints.csv
    exit /b 1
  )
)

echo.
echo Done. Rebuild the .zippp caches next if you want the runtime LOD builder to consume the new policies.
