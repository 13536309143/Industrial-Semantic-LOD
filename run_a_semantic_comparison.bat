@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONNOUSERSITE=1"
call conda activate gpt-pointnext
if errorlevel 1 exit /b 1
python -u tools\pointclipv2\run_scene_semantic_comparison.py ^
  --glb "%cd%\_downloaded_resources\a.glb" ^
  --output-dir "%cd%\lod_analysis_outputs\a_seed2027_20261019"
if errorlevel 1 exit /b 1
python -u tools\pointclipv2\analyze_scene_semantics.py ^
  --glb "%cd%\_downloaded_resources\a.glb" ^
  --run-dir "%cd%\lod_analysis_outputs\a_seed2027_20261019" ^
  --report-dir "%cd%\analysis_reports\a_seed2027_20261019"
if errorlevel 1 exit /b 1
echo Complete. Report: analysis_reports\a_seed2027_20261019\analysis_report.md
endlocal
exit /b 0
