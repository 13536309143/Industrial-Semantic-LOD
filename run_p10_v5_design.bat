@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONNOUSERSITE=1"
call conda activate gpt-pointnext
if errorlevel 1 exit /b 1
python -m unittest discover -s tools\pointclipv2\tests -p test_adaptive_p10_v5.py -q
if errorlevel 1 exit /b 1
python -u tools\pointclipv2\run_p10_v5_design.py %*
set "P10_V5_RESULT=%ERRORLEVEL%"
echo.
echo V5 design report: analysis_reports\adaptive_p10_v5_20261021
echo Current V4 viewer policies are unchanged. This run evaluates allocations, not FPS.
endlocal & exit /b %P10_V5_RESULT%
