@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONNOUSERSITE=1"
call conda activate gpt-pointnext
if errorlevel 1 exit /b 1
cmake --build build --config Release --parallel 3
if errorlevel 1 exit /b 1
python -m unittest discover -s tools\pointclipv2\tests -q
if errorlevel 1 exit /b 1
python -u tools\pointclipv2\run_adaptive_p10.py %*
if errorlevel 1 exit /b 1
python -u tools\pointclipv2\summarize_adaptive_p10.py
set "RESULT=%ERRORLEVEL%"
echo.
echo Report: analysis_reports\adaptive_p10_v4_20261020
echo Use the seventh UI strategy, command-line semanticlodmode 6, and rebuild LOD.
endlocal & exit /b %RESULT%
