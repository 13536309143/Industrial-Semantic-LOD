@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONNOUSERSITE=1"
call conda activate gpt-pointnext
if errorlevel 1 exit /b 1
python -u tools\pointclipv2\run_assembly_context_experiment.py %*
set "RESULT=%ERRORLEVEL%"
echo.
echo Reports: analysis_reports\assembly_context_v3_20261020
echo Select the seventh strategy and rebuild LOD to use assembly/CAD policies.
endlocal & exit /b %RESULT%
