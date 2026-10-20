@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONNOUSERSITE=1"
call conda activate gpt-pointnext
if errorlevel 1 exit /b 1
python -u tools\pointclipv2\run_open_semantic_upgrade.py %*
set "RESULT=%ERRORLEVEL%"
echo.
echo Report: analysis_reports\open_semantic_v2_20261019
echo Failed scenes are recorded and skipped; rerun reuses verified completed inference.
endlocal & exit /b %RESULT%
