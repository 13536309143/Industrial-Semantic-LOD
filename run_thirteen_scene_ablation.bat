@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONNOUSERSITE=1"
call conda activate gpt-pointnext
if errorlevel 1 exit /b 1
python -u -X utf8 tools\pointclipv2\run_thirteen_scene_ablation.py %*
if errorlevel 1 exit /b 1
python -u -X utf8 tools\pointclipv2\visualize_thirteen_scene_ablation.py
if errorlevel 1 exit /b 1
python -u -X utf8 tools\pointclipv2\summarize_thirteen_scene_findings.py
if errorlevel 1 exit /b 1
python -u -X utf8 tools\pointclipv2\validate_thirteen_scene_ablation.py
exit /b %ERRORLEVEL%
