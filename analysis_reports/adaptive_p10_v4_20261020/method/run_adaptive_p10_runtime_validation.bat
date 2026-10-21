@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONNOUSERSITE=1"
call conda activate gpt-pointnext
if errorlevel 1 exit /b 1
python -u tools\pointclipv2\validate_adaptive_p10_runtime.py %*
if errorlevel 1 exit /b 1
python -u tools\pointclipv2\summarize_adaptive_p10.py
set "RESULT=%ERRORLEVEL%"
endlocal & exit /b %RESULT%
