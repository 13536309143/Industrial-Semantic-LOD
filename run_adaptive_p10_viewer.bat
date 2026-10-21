@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "P10_SCENE=%~1"
if not defined P10_SCENE set "P10_SCENE=%~dp0_downloaded_resources\a.glb"
if not exist "%P10_SCENE%" (
  echo Scene not found: "%P10_SCENE%"
  exit /b 1
)
if not exist "%~dp0_bin\Release\t1.exe" (
  echo Build the Release application first with run_adaptive_p10_comparison.bat.
  exit /b 1
)
rem Detail is the tested visual-quality preset. Pass balanced as the second argument for screen error 2.
set "P10_PIXEL_ERROR=0.5"
if /i "%~2"=="balanced" set "P10_PIXEL_ERROR=2"
"%~dp0_bin\Release\t1.exe" --scene "%P10_SCENE%" --semanticlodmode 6 --loderror %P10_PIXEL_ERROR% --mappedcache 1 --compressed 0 --visualize 0 --cachesuffix .adaptive_p10_v4
set "RESULT=%ERRORLEVEL%"
endlocal & exit /b %RESULT%
