@echo off
setlocal EnableExtensions
cd /d "%~dp0..\.."
set "LOD_SCENE=%~1"
if not defined LOD_SCENE set "LOD_SCENE=%cd%\_downloaded_resources\1.glb"
set "LOD_ERROR=%~2"
if not defined LOD_ERROR set "LOD_ERROR=0.5"
for %%D in ("%cd%") do set "LOD_TARGET=%%~nxD"
set "LOD_EXE=%cd%\_bin\Release\%LOD_TARGET%.exe"
if not exist "%LOD_SCENE%" (
  echo Scene not found: "%LOD_SCENE%"
  exit /b 1
)
if not exist "%LOD_EXE%" (
  echo Build the Release renderer first. Missing: "%LOD_EXE%"
  exit /b 1
)
"%LOD_EXE%" --scene "%LOD_SCENE%" --semanticlodmode 6 --loderror %LOD_ERROR% --mappedcache 1 --compressed 0 --visualize 0
exit /b %ERRORLEVEL%
