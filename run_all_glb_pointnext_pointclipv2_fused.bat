@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "RESOURCES_DIR=%cd%\_downloaded_resources"
set "OUT=%cd%\lod_analysis_outputs"

if "%POINTNEXT_CONDA_ENV%"=="" set "POINTNEXT_CONDA_ENV=gpt-pointnext"
if "%POINTCLIP_CONDA_ENV%"=="" set "POINTCLIP_CONDA_ENV=PointCLIPV2_LOD"

set "POINTNEXT_ROOT=%cd%\tools\pointnext_lod\PointNeXt"
set "POINTNEXT_CFG=%POINTNEXT_ROOT%\cfgs\industrial_part\pointnext-s-14class-randomrot-8192-fast.yaml"
if "%POINTNEXT_CKPT%"=="" set "POINTNEXT_CKPT=%cd%\models\pointnext\industrial-part-14class-best.pth"
set "POINTNEXT_CLASSES=%cd%\tools\pointnext_lod\industrial_part_14classes.txt"

set "POINTCLIP_ROOT=%cd%\tools\pointclipv2\PointCLIP_V2"
set "POINTCLIP_PROMPTS=%cd%\tools\pointclipv2\industrial_open_vocab_prompts.json"

if not exist "%POINTNEXT_CKPT%" (
  echo Missing trained PointNeXt checkpoint:
  echo   %POINTNEXT_CKPT%
  echo Train the 14-class model and copy its best checkpoint here, or set POINTNEXT_CKPT.
  echo See docs\guides\end-to-end.md for the complete workflow.
  exit /b 1
)

if not exist "%RESOURCES_DIR%" (
  echo Missing resources directory:
  echo   %RESOURCES_DIR%
  exit /b 1
)

if not exist "%OUT%" mkdir "%OUT%"

echo.
echo ===== GLB files under _downloaded_resources =====
dir /b "%RESOURCES_DIR%\*.glb" >nul 2>nul
if errorlevel 1 (
  echo No .glb files found in:
  echo   %RESOURCES_DIR%
  exit /b 1
)

for %%G in ("%RESOURCES_DIR%\*.glb") do (
  echo   %%~nxG
)

echo.
echo ===== Stage 1/3: PointNeXt analysis for all GLB files =====
echo Activating conda environment: %POINTNEXT_CONDA_ENV%
call conda activate %POINTNEXT_CONDA_ENV%
if errorlevel 1 (
  echo Failed to activate conda environment: %POINTNEXT_CONDA_ENV%
  exit /b 1
)

python -c "import sys; sys.path.insert(0, r'%POINTNEXT_ROOT%'); import torch; print('python', sys.version); print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); from openpoints.cpp.pointnet2_batch import pointnet2_cuda; print('pointnet2 ok')"
if errorlevel 1 (
  echo.
  echo PointNeXt CUDA environment check failed.
  echo Check conda environment: %POINTNEXT_CONDA_ENV%
  exit /b 1
)

python tools\pointnext_lod\batch_analyze_glb_lod.py ^
  --resources-dir "%RESOURCES_DIR%" ^
  --output-dir "%OUT%" ^
  --pointnext-root "%POINTNEXT_ROOT%" ^
  --cfg "%POINTNEXT_CFG%" ^
  --ckpt "%POINTNEXT_CKPT%" ^
  --classes-file "%POINTNEXT_CLASSES%" ^
  --num-points 8192 ^
  --batch-size 16 ^
  --pointnext-only ^
  --skip-existing
if errorlevel 1 (
  echo.
  echo PointNeXt batch analysis failed.
  exit /b 1
)

echo.
echo ===== Stage 2/3: PointCLIP V2 zero-shot for low-confidence PointNeXt rows =====
echo Activating conda environment: %POINTCLIP_CONDA_ENV%
call conda activate %POINTCLIP_CONDA_ENV%
if errorlevel 1 (
  echo Failed to activate conda environment: %POINTCLIP_CONDA_ENV%
  exit /b 1
)

python -c "import torch, numpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('numpy', numpy.__version__)"
if errorlevel 1 (
  echo.
  echo PointCLIP V2 environment check failed.
  echo Check conda environment: %POINTCLIP_CONDA_ENV%
  exit /b 1
)

for %%G in ("%RESOURCES_DIR%\*.glb") do (
  set "STEM=%%~nG"
  set "STEM=!STEM: =_!"
  echo.
  echo ===== PointCLIPV2 %%~nxG =====
  if exist "%OUT%\!STEM!_pointclipv2_candidates.csv" (
    echo Skip existing PointCLIPV2 candidate output: %OUT%\!STEM!_pointclipv2_candidates.csv
  ) else (
    python tools\pointclipv2\run_pointclipv2_zeroshot_glb.py ^
      --glb "%%~fG" ^
      --pointclip-root "%POINTCLIP_ROOT%" ^
      --prompt-json "%POINTCLIP_PROMPTS%" ^
      --candidate-csv "%OUT%\!STEM!_pointnext_analysis.csv" ^
      --num-points 8192 ^
      --batch-size 8 ^
      --top-k 5 ^
      --output-csv "%OUT%\!STEM!_pointclipv2_candidates.csv" ^
      --output-json "%OUT%\!STEM!_pointclipv2_candidates.json" ^
      --summary-csv "%OUT%\!STEM!_pointclipv2_candidates_summary.csv"
    if errorlevel 1 (
      echo PointCLIPV2 failed for %%~nxG
      exit /b 1
    )
  )
)

echo.
echo ===== Stage 3/3: Fuse PointNeXt + PointCLIP V2 into semantic LOD policies =====
for %%G in ("%RESOURCES_DIR%\*.glb") do (
  set "STEM=%%~nG"
  set "STEM=!STEM: =_!"
  echo.
  echo ===== Fuse %%~nxG =====
  if not exist "%OUT%\!STEM!_pointnext_analysis.csv" (
    echo Missing PointNeXt CSV: %OUT%\!STEM!_pointnext_analysis.csv
    exit /b 1
  )
  if not exist "%OUT%\!STEM!_pointclipv2_candidates.csv" (
    echo Missing PointCLIPV2 candidate CSV: %OUT%\!STEM!_pointclipv2_candidates.csv
    exit /b 1
  )
  python tools\pointclipv2\fuse_pointnext_pointclip_lod.py ^
    --pointnext-csv "%OUT%\!STEM!_pointnext_analysis.csv" ^
    --pointclip-csv "%OUT%\!STEM!_pointclipv2_candidates.csv" ^
    --merged-csv "%OUT%\!STEM!_pointnext_pointclip_merged.csv" ^
    --output-csv "%OUT%\!STEM!_lod_constraints_fused.csv" ^
    --summary-json "%OUT%\!STEM!_lod_constraints_fused_summary.json"
  if errorlevel 1 (
    echo Fusion failed for %%~nxG
    exit /b 1
  )
)

echo.
echo ===== Done =====
echo Generated fused LOD constraints in:
echo   %OUT%
echo.
echo Next, run renderer preprocessing with --processingonly 1 --autoloadcache 0 to build the policy-controlled LOD cache.
echo Then open the scene in the renderer. See docs\guides\end-to-end.md for commands.

endlocal
