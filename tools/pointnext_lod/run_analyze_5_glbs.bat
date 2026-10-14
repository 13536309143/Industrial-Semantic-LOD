@echo off
setlocal

cd /d "%~dp0\..\.."

if "%PYTHON_EXE%"=="" set "PYTHON_EXE=python"
if "%POINTNEXT_CKPT%"=="" set "POINTNEXT_CKPT=%cd%\models\pointnext\industrial-part-14class-best.pth"

if not exist "%POINTNEXT_CKPT%" (
  echo Missing PointNeXt checkpoint: %POINTNEXT_CKPT%
  echo See docs\guides\end-to-end.md to train or supply the industrial checkpoint.
  exit /b 1
)

"%PYTHON_EXE%" -c "import sys; sys.path.insert(0, r'%cd%\tools\pointnext_lod\PointNeXt'); import torch; print('python', sys.version); print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); from openpoints.cpp.pointnet2_batch import pointnet2_cuda; print('pointnet2 ok')"
if errorlevel 1 (
  echo.
  echo PointNeXt CUDA environment check failed.
  echo Run this script from the Python 3.10 CUDA environment that can import pointnet2_batch_cuda.
  echo You can also set PYTHON_EXE before running, for example:
  echo   set PYTHON_EXE=D:\path\to\python.exe
  exit /b 1
)

"%PYTHON_EXE%" tools\pointnext_lod\batch_analyze_glb_lod.py ^
  --resources-dir "%cd%\_downloaded_resources" ^
  --output-dir "%cd%\lod_analysis_outputs" ^
  --pointnext-root "%cd%\tools\pointnext_lod\PointNeXt" ^
  --cfg "%cd%\tools\pointnext_lod\PointNeXt\cfgs\industrial_part\pointnext-s-14class-randomrot-8192-fast.yaml" ^
  --ckpt "%POINTNEXT_CKPT%" ^
  --classes-file "%cd%\tools\pointnext_lod\industrial_part_14classes.txt" ^
  --models a.glb b.glb c.glb o1778.glb o3049.glb ^
  --num-points 8192 ^
  --batch-size 16

endlocal
