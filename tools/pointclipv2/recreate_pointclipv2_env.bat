@echo off
setlocal

cd /d "%~dp0\..\.."

echo This will remove and recreate the PointCLIPV2_LOD conda environment.
echo Press Ctrl+C now if you do not want to remove it.
pause

conda env remove -n PointCLIPV2_LOD -y
conda create -n PointCLIPV2_LOD --clone gpt-pointnext -y
conda run -n PointCLIPV2_LOD python -m pip install ftfy regex tqdm scipy scikit-learn yacs pillow pandas

echo.
echo Activate the environment:
echo   conda activate PointCLIPV2_LOD
echo.
echo Validate PyTorch:
echo   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

endlocal
