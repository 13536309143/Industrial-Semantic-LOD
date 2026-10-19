# Training, semantic analysis, and rendering

[English](end-to-end.md) · [简体中文](end-to-end.zh-CN.md) · [Project README](../../README.md)

The default checkpoint was updated on 2026-10-19 to PointNeXt-S, seed 2027, epoch 98, selected by validation macro-F1 from the retained `gpt` runs. Start at sections 5–6 to use it. Sections 3–4 remain an optional from-scratch training reference; their legacy YAML does not reproduce the two-seed training implementation. See the [current experiment record](../../experiments/pointnext/encoder-compare-two-seeds-20261019/README.md). Previous scene outputs were archived; new scene and LOD results still require recomputation.

The complete research workflow is **labeled industrial point clouds → supervised PointNeXt training → scene-level multi-source analysis → policy-controlled cluster LOD construction → Vulkan rendering**. Train the classifier once for a compatible category set, then reuse its best checkpoint for new scenes. PointCLIP V2 supplies pretrained open-vocabulary evidence and is not trained by the industrial scene-analysis scripts.

## 1. Prerequisites and software environments

| Component | Requirements |
| :--- | :--- |
| PointNeXt training/inference | Python 3.10, PyTorch 2.2.0, torchvision 0.17.0, CUDA runtime 12.1, NumPy 1.26.4, OpenPoints CUDA extension |
| Extension compilation on Windows | CUDA Toolkit 12.1 with `nvcc`, Visual Studio C++ build tools, a Developer/Native Tools command prompt |
| PointCLIP V2 inference | Separate `PointCLIPV2_LOD` environment, CUDA GPU, pretrained CLIP ViT-B/16 |
| Renderer | Visual Studio 2022/C++20, CMake 3.22+, Git, Vulkan SDK 1.4.309.0+, compatible mesh-shader device |

The PointNeXt version family above was checked in the existing local environment: Python 3.10.20, PyTorch 2.2.0+cu121, NumPy 1.26.4, and a loadable `pointnet2_batch_cuda` extension. Environment recipes describe installation inputs; they are not a claim that every GPU/driver combination has been tested.

Use **Anaconda Prompt / cmd.exe** for the following commands. Set the repository root to your checkout path:

```bat
set PROJECT_ROOT=E:\vk_lod_clusters1\t1
cd /d "%PROJECT_ROOT%"
conda env create -f tools\pointnext_lod\environment_pointnext.yml
conda activate gpt-pointnext
python -c "import torch,numpy; print(torch.__version__,torch.version.cuda,numpy.__version__,torch.cuda.is_available())"
```

If this environment already exists and works, activate it instead of creating it again. The Conda CUDA runtime supports PyTorch execution; **compiling the extension also needs the system CUDA Toolkit and C++ compiler**. From a Visual Studio Native Tools prompt with Conda activated:

```bat
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1
set PATH=%CUDA_HOME%\bin;%PATH%
where cl
where nvcc
cd /d "%PROJECT_ROOT%\tools\pointnext_lod\PointNeXt\openpoints\cpp\pointnet2_batch"
python -m pip install . --no-build-isolation
cd /d "%PROJECT_ROOT%"
python -c "import sys; sys.path.insert(0,r'tools/pointnext_lod/PointNeXt'); from openpoints.cpp.pointnet2_batch import pointnet2_cuda; from openpoints.models import build_model_from_cfg; from openpoints.dataset import IndustrialPartPointCloud; print('PointNeXt imports OK')"
```

If you switch terminals, set `PROJECT_ROOT` again and activate the environment. CUDA extension binaries depend on the Python, PyTorch, CUDA, compiler, and platform combination. The local `.pyd` is kept for the current installation but ignored by Git; new checkouts should build their own extension.

Create the open-vocabulary environment independently:

```bat
cd /d "%PROJECT_ROOT%"
conda env create -f tools\pointclipv2\environment_pointclipv2.yml
conda activate PointCLIPV2_LOD
python -c "import torch,sys; sys.path.insert(0,r'tools/pointclipv2/PointCLIP_V2/zeroshot_cls'); from clip import clip; from trainers.mv_utils_zs import Realistic_Projection; print(torch.__version__,torch.cuda.is_available(),'PointCLIP imports OK')"
```

The industrial wrapper has a PyTorch scatter fallback for its projection path; installing the old upstream `torch-scatter==2.0.9` requirement is not mandatory for that path. The default CLIP loader may download ViT-B/16 weights on first use. For offline execution, populate its cache before starting scene analysis.

## 2. Prepare labeled training data

Training samples and renderable scene assets serve different purposes. Training needs category labels; scene inference accepts unlabeled mesh geometry.

```text
datasets/industrial_part_14class/          # local, ignored by Git
  classes.txt
  train_list.txt
  val_list.txt                           # recommended for new training
  test_list.txt
  screws_bolts_studs/part_0001.npy
  nuts/part_0002.npy
  ...
```

Each `.npy` stores a finite floating-point point cloud of shape `[N, 3]` containing XYZ coordinates. Prepare area-weighted surface samples of the part mesh and normalize consistently with inference (centered unit-sphere point clouds). The active recipe uses 8,192 points; the dataset resamples inputs with more or fewer points. Additional columns are ignored by the current adapter, so normals are not supervised input features in this configuration.

`classes.txt` uses **real tab separators**, with class indices 0–13 in exactly the order of [industrial_part_14classes.txt](../../tools/pointnext_lod/industrial_part_14classes.txt):

```text
0<TAB>screws_bolts_studs
1<TAB>nuts
...
13<TAB>springs
```

Each list row is `relative/path.npy<TAB>class_index<TAB>class_name`. `<TAB>` above is notation, not literal text. Example:

```text
screws_bolts_studs/part_0001.npy<TAB>0<TAB>screws_bolts_studs
```

Split by original CAD part, family, or source assembly so resampled/rotated copies do not cross subsets. The repository includes the dataset adapter, **not a raw-CAD dataset converter**; labeled, normalized `.npy` samples must be prepared before training.

The recorded recipe uses the same `test` split for validation and test. For a new independent evaluation, keep `val_list.txt` separate and explicitly pass `dataset.val.list_file=val_list.txt` and `dataset.test.list_file=test_list.txt`. The adapter accepts these explicit list files without changing its algorithm. Historical README scores retain their original shared-split interpretation.

## 3. Prepare a complete PointNeXt training project

**The vendored PointNeXt directory in this repository contains inference code and industrial recipes, but no `examples/classification/main.py` training launcher.** The original training log used that launcher in an external full PointNeXt project. Do not run an absent path from this checkout.

Use a compatible complete [PointNeXt training checkout](https://github.com/guochengqian/PointNeXt), including its classification examples and inherited default configurations. The author's [classification launcher](https://github.com/guochengqian/PointNeXt/blob/master/examples/classification/main.py) accepts `--cfg` and configuration overrides. Upstream revisions can change APIs; record the checkout commit and verify compatibility with the industrial recipe. The exact external launcher used by the historical run is not bundled here.

For a new external checkout:

```bat
set TRAINING_ROOT=D:\PointNeXt-training
git clone --recursive https://github.com/guochengqian/PointNeXt.git "%TRAINING_ROOT%"
cd /d "%TRAINING_ROOT%"
git rev-parse HEAD
```

Integrate the project's industrial dataset adapter and recipe into that checkout:

```bat
copy "%PROJECT_ROOT%\tools\pointnext_lod\PointNeXt\openpoints\dataset\industrial_part.py" "%TRAINING_ROOT%\openpoints\dataset\industrial_part.py"
if not exist "%TRAINING_ROOT%\cfgs\industrial_part" mkdir "%TRAINING_ROOT%\cfgs\industrial_part"
copy "%PROJECT_ROOT%\tools\pointnext_lod\PointNeXt\cfgs\industrial_part\pointnext-s-14class-randomrot-8192-fast.yaml" "%TRAINING_ROOT%\cfgs\industrial_part\pointnext-s-14class-randomrot-8192-fast.yaml"
```

Register the adapter by adding the following import to the **external training checkout's** `openpoints/dataset/__init__.py`, unless already present:

```python
from .industrial_part import IndustrialPartPointCloud
```

Use the complete checkout's installation guidance for its additional training dependencies and CUDA extensions. Verify the external adapter imports before launching. The new environment recipe covers this repository's inference modules; it does not replace all dependencies of arbitrary upstream training revisions.

## 4. Train and select the supervised checkpoint

From the complete training checkout, with its compatible environment active:

```bat
conda activate gpt-pointnext
cd /d "%TRAINING_ROOT%"
set CUDA_VISIBLE_DEVICES=0
python examples/classification/main.py --cfg cfgs/industrial_part/pointnext-s-14class-randomrot-8192-fast.yaml ^
  dataset.common.data_dir="%PROJECT_ROOT%\datasets\industrial_part_14class" ^
  dataset.val.list_file=val_list.txt dataset.test.list_file=test_list.txt ^
  root_dir="%PROJECT_ROOT%\.artifacts\training\pointnext" seed=3098
```

This shows the verified launcher interface and dataset adapter contract; the external training checkout itself has not been installed or rerun during this repository cleanup. Single-GPU training is the documented starting point. On Windows, ensure any distributed launcher settings are compatible with the backend; the historical log does not establish multi-GPU Windows support.

The recipe specifies 14 classes, 8,192 XYZ points, 100 epochs, batch 16, AdamW, cosine scheduling, random rotation, label smoothing, and AMP. Reduce batch size through an override if memory is insufficient; record the effective configuration. The best checkpoint should be selected by validation, then evaluated on the separate test set.

After training, locate the printed run directory and copy **its exact best checkpoint**:

```bat
cd /d "%PROJECT_ROOT%"
if not exist models\pointnext mkdir models\pointnext
copy "D:\path\to\your\run\checkpoint\your_run_ckpt_best.pth" "models\pointnext\industrial-part-14class-best.pth"
```

Replace the source path with the actual training output. Alternatively:

```bat
set POINTNEXT_CKPT=D:\path\to\your\best_checkpoint.pth
```

Requirements for that file: the same encoder/head configuration, 14-class output, and class index ordering matching this project's class file. A generic ModelNet40 checkpoint cannot be substituted just by changing its filename. Latest checkpoints and TensorBoard events belong under `.artifacts/`; compact run evidence can be curated under `experiments/`.

If the existing compatible industrial checkpoint is already available under `models/pointnext/`, reuse it and begin with the next step. Retraining is not required for each new scene.

## 5. Prepare scene models

Place each scene as `_downloaded_resources/<scene>.glb`. The renderer accepts `.gltf`, `.glb`, and scene `.cfg`; the root semantic batch scans **GLB files**.

The Python analysis assumes GLB 2.0 mesh POSITION accessors and triangle-list primitives, with ordinary readable buffer data. Preserve mesh/node identities between analysis and rendering, avoid unsupported compressed input in the analysis path, and export scene units consistently. Material/camera support in the renderer does not imply equivalent support for every glTF extension in the Python tools. STEP/IGES are not direct inputs: export triangulated geometry to GLB and prepare labeled point clouds separately.

## 6. Run multi-source semantic analysis

At the repository root, from an Anaconda Prompt:

```bat
cd /d "%PROJECT_ROOT%"
set POINTNEXT_CONDA_ENV=gpt-pointnext
set POINTCLIP_CONDA_ENV=PointCLIPV2_LOD
run_all_glb_pointnext_pointclipv2_fused.bat
```

The batch performs full PointNeXt classification/geometry extraction, uncertainty-gated PointCLIP V2 inference, and semantic/structure policy fusion. Open inference triggers for confidence below 0.70 **or** top1–top2 margin below 0.08. Geometry descriptors and the resulting role evidence determine P1–P10 and control ratios/weights.

Required output for a scene is `lod_analysis_outputs/<scene>_lod_constraints_fused.csv`. Intermediate classification, candidate, and merged CSVs remain useful locally for diagnosis and regeneration; they are ignored by Git rather than deleted during this cleanup.

The batch reuses only complete outputs with matching content provenance. The PointNeXt stage checks scene/checkpoint/configuration/classes/code and sampling settings; the candidate stage checks the scene, PointNeXt CSV, prompts and inference code/settings. Legacy unstamped, truncated or stale outputs are rebuilt. Fusion regenerates policies. Set SEMANTIC_OUTPUT_DIR for an independent output directory. Rebuild renderer LOD caches with --processingonly 1 --autoloadcache 0; previous caches are not new-checkpoint results.

For six-configuration comparisons on `a`, `b`, `c`, `o1778`, and `o3049`, generate full PointCLIP outputs first:

```bat
conda activate PointCLIPV2_LOD
tools\pointclipv2\run_pointclipv2_full_5_glbs.bat
tools\pointclipv2\run_lod_ablation_5_glbs.bat
```

These outputs go under `lod_analysis_outputs/lod_ablation/`. Full PointCLIP is for comparison modes; the default fused path uses candidate-only inference.

## 7. Build the renderer and precompute LOD

```bat
cd /d "%PROJECT_ROOT%"
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

The local `t1` checkout outputs `_bin/Release/t1.exe`. The target name follows the source directory name. `nvpro_core2` is found locally or downloaded at the pinned commit recorded in CMake; Vulkan device extensions are requested in [main.cpp](../../src/app/main.cpp).

After semantic analysis completes, explicitly build a policy-controlled cache:

```bat
_bin\Release\t1.exe --scene "_downloaded_resources\your_scene.glb" ^
  --semanticlodmode 5 --processingonly 1 --autoloadcache 0 --processingpartial 1
```

Replace `your_scene` with the model stem. CPU processing imports geometry, matches policy records, constructs feature-constrained cluster LOD, and stores `your_scene.glb.zippp`. Partial processing can retain resumable preprocessing state. Semantic analysis itself does not generate this renderer cache.

`--semanticlodmode` values 0–5 correspond to the six configurations; 5 is full fusion, and 0 is PointNeXt rather than semantic-disable. The selected CSV is searched before fused/ordinary fallback files. Check the log for the actual CSV path and policy match information. Semantic hashes participate in cache compatibility; different experiments can use different `--cachesuffix` values.

## 8. Open the scene and configure runtime features

```bat
_bin\Release\t1.exe --scene "_downloaded_resources\your_scene.glb" --semanticlodmode 5
```

For automatic group-level streaming with explicit geometry/transfer budgets:

```bat
_bin\Release\t1.exe --scene "_downloaded_resources\your_scene.glb" ^
  --semanticlodmode 5 --streaming 1 --maxgeomegabytes 2048 --maxtransfermegabytes 64
```

The values are example starting budgets, not benchmark recommendations or a guarantee that a scene fits. GPU demand generates requests; CPU processing uploads groups and publishes residency updates. Lowest-detail data and hierarchy remain available, while finer groups are requested and older unused groups are reclaimed.

To combine streaming with adaptive two-pass culling:

```bat
_bin\Release\t1.exe --scene "_downloaded_resources\your_scene.glb" ^
  --semanticlodmode 5 --streaming 1 --culling 1 ^
  --twopassculling 1 --adaptivetwopassculling 1
```

Both two-pass flags matter: the adaptive flag alone does not enable a second pass. Fixed two-pass uses `--twopassculling 1 --adaptivetwopassculling 0`. Inspect **Semantic LOD Policy Distribution**, viewport visualization, Statistics, Streaming memory, and Profiler. Runtime `--loderror` controls the display precision threshold; assembly filtering uses real imported hierarchy and its own screen-space controls.

## Troubleshooting

| Symptom | Check |
| :--- | :--- |
| No training `main.py` | Use a compatible complete external PointNeXt training checkout |
| `pointnet2_batch_cuda` import or DLL failure | Match Python/PyTorch/CUDA, build the extension, check CUDA and torch DLL paths |
| CUDA unavailable | Confirm the selected environment, driver, and CUDA-enabled PyTorch package |
| Missing dataset lists / class mismatch | Use absolute data directory, actual tab-separated lists, and the fixed 14-class order |
| Checkpoint tensor-size mismatch | Match encoder configuration and classification head; do not use generic 40-class weights |
| Wrong comparison mode appears active | Check resolved CSV path; missing selected files can fall back to fused |
| New settings appear unchanged | Regenerate neural outputs if needed, then rebuild the renderer cache |
| Second pass never runs | Enable culling and two-pass; inspect cluster-count/matrix-delta thresholds |
| No assembly filtering benefit | Confirm the GLB has usable assembly hierarchy, not only flat nodes |
| Large streaming delays | Check actual residency/transfer/request budgets and coarse fallback behavior |

See [repository layout](../repository-layout.md) for tracked evidence versus local data and [technical review](../../analysis_reports/PROJECT_REVIEW_20261014.md) for implementation details and experiment limits.
