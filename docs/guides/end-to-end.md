# Local workflow

This guide covers program interfaces. Manuscript protocols, dataset splits and measurement records are not distributed.

Build the renderer with Visual Studio 2022 C++ tools, CMake and a Vulkan SDK, following the repository README. Required GPU features are defined by `src/app/main.cpp`.

Use separate inference environments described by `tools/pointnext_lod/environment_pointnext.yml` and `tools/pointclipv2/environment_pointclipv2.yml`. Build the OpenPoints extension for the actual Python/PyTorch/CUDA/compiler combination. The open branch uses pretrained weights managed by its CLIP loader.

Prepare authorized GLBs, compatible local weights and the correct class order. See [model interfaces](../../models/README.md). Store inputs in `_downloaded_resources/` and generated data in `lod_analysis_outputs/`; neither is published. Existing external directories can be bound with `scripts/windows/setup-data.ps1 -DataRoot YOUR_DATA_ROOT` after preparing its required subdirectories.

```powershell
.\scripts\windows\run_all_glb_pointnext_pointclipv2_fused.bat
.\scripts\windows\viewer.bat "path\to\scene.glb"
```

Batch inference needs the configured Conda environments. The viewer defaults to the seventh fused policy CSV, which must be generated for the actual scene first. Use each tool's `--help` for interfaces; the open upgrade runner requires explicit scene stems with `run_open_semantic_v2.bat --models SCENE_STEM`.

Legacy policy tools are retained as development source, without private scenes or results. Use a fresh local output directory for new runs. Follow [the publication policy](../publication-policy.md) before committing generated material.
