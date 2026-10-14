# Repository layout and artifact policy

The repository separates implementation, selected research evidence, model weights, and regenerable output.

| Directory | Contents | Git policy |
| :--- | :--- | :--- |
| `src/`, `shaders/`, `cmake/` | Renderer implementation and build configuration | Track |
| `thirdparty/` | Required vendor code, including generated shader headers | Track |
| `tools/` | Analysis scripts, dataset adapter, model configurations, environment recipes | Track source; ignore caches/build output |
| `docs/guides/` | English/Chinese end-to-end instructions | Track |
| `docs/methods/`, `docs/experiments/` | Method analysis and comparison design | Track |
| `docs/assets/`, `docs/data/` | Presentation figures and compact evidence metadata | Track |
| `analysis_reports/` | Selected policy statistics and technical reviews | Track |
| `experiments/pointnext/` | Selected training log and original run configuration | Track compact evidence |
| `models/pointnext/` | Best checkpoint used for inference | Weights stay local; directory README is tracked |
| `datasets/` | Labeled point clouds and split lists | Local, ignored |
| `_downloaded_resources/` | Scene GLBs and renderer caches | Local, ignored |
| `lod_analysis_outputs/` | Per-mesh neural inference, fusion, and comparison CSV/JSON | Local, ignored |
| `.artifacts/training/` | Latest checkpoint, TensorBoard events, and detailed trainer logs | Local, ignored |
| `.artifacts/documentation/` | Preserved obsolete workflow document | Local, ignored |
| `.artifacts/organization/` | Cleanup/move manifest | Local, ignored |
| `build/`, `_bin/` | Current renderer build and runtime files | Local, ignored |

Python bytecode and CUDA `.obj`/temporary build files are regenerable. The currently working `.pyd` and installed package metadata are retained locally for compatibility but removed from Git tracking. `shaders/build/` and `thirdparty/vulkan_radix_sort/src/generated/` are source inputs and are deliberately preserved.

## Cleanup performed on 2026-10-14

Removed generated Python cache files and CUDA intermediate/distribution files. Retained the active extension binary. Moved both distinct model checkpoints without changing their contents: best to `models/pointnext/`, latest to `.artifacts/training/pointnext/industrial-part-14class/`. Preserved the complete console log in `experiments/pointnext/industrial-part-14class/training.log`; moved the original runtime YAML there as provenance. The duplicated recipe was byte-identical to the retained portable configuration and was removed.

From the original tracked tree, **458 generated files (137,603,190 bytes, approximately 131.2 MiB)** were deleted locally. **475 generated/runtime paths** were removed from the Git index, including the preserved extension, package metadata, and old training-output paths. The source recipe and both distinct checkpoints were retained. A local manifest records the removed tracked paths and preserved artifact hashes.

The obsolete root workflow document was archived locally and replaced by the maintained bilingual guide. Method and comparison-design documents moved into the corresponding `docs/` subdirectories. Analysis scripts now use the organized best-checkpoint path and allow `POINTNEXT_CKPT` overrides. Renderer and neural algorithm source files were not changed.

Adding ignore patterns alone does not remove already tracked files. Generated files and training runtime artifacts were explicitly removed from the Git index; the working-tree changes should be reviewed and committed normally. This cleanup does **not** rewrite previous Git history, remove existing large blobs from old commits, or push to GitHub.

For future experiments, retain raw outputs locally and publish only selected summaries with source hashes and effective configurations. Model weights can be distributed separately through an intentional release/storage workflow rather than mixed into source commits.

## 本地目录约定

源码、环境文件和精简实验依据纳入 Git；点云数据、场景、逐网格输出、权重和训练过程产物保留在本地。`models/pointnext/` 保存推理 best，`.artifacts/training/` 保存恢复训练 latest 及 TensorBoard，`experiments/` 保存论文/README 可追溯的日志和配置。

清理没有删除有价值的模型或实验结果，也没有改动算法。对已经上传过的产物，需要提交当前整理变更后，GitHub 的当前文件树才会更新；旧提交里的大文件仍存在。
