# 本地运行流程

本指南说明程序接口，不包含论文实验方案、数据划分或测试记录。输入场景、兼容权重和生成结果由本地环境管理。

## 环境

渲染器需要Windows、Visual Studio 2022 C++工具、CMake及Vulkan SDK。GPU功能以`src/app/main.cpp`请求的特性为准。

监督与开放推断分别使用独立环境，依赖配方为：

```powershell
conda env create -f tools\pointnext_lod\environment_pointnext.yml
conda env create -f tools\pointclipv2\environment_pointclipv2.yml
```

PointNeXt的OpenPoints扩展需要匹配的CUDA Toolkit、PyTorch及C++编译器。在对应环境中编译`tools/pointnext_lod/PointNeXt/openpoints/cpp/pointnet2_batch`。开放分支使用CLIP加载器管理的预训练权重。新机器需要按实际平台编译扩展，不能直接复用其他环境的二进制。

## 本地资产

准备具有使用权限的GLB、兼容的监督检查点及类别顺序。默认权重路径和接口见[模型说明](../../models/README.md)。已有权重可通过`POINTNEXT_CKPT`指定。

将场景放入本地`_downloaded_resources/`，分析输出写入`lod_analysis_outputs/`。两个目录均不上传。已有外部数据目录也可通过`scripts/windows/setup-data.ps1 -DataRoot 数据根目录`绑定，所需子目录结构以脚本为准。

## 分析与查看

```powershell
.\scripts\windows\run_all_glb_pointnext_pointclipv2_fused.bat
.\scripts\windows\viewer.bat "模型路径\scene.glb"
```

批处理需要可用的Conda环境。查看器默认读取第七份融合策略文件；需先生成与场景一致的CSV。各分析工具接受的参数请查看对应`--help`。开放语义升级入口使用`run_open_semantic_v2.bat --models 场景文件名主干`显式选择输入。

历史策略工具保留源码供开发使用，不包含预先配置的论文场景或完整测量结果。新运行请使用新的本地输出目录，避免覆盖已有记录。

## 检查

Python回归测试及CPU几何检查见项目README和`tools/feature_protection_tests/README.md`。发布前运行`tools/maintenance/check_publication.py`，并按[发布约定](../publication-policy.md)审阅新增材料。
