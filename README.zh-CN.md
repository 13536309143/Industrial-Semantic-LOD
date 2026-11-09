# Industrial Semantic LOD

面向工业装配体的语义结构感知集群LOD构建与GPU驱动渲染源码。离线融合监督语义、开放语义与几何信息，生成简化策略；在线通过Vulkan完成层次遍历、局部精度选择、可见性筛选、网格着色器绘制与按需驻留。

[English](README.md) · [关键对比摘要](docs/experiments/key-comparisons.md) · [使用流程](docs/guides/end-to-end.zh-CN.md) · [目录说明](docs/repository-layout.md)

## 关键对比

以下仅公开2026年11月4日版本中的两项汇总结果，数值已降低展示精度。结果属于内部测试，不代表外部数据泛化结论。

| 对比项目 | 公开摘要 |
| --- | --- |
| 监督分类OA | PointNet++约94.6%；适配的PointNeXt分支约95.4% |
| 按需驻留相对全量驻留 | 已评估场景的几何缓冲区分配峰值降低约41%–71% |

几何分配峰值不等于总显存。两项结果分别评价不同模块，不能据此推断严格等画质下的渲染加速。完整表格、逐场景指标、训练明细、实验图片、论文导出文件及权重不随公开源码分发。[比较范围与局限](docs/experiments/key-comparisons.md)

## 实现概览

源码保留渲染器、着色器、神经推断封装、策略代码和回归测试。模型级预算与局部几何约束共同参与集群LOD构建；集群DAG描述跨层几何关系，空间层次用于运行时搜索。GPU完成有效任务压缩和间接网格绘制，按需驻留按访问需求管理细节数据。

实验输入与生成结果保存在本地数据目录，公共仓库仅保留程序接口和上述关键摘要。

## 构建与运行

使用Windows、Visual Studio 2022 C++工具、CMake 3.22+、Git及Vulkan SDK。GPU特性要求以 [main.cpp](src/app/main.cpp) 为准。神经推断另需配置CUDA/PyTorch环境。

```powershell
git clone https://github.com/13536309143/Industrial-Semantic-LOD.git t1
cd t1
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --parallel 3
.\scripts\windows\viewer.bat "模型路径\scene.glb"
```

CMake获取固定版本的nvpro_core2。目录名为`t1`时，程序为`_bin/Release/t1.exe`。查看器默认0.5 px误差和语义索引6；需在GLB旁准备兼容的策略CSV。环境及本地权重配置见[使用流程](docs/guides/end-to-end.zh-CN.md)。

## 目录

| 目录 | 内容 |
| --- | --- |
| `src/`、`shaders/` | 渲染、场景、集群几何、可见性与流送 |
| `cmake/`、`thirdparty/` | 构建及第三方源码 |
| `tools/pointnext_lod/`、`tools/pointclipv2/` | 推断、融合、策略工具及测试 |
| `tools/feature_protection_tests/` | CPU几何验证 |
| `scripts/windows/` | 启动及本地数据绑定 |
| `models/` | 推理接口元数据，权重保留在本地 |
| `docs/` | 使用说明与有限结果摘要 |

## 检查

```powershell
python -s -m unittest discover -s tools\pointclipv2\tests -q
python -s -m unittest discover -s tools\pointnext_lod\tests -q
python -s tools\maintenance\check_publication.py --staged
```

依赖测试使用已配置的分析环境。启用本地提交及推送检查：

```powershell
git config core.hooksPath .githooks
```

公开材料的边界见[发布约定](docs/publication-policy.md)。

## 基础组件

[Vulkan Cluster LOD](https://github.com/nvpro-samples/vk_lod_clusters)、[nvpro_core2](https://github.com/nvpro-samples/nvpro_core2)、[meshoptimizer](https://github.com/zeux/meshoptimizer)、[PointNeXt](https://github.com/guochengqian/PointNeXt)、[PointCLIP V2](https://github.com/yangyangyang127/PointCLIP_V2)。请保留对应源码的上游许可声明。
