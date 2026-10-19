# 从模型训练到语义分析与渲染

[English](end-to-end.md) · [简体中文](end-to-end.zh-CN.md) · [项目说明](../../README.zh-CN.md)

当前默认权重已于 2026-10-19 更新为 `gpt` 双种子实验的 PointNeXt-S、seed=2027、epoch=98，按验证 Macro-F1 选择。可直接从第 5、6 节准备场景并推理；第 3、4 节保留从头训练的参考流程。新模型的训练流程、划分和结果见[当前实验说明](../../experiments/pointnext/encoder-compare-two-seeds-20261019/README.md)，不应以旧 YAML 重现该双种子实验。旧场景输出已归档，新模型尚需重新生成场景和 LOD 结果。

完整研究流程为：**带标签工业点云 → PointNeXt 监督训练 → 工业场景多源语义结构分析 → 策略控制的集群 LOD 构建 → Vulkan 渲染**。分类器按类别体系训练一次，后续新场景复用最佳检查点。PointCLIP V2 使用预训练 CLIP 提供开放词汇证据，当前工业场景脚本不对它重新训练。

## 1. 环境准备

| 部分 | 要求 |
| :--- | :--- |
| PointNeXt 训练/推理 | Python 3.10、PyTorch 2.2.0、torchvision 0.17.0、CUDA runtime 12.1、NumPy 1.26.4、OpenPoints CUDA 扩展 |
| Windows 扩展编译 | CUDA Toolkit 12.1 的 `nvcc`、Visual Studio C++ Build Tools、Native Tools 命令行 |
| PointCLIP V2 | 独立 `PointCLIPV2_LOD` 环境、CUDA GPU、预训练 CLIP ViT-B/16 |
| 渲染引擎 | Visual Studio 2022/C++20、CMake 3.22+、Git、Vulkan SDK 1.4.309.0+、支持所需设备功能的 GPU |

现有本机环境已核对为 Python 3.10.20、PyTorch 2.2.0+cu121、NumPy 1.26.4，且 `pointnet2_batch_cuda` 可导入。环境文件用于描述安装依赖，不代表所有 GPU/驱动组合已经测试。

以下命令使用 **Anaconda Prompt / cmd.exe**；先设置实际仓库目录：

```bat
set PROJECT_ROOT=E:\vk_lod_clusters1\t1
cd /d "%PROJECT_ROOT%"
conda env create -f tools\pointnext_lod\environment_pointnext.yml
conda activate gpt-pointnext
python -c "import torch,numpy; print(torch.__version__,torch.version.cuda,numpy.__version__,torch.cuda.is_available())"
```

已有可用环境直接激活即可。Conda 的 CUDA runtime 用于运行 PyTorch；**编译 CUDA 扩展还需要系统 CUDA Toolkit 和 C++ 编译器**。在 Visual Studio Native Tools 命令行中激活 Conda 后：

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

切换终端后需重新设置 `PROJECT_ROOT` 并激活环境。扩展二进制与 Python、PyTorch、CUDA、编译器及平台相关。现有本机 `.pyd` 保留使用，但不再上传 Git；新机器应按自己的环境编译。

开放分支环境独立创建：

```bat
cd /d "%PROJECT_ROOT%"
conda env create -f tools\pointclipv2\environment_pointclipv2.yml
conda activate PointCLIPV2_LOD
python -c "import torch,sys; sys.path.insert(0,r'tools/pointclipv2/PointCLIP_V2/zeroshot_cls'); from clip import clip; from trainers.mv_utils_zs import Realistic_Projection; print(torch.__version__,torch.cuda.is_available(),'PointCLIP imports OK')"
```

工业投影路径已有 PyTorch scatter 回退，因此不需要直接照搬旧上游 `torch-scatter==2.0.9` 的依赖。首次加载 ViT-B/16 可能下载权重，离线使用前需准备好 CLIP 缓存。

## 2. 准备工业分类数据

训练需要类别标签，场景推理只需要网格几何，两类输入不能混为一谈。

```text
datasets/industrial_part_14class/        # 本地数据，不上传 Git
  classes.txt
  train_list.txt
  val_list.txt                         # 新训练建议独立验证集
  test_list.txt
  screws_bolts_studs/part_0001.npy
  nuts/part_0002.npy
  ...
```

每个 `.npy` 保存有限浮点数的 `[N,3]` XYZ 点云。对零件网格做面积加权表面采样，并采用与推理一致的归一化：居中后缩放到单位球。活动配置使用 8,192 点，数据集会对不同点数重采样；额外列被忽略，当前不使用法向作为监督输入。

`classes.txt` 用真正的制表符分隔，0–13 的类别顺序必须与 [industrial_part_14classes.txt](../../tools/pointnext_lod/industrial_part_14classes.txt) 一致：

```text
0<TAB>screws_bolts_studs
1<TAB>nuts
...
13<TAB>springs
```

列表每行为 `相对路径.npy<TAB>类别编号<TAB>类别名`，例如：

```text
screws_bolts_studs/part_0001.npy<TAB>0<TAB>screws_bolts_studs
```

这里 `<TAB>` 是格式说明，文件中应写实际制表符。按原始 CAD 零件、家族或来源装配划分，避免同一零件的重采样/旋转副本跨集合。仓库有数据集适配器，**没有原始 CAD 到训练数据的转换器**，训练前需要自行准备带标签、归一化的 `.npy` 数据。

历史配置的验证和测试共用 `test`。新训练可以准备独立 `val_list.txt`，通过 `dataset.val.list_file=val_list.txt` 和 `dataset.test.list_file=test_list.txt` 显式指定不同列表；适配器已经支持这一参数，无需改变算法。已有准确率仍按历史共享划分解释。

## 3. 准备完整训练工程

**当前仓库的 PointNeXt 子目录包含推理实现与工业配置，但没有 `examples/classification/main.py` 训练入口。** 原始日志是在仓库外的完整 PointNeXt 工程中运行该入口，不能在本仓库直接执行缺失路径。

请准备兼容的 [完整 PointNeXt 工程](https://github.com/guochengqian/PointNeXt)，包含 classification examples 和父级默认配置。[作者训练入口](https://github.com/guochengqian/PointNeXt/blob/master/examples/classification/main.py) 支持 `--cfg` 与配置覆盖。上游接口会随版本变化，需要记录提交并验证工业配置兼容性；历史训练所用外部入口未随本仓库提供。

```bat
set TRAINING_ROOT=D:\PointNeXt-training
git clone --recursive https://github.com/guochengqian/PointNeXt.git "%TRAINING_ROOT%"
cd /d "%TRAINING_ROOT%"
git rev-parse HEAD
```

将本项目工业适配器和配置接入外部工程：

```bat
copy "%PROJECT_ROOT%\tools\pointnext_lod\PointNeXt\openpoints\dataset\industrial_part.py" "%TRAINING_ROOT%\openpoints\dataset\industrial_part.py"
if not exist "%TRAINING_ROOT%\cfgs\industrial_part" mkdir "%TRAINING_ROOT%\cfgs\industrial_part"
copy "%PROJECT_ROOT%\tools\pointnext_lod\PointNeXt\cfgs\industrial_part\pointnext-s-14class-randomrot-8192-fast.yaml" "%TRAINING_ROOT%\cfgs\industrial_part\pointnext-s-14class-randomrot-8192-fast.yaml"
```

在**外部训练工程** `openpoints/dataset/__init__.py` 中注册适配器；已有相同导入时无需重复添加：

```python
from .industrial_part import IndustrialPartPointCloud
```

外部完整工程所需的额外训练依赖和 CUDA 扩展按对应版本安装指南准备，并先验证工业适配器可导入。这里新增的环境文件面向本仓库模块，不代替任意上游版本的全部训练依赖。

## 4. 监督训练与检查点选择

在兼容的完整训练工程中：

```bat
conda activate gpt-pointnext
cd /d "%TRAINING_ROOT%"
set CUDA_VISIBLE_DEVICES=0
python examples/classification/main.py --cfg cfgs/industrial_part/pointnext-s-14class-randomrot-8192-fast.yaml ^
  dataset.common.data_dir="%PROJECT_ROOT%\datasets\industrial_part_14class" ^
  dataset.val.list_file=val_list.txt dataset.test.list_file=test_list.txt ^
  root_dir="%PROJECT_ROOT%\.artifacts\training\pointnext" seed=3098
```

这里给出已核对的入口接口与数据适配器契约，本次整理没有安装或重跑外部训练工程。先从单 GPU 开始；Windows 分布式训练还需要兼容的 backend 设置，已有日志不能证明多 GPU Windows 可用。

配置为 14 类、8,192 XYZ 点、100 轮、batch 16、AdamW、余弦调度、随机旋转、标签平滑及 AMP。显存不足时可通过命令覆盖 batch size，并记录有效配置。通过验证集选最佳检查点，之后在独立测试集评估。

训练结束后，从输出的 run 目录找到**确切的 best 文件**并复制到推理位置：

```bat
cd /d "%PROJECT_ROOT%"
if not exist models\pointnext mkdir models\pointnext
copy "D:\path\to\your\run\checkpoint\your_run_ckpt_best.pth" "models\pointnext\industrial-part-14class-best.pth"
```

将源路径替换成真实训练输出，也可以指定外部权重：

```bat
set POINTNEXT_CKPT=D:\path\to\your\best_checkpoint.pth
```

检查点需匹配当前编码器/分类头、14 类输出及类别索引顺序，不能把通用 ModelNet40 权重改名后直接使用。latest 和 TensorBoard 放入 `.artifacts/`，选定训练日志与配置放入 `experiments/`。已有兼容工业 best 时可直接复用，不必为每个新场景重新训练。

## 5. 准备待渲染模型

场景放在 `_downloaded_resources/<scene>.glb`。渲染器接受 `.gltf`、`.glb` 和场景 `.cfg`，根目录语义批处理扫描 **GLB 文件**。

Python 路径要求 GLB 2.0、可读取的 POSITION accessor 与三角形列表 primitive；分析与渲染之间应保持 mesh/node 身份不变，统一单位，避免在分析输入中使用该路径不支持的压缩表示。渲染器能处理材质/相机或某扩展，不意味着 Python 也支持所有 glTF 扩展。STEP/IGES 不能直接打开，应先导出三角网格 GLB，分类训练数据另行准备。

## 6. 多源语义结构分析

仓库根目录、Anaconda Prompt：

```bat
cd /d "%PROJECT_ROOT%"
set POINTNEXT_CONDA_ENV=gpt-pointnext
set POINTCLIP_CONDA_ENV=PointCLIPV2_LOD
run_all_glb_pointnext_pointclipv2_fused.bat
```

依次执行全量 PointNeXt 分类和几何提取、门控 PointCLIP V2、语义结构融合。置信度低于 0.70 **或** top1–top2 间隔低于 0.08 时触发开放推断。角色证据与结构描述生成 P1–P10、保留比例和误差/保护权重。

每场景关键产物是 `lod_analysis_outputs/<scene>_lod_constraints_fused.csv`。分类、候选和 merged 中间 CSV 对诊断与重新融合有价值，本次保留在本地并由 Git 忽略。

批处理仅复用通过内容哈希核验的完整输出。监督阶段检查场景、权重、推理配置、类别顺序、代码及采样设置；候选开放阶段检查场景、监督 CSV、提示词与代码/设置。没有来源侧车文件的旧输出、被截断的输出及依赖变化后的输出都会重算。融合阶段重新生成策略。可设置 `SEMANTIC_OUTPUT_DIR` 使用独立目录；渲染器读取策略时仍应指向相应结果路径。第一次使用新权重必须生成新策略，并以 `--processingonly 1 --autoloadcache 0` 重建 LOD，旧缓存不能当作新模型结果。

`a b c o1778 o3049` 的六组配置对比需先生成 full PointCLIP：

```bat
conda activate PointCLIPV2_LOD
tools\pointclipv2\run_pointclipv2_full_5_glbs.bat
tools\pointclipv2\run_lod_ablation_5_glbs.bat
```

对比输出位于 `lod_analysis_outputs/lod_ablation/`。full 开放推断供比较配置使用，默认 fused 仍使用 candidate-only。

## 7. 编译引擎并构建 LOD 缓存

```bat
cd /d "%PROJECT_ROOT%"
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

当前 `t1` 目录产物为 `_bin/Release/t1.exe`，target 名随源码目录名变化。`nvpro_core2` 从本地寻找或按固定提交下载；实际 Vulkan 功能需求参见 [main.cpp](../../src/app/main.cpp)。

完成语义分析后，显式构建策略控制的 LOD：

```bat
_bin\Release\t1.exe --scene "_downloaded_resources\your_scene.glb" ^
  --semanticlodmode 5 --processingonly 1 --autoloadcache 0 --processingpartial 1
```

将 `your_scene` 替换成实际名称。CPU 导入几何、匹配策略、构建约束集群 LOD，保存 `your_scene.glb.zippp`。partial 允许保留可续处理状态。**语义脚本只生成策略文件，渲染引擎的预处理才生成 LOD 缓存。**

`--semanticlodmode` 为六组配置的 0–5，完整融合是 5，0 是 PointNeXt 而非关闭语义。加载器先找所选组，缺失后回退 fused/普通策略，必须检查日志中的实际 CSV 与匹配信息。策略哈希参与缓存兼容性检查，不同实验可用不同 `--cachesuffix`。

## 8. 打开场景与运行时设置

```bat
_bin\Release\t1.exe --scene "_downloaded_resources\your_scene.glb" --semanticlodmode 5
```

启用自动集群组流式加载，并指定几何/传输预算：

```bat
_bin\Release\t1.exe --scene "_downloaded_resources\your_scene.glb" ^
  --semanticlodmode 5 --streaming 1 --maxgeomegabytes 2048 --maxtransfermegabytes 64
```

预算只是示例起点，不是性能推荐或适配保证。GPU 自动产生需求请求，CPU 异步准备并上传组数据，更新驻留映射；最低细节和搜索层次提供基础可访问数据，旧的未访问细组可回收。

同时启用自适应双阶段剔除：

```bat
_bin\Release\t1.exe --scene "_downloaded_resources\your_scene.glb" ^
  --semanticlodmode 5 --streaming 1 --culling 1 ^
  --twopassculling 1 --adaptivetwopassculling 1
```

自适应开关不会单独开启第二阶段；固定双阶段使用 `--twopassculling 1 --adaptivetwopassculling 0`。UI 可查看 Semantic LOD Policy Distribution、策略可视化、Statistics、Streaming memory 与 Profiler。`--loderror` 控制显示精度阈值；装配筛选依赖实际导入层级及其屏幕尺寸条件。

## 常见问题

| 现象 | 检查方法 |
| :--- | :--- |
| 找不到训练 `main.py` | 使用完整且兼容的外部 PointNeXt 训练工程 |
| CUDA 扩展导入/DLL 失败 | 核对 Python/PyTorch/CUDA，编译扩展并检查 DLL 搜索路径 |
| CUDA 不可用 | 核对所激活环境、驱动及 CUDA 版 PyTorch |
| 数据列表或类别不匹配 | 使用绝对数据路径、真实 TAB 分隔和固定 14 类顺序 |
| 检查点张量尺寸不匹配 | 核对 encoder/head，不能直接用通用 40 类权重 |
| 对比配置似乎没有生效 | 检查实际 CSV 路径，所选组缺失可能回退 |
| 新设置没有反映出来 | 按需重做神经分析，再重建引擎缓存 |
| 第二阶段不运行 | 打开剔除与双阶段，并检查规模/矩阵变化阈值 |
| 装配预筛选没有收益 | 确认模型含可用装配层级，而非平铺节点 |
| 流式加载延迟较大 | 检查驻留、传输、请求预算及粗精度回退 |

文件分类见 [仓库目录说明](../repository-layout.md)，实现与实验边界见 [技术审查](../../analysis_reports/PROJECT_REVIEW_20261014.md)。
