# 工业语义结构 LOD 项目代码与实验审查

审查日期：2026-10-14。对象：当前 `E:\vk_lod_clusters1\t1` 工作区、提供的硕士论文与图片资料。审查覆盖项目自己的 C++/GLSL 主链路、Python 语义分析与融合工具、配置、训练日志及五场景实验输出；第三方训练框架不做逐行完整审计。本文是静态代码分析与已有数据核验，没有重新训练、编译或执行 GPU 性能基准。修改范围仅为 README、图片、证据元数据及本报告。

## 1. 项目定位与 GitHub 标题

本项目最有价值的叙述主线，是工业零件的语义与结构分析如何转化为集群组简化约束，并最终参与 GPU 侧局部 LOD 选择。只使用 “Vulkan Cluster LOD Renderer” 会遗漏论文第三章的主要工作；只使用 “ISPointNet” 又会遗漏完整渲染系统，并让读者误以为仓库主要发布一个独立的新网络架构。

建议使用以下组合，README 已采用这一展示标题：

| GitHub 字段 | 建议内容 |
| :--- | :--- |
| Repository name | `industrial-semantic-lod` |
| README title | **Industrial Semantic LOD** |
| English subtitle | Semantic and Structure-Aware Cluster LOD for GPU-Driven Industrial Rendering |
| 中文副标题 | 工业三维模型的语义结构感知集群 LOD 与 GPU 驱动渲染 |
| About description | Semantic and structure-aware cluster LOD for industrial 3D models, integrating PointNeXt, gated PointCLIP V2 inference, and Vulkan GPU-driven rendering. |
| Topics | `vulkan`, `gpu-driven-rendering`, `mesh-shaders`, `level-of-detail`, `mesh-simplification`, `industrial-visualization`, `point-cloud`, `pointnext`, `pointclip`, `gltf` |

备选仓库名 `vk-industrial-lod` 更强调图形学，`semantic-cluster-lod` 更强调方法。优先推荐 `industrial-semantic-lod`，因为应用对象、语义与 LOD 都能从名字中读出。这里的名称建议不表示 GitHub 远端已经改名或推送。

当前 [CMakeLists.txt](../CMakeLists.txt) 从源码目录名生成 target，GitHub 仓库名、README 标题与本地 target 是不同概念。若将本地目录也改名，重新配置后可执行文件名称会变化。现有本地程序位于 `_bin/Release/t1.exe`；README 已纠正旧版 `build/Release/t1.exe` 示例。

## 2. 总体架构与论文对应关系

系统分成离线语义分析、CPU 集群构建与在线 GPU 渲染三段，通过 CSV、缓存及共享 CPU/GPU 结构连接。

| 论文内容 | 实现入口 | 实际工作 |
| :--- | :--- | :--- |
| 工业语义分析 | [PointNeXt 配置](../tools/pointnext_lod/PointNeXt/cfgs/industrial_part/pointnext-s-14class-randomrot-8192-fast.yaml)、[大 GLB 分析脚本](../tools/pointnext_lod/analyze_large_glb_parts_pointnext.py) | 采样、分类、置信度与几何统计 |
| 开放词汇补充 | [PointCLIP 推断](../tools/pointclipv2/run_pointclipv2_zeroshot_glb.py) | 候选门控、多视图投影、图文匹配 |
| 结构角色与策略生成 | [策略生成器](../tools/pointnext_lod/build_lod_constraints_from_analysis.py)、[融合入口](../tools/pointclipv2/fuse_pointnext_pointclip_lod.py) | 证据融合、结构角色映射、P1–P10 配置输出 |
| 语义策略接入 | [scene_semantic_lod.cpp](../src/scene/scene_semantic_lod.cpp) | CSV 搜索、行匹配、连续参数派生与策略哈希 |
| 集群 LOD 构建 | [clusterlod.cpp](../src/scene/clusterlod.cpp)、[meshlod_build.h](../src/meshlod/meshlod_build.h) | 集群分组、简化、重聚类与生成关系维护 |
| 局部结构保护 | [meshlod_simplify.h](../src/meshlod/meshlod_simplify.h) | 特征启发式、属性加权与顶点保护 |
| 多粒度可见性 | [scene_gltf.cpp](../src/scene/scene_gltf.cpp)、[assembly_visibility.comp.glsl](../shaders/traversal/assembly_visibility.comp.glsl) | 装配节点与 GPU 粗粒度筛选 |
| GPU 驱动绘制 | [renderer_clusters_lod.cpp](../src/renderer/renderer_clusters_lod.cpp)、[traversal_run.comp.glsl](../shaders/traversal/traversal_run.comp.glsl)、[clusters.mesh.glsl](../shaders/render/clusters.mesh.glsl) | 遍历、选择、任务压缩与间接网格绘制 |
| 流式支持 | [streaming.cpp](../src/streaming/streaming.cpp)、[stream_agefilter_groups.comp.glsl](../shaders/streaming/stream_agefilter_groups.comp.glsl) | 请求、传输、驻留更新及年龄回收 |

论文将三类证据统一描述为 ISPointNet，当前可执行实现应说明：监督分支是 PointNeXt 的 `PointNextEncoder`，开放分支基于 PointCLIP V2，结构分支与融合是确定性分析和规则，而不是另有一个端到端训练的三分支融合网络。论文中的整体框架名称可以保留，但应准确区分基础编码器与面向 LOD 的新增集成。

## 3. 离线语义分析

### 3.1 PointNeXt 工业配置

活动配置使用 8,192 点、14 类、XYZ 输入，训练 batch 为 16、评估 batch 为 32，训练 100 轮、每 5 轮评估。优化器为 AdamW，初始学习率 0.001、weight decay 0.05，余弦调度，标签平滑 0.2，并启用 AMP。训练增强包含三轴随机旋转与尺度/平移。

编码器宽度 32，`strides=[1,2,2,2,2,1]`，半径 0.15、缩放 1.5，球邻域上限 32，最大聚合与残差连接。数据集实际只读取前三列作为位置，并将位置用于输入特征；当前不能将法向描述成已被监督输入使用。

大 GLB 脚本直接解析 JSON/BIN，用 mmap 逐个读取带 mesh 的 node，进行面积加权表面采样、归一化及批量推断，逐步写出 CSV/JSON。这一处理方式避免同时展开整场景全部点云。脚本另提供 `grid-group` 模式，将空间组的预测传播回成员；根目录默认批处理使用逐 mesh 模式。两者不能作为完全相同的逐零件识别结果混报。

### 3.2 门控 PointCLIP V2

候选门控跳过置信度至少 0.70 且 top1–top2 间隔至少 0.08 的记录，其余送入开放分支。默认使用 CLIP ViT-B/16，按多视图投影获得视觉特征，逐视图归一化后平均聚合并再次归一化；文本通过类别及别名的多个模板聚合。

匹配实际执行 `100 * normalized_image_features @ normalized_text_features.T`，再 softmax，导出 top-k 与 top1–top2 间隔。CSV 的 score 是当前词库上的 softmax 值，不能直接称为原始余弦相似度，也不能当成经过校准的正确率。

默认候选推断排除 `micro_uncertain` 角色，而全量五场景批处理传入 `--exclude-roles ""`，包含该角色。这意味着全量与候选输入的差异不只有记录数量，也涉及词库过滤，比较时需要保留这一条件。

### 3.3 结构证据与融合

策略生成器通过场景记录分位数与网格统计得到尺度、细节、细长、板状、环盘、紧致及不确定性等连续响应，再与监督语义和开放语义角色证据匹配。融合是角色先验、可靠性权重和修正规则的组合；结构角色是面向简化控制的代理，不能在没有 BOM/工程标注时视为零件真实功能。

融合与比较脚本使用 `(order, node_index, mesh_index)` 连接记录。这一键能保持当前五场景的记录对应，但输入遍历顺序变化后需要重新生成相互配套的文件。根目录脚本跳过已有推断结果，模型或提示词变化后不应直接把旧 CSV 当作新实验。

### 3.4 与论文表述需要核对的实现细节

在当前大 GLB 脚本中，`basic_stats()` 的 `obb_size_long/mid/short` 实际是排序后的 AABB 三个轴长，并非计算得到的定向包围盒；`mesh_volume` 留空。论文若将这条活动路径描述成计算真实 OBB 和实体体积，需要调整表述或另提供对应实现证据。

PointNeXt 与 PointCLIP 的 `mesh_for_node()` 仅使用当前 node 的局部变换，没有在这条路径累乘父节点变换。对带父级旋转或缩放的通用 glTF，尺度与朝向可能与 C++ 场景导入不一致。这里是代码层面的适用范围观察；没有据此断言当前五场景都受到同样影响。

两条 Python 读取路径按三角面解释 primitive 的索引；不能将 C++ 的 glTF/meshopt 导入能力自动扩展为 Python 分析工具已支持全部 primitive 模式或压缩扩展。

## 4. P1–P10 如何影响真实构建

P 编号表示保护策略类别，而不是从近到远的十个几何 LOD。当前策略表如下，比例为输出的 near/mid/far 控制值：

| P | 角色 | near / mid / far | 输出 allow_cull | 屏幕误差权重 |
| ---: | :--- | :--- | :--- | ---: |
| 1 | micro_uncertain | 0.30 / 0.10 / 0.03 | True | 0.40 |
| 2 | repeated_fastener | 0.42 / 0.18 / 0.06 | True | 0.55 |
| 3 | large_static_bulk | 0.55 / 0.30 / 0.12 | False | 0.70 |
| 4 | ordinary_low_detail | 0.60 / 0.34 / 0.15 | False | 0.82 |
| 5 | balanced_visible | 0.68 / 0.42 / 0.20 | False | 0.98 |
| 6 | high_detail_shape | 0.76 / 0.50 / 0.28 | False | 1.12 |
| 7 | interface_fluid | 0.82 / 0.58 / 0.36 | False | 1.25 |
| 8 | structural_control | 0.86 / 0.64 / 0.42 | False | 1.38 |
| 9 | motion_precision | 0.92 / 0.74 / 0.52 | False | 1.55 |
| 10 | critical_preserve | 0.96 / 0.84 / 0.65 | False | 1.80 |

C++ 的 `derivePolicy()` 并非原样使用三个比例切换距离档位：它将 near 值与稳定比例混合生成 `simplifyRatio`，由 near/mid/far 差异派生层次衰减与最低保留率，再依据 P 派生误差合并、特征权重、软约束、硬保护预算和分组规模。`lodErrorScale` 来自屏幕误差权重的倒数并经过限幅。README 因此描述其控制关系，不承诺输出网格一定达到表中比例。

`applySemanticPolicyToConfig()` 将这些参数写入 `clodConfig`；`buildGeometryLod()` 在实际构建前调用它，并保存运行时 error scale 和策略 flags。离线简化与在线误差选择都可能受策略影响，所以后续质量评价需要把这两类作用拆开。

策略文件搜索先尝试所选六组后缀，再尝试 fused，最后普通 constraints，并在多个相邻输出目录寻找。默认 mode 为 5；mode 0 是 PointNeXt，并非纯几何基线。若所选策略文件缺失，运行时可能回退，不能仅凭 UI 的选择名称判断实验用了哪份数据。

语义行哈希参与 geometry 的缓存匹配，几何/策略不一致时有 stale-cache 重建路径。不同实验保留独立缓存后缀能减少混淆；这一机制不意味着任何旧缓存或旧推断 CSV 在任意改动后都必然有效。

## 5. 局部结构约束与层次构建

`meshlod_simplify.h` 构造焊接拓扑与边关系，检测边界和非流形顶点，通过法向夹角识别锐边，通过连通边界的近圆性质建立圆环线索，并结合邻接法向与短边尺度构造柱面和薄壁提示。

特征响应组合为 per-vertex importance，并作为附加属性进入 `meshopt_simplifyWithAttributes`；选出的关键顶点与跨组边界保护共同构成约束。非流形顶点直接保护，其余候选按重要度排序，在策略给定的保护预算内设置 protect 标志。约束饱和时有 permissive/sloppy 等回退路径，需要在质量实验中记录实际使用情况。

这里的圆孔与薄壁检测是启发式。封闭 CAD 通孔不一定对应开放边界环，短边提示也不等同于物理壁厚测量。现有 feature metrics 描述输入检测与保护配置，不能直接拿来证明简化后的孔保持率或薄壁完整性。

层次构建在集群组范围内简化，再重新聚类并维护粗集群对应的生成组。几何生成关联与用于搜索的空间层次职责不同：前者决定局部精度替代与驻留回退，后者减少搜索任务。论文中宜保留这一区分。

## 6. GPU 渲染、可见性与流式管理

### 6.1 任务组织

应用负责场景状态、资源生命周期、参数更新和命令调度。CPU 侧生成缓存与运行时数据布局，计算着色器执行层次搜索及候选筛选，子组投票、前缀定位和批量原子申请构造连续任务列表，再生成间接网格绘制参数。网格着色器根据集群索引展开局部顶点和图元，片段阶段完成着色及深度测试。

这条链路符合 GPU 驱动任务生成的含义，但不意味着 CPU 完全不参与。CPU 仍负责初始化、同步、异步请求消费及传输调度。没有受控 CPU 对照时，不能量化声称相对 CPU 方案的加速比例。

### 6.2 装配级预筛选

C++ 从场景层级建立符合深度、后代实例数量及包围有效性要求的装配节点，GPU 对其进行视锥、屏幕尺寸和 Hi-Z 测试，符合小投影条件时标记粗 LOD 快速路径。默认 `assemblymininstances=8`、`assemblylodpixels=24`。

重新读取 GLB 元数据确认：`a` 有 14,474 个包含 children 的 node；其余四个文件均为平铺节点。因此全部五场景可用于常规集群 LOD 与渲染评价，装配模块应重点使用 `a`，并单独报告没有层级时的行为。不能以场景零件很多推断其一定能获得装配级筛选收益。

### 6.3 双阶段遮挡

第一阶段以历史视图/Hi-Z 筛选，但通过的几何用当前变换绘制。启用第二阶段时，第一阶段深度构建中间 Hi-Z，重新判断候选并补绘新显露几何，随后更新最终历史深度。

默认 `useTwoPassCulling=false`、`useAdaptiveTwoPassCulling=true`；后者并不独立开启第二阶段。自适应判定还要求剔除与双阶段开关启用、未冻结剔除，并满足集群数与矩阵变化阈值。默认最小集群数 4,096，矩阵最大元素差阈值 0.0005。这是计算量控制启发式，性能和恢复效果需通过静止、缓慢运动、快速转向和显露轨迹验证。

### 6.4 按需驻留

预加载路径一次上传所需几何。流式路径以集群组为单位进行请求、加载、上传、地址补丁与卸载；基础层次和最低细节提供可访问表示，细组尚未驻留时依生成关系保留粗候选。回收依据访问年龄与预算，而非物理时间。

显存预算、传输预算及每帧请求上限均已注册，但“具备预算控制”不等于已证明任意预算下可保持目标质量或帧率。运行时工作集、请求延迟和质量收敛曲线仍需实测。

### 6.5 策略位与统计口径

全仓搜索 `SEMANTIC_LOD_ALLOW_CULL_BIT` 只找到共享定义和 C++ 写入，没有找到 GPU 读取该位决定可见性的分支。README 保留该输出统计，但明确它不是实际剔除率。

`numRenderClusters` 使用 atomicMax 记录阶段任务峰值，`numRenderedClusters` 累加实际提交量，并另有 pass0/pass1 字段。双阶段总工作量不能用峰值替代。三角形量应使用实际集群图元计数，不宜使用集群数乘 128。任务容量不足导致截断也不能被当成剔除收益。

## 7. 已完成实验与核验结果

### 7.1 分类训练

当前找到一组完整 14 类 PointNeXt 训练记录，种子 3098，61,711 训练样本、15,372 评估样本、100 轮。最佳检查点第 85 轮，重新加载后 OA 96.19%、mAcc 93.86%。日志还包含另一段 OA 96.14%、mAcc 94.10% 的评估；README 采用明确加载最佳检查点后的最后一段，不混用两段指标。

验证和测试使用同一划分，评估过程还可能随机重采样。因此准确率属于已有开发结果，不构成独立测试泛化证据。虽然仓库包含 PointNet、PointNet++ 和多个 PointNeXt YAML，本次只确认这一组完整训练结果，配置存在不能视为各基线实验已完成。

### 7.2 五场景推断与六组配置

五个模型均有 PointNeXt、candidate PointCLIP、full PointCLIP、融合策略和六组对比输出。逐份读取 30 个 CSV，验证行数、唯一 mesh 数与各 P 等级占比；每组均有 153,616 条，所有比例与已有汇总在舍入误差内一致。[核验元数据](../docs/data/readme-evidence.json)

| 场景 | Mesh | 原始三角形 | PointCLIP 候选 |
| :--- | ---: | ---: | ---: |
| a | 24,259 | 50,138,563 | 12,252 |
| b | 63,063 | 78,514,530 | 29,115 |
| c | 61,467 | 126,985,435 | 35,437 |
| o1778 | 1,778 | 5,469,879 | 920 |
| o3049 | 3,049 | 6,087,168 | 1,762 |
| 合计 | 153,616 | 267,195,575 | 79,486 |

原始三角形数量来自文件的三角 primitive accessor，不能称为单帧渲染量；mesh 记录也不代表来自独立模型家族的 153,616 个分类测试样本。

最值得展示的已有观察为：N 加 G 后低保护占比从 73.78% 变为 45.50%；N 加候选 C 后策略变化 8.41%；N+C 加 G 后变化 30.82%；完整配置低保护 46.27%、高保护 22.08%。这些是策略分配的证据，不预设“更保守”即“更正确”。

### 7.3 六组比较为何不能直接称严格分支消融

完整模式调用 `infer_structural_semantics()`，其他模式调用 `infer_ablation_semantics()`。开放证据权重分别出现 `0.18 * pointclip_score` 和 `0.35 + 0.65 * pointclip_score`，结构权重及部分规则也不同。纯 PointCLIP 配置仍携带监督 confidence 列，C++ 又用该列调节低置信保护；全量与候选开放词库过滤也不同。

因此现有六组适合表达工程策略配置的行为差异，不能把完整组相对删减组的全部差异归因于单一分支。本文不修改这些算法，只在 README 使用准确的实验名称。

### 7.4 暂不能写成已经完成的结果

本次未识别出完整、可配对复核的 GPU FPS/帧时基准、相同三角预算的几何质量结果、双阶段显露恢复统计、流式显存预算曲线、独立角色标注准确率，以及多模型分类基线结果。代码有实现与 profiler，并不自动构成已完成实验。

提供的论文正文主要包含第二至第四章方法内容；实验章节的组织安排出现在目录/研究安排中，但没有找到可直接提取并核验的完整实验表格。图片资料为 12 页方法图，含装配展示、监督/开放分支、层次、双阶段与任务压缩示意。README 只将它们标作方法示意图，不当成真实程序截图或定量简化结果。

## 8. 后续最值得补充的实验

以下仅为建议，没有更改实现或执行新实验。

1. **独立分类评估。** 按 CAD 来源模型或零件家族建立 train/validation/test，记录宏 F1、每类召回、混淆矩阵及多个种子。旧共享 test 可作为开发集，另准备未参与选择的测试集合。
2. **结构质量与预算。** 选孔洞、齿廓、薄壁、细长件，在相同实际三角数量下比较统一策略、模型级策略、局部约束及二者结合；报告距离、法向、孔径/边界误差与失败案例。
3. **统一融合消融。** 若要做分支贡献研究，固定同一融合函数、角色映射和下游参数，只屏蔽输入分支；增加纯几何与纯结构基线，并锁定词库与阈值。
4. **动态可见性。** 固定相机轨迹比较单阶段、固定双阶段和自适应双阶段，联合报告 GPU 毫秒、实际提交量、错误漏绘与恢复延迟。
5. **装配及流式。** 对有真实层级的场景评估装配模块，对大场景扫描驻留与传输预算，区分冷启动、稳定工作集和快速往返；补充质量收敛与帧时尾部。

README 后续优先加入真实应用截图、固定视角局部对比和质量—GPU 时间曲线。这些资料比仅扩大 P 等级分布展示更能解释方法的下游价值。

## 9. 本次文档改动与验证

主页面改为英文，单独提供中文版本，包含展示标题、双语切换、方法流程、训练曲线、五场景规模、六组配置、正确运行路径、参数入口及源码导航。保留基础组件归属说明，不添加虚构论文发表状态、性能徽章或统一许可证。

旧 README 中不对应当前注册代码的 `learnedimportance` 等参数没有继续列出；原先指向空 `docs` 目录的文档链接改为实际存在的源码与报告。旧流程文档中“优先 fused”的说法与当前所选配置优先的搜索逻辑存在时间差，主 README 以当前代码为准。

展示图直接提取两个原始嵌入图片，保持内容不变；两张统计图由原日志/汇总生成。核验覆盖 30 个策略文件、五个 GLB 元数据与最佳检查点日志，并将公开汇总来源的 SHA-256 写入证据文件。

上游基础能力参考 [NVIDIA Vulkan Cluster LOD](https://github.com/nvpro-samples/vk_lod_clusters)、[meshoptimizer](https://github.com/zeux/meshoptimizer)、[PointNeXt](https://github.com/guochengqian/PointNeXt) 及本地 PointCLIP V2 作者说明。这里没有声称完成与上游全部提交的逐行归属比对。
