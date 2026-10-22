# ISPointNet监督分支与PointNeXt的实际差异

核验日期：2026-10-21。只读取代码、配置与部署记录，没有训练、推理或更改实验结果。

## 核验结论

项目确实有多项工业适配和多源融合改动，不能将整个ISPointNet概括为“只修改采样步长”。但当前5.2的两种PointNeXt实验配置共用这些工业训练适配，model字典唯一的差异是strides。因此，5.2中的两个实验配置差异、工业任务适配与完整框架贡献，必须分开解释。

## 配置级证据

对保存的官方ModelNet40 PointNeXt-S模型配置、当前实验JSON和部署推理YAML进行结构化比较，结果见configuration_audit.json。

- 保存的官方配置与当前pointnext_s模型字典只在分类头num_classes上不同：40变为14；encoder_args完全相同。
- industrial.model与pointnext_s.model完全相同，不是另一套改进骨干。
- pointnext_s与pointnext_stride4唯一的模型配置差异为strides，从[1,2,2,2,2,1]变为[1,4,4,4,4,1]。
- 当前部署推理YAML的model与实验pointnext_s.model相同。
- gpt与t1中的pointnext.py、旧RandomRotate实现、工业数据集实现逐文件SHA256相同。这证明两个工程所用这些文件一致，不单凭该哈希断言与所有上游版本相同。

官方配置本身采用二分下采样。源码构造函数存在四分下采样默认值，也不等于实验实际使用该默认值；实验显式传入strides，因此以实际配置为准。

## 修改分层

| 工作 | 已有实现或记录 | 贡献与实验边界 |
|---|---|---|
| 工业类别整理 | 90类来源整理与14类工业映射，分类头适配14类 | 数据与任务适配；四种当前编码器共享14类数据 |
| 高密度输入 | 旧工业配置4096点，后续配置8192点 | 输入与训练配置适配；当前对比四种方法均输入8192点 |
| 三轴随机旋转 | 训练阶段Rx/Ry/Rz组合，每轴角度在[-pi,pi]采样 | 确实存在；当前四种方法共用，5.2不能单独量化旋转贡献；不是监督推理多旋转投票 |
| 缩放、平移与训练实现 | 当前统一缩放0.8～1.2、平移±0.2，AMP、梯度累积、AdamW等 | 训练适配和执行机制；新旧配方存在差别，不全属于新网络算子 |
| 采样与标准化 | 面积加权表面采样、质心归零、单位球归一化 | 实际预处理流程；不将常规操作本身宣称为新算子 |
| 层次局部编码 | FPS、球查询、相对位置归一化、局部聚合、残差等 | 当前官方来源配置和骨干已有，不宜整体写成自研新模块 |
| 可靠性输出 | 从softmax提取Top-1、Top-2与差值，用于候选和融合 | 由分类器构造的下游接口，不是新增可训练可靠性分支；不自动构成概率校准 |
| 分类实验规范 | 分组划分、冲突排除、独立验证、验证Macro-F1选权重、两个种子 | 评价协议改进，提高结果可追溯性；不等于提高网络表征能力的新结构 |
| GLB部署与装配输入 | 节点解析、世界变换、可选分组几何采样、批量推断及缓存追溯 | 工程与输入组织改动；不能归入分类骨干结构变化 |
| 开放语义v2 | 工业词库、深度与表面视图、姿态/局部补充、视图一致性、分层拒识 | 开放语义分支改动；OPEN_SEMANTIC_V2.md注明未训练新的PointNeXt/CLIP网络 |
| 显式几何与装配树 | 尺度、形态、复杂度、祖先与局部装配上下文、身份与功能分离 | 完整ISPointNet语义结构推断的组成；5.2闭集分类不含这些输入 |
| P1～P10 | 融合证据转为整体预算与局部保护约束 | 下游LOD策略；V5仍属于候选设计，不能以分配结果替代真实质量/渲染验证 |

## 5.2究竟验证了什么

第一层：在相同工业数据、输入点数和训练协议下，选择监督编码器。当前PointNeXt-S配置的内部分类均值最好。

第二层：在同一编码骨干中比较两种下采样步长。步长2的平均OA/mAcc/Macro-F1比步长4高0.61/0.93/1.21个百分点。这是当前采样配置的分类收益，不是所有工业适配、多源融合或自研全新骨干的总收益。

当前四种模型共享随机旋转、高密度输入与工业数据，不能由模型间差值推出这些共同设置各自带来的提升。完整ISPointNet是否改善身份判断、装配功能与LOD保持，应依靠对应场景与消融实验。

## 推荐论文表述

本文以PointNeXt层次化局部编码框架为基础，构建面向工业零部件的ISPointNet监督语义分支，完成工业类别适配、高密度点云输入、姿态增强训练及可靠性信息输出，并与开放语义、显式几何和装配上下文协同推断。监督编码器实验用于验证所选编码器及下采样配置的分类表现；完整框架的贡献通过后续多源融合与LOD实验评价。

表格可写“PointNeXt（快速下采样对照）”和“ISPointNet监督语义分支（本文配置）”。若表内为排版而简写PointNeXt，必须在表注明确其为四分下采样受控配置。不能将该行写成官方原始PointNeXt-S，也不能将二分采样本身写成首次提出的网络结构。

## 关键位置

- 当前实验模型：experiments/pointnext/encoder-compare-two-seeds-20261019/models/。
- 当前训练实现快照：experiments/pointnext/encoder-compare-two-seeds-20261019/source_worker.py。
- 训练与划分：同目录source_protocol.json、dataset_summary.json。
- 当前部署：models/pointnext/deployment.json。
- 监督骨干：tools/pointnext_lod/PointNeXt/openpoints/models/backbone/pointnext.py。
- 场景监督推理：tools/pointnext_lod/analyze_large_glb_parts_pointnext.py，predict_batch为单次前向；sample_group_points属于可选组装输入路径。
- 开放分支：tools/pointclipv2/OPEN_SEMANTIC_V2.md。
- 装配上下文：tools/pointclipv2/ASSEMBLY_CONTEXT_V3.md。
- V5候选策略：tools/pointclipv2/ADAPTIVE_P10_V5.md。
- 保存的官方配置：D:/111111111111111/gpt/experiments/encoder_compare/official_pointnext_s_source.yaml。
