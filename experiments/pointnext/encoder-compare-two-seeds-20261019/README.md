# 当前监督编码器与部署记录

更新日期：2026-10-19。当前 `t1` 默认模型来自 `D:/111111111111111/gpt` 的编码器对比实验，统一采用 seed=2026、2027，共四种配置、八次完整训练。第三个种子未纳入本次汇总。

当前默认部署为 **PointNeXt-S、seed=2027、最佳 epoch=98**。选择规则是比较两个保留种子的验证 Macro-F1，再比较验证 OA；没有依据测试集成绩挑选。2027 的验证 Macro-F1 为 92.1689%，2026 为 91.6144%。

## 正式分类结果

以下为内部测试集指标，单位 %；均值±样本标准差，n=2、ddof=1。

| 模型 | OA | mAcc | Macro-F1 |
|---|---:|---:|---:|
| PointNet | 67.64±0.47 | 39.67±0.33 | 39.97±0.43 |
| PointNet++ | 94.63±0.23 | 91.04±0.58 | 91.26±0.46 |
| PointNeXt-S | 95.40±0.03 | 92.45±0.04 | 92.91±0.21 |
| PointNeXt stride=4 | 94.80±0.13 | 91.52±0.21 | 91.70±0.48 |

![双种子结果](two_seed_accuracy.png)

单独部署的 seed=2027 权重测试 OA/mAcc/Macro-F1 为 **95.4254%/92.4768%/93.0601%**。不能把这个单次部署分数当成两次均值，也不能把内部测试成绩当作实际装配场景准确率。

数据共 14 类、8,192 XYZ 点，训练/验证/测试为 53,654/11,517/11,542。分组控制模型 ID 和重复几何关系，排除跨标签冲突组。两次训练共享固定数据划分，不能视为两套独立测试集；旧数据来源曾被查看，不能称完全外部未见数据。两次重复不用于宣称统计显著性。

`industrial` 与 `pointnext_s` 网络配置相同，不另计一种模型。stride=4 为受控改动，不是官方原始 PointNeXt-S。

## 可追溯材料

- [双种子统计与逐份源文件哈希](summary.json)。
- [数据划分与审计摘要](dataset_summary.json)。
- 各模型/seed 子目录保留 `result.json`、`history.csv`、`confusion.json`、`per_class.csv`、`environment.json`、`job.json`；八份混淆矩阵均重新计算并核对指标，训练历史均覆盖 1～100 轮。
- 部署 seed=2027 另保留逐样本 `test_predictions.csv`。
- `models/*.json`、`source_protocol.json`、`source_worker.py` 为来源快照。来源协议仍记载原计划三个 seed；本报告的统计范围固定为前两个。该 worker 文件用于追溯，不是本目录可单独执行的训练入口。
- [部署检查](deployment_validation.json)：全部参数严格加载、与源检查点逐项相等，RTX 3060 上 8,192 点单次前向输出形状为 1×14，数值有限。
- [当前权重与选择清单](../../../models/pointnext/deployment.json)。

SHA256：`238ed37c8af354ea5bb6532f371ef6003d8fb43e0d6124f44363c55b66ea0493`。

双种子的原始训练工程和断点继续保留在 `gpt`，本次没有修改或重跑它们。

## 已替换的内容与待重算内容

已替换 `models/pointnext/industrial-part-14class-best.pth`，并使用专门的 `pointnext-s-14class-encoder-compare-inference.yaml`。旧工业 YAML 继续作为历史训练参考，不能用它精确复现本次独立训练器。

旧权重和 `lod_analysis_outputs/` 中 130 个文件、共 2,936,608,725 字节已归档至 `.artifacts/migrations/pointnext_replaced_20261019_094119/`。归档包含旧监督 CSV、候选/全量 PointCLIP、融合与六配置输出。旧 seed=3098 日志及历史汇总仍保留，并标注历史用途。

**模型和内部分类结果已更新；新权重的真实场景推理、融合策略、LOD 缓存尚未生成。** 历史候选占比 51.74% 不代表新模型结果，也从来不是带标签准确率。

当前资源目录包含 `o1778.glb`、`o3049.glb`、`臂架生产线.glb`、`车架生产线.glb`。旧记录中的 a/b/c 文件不在当前资源目录，不能假定新结果会与旧五场景一一对应。

## 运行新模型的场景流程

在 Anaconda Prompt 中运行：

```bat
cd /d E:\vk_lod_clusters1\t1
run_all_glb_pointnext_pointclipv2_fused.bat
```

默认使用 `gpt-pointnext` 和 `PointCLIPV2_LOD` 两个环境，枚举 `_downloaded_resources` 中全部 GLB，执行监督推理、候选开放分支和融合。完整流程需要相应环境和 CLIP 权重，本次仅做模型兼容性与小规模检查，没有启动完整场景任务。

如只想先运行监督分析与几何策略：

```bat
conda activate gpt-pointnext
tools\pointnext_lod\run_analyze_5_glbs.bat
```

第二个 BAT 保留历史文件名，但已改为枚举现有 GLB，不再要求 a/b/c 必须存在。

可设置 `POINTNEXT_CKPT`、`POINTNEXT_CFG` 覆盖模型输入，设置 `SEMANTIC_OUTPUT_DIR` 使用独立结果目录。总 BAT 还支持 `RESOURCES_DIR`。如果此前设置过旧权重覆盖变量，应先清除它们后使用默认新模型。

复用逻辑已更新为内容来源核验：监督阶段关联权重、GLB、配置、类别与代码，开放候选阶段关联监督 CSV、GLB、提示词与代码/设置；完成后记录输出哈希。没有侧车记录或内容不一致时重算，截断文件不会仅凭存在而被复用。成功完成整个流程后，输出目录的 `deployment_status.json` 记录实际权重和各场景产物；当前文件则明确记载等待重算。

生成新策略后，按[使用教程](../../../docs/guides/end-to-end.zh-CN.md)以 `--processingonly 1 --autoloadcache 0` 重建相应策略下的 LOD 缓存。旧渲染缓存、六配置统计与旧场景质量观察不能直接改名作为新模型结果。
