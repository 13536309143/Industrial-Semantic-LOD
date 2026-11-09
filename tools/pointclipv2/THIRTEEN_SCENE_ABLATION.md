# 十三个编号场景的全量V5融合消融

输入为项目`_downloaded_resources/1.glb`至`13.glb`，对应论文M1至M13。固定当前监督权重、开放语义v2词库与阈值，以及V5预算策略；不训练网络、不改部署CSV。

## 正常运行

项目根目录执行`run_thirteen_scene_ablation.bat`。需要现有`gpt-pointnext`和`PointCLIPV2_LOD`环境。GPU阶段串行，失败模型记录后继续；存在失败则整套实验不标记为完成。

## 阶段入口

在`gpt-pointnext`环境中执行：

```text
python -u -X utf8 tools/pointclipv2/run_thirteen_scene_ablation.py --inference-only
python -u -X utf8 tools/pointclipv2/analyze_thirteen_scene_ablation.py --wait-for-evidence
python -u -X utf8 tools/pointclipv2/visualize_thirteen_scene_ablation.py
python -u -X utf8 tools/pointclipv2/summarize_thirteen_scene_findings.py
python -u -X utf8 tools/pointclipv2/validate_thirteen_scene_ablation.py
```

CPU分析可在GPU证据获取期间启动，会等待当前模型完成。独立校验入口支持`--wait`，图表入口支持`--wait`。不要同时启动两个GPU证据获取任务。阶段选项直接用于Python入口；BAT面向整套实验。

## 条件定义

- N、C、G、NC、NG、CG、NCG七种非空分支组合。
- 每种组合分别H0、HT、H1。H0屏蔽全部父装配上下文；HT只使用拓扑与父组尺度；H1另使用范围受限的祖先名称。世界变换、输入几何不随开关变化。
- NCG_H0_M1和NCG_H1_M1额外使用叶名称，单列辅助，禁止用同一名称自评准确率。
- 每个场景23种条件，13场景共299种条件。主H0/H1对照182种，其余HT91种和M辅助26种。
- 原始68,974个网格引用；M13含2个无POSITION顶点的空网格，输入审计单列，不填造监督预测。全部68,972个有顶点几何实例形成1,586,356条策略记录；重复实例保留，不能当作独立几何样本或独立人工标注。

## 复用与冻结

仅当场景内容、权重、相关源文件和输出哈希一致时复用。允许字节完全相同的GLB重命名；仅名称相似或节点数相同不能复用。限制节点数的smoke结果不得复用为全场景结果。所有监督证据必须来自当前权重、8192点和mesh推理模式。

核心神经推断与策略方法发生变化时应使用新输出目录。调度、报告脚本的修复版本会保留旧快照；Windows状态文件发布遇到短暂读取锁时重试，不因已完成推断后的进度发布冲突而丢弃模型。

## 输出

- `lod_analysis_outputs/thirteen_scene_ablation_v5_20261021/scene_sources.json`：来源清单。
- 各模型`policies/*.csv.gz`：所有节点的完整策略字段。
- `analysis_reports/thirteen_scene_ablation_v5_20261021/`：合并及逐场景表、P1—P10节点/源面数/表面积占比、低置信候选类型、树切换配对、10×10迁移矩阵、图表及逐模型报告。
- `run_status.json`记录证据获取；`analysis_status.json`记录矩阵完成情况；`full_validation.json`独立逐行核验全部压缩CSV；`chart_validation.json`核对绘图数据。
- 三角形是原始实例面数，请求保留率不是实际LOD压缩比。没有独立真值与渲染测量，不报告语义准确率、画质、显存或帧率收益。

临时文件写在E盘工作区，避免系统盘空间不足影响长任务。原历史结果、论文源文件与查看器当前策略保持原样。
