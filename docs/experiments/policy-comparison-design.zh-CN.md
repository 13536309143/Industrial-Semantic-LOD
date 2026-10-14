# LOD 六组对比实验设计

本文保留六组工程配置的原始实验设计。完整训练/分析/渲染流程见 [使用教程](../guides/end-to-end.zh-CN.md)。当前融合规则、权重及词库过滤并非全部固定，因此正式分支消融的边界以 [项目审查](../../analysis_reports/PROJECT_REVIEW_20261014.md) 为准。

本文档用于单独记录当前项目的 LOD 策略对比实验。实验目标是比较不同语义证据组合对 P1-P10 保护策略的影响，并保证默认策略 `PointNeXt + PointCLIP + structure` 仍然遵守“PointNeXt 低置信样本才交给 PointCLIP 补充推断”的覆盖逻辑。

## 1. 实验目标

当前系统有三类证据来源：

```text
PointNeXt
  闭集工业零件分类，提供类别、置信度、margin 等信息。

PointCLIP V2
  开放词库 zero-shot 推断，补充 PointNeXt 低置信或歧义样本。

Structure
  基于 mesh 几何结构的判断，包括尺寸、面数、细长性、扁平性、紧凑性、shape_hint 等。
```

本实验需要回答：

```text
1. 单独使用 PointNeXt 的 P1-P10 策略是否稳定。
2. 单独使用 PointCLIP 的策略是否比 PointNeXt 更合理。
3. 加入结构判断后，是否能减少语义误判导致的过度简化或过度保护。
4. PointNeXt + PointCLIP 的融合是否优于单模型。
5. PointNeXt + PointCLIP + structure 是否是当前最合理的默认方案。
```

## 2. 六组对比策略

| 编号 | 策略 | CSV 后缀 | PointCLIP 输入 | 用途 |
|---|---|---|---|---|
| 1 | PointNeXt | `_lod_constraints_1_pointnext.csv` | 不使用 | 闭集分类基线 |
| 2 | PointCLIP | `_lod_constraints_2_pointclip.csv` | full PointCLIP | 开放词库单模型基线 |
| 3 | PointNeXt + structure | `_lod_constraints_3_pointnext_structure.csv` | 不使用 | 闭集分类加结构修正 |
| 4 | PointCLIP + structure | `_lod_constraints_4_pointclip_structure.csv` | full PointCLIP | 开放词库加结构修正 |
| 5 | PointNeXt + PointCLIP | `_lod_constraints_5_pointnext_pointclip.csv` | candidate-only PointCLIP | 低置信样本开放词库补充 |
| 6 | PointNeXt + PointCLIP + structure | `_lod_constraints_6_pointnext_pointclip_structure.csv` | candidate-only PointCLIP | 当前默认完整策略 |

重点约束：

```text
策略 2 和策略 4 使用 full PointCLIP，因为它们需要评估“只用 PointCLIP”的完整能力。

策略 5 和策略 6 必须使用 candidate-only PointCLIP，不能使用 full PointCLIP。
这样才能保持原始算法：PointNeXt 高置信样本不被 PointCLIP 覆盖，只有低置信或低 margin 样本进入 PointCLIP。
```

## 3. 数据输入与输出

以模型 `a.glb` 为例：

```text
输入模型：
_downloaded_resources\a.glb

PointNeXt 全量输出：
lod_analysis_outputs\a_pointnext_analysis.csv

PointCLIP 低置信候选输出：
lod_analysis_outputs\a_pointclipv2_candidates.csv

PointCLIP 全量输出：
lod_analysis_outputs\a_pointclipv2_full.csv

六组 LOD 输出目录：
lod_analysis_outputs\lod_ablation\
```

六组策略输出：

```text
lod_analysis_outputs\lod_ablation\a_lod_constraints_1_pointnext.csv
lod_analysis_outputs\lod_ablation\a_lod_constraints_2_pointclip.csv
lod_analysis_outputs\lod_ablation\a_lod_constraints_3_pointnext_structure.csv
lod_analysis_outputs\lod_ablation\a_lod_constraints_4_pointclip_structure.csv
lod_analysis_outputs\lod_ablation\a_lod_constraints_5_pointnext_pointclip.csv
lod_analysis_outputs\lod_ablation\a_lod_constraints_6_pointnext_pointclip_structure.csv
```

汇总对比表：

```text
lod_analysis_outputs\lod_ablation\a_lod_ablation_compare.csv
```

## 4. 推荐执行流程

### 4.1 生成 PointNeXt 结果

先运行当前项目已有的 PointNeXt 分析流程，生成：

```text
lod_analysis_outputs\<model>_pointnext_analysis.csv
```

PointNeXt 必须是全量 mesh 推断，因为它是后续 candidate 筛选和结构阈值统计的共同基准。

### 4.2 生成 candidate-only PointCLIP

继续使用原来的 PointCLIP 候选流程，生成：

```text
lod_analysis_outputs\<model>_pointclipv2_candidates.csv
```

这个文件只包含 PointNeXt 低置信、低 margin 或语义不确定的样本。它用于策略 5 和策略 6。

### 4.3 生成 full PointCLIP

运行：

```bat
tools\pointclipv2\run_pointclipv2_full_5_glbs.bat
```

该脚本生成：

```text
lod_analysis_outputs\<model>_pointclipv2_full.csv
```

这个文件只用于策略 2 和策略 4。它不会覆盖 candidate-only 文件，也不应该用于默认完整策略。

### 4.4 生成六组 LOD CSV

运行：

```bat
tools\pointclipv2\run_lod_ablation_5_glbs.bat
```

该脚本会为 `a b c o1778 o3049` 五个模型生成六组策略 CSV，并输出每个模型的 compare 表。

## 5. 前端验证方式

前端已经增加 `Semantic LOD Strategy` 选择项，可选择：

```text
PointNeXt
PointCLIP
PointNeXt + structure
PointCLIP + structure
PointNeXt + PointCLIP
PointNeXt + PointCLIP + structure
```

默认值：

```text
PointNeXt + PointCLIP + structure
```

选择策略后点击 `Reload Strategy`，场景会重新加载对应 CSV。

C++ 读取顺序：

```text
优先读取：
lod_analysis_outputs\lod_ablation\<model>_lod_constraints_<对应策略>.csv

如果找不到，再回退：
lod_analysis_outputs\<model>_lod_constraints_fused.csv
lod_analysis_outputs\<model>_lod_constraints.csv
```

为了保证对比实验严格，建议先确认 `lod_analysis_outputs\lod_ablation` 中六个 CSV 都存在，再进行前端切换。

## 6. 建议分析指标

每个模型至少比较以下指标：

```text
P1-P10 数量分布
P1-P3 激进简化比例
P8-P10 高保护比例
allow_cull 数量和比例
target_ratio 分布
screen_error_weight 分布
PointCLIP matched rows 数量
```

重点观察：

```text
1. 策略 1 vs 策略 3
   结构判断是否减少了明显不该激进简化的零件。

2. 策略 2 vs 策略 4
   PointCLIP 加结构后是否降低开放词库误判的风险。

3. 策略 1 vs 策略 5
   PointCLIP 是否有效修正了 PointNeXt 低置信样本。

4. 策略 5 vs 策略 6
   structure 是否进一步提升 P1-P10 的保护合理性。

5. 策略 2 vs 策略 6
   单独开放词库是否不如“PointNeXt 主导 + PointCLIP 补充 + structure 校正”稳定。
```

## 7. 合理性判断标准

一个更合理的策略通常应该满足：

```text
1. 大型主体、外壳、底座、框架不应大量落入 P1-P3。
2. 连接件、接口、轴承、齿轮、导轨、阀门等功能关键件应更多进入 P6-P10。
3. 大量重复小件可以进入 P1-P4，但不能误删关键连接结构。
4. allow_cull 应主要出现在语义弱、重复性强、视觉贡献低的 mesh。
5. P8-P10 不应无限膨胀，否则会导致 LOD 保护过强、性能收益下降。
6. PointCLIP 的引入应主要影响低置信样本，而不是推翻 PointNeXt 高置信样本。
```

## 8. 重要注意事项

不要把：

```text
lod_analysis_outputs\<model>_pointclipv2_full.csv
```

手动改名或覆盖成：

```text
lod_analysis_outputs\<model>_pointclipv2_candidates.csv
```

否则策略 5 和策略 6 会退化成 full PointCLIP 覆盖，破坏“PointNeXt 低置信样本才进入 PointCLIP”的原始设计。

如果 full PointCLIP 文件缺失，`run_lod_ablation_5_glbs.bat` 会临时回退到 candidate-only 文件，但此时策略 2 和策略 4 不再是严格的 PointCLIP-only 对比，只能作为不完整结果参考。

## 9. 当前结论假设

在没有完整人工标注 ground truth 的情况下，当前推荐默认策略仍为：

```text
PointNeXt + PointCLIP + structure
```

原因：

```text
1. PointNeXt 负责稳定的闭集工业类别判断。
2. PointCLIP 只补充 PointNeXt 不确定的样本，降低开放词库误覆盖风险。
3. structure 对语义结果进行几何校正，能减少“大件被过度简化”和“小关键件被误删”的情况。
4. P1-P10 策略最终服务于 LOD 保护强度，而不是单纯追求分类名称完全正确。
```
