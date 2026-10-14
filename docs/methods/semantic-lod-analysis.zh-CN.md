# 语义与结构感知 LOD 策略判断技术分析

本文保留原有策略设计分析。实际运行步骤见 [完整流程](../guides/end-to-end.zh-CN.md)，当前代码与实验口径见 [项目审查](../../analysis_reports/PROJECT_REVIEW_20261014.md)。保护等级分布属于策略行为，不能单独证明几何质量或渲染性能提升。

本文分析当前项目中“语义 + 结构感知 LOD 策略判断”的关键技术点。它不是操作手册，而是说明系统如何判断一个工业模型子 mesh 应进入 P1-P10 哪一类，如何把该判断转化为底层 LOD 构建参数，以及为什么这种方法比单纯几何误差或单一神经分类更适合工业装配模型。

## 1. 问题定义

传统 LOD 通常依据三类信息：

```text
几何误差
屏幕空间误差
三角形 / cluster 数量
```

这对通用模型有效，但对工业模型不够。工业模型里的零件具有明显功能差异：

```text
螺栓、螺母、垫片、销、铆钉
  -> 数量多、重复性强、远处可快速简化。

地基、外壳、墙体、大板件、大型静态支撑
  -> 尺寸大，但通常可较快降面，主要保留外轮廓。

夹具、连接件、控制件、铰链、把手
  -> 对结构理解和交互视觉重要，应适度保护。

齿轮、轴承、电机、导向件、弹簧
  -> 功能关键，应保留更多轮廓、轴线和接触面。

管件、阀门、喷嘴、接口
  -> 接口边界重要，不能只按普通小件处理。
```

因此，本项目的策略判断目标不是单纯追求“分类名称完全正确”，而是：

```text
判断该 mesh 在 LOD 构建中应采用怎样的简化约束强度。
```

最终输出是 P1-P10 十级语义结构策略。

## 2. 总体技术路线

当前系统使用三路证据融合：

```text
PointNeXt 闭集分类
  -> 训练过的工业零件类别概率。

PointCLIP V2 开放词表推断
  -> 对未知或泛化类别提供开放语义候选。

结构特征语义库
  -> 从 mesh 几何本身提取尺寸、形状、复杂度和结构倾向。
```

完整判断链路如下：

```text
GLB mesh
  -> 点云采样
  -> PointNeXt: predicted_class, confidence, second_class
  -> PointCLIP V2: top-k open-vocab label, role, score, margin
  -> Geometry: bbox, OBB, face_count, area, shape_hint
  -> 结构特征向量
  -> role prototype matching
  -> 语义结构融合
  -> inferred_role
  -> P1-P10 policy
  -> target_ratio / allow_cull / screen_error_weight
  -> clodConfig
  -> meshoptimizer / cluster LOD 构建
```

关键文件：

| 功能 | 文件 |
|---|---|
| PointNeXt GLB 分析 | `tools/pointnext_lod/analyze_large_glb_parts_pointnext.py` |
| PointNeXt 批量分析 | `tools/pointnext_lod/batch_analyze_glb_lod.py` |
| PointCLIP V2 GLB 推断 | `tools/pointclipv2/run_pointclipv2_zeroshot_glb.py` |
| PointNeXt + PointCLIP 融合 | `tools/pointclipv2/fuse_pointnext_pointclip_lod.py` |
| P1-P10 策略生成 | `tools/pointnext_lod/build_lod_constraints_from_analysis.py` |
| C++ 读取语义 CSV | `src/scene/scene_semantic_lod.cpp` |
| GLB 加载并绑定 policy | `src/scene/scene_gltf.cpp` |
| 底层 LOD 构建配置 | `src/meshlod/lod.h`、`src/meshlod/meshlod*.h` |
| 前端策略统计 | `src/app/lodclusters_ui.cpp` |
| 着色可视化 | `shaders/common/render_shading.glsl` |

## 3. P1-P10 策略空间

P1-P10 不是简单的“编号越大越重要”，而是从快速简化到强保护的语义结构约束空间。

| 策略 | role | 工程含义 | 简化倾向 |
|---|---|---|---|
| P1 | `micro_uncertain` | 微小或低置信零件 | 最激进，远处可剔除 |
| P2 | `repeated_fastener` | 螺栓、螺母、垫片、销等重复件 | 激进简化，保留位置和粗轮廓 |
| P3 | `large_static_bulk` | 大型静态构件、壳体、地基、板件 | 快速降面，保留大轮廓 |
| P4 | `ordinary_low_detail` | 普通低细节零件 | 中等偏激进 |
| P5 | `balanced_visible` | 普通可见结构 | 平衡质量和性能 |
| P6 | `high_detail_shape` | 高细节复杂形状 | 保护轮廓、高曲率和特征边 |
| P7 | `interface_fluid` | 管件、阀门、喷嘴、接口 | 保护接口边界和安装面 |
| P8 | `structural_control` | 结构连接、夹具、把手、控制件 | 保护接触面和控制轮廓 |
| P9 | `motion_precision` | 导向、运动、精密配合件 | 强保护轴线和接触面 |
| P10 | `critical_preserve` | 齿轮、轴承、电机、传动件 | 最高保护 |

策略表在：

```text
tools/pointnext_lod/build_lod_constraints_from_analysis.py
```

其中 `P10_POLICY_TABLE` 定义了每类策略的：

```text
target_ratio_near
target_ratio_mid
target_ratio_far
allow_cull
screen_error_weight
```

例如：

```text
P2 repeated_fastener:
  near = 0.38
  mid  = 0.16
  far  = 0.05
  allow_cull = true

P10 critical_preserve:
  near = 1.00
  mid  = 0.80
  far  = 0.55
  allow_cull = false
```

这意味着 P2 更偏性能，P10 更偏质量。

## 4. PointNeXt 闭集证据

PointNeXt 的作用是提供训练过的工业类别先验。它输出：

```text
predicted_class
confidence
second_class
second_confidence
confidence_margin
```

它的优势：

- 对训练集中出现过的工业零件类别稳定。
- 对螺栓、垫片、轴承、齿轮、管件等闭集类有明确概率。
- 可以提供 `confidence` 和 `margin`，用于判断是否可靠。

它的局限：

- 只能识别训练类别。
- 对复合件、非标准件、建筑构件、壳体类大件可能泛化不足。
- 对非常小的 mesh 或低面数 mesh 可能置信度不稳定。

因此 PointNeXt 不直接决定最终 P1-P10，而是先映射为候选 role prior。

示例：

```text
screws_bolts_studs      -> repeated_fastener
nuts                    -> repeated_fastener
washers_rings_spacers   -> repeated_fastener / interface_fluid
bearings_bushings_guides -> critical_preserve / motion_precision
gears_pulleys_chains    -> critical_preserve
pipe_fittings_valves_nozzles -> interface_fluid
joints_clamps_structural_connectors -> structural_control
plates_discs_shapes     -> large_static_bulk / ordinary_low_detail
handles_controls        -> structural_control / high_detail_shape
```

这部分由 `CLASS_ROLE_PRIORS` 描述。

## 5. PointCLIP V2 开放词表证据

PointCLIP V2 用来补足闭集分类的不足。它通过点云多视角投影，将 3D mesh 转为 CLIP 可理解的多视角图像，再与开放词表 prompt 对齐。

输出字段包括：

```text
pointclip_top1_id
pointclip_top1_name
pointclip_top1_role
pointclip_top1_score
pointclip_top2_name
pointclip_top2_score
pointclip_margin
pointclip_topk_json
```

PointCLIP V2 的主要价值：

- 遇到训练集外类别时，仍能给出开放语义候选。
- 能把一些 “PointNeXt 不确定件” 拉回到明确 role。
- 能识别更抽象的功能角色，例如 structural_control、large_static_bulk。
- 能减少 old 策略中 P1 堆积的问题。

实际分析中，`b`、`c`、`o1778` 的 fused 结果明显减少了 P1：

```text
b:     P1 19772 -> 1765
c:     P1 23198 -> 3085
o1778: P1 523   -> 57
```

这说明开放词表分支有效提升了低置信样本的可解释性。

PointCLIP V2 的风险是容易偏向宽泛语义，例如 `structural_control`。因此融合时不能只看 PointCLIP top1，而要结合结构特征和 PointNeXt 置信度。

## 6. 结构特征语义库

结构特征不是越多越好。本项目只保留能直接影响 LOD 策略的特征：

```text
bbox_diagonal
bbox_volume
face_count
surface_area
obb_size_long / mid / short
elongation
flatness
compactness
shape_hint
confidence / margin
```

这些原始特征进一步转成归一化结构分数：

```text
micro_score
size_score
medium_size_score
detail_score
high_detail_score
density_score
slender_score
plate_score
compact_score
ring_disk_score
bulk_score
simple_score
uncertainty_score
```

这些分数的目的不是做通用几何描述，而是服务 P1-P10 判定：

- `micro_score` 高：更可能 P1 或 P2。
- `bulk_score` 高：更可能 P3。
- `high_detail_score` 高：更可能 P6 或 P10。
- `ring_disk_score` 高：可能是垫片、轴承、接口件。
- `slender_score` 高：可能是销、杆、螺栓、导向件。
- `plate_score` 高：可能是板件、薄壁、连接面。
- `uncertainty_score` 高：降低神经语义权重，增强几何 fallback。

结构特征语义库的核心价值是：即使神经分类不稳定，也能通过几何形态把零件归入合理 LOD 策略。

## 7. Role Prototype 匹配

每个 role 都有一个结构原型，定义在 `ROLE_PROTOTYPES` 中。例如：

```text
repeated_fastener:
  期望 micro / slender / ring_disk 有一定响应
  策略 repeated_fastener_aggressive
  保护 placement、head_silhouette、axis_if_visible

large_static_bulk:
  期望 bulk_score 和 size_score 高
  策略 large_static_bulk_fast_simplify
  保护 outer_silhouette、major_openings

critical_preserve:
  期望 high_detail、ring_disk、density、compact 有响应
  策略 critical_preserve
  保护 periodic_profile、center_axis、contact_surfaces
```

融合脚本会计算：

```text
PointNeXt role score
PointCLIP role score
Structural prototype score
```

再得到：

```text
inferred_role
semantic_structural_score
structural_match_role
structural_match_score
```

最终输出到 CSV，便于后续审查。

## 8. 融合判断逻辑

融合判断的基本原则是：

```text
高置信 PointNeXt 不轻易推翻。
低置信 PointNeXt 更多参考 PointCLIP 和结构特征。
PointCLIP 只能作为开放语义补充，不能单独决定高保护策略。
结构特征用于纠偏和约束最终策略。
```

典型修正：

```text
低置信 + 微小 + 形状简单
  -> P1 micro_uncertain

低置信 + 细长 / 环盘 / 重复标准件语义
  -> P2 repeated_fastener

大尺寸 + 低细节 + 板/块体
  -> P3 large_static_bulk

高面数 + 高复杂度 + 轮廓复杂
  -> P6 high_detail_shape

接口语义 + 环形/管状/紧凑结构
  -> P7 interface_fluid

结构连接语义 + 接触面/板状/夹具形态
  -> P8 structural_control

运动关键语义 + 轴线/环盘/高细节
  -> P9 或 P10
```

这套逻辑避免两个极端：

- 所有低置信件都变成 P1，导致重要小件被过度简化。
- 所有复杂件都变成 P10，导致 LOD 失去性能收益。

## 9. CSV 到 C++ 的转换

融合后的 CSV 被 C++ 在加载 GLB 时读取：

```text
src/scene/scene_semantic_lod.cpp
```

读取顺序：

```cpp
const std::string fusedCsvName = m_filePath.stem().string() + "_lod_constraints_fused.csv";
const std::string csvName      = m_filePath.stem().string() + "_lod_constraints.csv";
```

程序优先查找 fused，再查找普通 constraints。

每行 CSV 转为：

```text
SemanticLodPolicy
```

核心字段包括：

```text
meshIndex
nodeIndex
priority
allowCull
confidence
targetRatioNear
targetRatioMid
targetRatioFar
screenErrorWeight
rowHash
```

`rowHash` 和整体 fingerprint 会进入 cache 判断。如果 CSV 改变，旧 `.zippp` 会被视为不匹配，需要重建，避免使用过时 LOD。

## 10. 对底层 LOD 构建的影响

CSV 策略不会只停留在 UI 显示，而是会真正影响底层 `clodConfig`：

```text
src/scene/scene_semantic_lod.cpp
Scene::applySemanticPolicyToConfig(...)
```

主要影响：

```text
simplify_ratio
simplify_error_merge_previous
semantic_priority
semantic_confidence
feature_soft_scale
feature_hard_lock_ratio
hierarchy_depth_decay
hierarchy_min_ratio
partition_size
feature_attribute_weight
feature_protect_threshold
feature_critical_threshold
simplify_regularize
```

其中：

- `simplify_ratio` 控制目标简化比例。
- `screenErrorWeight` 间接影响运行时 LOD 切换阈值。
- `feature_soft_scale` 控制特征保护权重。
- `feature_hard_lock_ratio` 控制关键特征硬锁比例。
- `hierarchy_depth_decay` 和 `hierarchy_min_ratio` 影响层级构建深度。
- `partition_size` 影响 cluster/group 构建粒度。

因此语义策略会进入两个层面：

```text
离线 LOD 构建阶段
  -> 影响 mesh 简化、特征保护、层级构建。

运行时渲染阶段
  -> 影响 LOD 选择、屏幕误差权重、是否允许远处剔除。
```

## 11. 与 meshoptimizer 的关系

当前底层仍然基于 meshoptimizer / clustered LOD 的思想：

```text
先构建 meshlet / cluster
再进行 group partition
再进行简化
再生成层级 LOD
```

语义结构策略不是替代 meshoptimizer，而是给它提供更高层的约束：

```text
meshoptimizer 负责几何优化；
语义结构策略负责告诉它哪些区域应该更激进，哪些区域应该更保守。
```

这种设计比较合理，因为：

- 不破坏现有高性能 LOD 管线。
- 不重新发明 mesh 简化算法。
- 只在策略层和约束层增加工业语义。
- 易于调参、可解释、可可视化。

## 12. 可视化和验证

前端支持 `semantic lod policy` 可视化模式。颜色大致为：

```text
P1  灰色
P2  棕色
P3  橙黄色
P4  黄色
P5  绿色
P6  青绿色
P7  浅蓝色
P8  蓝色
P9  紫色
P10 粉红 / 红色
```

验证时不应只看 FPS，而应同时看：

```text
P1-P10 数量分布
Enqueued Clusters
Enqueued Triangles
LOD pixel error
模型近处和远处视觉质量
关键结构是否破坏
重复小件是否快速简化
```

特别需要检查：

- P8 是否过多，导致结构控制件保护偏保守。
- P6 是否过多，导致复杂形状件保面过多。
- P2 是否包含明显非紧固件。
- P10 是否稳定，不能大规模膨胀。
- P1 是否显著减少但没有吞掉重要小件。

## 13. old 与 fused 的评估方法

融合策略的效果不能只看单个 mesh 的类别是否变化，而应看整体 LOD 策略是否更符合工程目标。推荐从以下维度比较：

```text
1. 行数是否一致
   old 和 fused 应能按 order + node_index + mesh_index 完整匹配。

2. P1 是否合理减少
   P1 是微小或不确定件。过多 P1 说明策略无法理解零件功能。

3. P10 是否稳定
   P10 是最高保护类，不应因为开放词表推断而大规模膨胀。

4. P2 / P3 是否增加合理
   重复紧固件和大型静态件应能更快简化，是性能收益的主要来源。

5. P6 / P8 是否过度增长
   高细节件和结构控制件过多会使策略偏保守。

6. 变化是否集中在低置信样本
   如果高置信 PointNeXt 样本大量被推翻，说明融合权重可能过强。
```

当前项目中，`a`、`b`、`c`、`o1778`、`o3049` 五个模型的 old 与 fused 对比可以作为基准观察：

| 模型 | mesh 数 | 变化比例 | 主要变化 |
|---|---:|---:|---|
| `a` | 24259 | 5.87% | 小幅修正，old 已较稳定 |
| `b` | 63063 | 43.55% | P1 大幅减少，P2/P3/P6/P8 增加 |
| `c` | 61467 | 48.36% | P1 大幅减少，复杂件和结构件被细分 |
| `o1778` | 1778 | 50.39% | P1 从 523 降到 57，P10 基本稳定 |
| `o3049` | 3049 | 100% | old 是旧 5 类策略，fused 升级为 P1-P10 |

其中 `b`、`c`、`o1778` 的核心改进是减少 old 中过多的 `P1_micro_uncertain`。这说明 PointCLIP V2 和结构特征有效补足了低置信样本。

`a` 的变化比例只有 5.87%，说明 old 已经相对稳定，fused 主要做小范围精修。`o3049` 显示 100% 变化，是因为 old 文件使用旧的五类策略命名，fused 则升级为 P1-P10 十类体系，不能简单理解为每个 mesh 都被错误改变。

后续加入更多 GLB 后，应继续使用同样方法分析：

```text
*_lod_constraints_old.csv
*_lod_constraints_fused.csv
```

重点不是追求变化越多越好，而是让变化集中在原先低置信、不确定、过粗分类的 mesh 上。

## 14. 技术优势

当前方案的优势在于：

- 多证据融合，不依赖单一分类器。
- P1-P10 策略可解释，可直接映射到底层 LOD 参数。
- 保留 mesh 级输出，兼容原项目按 mesh 做 LOD 的流程。
- fused 文件优先读取，不破坏旧的 `_lod_constraints.csv` 兼容逻辑。
- cache fingerprint 包含语义策略变化，避免旧 cache 污染结果。
- UI 可显示每类数量并用颜色检查策略合理性。

## 15. 局限和后续优化

当前仍有几个需要注意的点：

1. PointCLIP V2 对 `structural_control` 可能偏保守。
   如果 P8 面积过大，可以调低 PointCLIP 对 P8 的融合权重。

2. P6 可能在复杂模型中增长较多。
   如果简化不够激进，可以降低 P6 的 `target_ratio_*` 或 `screen_error_weight`。

3. 结构特征仍然是 mesh 局部判断。
   如果需要更强的装配级理解，可以加入父子节点关系、重复实例模式、空间邻接关系。

4. 小件的 PointCLIP 投影可能不稳定。
   对极小 mesh 应更多依赖尺寸、重复性和 PointNeXt 类别。

5. Runtime / Cache Parameters 面板不应每帧重新统计大场景。
   P1-P10 分布适合在加载后缓存，而不是每帧遍历所有 instance。

## 16. 结论

当前语义结构 LOD 策略的核心不是“给模型贴标签”，而是把工业零件的功能语义转化为可执行的 LOD 约束。

它通过：

```text
PointNeXt 提供闭集工业类别先验；
PointCLIP V2 提供开放词表语义补充；
结构特征库提供几何形态约束；
融合模块输出 P1-P10 策略；
C++ 运行时把策略转成 clodConfig；
底层 meshoptimizer / cluster LOD 根据约束构建层级。
```

这种方法保留了原 LOD 系统的高性能基础，同时引入了工业模型所需的结构感知能力。对于螺栓、地基、运动件、结构控制件和关键传动件，它能给出不同的简化策略，从而在性能和视觉/功能保真之间取得更合理的平衡。
