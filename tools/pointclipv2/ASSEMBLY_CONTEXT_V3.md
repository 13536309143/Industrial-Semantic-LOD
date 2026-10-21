# 装配上下文与实测几何反馈策略

本文保留v3实验口径。当前十档策略已升级，见[ADAPTIVE_P10_V4.md](ADAPTIVE_P10_V4.md)；v3复现单独使用`run_assembly_context_v3_legacy.bat`。

本模块把物体身份、装配作用和几何保真要求分开。网络权重和开放词库保持v2；新增的关系推断与实测反馈不等同于分类网络精度提升。

## 运行

项目根目录运行 `run_assembly_context_v3_legacy.bat`，默认处理 `_downloaded_resources/a.glb` 至 `f.glb`。需要与已有项目一致的 `gpt-pointnext` 和 `PointCLIPV2_LOD` 环境，以及已经配置的项目 `build` 目录。入口会构建CPU几何诊断动态库；迁移设备时先按项目README配置CMake工程。

```bat
run_assembly_context_v3_legacy.bat --policies-only
run_assembly_context_v3_legacy.bat --no-export
```

`--policies-only`要求已有完整的 `scene_sources.json` 和推断文件。普通运行会按内容核验并复用；即使GLB重命名，也只在文件SHA256完全相同、相关代码/权重和输出完整时复用原结果，原有来源记录不改写。每个场景推断失败会记录并继续，其余阶段失败返回非零；存在失败时不导出整套结果。

流程为：全量N/C推断 → 独立几何描述 → v2/v3策略与上下文消融 → 全量单网格QEM诊断 → 完整配置的实测保留比例反馈 → 报告 → 验证后导出。没有调用Vulkan图形界面，也没有测FPS。

## 决策逻辑

- **物体身份**保留N/C来源、对象大类、拒识和冲突。父节点名称不复制给子零件；功能名称冲突也不用于虚构精确物体型号。
- **装配上下文H**只使用祖先名称和子树几何。排除覆盖整机的节点，选择最近的有作用语义的局部装配；按距离衰减。安装件在检测组可作为检测安装支撑，在运动组可作为运动安装支撑。普通紧固件不会仅因为属于大工位就全部升级。
- **名称M**单独开关，复合名称先分清安装板/支架与器件本体。送料物料候选要求类型、物料上下文及局部尺度共同支持；不是工位所有子件都变工件。`metadata_candidate`、功能名称与视觉身份冲突都保留复核标记。
- **几何G**不使用监督类别或置信度：归一化焊接、二面角、特征边长度、显著分离表面分量、主轴尺寸比例，以及全局/局部相对尺度。表面描述按两倍RMS半径归一化，避免单位和刚体旋转影响。几何只能提出形状保护，不能凭细长就断言是气管或导轨。
- **策略**综合作用和保真下限；未知不直接丢弃。薄壁/细长件有近中远保留比例与特征保护下限；有名称和大平面几何共同支持的地面参考件单列，仍不自动删除。
- **实测迁移反馈**仅给NCG/NCGH/NCGHM使用旧NCG作为几何参照。QEM误差、主轴包围范围收缩或显著分量丢失明显退步时，提高保留比例；不改对象身份与功能候选。它有现有复合配置依赖，不能混称为纯G的独立收益。纯C/G消融不读取N。

阈值在 `assembly_context_config.json` 中。它们是工程规则，不是校准概率；不存在由softmax分数证明识别正确的逻辑。

## 结果与前端

报告：`analysis_reports/assembly_context_v3_20261020/六模型结构树辅助与策略优化实验报告.md`。

原始目录：`lod_analysis_outputs/assembly_context_v3_20261020/<a-f>/`。

- `geometry_descriptors.json`、`assembly_geometry_audit.csv`：完整几何与父结构证据。
- `r1_geometry_context`、`r2_scoped_context`：开发阶段保留的结果（b–e）。
- `r3_frozen_policy`：第三轮及f确认时的结果。
- `r4_final_policy`：修正“折弯头”覆盖后全量回放的16种策略结果。
- `final_guarded`：实测反馈后的策略，加一个仅叶名称、关闭父结构的对照，共17种。
- `geometry_proxy.csv`：全量节点、五种请求分配的QEM诊断，包含同请求预算对照；不是同实际预算或真实cluster LOD测试。

既有六项策略文件名保持兼容；第六项是无CAD名称/无父结构的NCG，第七项为NCGHM。另导出 `*_lod_constraints_assembly_tree_only.csv`、`*_lod_constraints_geometry_only.csv` 供诊断。不同内容的旧CSV先备份到 `.artifacts/migrations`。

重新打开模型、选择第七项 **PointNeXt + open semantics + geometry + assembly/CAD context**，重建LOD后才会使用新策略。新增 `feature_lock_floor` 和 `feature_soft_floor` 已接入前端，并在低语义置信度时保留几何下限。共用网格的几何仍按最严格实例合并；没有创建独立的每实例简化网格。

## 实验边界

本轮a–f共11,000个节点。b–e与之前四场景相同；f确认后发现名称覆盖缺口并参与修正，因此六个场景都属于开发诊断，没有独立的最终测试集。

名称核验只覆盖能保守映射到14类的明确名称，重复实例不独立。M辅助输出不参加名称一致率评分。名称提示的关键件低保护计数属于策略规则一致性，不能作为关键件识别准确率。

探针使用单网格meshoptimizer QEM；误差不是Hausdorff距离，显著分离分量保持不等于孔洞保持，未应用前端特征锁或cluster层次。相同请求预算不保证相同实际几何预算。f原策略已有的11个显著分量丢失节点仍存在。尚不能声称真实分类准确率、全局视觉质量或FPS已经提升。
