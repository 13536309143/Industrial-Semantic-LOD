# 当前 PointNeXt 部署

当前默认权重：`industrial-part-14class-best.pth`，PointNeXt-S、seed=2027、epoch=98。

按验证 Macro-F1 从保留的两个种子中选择；来源、文件哈希、类别顺序、单次内部测试成绩与历史归档位置见 [deployment.json](deployment.json)。

[正式双种子实验与场景重算说明](../../experiments/pointnext/encoder-compare-two-seeds-20261019/README.md)。权重文件按项目约定由 Git 忽略，复制项目到其他设备时需要一并复制此目录下的 `.pth`；JSON 清单用于校验，不包含模型张量。

旧权重保存在 `.artifacts/migrations/pointnext_replaced_20261019_094119/weights/`。当前 a 的新权重全量推理和策略分析已完成，见[场景报告](../../analysis_reports/a_seed2027_20261019/analysis_report.md)；其他场景与 LOD 质量仍需验证。历史候选比例不代表当前模型表现。
