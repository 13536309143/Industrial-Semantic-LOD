# Windows启动脚本

所有入口从脚本自身位置定位仓库，可从任意工作目录启动。主构建与语义流程仍使用原代码定义。

| 入口 | 用途与版本 |
| --- | --- |
| `setup-data.ps1` | 将已准备的外部数据目录绑定到本地兼容路径；不删除或替换现有数据 |
| `viewer.bat [GLB] [像素误差]` | 默认`1.glb`、0.5 px、语义模式6，使用第七份融合约束 |
| `run_all_glb_pointnext_pointclipv2_fused.bat` | 批量PointNeXt、门控PointCLIP及融合 |
| `run_thirteen_scene_ablation.bat` | 历史编号13场景V5消融 |
| `run_adaptive_p10_comparison.bat` | 历史a–f场景V4策略及比较 |
| `run_adaptive_p10_runtime_validation.bat` | 历史V4运行时验证 |
| `run_adaptive_p10_viewer.bat` | 历史a.glb/V4缓存查看入口 |
| `run_assembly_context_comparison.bat` | 转入V4比较的兼容入口 |
| `run_assembly_context_v3_legacy.bat` | 保留的V3比较 |
| `run_a_semantic_comparison.bat` | 历史a.glb单场景比较 |
| `run_open_semantic_v2.bat --models 场景主干名` | 显式选择本地场景的开放语义V2流程 |
| `run_p10_v5_design.bat` | V5预算设计评估，不能作为FPS测试 |

旧a–f入口仅保留开发接口，不能假定与编号场景一一对应。私有案例、参考数据、冻结实验包及完整记录不随源码分发。公开范围见[发布约定](../../docs/publication-policy.md)。
