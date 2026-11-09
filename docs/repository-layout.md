# 源码与本地数据

| 路径 | 用途 |
| --- | --- |
| `src/`、`shaders/` | 程序与GPU实现 |
| `cmake/`、`thirdparty/` | 构建与第三方代码 |
| `tools/` | 推断、策略、通用分析及测试 |
| `scripts/windows/` | 启动与本地数据绑定 |
| `models/` | 推理接口说明，不分发权重或评估记录 |
| `docs/experiments/key-comparisons.md` | 唯一公开的结果摘要 |

实验目录、逐网格推断、场景文件和历史报告均不跟踪。Windows本地环境可使用`scripts/windows/setup-data.ps1 -DataRoot 数据根目录`建立兼容联接；公共克隆不包含这些联接或其目标。

`build/`、`_bin/`及Python缓存为本机构建产物。`shaders/build/`是着色器源码，不能作为临时构建目录清除。第三方源码中的生成头文件属于构建输入，需要保留。

请使用自行准备且具有使用权限的数据和兼容权重。公开材料边界及检查方式见[发布约定](publication-policy.md)。
