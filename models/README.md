# 本地推理权重

权重与评估记录不随公开源码分发。监督分类器需要与代码匹配的类别头、输入点数和标签顺序。

默认部署路径为`pointnext/industrial-part-14class-best.pth`，也可通过`POINTNEXT_CKPT`指定。不要将其他类别体系的检查点直接重命名使用。类别索引顺序见[类别定义](../tools/pointnext_lod/industrial_part_14classes.txt)，接口说明见[PointNeXt模型目录](pointnext/README.md)。

开放分支使用CLIP加载器管理的本地预训练权重缓存。完整训练来源、测试成绩、数据指纹及私人文件路径保留在本地，不写入公开部署描述。
