# PointNeXt推理接口

使用适配的PointNeXt-S编码器，输入8,192个XYZ坐标点，输出14类标签。类别顺序必须与`tools/pointnext_lod/industrial_part_14classes.txt`一致。

本地检查点文件名为`industrial-part-14class-best.pth`。`deployment.json`只描述公开接口，不包含权重、训练种子、评估成绩或来源路径。检查点及其完整校验记录由本地数据环境管理。
