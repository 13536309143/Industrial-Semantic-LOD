# gpt 与 t1 的 PointNeXt 使用及旋转增强核验

> 历史快照说明：本文记录替换前的状态。2026-10-19 已将默认部署更新为 seed=2027、epoch=98；旧场景结果已归档，新场景指标待重算。当前状态以[部署说明](../experiments/pointnext/encoder-compare-two-seeds-20261019/README.md)为准。

核验日期：2026-10-19。本次读取代码、配置、日志、已有概率文件与检查点；CPU 读取权重核验，没有启动训练或 GLB 推理，没有修改模型、训练协议、BAT 或场景 CSV。

## 1. 主要结论

用户记得的旋转改动确实存在：在工业零件训练配置中启用了 RandomRotate，并将 angle 设置为 [1,1,1]，对 X、Y、Z 三轴分别从 [-π,π] 采样角度。这项改动在旧实验日志中实际生效；当前双种子对比的独立训练器也实现了同样的旋转构造，四种模型均使用它。

这属于针对工业零件姿态变化的训练增强。当前 GLB 推理没有多旋转投票，没有 PCA 主轴对齐，也没有使网络在数学上严格旋转不变。文件名带 randomrot 表示训练配方，不意味着 GLB 推理会自动执行训练增强。

当前 t1 默认部署权重与 gpt 的旧训练最佳权重逐字节相同，均为旧实验 epoch=85 的模型；它们不同于新实验 seed=2026、2027 的最佳权重。两个目录中的网络骨干、旧旋转实现、工业数据集实现与旧 fast YAML 均逐字节相同，两个主推理脚本也相同；它们不是两套不同的 PointNeXt 网络。

## 2. 旋转具体在哪里

旧无旋转工业配置：

[D:/111111111111111/gpt/PointNeXt/cfgs/industrial_part/pointnext-s.yaml](D:/111111111111111/gpt/PointNeXt/cfgs/industrial_part/pointnext-s.yaml:20)

训练增强为 PointsToTensor、PointCloudScaleAndTranslate，没有 RandomRotate。该文件同时为 90 类、4096 点，不能直接与 14 类、8192 点实验相减后把差值全部归因于旋转。

旧 randomrot 配置：

[pointnext-s-14class-randomrot-8192-fast.yaml](D:/111111111111111/gpt/PointNeXt/cfgs/industrial_part/pointnext-s-14class-randomrot-8192-fast.yaml:21)

```yaml
datatransforms:
  train: [RandomRotate, PointsToTensor, PointCloudScaleAndTranslate]
  val: [PointsToTensor]
  test: [PointsToTensor]
  kwargs:
    angle: [1, 1, 1]
    shift: [0.2, 0.2, 0.2]
```

执行代码是 [point_transform_cpu.py](D:/111111111111111/gpt/PointNeXt/openpoints/transforms/point_transform_cpu.py:23)。角度的单位是 π 的倍数，[1,1,1] 表示每轴 -180°～180°，不是每轴旋转 1°。

令三个角分别为 α、β、γ，代码构造 R=Rz(γ)Ry(β)Rx(α)，按行存储的点执行 P'=P Rᵀ。旧 dataset 的 __getitem__ 调用 transform，随后令 x=pos，因此旋转后的坐标同时进入 pos 与 XYZ 特征。

训练日志 [training.log](../experiments/pointnext/industrial-part-14class/training.log) 明确记录了 RandomRotate、angle=[1,1,1]、seed=3098、use_voting=False，证明这不只是保留在目录里的配置文件。

RandomRotate 本身是上游 OpenPoints 已有实现：[上游旋转实现](https://raw.githubusercontent.com/guochengqian/openpoints/master/transforms/point_transform_cpu.py)。可以把启用和配置三轴旋转描述为工业任务的训练适配，不宜声称发明了新的旋转算子。

独立均匀采样三个欧拉角并不等于在 SO(3) 上均匀采样旋转。论文应使用“三轴随机旋转增强”等准确名称；不应写成“SO(3) 均匀采样”或“严格旋转不变网络”。

## 3. 两个目录各自负责什么

| 位置/入口 | 实际用途 | 配置来源 | 权重/结果来源 |
|---|---|---|---|
| gpt/PointNeXt/examples/classification/main.py | 旧 OpenPoints 分类训练入口，转入 train.py | cfgs/industrial_part/*.yaml | 旧 PointNeXt/log 下的检查点 |
| gpt/run_encoder_comparison.bat | 当前编码器对比实验 | experiments/encoder_compare/protocol.json 与 models/*.json | encoder_compare_outputs/方法/seed_*/ |
| gpt/experiments/encoder_compare/worker.py | 当前独立训练、验证、测试实现 | 上述 JSON；不读取旧 YAML 的训练增强 | 自己训练得到的 best.pth/latest.pth |
| gpt/pointnext_stage1_glb.py | 旧双环境流程的监督分支，输出完整概率等 | 旧 14 类 fast YAML | --ckpt 明确指定 |
| gpt/pointnext_clip_openworld_glb.py | 旧单脚本监督与开放语义流程 | 旧 YAML + 开放提示词 | --ckpt 明确指定 |
| gpt/zeroshot_cls/step1_pointnext_infer.py | 对旧点云 test_list 逐样本推理 | 旧 YAML | 写死指向旧 log 最佳权重 |
| t1/tools/pointnext_lod/run_analyze_5_glbs.bat | 五个 GLB 的监督分类及几何/LOD 分析 | t1 内的旧 fast YAML | models/pointnext/industrial-part-14class-best.pth，可环境变量覆盖 |
| t1/run_all_glb_pointnext_pointclipv2_fused.bat | 场景监督推理→候选开放分支→融合策略 | t1 内旧 fast YAML + PointCLIP 提示词 | 同一默认部署权重；复用已有 CSV |
| t1/src/scene/scene_semantic_lod.cpp | 加载和解释离线语义策略 | 策略 CSV | 此处不直接加载 .pth 或运行 PointNeXt |

因此 gpt 当前主要承担实验训练，t1 主要承担模型部署、场景分析及语义 LOD 集成。不过两个目录都保留了旧训练和推理代码，不能仅根据目录名判断一次结果来自哪个流程。

## 4. 现在的双种子实验保留了旋转吗

保留了。关键代码是 [worker.py:68](D:/111111111111111/gpt/experiments/encoder_compare/worker.py:68)：

```python
if self.split == 'train':
    angles = rng.uniform(-np.pi, np.pi, 3)
    # 构造 rx、ry、rz
    xyz = xyz @ (rz @ ry @ rx).T
    xyz *= rng.uniform(*self.augmentation['scale'])
    xyz += rng.uniform(-self.augmentation['shift'], self.augmentation['shift'], 3)
    rng.shuffle(xyz)
```

每个训练样本的随机数由训练 seed、epoch 与 source 派生。验证与测试不进入这个分支，也没有测试时多旋转投票。因此双种子实验已经包含了三轴旋转训练增强，并不是把旧工业旋转处理去掉后重训。

但新旧训练配方不完全一致：

| 项目 | 旧 14 类 fast 训练 | 当前 encoder_compare |
|---|---|---|
| 三轴旋转 | 每轴 [-π,π] | 每轴 [-π,π] |
| 缩放 | 默认各向异性，每轴 [2/3,1.5] | 各向同性统一缩放 [0.8,1.2] |
| 平移 | 每轴 ±0.2 | 每轴 ±0.2 |
| BatchNorm 所见批次 | 实际 batch=16 | 微批次=4，累积至有效 16 |
| 数据划分 | 旧 train/test；val 和 test 共用 test | 当前 train/val/test 分开 |
| 最佳权重选择 | 旧流程按验证 OA | 验证 Macro-F1，再以 OA 决定并列 |
| 评估周期 | 每 5 轮 | 每轮 |

新实验的旋转参数说明位于 protocol.json，但实际欧拉角构造直接写在 worker.py 中；调整旧 YAML 的 angle 不会改变当前对比实验。旧配置文件名中的 fast、randomrot、8192 不能替代对当前执行代码的检查。

当前四种独立模型都应用同一旋转增强，所以其比较支持“在统一旋转增强条件下，哪个模型/下采样配置表现更好”，不能单独量化旋转增强的贡献。

## 5. t1 实际部署的是哪份模型

本次对以下文件计算 SHA256，并在 CPU 上加载检查点对比 model 张量：

| 检查点 | 最佳 epoch | SHA256 |
|---|---:|---|
| t1/models/pointnext/industrial-part-14class-best.pth | 85 | 094de0ac8f1b58fbf37fb5fa06688c498da41a8e99e7a261356d2a3ce908b0fe |
| gpt/PointNeXt/log/industrial_part/industrial_part-train-pointnext-s-14class/checkpoint/industrial_part-train-pointnext-s-14class_ckpt_best.pth | 85 | 094de0ac8f1b58fbf37fb5fa06688c498da41a8e99e7a261356d2a3ce908b0fe |
| gpt/encoder_compare_outputs/pointnext_s/seed_2026/best.pth | 99 | f9351b7bfeb6f1bf978159b1fc130eb7ec172c7c6e467228c6124b60ea340fdf |
| gpt/encoder_compare_outputs/pointnext_s/seed_2027/best.pth | 98 | 238ed37c8af354ea5bb6532f371ef6003d8fb43e0d6124f44363c55b66ea0493 |

前两份文件哈希完全一致，model 张量也全部一致，检查点内 best_val=96.19438171386719、epoch=85。旧运行记录的训练 seed=3098，验证与测试共用旧 test，不能把这项开发评估当作当前内部独立测试。

后两份新权重与部署旧权重的张量不同。当前 t1 总 BAT 默认仍指向 models/pointnext 下的旧文件；除非显式设置 POINTNEXT_CKPT，训练新实验不会自动替换这份部署文件。

t1 与 gpt 的工业类别文件、旧概率文件的类别顺序一致，均为 14 类。不存在本次检查所见的类别顺序差异。

旧检查点约 16.5 MB，新 best.pth 约 5.5 MB，主要原因是旧文件还保存了 optimizer、scheduler 等训练状态，不能据文件大小判断网络参数减少。它们都含 84 项模型状态记录。

两个目录间以下文件逐字节相同：

- openpoints/models/backbone/pointnext.py。
- openpoints/transforms/point_transform_cpu.py。
- openpoints/dataset/industrial_part.py。
- cfgs/industrial_part/pointnext-s-14class-randomrot-8192-fast.yaml。
- analyze_large_glb_parts_pointnext.py。
- analyze_gltf_parts_pointnext.py。

batch_analyze_glb_lod.py 的主要差别是：gpt 默认固定五场景，t1 未指定 --models 时枚举资源目录全部 GLB。分类算法没有因为这个包装差别而改变。

官方 ModelNet40 PointNeXt-S 的模型参数与当前网络核心配置一致，分类头适配为 14 类；官方配置也使用 [1,2,2,2,2,1]：[官方配置](https://raw.githubusercontent.com/guochengqian/PointNeXt/master/cfgs/modelnet40ply2048/pointnext-s.yaml)。所以工业适配主要体现在数据、类别、输入点数、训练增强与下游融合，不应把缓慢下采样本身当成新发明。

## 6. t1 GLB 使用流程与几个容易误解的“旋转”

主链路是：

GLB 的 mesh 节点→读取三角面和顶点→应用当前节点局部变换→按表面积采样→点云质心归零与单位球归一化→单次 PointNeXt 前向→softmax/top2→输出类别、置信度和几何量→候选 PointCLIP→角色/优先级策略→C++ 语义 LOD 预处理。

[analyze_large_glb_parts_pointnext.py:288](../tools/pointnext_lod/analyze_large_glb_parts_pointnext.py:288) 的 predict_batch 只调用一次 model(data)，没有旋转循环或投票。此路径读取 YAML 主要用于建模、类别与 AMP 等设置，没有构建训练 datatransforms。

需要区分四种不同操作：

1. 训练增强旋转：改变训练样本姿态，意图减弱模型对固定朝向的依赖。
2. GLB 节点旋转：还原输入节点本身的变换，不是新增训练增强。
3. PointCLIP 多视图旋转：用于投影视图，属于开放分支，不是 PointNeXt 多旋转投票。
4. 测试时旋转投票：同一样本多姿态推理并聚合概率；当前核查的监督推理路径没有使用。

另外，推理脚本 --seed 默认 2026 控制采样随机性，不表示它加载了训练 seed=2026 的权重；加载哪份模型始终由 --ckpt 决定。

## 7. 影响结果解释和后续使用的细节

### 7.1 结果复用不绑定权重

[t1 总 BAT](../run_all_glb_pointnext_pointclipv2_fused.bat:66) 传入 --skip-existing。batch_analyze_glb_lod.py 只根据 analysis_csv.exists() 决定跳过；它不验证权重哈希、采样参数或脚本版本。PointCLIP 候选输出也按文件存在性复用。

因此更换权重并重跑总 BAT，不一定会重新生成监督输出。若后续评估新模型，应使用独立输出目录并重新生成候选与融合文件，保留旧输出以便追溯。

本次确认的是当前默认权重来源。历史五场景 CSV 没有记录权重哈希及完整执行命令，因此不能仅凭当前默认路径，严格证明所有历史 CSV 的实际生成权重。

### 7.2 输入点数由命令行控制

大 GLB 推理脚本单独启动时 --num-points 默认 4096，而旧 fast YAML 写的是 8192。两种 t1 BAT 都显式传入 8192，所以正常 BAT 路径与训练点数一致；手工运行脚本时如果省略该参数，会使用 4096，不能仅因 YAML 文件名带 8192 就认定实际输入也是 8192。

### 7.3 局部节点变换和父级变换

大 GLB 路径的 mesh_for_node 只调用当前 node 的 node_matrix，没有在此处累乘祖先变换。带父级缩放/旋转的通用 glTF，其输入几何和几何统计可能与最终渲染坐标不一致。不能据此断言当前五场景均已受影响，但应在迁移到新装配场景时核验。

### 7.4 不同候选统计不是同一个阈值

监督分析脚本的 review_candidates 使用 confidence<0.40 或 margin<0.08。PointCLIP 候选门控默认使用更高的直接接受置信度 0.70。前者是低置信复核表，后者是开放分支候选，两者数量不能混为一谈。gpt 旧 stage1_summary 又包含 <0.55 和 margin<0.10 的摘要统计，比较前需先统一定义。

### 7.5 原有“约五十多”仍不是准确率

五场景统计中 51.74% 是未达直接接受条件的记录占比。它没有人工类别标签，不能解释成识别准确率约 50%。本次另外读取 gpt/zeroshot_cls/pointnext_probs.npz：共 15,372 条带标签记录，重新计算 OA 为 96.0643%、mAcc 为 93.4999%。这份旧点云评估同样不等于真实装配场景评估，也不等于旧日志最佳检查点评估的 96.19%；采样、精度与生成过程可能不同，未确定的差异不能强行归因。

## 8. 能否说旋转提高了识别率

代码与运行日志能证明“启用了旋转增强”，不能单独证明“提高了多少”。本次核查的双种子结果全部包含旋转增强，缺少同数据、同划分、同模型、同优化设置下关闭旋转的配对对照。

可以在方法章写：为适应工业装配中零件姿态变化，训练阶段对点云施加三轴随机旋转，并结合尺度和平移增强，以减弱编码器对固定朝向的依赖。

不能在结果章直接写：旋转增强将识别率提高了某个数值；该网络保证任意旋转不变；双种子编码器对比已经证明了旋转增强的独立增益。

若论文保留旋转增强的独立效果主张，最直接的验证设计是同一 PointNeXt-S 下比较无旋转、仅单轴旋转、三轴旋转三个训练设置，在原姿态与固定随机旋转测试副本上分别测 OA/Macro-F1。保持当前两个训练 seed 即可形成对称比较，不需要因此增加第三个 seed。旋转副本必须与原样本保持同一数据划分，不能作为额外独立样本增加统计数量。

资源有限时，可先仅对现有模型测试旋转扰动前后的指标和预测一致率，了解当前模型的姿态敏感性；这不需要重新训练，但它只能测现有模型鲁棒性，不能代替无增强训练对照证明增强收益。若不开展对照，可把旋转作为训练设置如实描述，不将它单列为已证实的创新贡献。

## 9. 与当前论文实验的衔接

双种子编码器实验支持 PointNeXt-S 在当前统一旋转增强条件下的内部分类表现：平均 OA 95.40%、Macro-F1 92.91%。该结论与“使用旧旋转增强权重进行场景分析”是不同层面的证据。

当前更合理的证据链为：

- 监督编码器实验：验证模型选择和下采样精度/代价。
- 姿态分析：描述已有旋转训练设置；如有额外主张再设计严格旋转对照。
- 场景融合实验：锁定明确权重，人工核验角色/类别，评估门控与融合纠错。
- LOD 实验：在匹配实际三角预算或质量条件下，测关键结构保留和效率。

后续使用两个 seed 做完整系统实验时，可以分别让两份监督权重通过同一套场景流程并报告对应结果；如果只部署其中一份，应预先指定选择规则，避免根据最终测试场景表现挑选权重。现有流程并没有把两份权重自动集成为一个模型。

本次只生成分析说明，没有替换部署权重、重跑场景或改写已有实验输出。
