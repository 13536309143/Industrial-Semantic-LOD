"""Compare unverified user-supplied aggregate references with existing outputs.

No per-instance labels, accuracy estimates, or invented results are generated.
"""
from pathlib import Path
import hashlib
import json
import re
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'analysis_reports/thirteen_scene_ablation_v5_20261021/models'
REFERENCE_TEXT = '''
M1 0.26 8.57 35.22 29.05 17.05 7.28 0.86 1.11 0.60 0.00
M2 0.47 6.89 38.51 29.73 9.53 5.27 4.86 1.89 2.84 0.00
M3 0.27 10.06 42.34 30.38 7.36 2.50 2.36 2.63 2.09 0.00
M4 0.24 14.53 60.76 17.48 2.61 0.26 0.57 1.28 2.27 0.00
M5 0.46 15.05 39.78 23.98 10.53 3.03 3.03 2.12 2.00 0.00
M6 0.24 16.37 39.44 24.20 6.36 2.36 4.63 2.03 4.36 0.00
M7 0.19 10.62 60.33 16.08 7.99 1.85 0.39 0.78 1.75 0.00
M8 1.43 18.84 50.88 14.99 5.00 2.39 1.80 2.74 1.90 0.03
M9 0.16 8.47 35.38 24.89 9.56 4.33 10.42 3.21 3.59 0.00
M10 2.79 14.31 46.59 20.03 5.29 4.58 2.04 1.29 3.04 0.04
M11 1.37 28.04 37.35 17.24 4.80 2.42 3.12 3.17 2.45 0.04
M12 0.27 22.60 59.71 8.04 3.62 0.91 0.52 1.20 3.14 0.00
M13 4.63 33.78 42.31 9.60 1.92 2.44 0.91 1.39 2.99 0.01
'''
BRANCHES = ('N', 'C', 'G', 'NC', 'NG', 'CG', 'NCG')
VARIANTS = [f'{branch}_H{h}' for branch in BRANCHES for h in (0, 1)]


def normalize(values):
    return [value * 100 / sum(values) for value in values]


def distance(left, right):
    return sum(abs(a - b) for a, b in zip(left, right)) / 2


references = {}
for line in REFERENCE_TEXT.strip().splitlines():
    fields = line.split()
    values = list(map(float, fields[1:]))
    assert len(values) == 10 and min(values) >= 0 and abs(sum(values) - 100) < .1
    references[fields[0]] = values

scenes = []
hashes = {}
for index in range(1, 14):
    identifier = f'M{index}'
    path = SOURCE / f'{identifier}.md'
    content = path.read_text(encoding='utf-8-sig')
    hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    title = re.search(r'^# (M\d+) (.+) 全量融合消融', content, re.M)
    population = int(re.search(r'共([\d,]+)个有顶点几何实例', content)[1].replace(',', ''))
    predictions = {}
    for line in content.splitlines():
        if not line.startswith('|'):
            continue
        row = [field.strip() for field in line.strip().strip('|').split('|')]
        if row[0] not in VARIANTS:
            continue
        assert row[0] not in predictions
        values = [float(value.rstrip('%')) if not value.startswith('<')
                  else float(value[1:].rstrip('%')) / 2 for value in row[4:]]
        assert len(values) == 10 and abs(sum(values) - 100) < .1
        predictions[row[0]] = normalize(values)
    assert set(predictions) == set(VARIANTS)
    ref = normalize(references[identifier])
    distances = {variant: distance(ref, predictions[variant]) for variant in VARIANTS}
    ranking = sorted(VARIANTS, key=distances.get)
    full_error = [predictions['NCG_H1'][i] - ref[i] for i in range(10)]
    scenes.append(dict(
        id=identifier, name=title[2], geometry_instances=population,
        user_reference_original_percent=references[identifier],
        user_reference_original_sum=sum(references[identifier]),
        normalized_reference_percent=ref,
        predictions_percent=predictions, tvd_percentage_points=distances,
        ranking=ranking, full_signed_error_pp=full_error,
        tree_gain_pp=distances['NCG_H0'] - distances['NCG_H1'],
        leave_one_out_gain_pp={branch: distances[variant] - distances['NCG_H1']
                              for branch, variant in [('N', 'CG_H1'), ('C', 'NG_H1'), ('G', 'NC_H1')]},
    ))
assert len(scenes) == 13
summary = {}
for variant in VARIANTS:
    values = [scene['tvd_percentage_points'][variant] for scene in scenes]
    summary[variant] = dict(macro_tvd_pp=statistics.mean(values), median_tvd_pp=statistics.median(values),
                            min_tvd_pp=min(values), max_tvd_pp=max(values),
                            strict_first_count=sum(scene['ranking'][0] == variant for scene in scenes))

def fmt(value):
    return f'{value:.2f}'


lines = [
    '# 十三模型：用户提供参考分布与融合消融结果的比较', '',
    '**评价对象是P1–P10汇总分布的接近程度，不是逐零件准确率。**', '',
    '参考数据为用户在本次消息中提供的13行百分比。用户将其描述为简单手动标记；'
    '尚无逐节点标注、实际标注数量、标注规则、盲标过程或独立复核记录。'
    '此前讨论中存在模拟数据需求，因此本报告将其保留为“用户提供、来源未核验的参考分布”，'
    '不认定为已核验人工真值，也不将本次距离排名作为方法有效性的独立证明。', '',
    '用户称呼的HCG_H1在本报告按现有实验的NCG_H1理解：N=监督语义，C=开放视觉语义，'
    'G=显式几何；H1启用装配上下文，H0关闭该上下文。'
    '比较7种非空分支组合×H0/H1，共182个场景条件。', '',
    '## 1. 比较方法与数据检查', '',
    '使用总变差距离（TVD）：D=½Σ|预测占比−参考占比|，占比以百分数表示，'
    '因此D的单位是百分点、取值0至100，越小表示分布越接近。'
    'D=0只说明边际分布相同，不说明零件逐一判断正确。本报告不将100−D命名为准确率。', '',
    '为消除两位小数的合计偏差，参考和算法分布均先归一化至100%。'
    '原报告的“<0.01%”按区间中点0.005%近似；非常接近的排名不宜过度解读。'
    '宏平均对13个场景等权，不按网格数量加权，因为参考实际标注样本数尚未提供。'
    '算法全量输出涵盖68,972个有效几何实例，M13的2个空网格除外；'
    '需要确认用户参考是否采用相同总体、是否只标记了子集。', '',
    '| 模型 | 名称 | 参考原始合计／% |', '|---|---|---:|',
]
for scene in scenes:
    lines.append(f"| {scene['id']} | {scene['name']} | {fmt(scene['user_reference_original_sum'])} |")

lines.extend(['', '## 2. 14种配置的整体比较', '',
              '以下均为分布距离，不是错误率。最佳场景数按未舍入距离严格排序统计。', '',
              '| 配置 | 宏平均TVD／百分点↓ | 中位数 | 最小值 | 最大值 | 最接近参考的场景数 |',
              '|---|---:|---:|---:|---:|---:|'])
for variant in sorted(VARIANTS, key=lambda key: summary[key]['macro_tvd_pp']):
    s = summary[variant]
    lines.append('| ' + ' | '.join([variant, *(fmt(s[key]) for key in
                  ('macro_tvd_pp', 'median_tvd_pp', 'min_tvd_pp', 'max_tvd_pp')), str(s['strict_first_count'])]) + ' |')

lines.extend(['', '## 3. 每个模型的完整融合表现', '',
              '树增益=NCG_H0距离−NCG_H1距离，正数表示启用上下文后更接近参考。'
              '相对消融增益=删除对应分支后的距离−NCG_H1距离，正数表示完整配置更接近参考。', '',
              '| 模型 | NCG_H0距离 | NCG_H1距离 | 树增益 | 去N增益 | 去C增益 | 去G增益 | 距离最小配置 |',
              '|---|---:|---:|---:|---:|---:|---:|---|'])
for scene in scenes:
    d = scene['tvd_percentage_points']
    gains = scene['leave_one_out_gain_pp']
    lines.append('| ' + ' | '.join([scene['id'], fmt(d['NCG_H0']), fmt(d['NCG_H1']),
                  fmt(scene['tree_gain_pp']), *(fmt(gains[b]) for b in ('N', 'C', 'G')),
                  scene['ranking'][0]]) + ' |')

lines.extend(['', '## 4. 逐场景、逐消融配置距离全表', ''])
for h in (0, 1):
    lines.extend([f'### H{h}（单位：百分点，越低越接近参考）', '',
                  '| 模型 | ' + ' | '.join(f'{branch}_H{h}' for branch in BRANCHES) + ' |',
                  '|---|' + '---:|' * len(BRANCHES)])
    for scene in scenes:
        lines.append('| ' + ' | '.join([scene['id'], *(fmt(scene['tvd_percentage_points'][f'{b}_H{h}'])
                                                      for b in BRANCHES)]) + ' |')
    lines.append('')

lines.extend(['## 5. 结构树在各组合中的作用', '',
              '| 分支组合 | H0宏平均距离 | H1宏平均距离 | 平均树增益 | 改善／持平／变差场景数 |',
              '|---|---:|---:|---:|---|'])
for branch in BRANCHES:
    changes = [scene['tvd_percentage_points'][branch + '_H0'] -
               scene['tvd_percentage_points'][branch + '_H1'] for scene in scenes]
    counts = [sum(value > 1e-8 for value in changes), sum(abs(value) <= 1e-8 for value in changes),
              sum(value < -1e-8 for value in changes)]
    lines.append('| ' + ' | '.join([branch, fmt(summary[branch + '_H0']['macro_tvd_pp']),
                  fmt(summary[branch + '_H1']['macro_tvd_pp']), fmt(statistics.mean(changes)),
                  '／'.join(map(str, counts))]) + ' |')

lines.extend(['', '## 6. 各模型的主要偏差与解释', '',
              '以下差值为NCG_H1输出占比减参考占比，正值表示该档分配偏多。'
              '占比差异不能定位到具体哪个零件判断有误。', ''])
for scene in scenes:
    d = scene['tvd_percentage_points']
    top = sorted(range(10), key=lambda i: abs(scene['full_signed_error_pp'][i]), reverse=True)[:3]
    errors = '、'.join(f"P{i+1} {scene['full_signed_error_pp'][i]:+.2f}个百分点" for i in top)
    strongest = max(scene['leave_one_out_gain_pp'], key=scene['leave_one_out_gain_pp'].get)
    tree_phrase = '更接近' if scene['tree_gain_pp'] > 0 else '更远离' if scene['tree_gain_pp'] < 0 else '不变'
    lines.extend([f"### {scene['id']} {scene['name']}", '',
                  f"NCG_H1的TVD为{fmt(d['NCG_H1'])}个百分点；启用结构树后{tree_phrase}参考，"
                  f"距离变化的改善量为{scene['tree_gain_pp']:+.2f}个百分点。"
                  f"14个配置中距离最小的是{scene['ranking'][0]}。"
                  f"完整配置相对删除{strongest}分支的距离优势最大，"
                  f"为{fmt(scene['leave_one_out_gain_pp'][strongest])}个百分点。主要剩余偏差为{errors}。", ''])

best_overall = min(summary, key=lambda variant: summary[variant]['macro_tvd_pp'])
full = summary['NCG_H1']
tree_gains = [scene['tree_gain_pp'] for scene in scenes]
lines.extend(['## 7. 可支持的结论与不可支持的结论', '',
    f"就本次提供的参考占比而言，宏平均距离最小的配置是{best_overall}。"
    f"NCG_H1的宏平均TVD为{fmt(full['macro_tvd_pp'])}个百分点，"
    f"在{full['strict_first_count']}个场景中取得最小距离。"
    f"完整融合加树相对不加树的平均距离改善为{statistics.mean(tree_gains):.2f}个百分点。", '',
    f"删除监督N、开放C、几何G分支后，宏平均距离分别增加"
    f"{summary['CG_H1']['macro_tvd_pp'] - full['macro_tvd_pp']:.2f}、"
    f"{summary['NG_H1']['macro_tvd_pp'] - full['macro_tvd_pp']:.2f}、"
    f"{summary['NC_H1']['macro_tvd_pp'] - full['macro_tvd_pp']:.2f}个百分点。"
    '这些是当前完整系统及其固定回退规则下的条件消融差值，不能分解为各分支可相加的独立贡献。', '',
    '开放分支的作用存在场景差异：NCG_H1在10个场景中较NG_H1更接近参考；'
    'M2、M7、M8则分别远离参考1.00、0.10、0.61个百分点。'
    '这提示可在后续独立验证集中研究开放分支的置信度、跨视图一致性与冲突门控，'
    '但不应依据当前参考表逐模型指定开关后，再将同一参考表作为最终测试结果。', '',
    '几何分支的增益应结合规则实现理解：缺少几何分支时，未知功能实例大量进入P5回退；'
    '加入几何后可依据大小、形状和复杂度分配至其他档位。因此分布改善首先支持几何分支的策略细化作用，'
    '不等于证明监督或开放语义分类错误了相同比例的零件。', '',
    '结构树与多源融合的作用可表述为“在所提供参考分布下，提高分配分布的一致性”。'
    '不能表述为“准确率超过90%”“统计显著优于”“结构保持更好”或“显存更低、帧率更高”。'
    '这些分别需要独立逐实例真值、适当统计设计、实际简化质量和运行时测量。', '',
    '建议保留每个被标注实例的node_index、mesh_index、参考P档位或可接受范围、理由、标注人及是否查看算法输出。'
    '若表格是在查看NCG_H1后修订形成，必须披露其依赖性；若只标记子集，'
    '应在相同子集上比较各配置，不能把全量输出分布直接当作子集准确率。', '',
    '## 8. 可复核文件', '',
    '本目录`comparison_results.json`保存用户原始参考、归一化分布、全部182个距离及来源SHA256；'
    '`analyze_reference.py`可复现。原有消融报告和策略输出没有修改。', '',
])
(HERE / '十三模型参考分布与消融对比分析.md').write_text('\n'.join(lines), encoding='utf-8')
(HERE / 'comparison_results.json').write_text(json.dumps(dict(
    reference_provenance='USER_SUPPLIED_AGGREGATES_UNVERIFIED',
    individual_ground_truth_available=False, accuracy_computed=False,
    metric='TOTAL_VARIATION_DISTANCE_IN_PERCENTAGE_POINTS',
    assumptions=['Normalize each row to 100 percent', 'Use midpoint 0.005 for <0.01 percent',
                 'Equal scene weighting; actual annotation counts unavailable'],
    source_sha256=hashes, summary=summary, models=scenes,
), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status='passed', scenes=len(scenes), comparisons=len(scenes)*len(VARIANTS),
    reference_sums={scene['id']: round(scene['user_reference_original_sum'], 2) for scene in scenes},
    summary=summary,
    per_scene=[dict(id=scene['id'], full_tvd=scene['tvd_percentage_points']['NCG_H1'],
        full_rank=scene['ranking'].index('NCG_H1')+1, best=scene['ranking'][0],
        tree_gain=scene['tree_gain_pp'], removal_gains=scene['leave_one_out_gain_pp']) for scene in scenes],
), ensure_ascii=False))
