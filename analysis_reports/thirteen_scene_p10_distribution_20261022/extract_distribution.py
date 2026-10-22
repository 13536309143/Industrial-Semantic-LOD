"""Extract measured policy assignments; never generate labels or ground truth."""
from pathlib import Path
import hashlib
import json
import re


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'analysis_reports/thirteen_scene_ablation_v5_20261021/models'


def cells(line):
    return [part.strip() for part in line.strip().strip('|').split('|')]


scenes = []
hashes = {}
for index in range(1, 14):
    path = SOURCE / f'M{index}.md'
    content = path.read_text(encoding='utf-8-sig')
    hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    title = re.search(r'^# (M\d+) (.+) 全量融合消融', content, re.M)
    population = re.search(r'共([\d,]+)个有顶点几何实例', content)
    assert title and population, path
    rows = [cells(line) for line in content.splitlines() if line.startswith('|')]
    selected = {}
    for variant in ('NCG_H0', 'NCG_H1'):
        matches = [row for row in rows if row[0] == variant]
        assert len(matches) == 1, (path, variant)
        values = matches[0][4:]
        assert len(values) == 10, (path, variant)
        assert all(re.fullmatch(r'(?:<)?\d+\.\d+%', value) for value in values)
        assert abs(sum(float(value.rstrip('%').lstrip('<')) for value in values) - 100) < .1
        selected[variant] = values
    pair = [row for row in rows if row[0] == 'NCG']
    assert len(pair) == 1 and len(pair[0]) == 6, path
    scenes.append({
        'id': title[1], 'name': title[2],
        'geometry_instances': int(population[1].replace(',', '')),
        'distributions': selected,
        'tree_changes': dict(zip(
            ('priority_changed', 'raised', 'lowered', 'function_changed', 'identity_changed'),
            map(int, pair[0][1:]))),
    })

population = sum(scene['geometry_instances'] for scene in scenes)
totals = {key: sum(scene['tree_changes'][key] for scene in scenes)
          for key in scenes[0]['tree_changes']}
assert population == 68972, population
assert totals == dict(priority_changed=11449, raised=11449, lowered=0,
                      function_changed=1075, identity_changed=77), totals

lines = [
    '# 十三模型P1–P10实际算法输出分布', '',
    '本表从已完成的十三模型消融实验逐模型报告提取，没有生成或修改任何策略分配。'
    '**“实际输出”指算法已经产生的结果，不是人工标注的真实分布，也不是策略正确率。**', '',
    'N为监督语义分支，C为开放语义分支，G为显式几何分支。NCG表示三分支完整融合；'
    'H0关闭父结构上下文，H1启用拓扑、父结构尺度及限定范围的祖先名称辅助。', '',
    '统计单位为有顶点的几何实例，同一网格被多个节点引用时按实例计数。'
    '共68,972个有效实例；M13另有2个无顶点空网格，不纳入分母。'
    '各项为实例数量占比，不是三角形数、显存或视觉重要性的占比。', '',
    '## 模型对应关系', '', '| 编号 | 模型 | 有效几何实例数 |', '|---|---|---:|',
]
for scene in scenes:
    lines.append(f"| {scene['id']} | {scene['name']} | {scene['geometry_instances']:,} |")

for variant, heading in (
    ('NCG_H1', '完整融合并启用结构树辅助'),
    ('NCG_H0', '完整融合且关闭结构树辅助'),
):
    lines.extend(['', f'## {heading}（{variant}）', '',
                  '表内所有P列单位均为%。保留原始报告的小数精度。', '',
                  '| 模型 | ' + ' | '.join(f'P{i}' for i in range(1, 11)) + ' |',
                  '|---|' + '---:|' * 10])
    for scene in scenes:
        values = [value.rstrip('%') for value in scene['distributions'][variant]]
        lines.append('| ' + ' | '.join([scene['id'], *values]) + ' |')

lines.extend(['', '## 结构树辅助引起的逐实例变化（NCG：H0→H1）', '',
              '| 模型 | 改变P的实例数 | 改变比例／% | 升档 | 降档 | 功能角色改变 | 身份类别改变 |',
              '|---|---:|---:|---:|---:|---:|---:|'])
for scene in scenes:
    changes = scene['tree_changes']
    lines.append('| ' + ' | '.join([
        scene['id'], f"{changes['priority_changed']:,}",
        f"{changes['priority_changed'] / scene['geometry_instances'] * 100:.2f}",
        *(f'{changes[key]:,}' for key in ('raised', 'lowered', 'function_changed', 'identity_changed')),
    ]) + ' |')
lines.append('| ' + ' | '.join([
    '合计', f"{totals['priority_changed']:,}", f"{totals['priority_changed'] / population * 100:.2f}",
    *(f'{totals[key]:,}' for key in ('raised', 'lowered', 'function_changed', 'identity_changed')),
]) + ' |')
lines.extend([
    '', '## 如何解释这些结果', '',
    '1. 结构树辅助改变11,449个实例的P档位，占16.60%；这些变化均为升档。'
    '这表明当前结构上下文机制主要增加保护，不能据此认定策略判断更准确。',
    '2. 档位分布不同可以描述各模型的策略分配差异。合理的分布应由零件几何、作用和可见性决定，'
    '不要求十档均匀，也不能以“各档都有实例”作为算法有效性的证据。',
    '3. 功能角色改变1,075个实例，身份类别改变77个实例。这些字段可以同时改变，不能相加为独立样本数。'
    '分布表还不能替代逐实例配对统计，因此树辅助变化另列第三张表。',
    '4. P10在完整融合加树的全体实例中实际只有1个，不能将低于0.01%的非零比例写成0。'
    'M10中该实例占0.02%；提取过程也会保留来源中可能出现的“<0.01”。各行因四舍五入可能不恰好合计100%。',
    '5. 要论证“算法有效”，仍需独立抽样审查或实际简化后的结构、视觉及运行性能对照。'
    '当前表格可用于论文中的“策略分配行为分析”，不能作为人工真值或准确率结果。',
    '', '## 来源与复核', '',
    '来源为 `analysis_reports/thirteen_scene_ablation_v5_20261021/models/M1.md` 至 `M13.md`。'
    '提取脚本只读取报告，保持原始实验文件不变。', '',
    '复核内容：13个模型、每模型两组10档比例、各行比例合计容差0.1个百分点、'
    '有效实例总数68,972及结构树配对变化总数。来源SHA256与提取字段见同目录JSON。', '',
])
(HERE / '十三模型P1-P10实际输出分布.md').write_text('\n'.join(lines), encoding='utf-8')
(HERE / 'extracted_distributions.json').write_text(json.dumps({
    'data_kind': 'EXISTING_ALGORITHM_OUTPUT_NOT_HUMAN_GROUND_TRUTH',
    'is_synthetic': False, 'is_human_annotated': False,
    'source_sha256': hashes, 'valid_geometry_instances': population,
    'tree_change_totals': totals, 'models': scenes,
}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': 'passed', 'models': len(scenes),
                  'valid_geometry_instances': population, 'tree_change_totals': totals}, ensure_ascii=False))
