"""Generate clearly marked synthetic percentages for formatting tests only."""
from pathlib import Path
import json
import random

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
source = ROOT / 'analysis_reports/thirteen_scene_p10_distribution_20261022/extracted_distributions.json'
models = json.loads(source.read_text(encoding='utf-8'))['models']
rng = random.Random(20261022)
rows = []

for model in models:
    # Arbitrary sampling parameters, unrelated to model geometry or classifier outputs.
    weights = [rng.gammavariate(shape, 1.0)
               for shape in (1.5, 4.0, 7.0, 5.0, 3.0, 3.0, 2.0, 2.0, 1.5, 0.8)]
    unrounded = [weight / sum(weights) * 10000 for weight in weights]
    units = [int(value) for value in unrounded]
    remainder_order = sorted(range(10), key=lambda i: unrounded[i] - units[i], reverse=True)
    for index in remainder_order[:10000 - sum(units)]:
        units[index] += 1
    assert len(units) == 10 and sum(units) == 10000 and min(units) >= 0
    rows.append({
        'simulation_id': 'SIM_ONLY_' + model['id'],
        'scene_name_for_layout_only': model['name'],
        'is_synthetic': True,
        'is_human_annotated': False,
        'is_measured': False,
        'actual_node_assignment': None,
        'hypothetical_percent_basis_points': {f'P{i + 1}': value for i, value in enumerate(units)},
    })

assert len(rows) == 13
metadata = {
    'data_kind': 'SYNTHETIC_P10_FORMAT_DEMO_NOT_GROUND_TRUTH',
    'is_synthetic': True, 'is_measured': False, 'is_human_annotated': False,
    'allowed_use': 'table_layout_and_software_pipeline_testing_only',
    'not_evidence_of_algorithm_effectiveness': True,
    'generation_seed': 20261022,
    'generation_method': 'Arbitrary gamma weights normalized using largest remainder to 10000 basis points',
    'source_fields_used': ['id', 'name'],
    'excluded_inputs': ['actual_strategy_distribution', 'actual_geometry', 'algorithm_predictions', 'human_annotations'],
    'models': rows,
}
(HERE / '虚构模拟P1-P10分布_非实测.json').write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

lines = [
    '# 十三模型P1–P10虚构模拟分布（非实测、非人工真值）', '',
    '**下表全部数字均为随机生成的模拟百分比，仅供排版与软件流程测试。'
    '它不描述这些GLB的实际零件构成，不是人工标注，也不能用于论文证明算法有效。**', '',
    '模型名称仅用于表格展示。生成过程没有读取真实P分布、算法预测、网格几何或人工判断；'
    '不构造与现有算法一致的参考答案，不生成准确率或方法优劣排名。', '',
    '表中P1–P10均为模拟占比，单位为%；每行严格合计100.00%。', '',
    '| 模拟编号 | 展示用模型名称 | ' + ' | '.join(f'P{i}（模拟）' for i in range(1, 11)) + ' |',
    '|---|---|' + '---:|' * 10,
]
for row in rows:
    values = [f'{value / 100:.2f}' for value in row['hypothetical_percent_basis_points'].values()]
    lines.append('| ' + ' | '.join([row['simulation_id'], row['scene_name_for_layout_only'], *values]) + ' |')
lines.extend(['', '**再次说明：这是虚构模拟表，不是实际分布、人工真值或实验结果。**', '',
              '复现脚本：`generate_p10_distribution_demo.py`；随机种子20261022。'
              '机器可读JSON在每条记录中保留模拟标志，不包含真实节点标签。', ''])
(HERE / '十三模型P1-P10虚构模拟分布_非实测.md').write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps({'status': 'passed', 'synthetic_models': len(rows),
                  'each_row_total_percent': '100.00', 'real_experiment_files_modified': False}))
