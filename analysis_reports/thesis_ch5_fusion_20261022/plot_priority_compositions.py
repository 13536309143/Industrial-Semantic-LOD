"""Publication figures from exact policy counts, not synthetic references."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'analysis_reports/thirteen_scene_ablation_v5_20261021/workbook_data.json'
COMPARISON = ROOT / 'analysis_reports/user_reference_distribution_20261022/comparison_results.json'
OUT = HERE / 'figures'
OUT.mkdir(exist_ok=True)
data = json.loads(SOURCE.read_text(encoding='utf-8'))
comparison = json.loads(COMPARISON.read_text(encoding='utf-8'))
models = comparison['models']
branches = ['N', 'C', 'G', 'NC', 'NG', 'CG', 'NCG']
variants = [f'{branch}_H{h}' for branch in branches for h in (0, 1)]
index = {(row['scene'], row['variant']): row for row in data['stats']}
counts = np.array([[[int(index[model['id'], variant][f'P{k}_count']) for k in range(1, 11)]
                    for variant in variants] for model in models], dtype=np.int64)
populations = np.array([model['geometry_instances'] for model in models], dtype=np.int64)
assert counts.shape == (13, 14, 10)
assert np.all(counts.sum(axis=2) == populations[:, None])
assert int(populations.sum()) == 68972
percentages = 100 * counts / populations[:, None, None]
assert np.allclose(percentages.sum(axis=2), 100)
assert int(counts[:, -1, -1].sum()) == 1
for model in models:
    for variant in variants:
        row = index[model['id'], variant]
        assert int(row['nodes']) == model['geometry_instances']
for filename, digest in comparison['source_sha256'].items():
    assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == digest

fonts = {font.name for font in font_manager.fontManager.ttflist}
font = next(name for name in ('Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC') if name in fonts)
plt.rcParams.update({'font.family': font, 'axes.unicode_minus': False,
                     'font.size': 8, 'svg.fonttype': 'none', 'savefig.dpi': 300,
                     'axes.spines.top': False, 'axes.spines.right': False})
colors = ['#d7edf2', '#b8dce8', '#8dc4d6', '#5eacc7', '#3292b9',
          '#3977ab', '#535f9d', '#74498c', '#9d3c75', '#be3556']
handles = [Patch(facecolor=color, label=f'P{i+1}') for i, color in enumerate(colors)]
saved = []


def save(fig, name):
    for ext in ('png', 'svg'):
        path = OUT / f'{name}.{ext}'
        fig.savefig(path, facecolor='white', bbox_inches='tight')
        saved.append(path)
    plt.close(fig)


# Main manuscript figure: paired configuration facets use the same model order.
fig, axes = plt.subplots(7, 2, figsize=(10.2, 11.7), sharex=True, sharey=True)
fig.subplots_adjust(left=.075, right=.99, top=.845, bottom=.14, hspace=.34, wspace=.13)
x = np.arange(13)
for branch_index, branch in enumerate(branches):
    for h in (0, 1):
        ax = axes[branch_index, h]
        variant_index = branch_index * 2 + h
        bottom = np.zeros(13)
        for k, color in enumerate(colors):
            values = percentages[:, variant_index, k]
            ax.bar(x, values, bottom=bottom, width=.78, color=color, edgecolor='white',
                   linewidth=.16, zorder=3)
            bottom += values
        ax.set_title(f'{branch}_H{h}', loc='left', fontsize=11.8, pad=3, fontweight='bold')
        ax.set(ylim=(0, 100), xlim=(-.65, 12.65), yticks=[0, 50, 100])
        ax.set_yticklabels(['0', '50', '100'], fontsize=11)
        ax.set_xticks(x, [str(i) for i in range(1, 14)], fontsize=11)
        ax.tick_params(axis='both', length=2, width=.45, pad=2)
        ax.grid(axis='y', color='#DCE3E8', linewidth=.4, zorder=0)
        ax.spines['left'].set_color('#9AA7B4')
        ax.spines['bottom'].set_color('#9AA7B4')
        if branch_index == 6:
            ax.set_xlabel('模型编号（M1—M13）', fontsize=11)
fig.suptitle('13个工业场景在14种融合配置下的P1–P10策略构成',
             fontsize=16, y=.985, color='#172B42')
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, .955),
           ncol=10, frameon=False, fontsize=11, handlelength=1.2, columnspacing=1.0)
fig.text(.29, .879, 'H0：关闭装配上下文', ha='center', fontsize=12)
fig.text(.78, .879, 'H1：启用装配上下文', ha='center', fontsize=12)
fig.text(.012, .49, '有效几何实例占比 / %', rotation=90, va='center', fontsize=12)
fig.text(.075, .047,
         '每个柱表示一个模型在对应配置下的P1–P10分配，合计100%；所有分面使用相同颜色与尺度。\n'
         'N：监督语义；C：开放语义；G：显式几何。占比来自实际策略计数，不表示逐实例准确率。\n'
         '极小占比按真实宽度绘制，可能无法肉眼分辨；精确值见逐模型明细。',
         fontsize=9, color='#425469', linespacing=1.65)
save(fig, '03_priority_composition_configuration_facets')

# Optional screen/appendix view: each model has its own fourteen horizontal bars.
fig = plt.figure(figsize=(16, 16.5))
grid = fig.add_gridspec(4, 4, left=.065, right=.985, top=.92, bottom=.075,
                       hspace=.42, wspace=.38)
for model_index, model in enumerate(models):
    ax = fig.add_subplot(grid[model_index // 4, model_index % 4])
    y = np.arange(14)
    left = np.zeros(14)
    for k, color in enumerate(colors):
        values = percentages[model_index, :, k]
        ax.barh(y, values, left=left, height=.72, color=color, edgecolor='white',
                linewidth=.18, zorder=3)
        left += values
    ax.set(yticks=y, yticklabels=variants, xticks=[0, 50, 100],
           xlim=(0, 100), ylim=(13.6, -.6))
    ax.tick_params(axis='both', length=2, width=.45, labelsize=7)
    for boundary in np.arange(1.5, 13, 2):
        ax.axhline(boundary, color='#C8D3DC', linewidth=.4, zorder=0)
    ax.grid(axis='x', color='#DFE5EB', linewidth=.4, zorder=0)
    ax.set_title(f"{model['id']}  {model['name']}\nn = {model['geometry_instances']:,}",
                 fontsize=9, loc='left', pad=6, linespacing=1.5)
    ax.set_xlabel('有效实例占比 / %', fontsize=7)
legend_ax = fig.add_subplot(grid[3, 1:])
legend_ax.axis('off')
legend_ax.text(0, .96, '统一策略图例', fontsize=15, color='#172B42', va='top')
legend_handles = [handles[i] for i in (0, 5, 1, 6, 2, 7, 3, 8, 4, 9)]
legend_ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(-.01, .84),
                 ncol=5, frameon=False, fontsize=12, handlelength=2, labelspacing=1.3,
                 columnspacing=1.8)
legend_ax.text(0, .36,
               '每个模型包含14种配置；每个堆积条合计100%。\n'
               '配置顺序、P档位顺序与颜色在13个模型中保持一致。\n'
               '不设置最小显示宽度，极小占比可能不可辨。\n'
               '本图展示算法分配构成，不表示准确率或保护效果。',
               fontsize=10, va='top', linespacing=2.0, color='#425469')
fig.suptitle('13个工业场景的P1–P10策略占比：逐模型分面',
             fontsize=18, y=.985, color='#172B42')
fig.text(.065, .023, 'N：监督语义；C：开放语义；G：显式几何；H0/H1分别关闭/启用装配上下文。',
         fontsize=10, color='#425469')
save(fig, '04_priority_composition_model_facets_overview')

summary_rows = []
for variant in variants:
    row = index['ALL', variant]
    assert int(row['nodes']) == 68972
    summary_rows.append({
        'variant': variant,
        'object_unresolved_percent': 100 * row['object_unresolved_count'] / row['nodes'],
        'unknown_function_percent': 100 * row['unknown_function_count'] / row['nodes'],
        'NC_conflict_percent': (100 * row['identity_conflict_count'] / row['nodes']
                                if variant.split('_')[0] in ('NC', 'NCG') else None),
        'macro_tvd_pp': comparison['summary'][variant]['macro_tvd_pp'],
    })

manifest = {
    'data_kind': 'EXACT_EXISTING_POLICY_COUNTS_NOT_HUMAN_LABELS',
    'source': str(SOURCE.relative_to(ROOT)),
    'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'comparison_source': str(COMPARISON.relative_to(ROOT)),
    'comparison_sha256': hashlib.sha256(COMPARISON.read_bytes()).hexdigest(),
    'scene_order': [model['id'] for model in models], 'variant_order': variants,
    'priority_order': [f'P{k}' for k in range(1, 11)], 'colors': colors,
    'populations': populations.tolist(), 'exact_counts': counts.tolist(),
    'percentages': percentages.tolist(), 'summary_rows': summary_rows,
    'main_figure': '03_priority_composition_configuration_facets',
    'optional_model_view': '04_priority_composition_model_facets_overview',
    'main_cells': int(counts.size), 'bars': 182,
    'NCG_H1_P10_count': int(counts[:, -1, -1].sum()),
    'minimum_visible_width_added': False, 'synthetic_demo_used': False,
    'accuracy_imputed': False,
    'figures': [{'file': str(path.relative_to(ROOT)),
                 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in saved],
}
(HERE / 'priority_composition_data_and_provenance.json').write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': 'generated', 'models': 13, 'configurations': 14,
                  'bars': 182, 'priority_counts': int(counts.size), 'files': len(saved),
                  'valid_instances': int(populations.sum()), 'NCG_H1_P10_count': 1,
                  'table': summary_rows}, ensure_ascii=False))
