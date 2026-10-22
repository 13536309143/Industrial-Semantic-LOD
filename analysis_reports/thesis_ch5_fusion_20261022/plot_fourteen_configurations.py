"""Plot fourteen configurations using existing distribution distances only."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch, Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'analysis_reports/user_reference_distribution_20261022/comparison_results.json'
OUT = HERE / 'figures'
OUT.mkdir(exist_ok=True)
data = json.loads(SOURCE.read_text(encoding='utf-8'))
branches = ['N', 'C', 'G', 'NC', 'NG', 'CG', 'NCG']
variants = [f'{branch}_H{h}' for branch in branches for h in (0, 1)]
models = data['models']
h0 = np.array([data['summary'][branch + '_H0']['macro_tvd_pp'] for branch in branches])
h1 = np.array([data['summary'][branch + '_H1']['macro_tvd_pp'] for branch in branches])
matrix = np.array([[model['tvd_percentage_points'][variant] for model in models] for variant in variants])
assert matrix.shape == (14, 13)
assert np.allclose(matrix.mean(axis=1)[::2], h0)
assert np.allclose(matrix.mean(axis=1)[1::2], h1)
assert not data['accuracy_computed']

available_fonts = {font.name for font in font_manager.fontManager.ttflist}
font = next((name for name in ('Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC')
             if name in available_fonts), None)
assert font, 'A Chinese font is required.'
plt.rcParams.update({
    'font.family': font, 'axes.unicode_minus': False, 'font.size': 10,
    'axes.titlesize': 12, 'axes.labelsize': 10, 'svg.fonttype': 'none',
    'savefig.dpi': 300, 'axes.spines.top': False, 'axes.spines.right': False,
})
H0_COLOR = '#9AA7B4'
H1_COLOR = '#187E86'
INK = '#172B42'
saved = []


def save(fig, name):
    for ext in ('png', 'svg'):
        path = OUT / f'{name}.{ext}'
        fig.savefig(path, bbox_inches='tight', facecolor='white')
        saved.append(path)
    plt.close(fig)


fig, (ax, gain_ax) = plt.subplots(1, 2, figsize=(12.8, 6.5),
                                gridspec_kw={'width_ratios': [1.9, 1]})
fig.subplots_adjust(left=.08, right=.98, top=.82, bottom=.19, wspace=.37)
y = np.arange(len(branches))
for index in range(len(branches)):
    if index % 2 == 0:
        ax.axhspan(index - .46, index + .46, color='#F3F6F8', zorder=0)
        gain_ax.axhspan(index - .46, index + .46, color='#F3F6F8', zorder=0)
ax.barh(y - .18, h0, height=.30, color=H0_COLOR, label='H0：关闭装配上下文', zorder=3)
ax.barh(y + .18, h1, height=.30, color=H1_COLOR, label='H1：启用装配上下文', zorder=3)
for index, (left, right) in enumerate(zip(h0, h1)):
    ax.text(left + .8, index - .18, f'{left:.2f}', va='center', fontsize=9, color=INK)
    ax.text(right + .8, index + .18, f'{right:.2f}', va='center', fontsize=9, color=H1_COLOR,
            fontweight='bold' if index == 6 else 'normal')
ax.set(yticks=y, yticklabels=branches, xlim=(0, 88), ylim=(6.55, -.55),
       xlabel='宏平均总变差距离 / 百分点（越低越接近参考）',
       title='（a）七种分支组合 × H0/H1：14种配置')
ax.grid(axis='x', color='#DFE5EB', linewidth=.65, zorder=0)
ax.set_axisbelow(True)
ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02), frameon=False, fontsize=9)
gains = h0 - h1
gain_ax.barh(y, gains, height=.52, color=H1_COLOR, zorder=3)
for index, value in enumerate(gains):
    gain_ax.text(value + .14, index, f'{value:.2f}', va='center', fontsize=9, color=INK)
gain_ax.set(yticks=y, yticklabels=branches, xlim=(0, 10.7), ylim=(6.55, -.55),
            xlabel='H0距离 − H1距离 / 百分点', title='（b）启用装配上下文的距离减少量')
gain_ax.grid(axis='x', color='#DFE5EB', linewidth=.65)
gain_ax.set_axisbelow(True)
fig.suptitle('多源语义与几何融合配置的分布对照', fontsize=15, x=.51, y=.98, color=INK)
fig.text(.08, .065,
         'N：监督语义；C：开放语义；G：显式几何。13个场景等权平均。\n'
         '参考为用户提供的P1–P10占比；本图比较分布距离，不表示逐实例准确率。',
         fontsize=9, color='#425469', linespacing=1.7)
save(fig, '01_fourteen_configurations_paired_comparison')

fig, ax = plt.subplots(figsize=(14.8, 9.2))
fig.subplots_adjust(left=.10, right=.90, top=.87, bottom=.18)
im = ax.imshow(matrix, cmap='YlOrRd', vmin=0, vmax=85, aspect='auto')
ax.set(xticks=np.arange(13), xticklabels=[model['id'] for model in models],
       yticks=np.arange(14), yticklabels=variants, xlabel='工业场景', ylabel='融合配置')
ax.tick_params(axis='both', length=0, pad=8)
for row in range(14):
    for col in range(13):
        value = matrix[row, col]
        ax.text(col, row, f'{value:.2f}', ha='center', va='center', fontsize=8.4,
                color='white' if value > 56 else INK)
for boundary in np.arange(1.5, 13, 2):
    ax.axhline(boundary, color='white', linewidth=2)
for row, col in zip(np.argmin(matrix, axis=0), range(13)):
    ax.add_patch(Rectangle((col-.46, row-.43), .92, .86,
                           fill=False, edgecolor='#00745C', linewidth=2))
counts = {variant: int(sum(np.argmin(matrix, axis=0) == i)) for i, variant in enumerate(variants)}
assert counts['NCG_H1'] == 10 and counts['NG_H1'] == 3
cax = fig.add_axes([.92, .23, .014, .59])
fig.colorbar(im, cax=cax, label='总变差距离 / 百分点（越低越接近参考）')
fig.suptitle('13个工业场景 × 14种融合配置：参考分布距离', fontsize=15, y=.97, color=INK)
fig.text(.10, .91, '绿色边框表示同一场景中距离最小的配置；各单元格使用同一色标。',
         fontsize=10, color='#425469')
fig.text(.10, .075,
         'N：监督语义；C：开放语义；G：显式几何；H1启用装配上下文，H0关闭。\n'
         '参考为用户提供的P1–P10占比；本图不报告准确率，不能据此判断逐实例纠错或渲染收益。',
         fontsize=9, color='#425469', linespacing=1.7)
save(fig, '02_thirteen_scenes_fourteen_configurations_heatmap')

notes = '''# 14种融合配置的论文插图与使用说明

## 主图：总体配置对比

文件：`figures/01_fourteen_configurations_paired_comparison.png`及同名SVG。

建议题注：**图5-X 多源语义与几何结构融合配置的宏平均分布距离及装配上下文作用**

（a）以七种分支组合为纵轴，每种组合并列展示H0与H1，涵盖全部14种配置；
（b）展示H0减H1的距离差，正值表示启用上下文后更接近参考。13个场景等权平均。
N为监督语义，C为开放语义，G为显式几何。H1含拓扑、局部尺度和限定范围的祖先名称，不能称为纯拓扑收益。

建议放在5.3.2节表5-5附近。如果篇幅有限，可用主图替代表5-5，详细数值表移至附录。
图中H0与H1保持一致颜色，避免给14种配置分别分配颜色而掩盖配对关系。

可用分析文字：

> 图5-X显示，在所提供的参考分布下，完整融合NCG在H0和H1条件下均取得最低宏平均分布距离，
> 其中NCG_H1为3.85个百分点。启用装配上下文后，完整融合的平均距离减少7.16个百分点。
> 几何分支参与后的各配置整体更接近参考；该现象同时受到缺少几何证据时P5回退规则的影响，
> 不能将差值解释为语义识别准确率增益。

## 辅图：逐场景对比

文件：`figures/02_thirteen_scenes_fourteen_configurations_heatmap.png`及同名SVG。

建议题注：**图5-Y 十三个工业场景在十四种融合配置下的策略分布距离**

纵轴按N_H0、N_H1、C_H0、C_H1……NCG_H1排列，横轴为M1—M13。
绿色边框表示同一场景的最低距离。使用统一色标，不对每个场景分别缩放。
建议放在总体对比后，用于说明场景差异，也可置于附录。

可用分析文字：

> NCG_H1在10个场景中取得最低距离；M2、M7和M8由NG_H1取得最低距离。
> 相比NG_H1，加入开放语义后这三个场景的距离分别增加约1.00、0.10和0.61个百分点，
> 说明开放证据的作用具有场景差异，不能将完整配置表述为所有场景均最优。

## 指标与来源

- 所有图均使用总变差距离D=½Σ|算法占比−参考占比|，单位为百分点，越低越接近参考。
- 参考占比由用户提供，独立标注过程尚未核验。图形结论以该参考为条件，不额外证明参考的真实性或独立性。
- 数据来自`analysis_reports/user_reference_distribution_20261022/comparison_results.json`。
- 两位小数占比先归一化；源报告中“<0.01%”按0.005%近似，处理与前一份对比报告一致。
- 13个场景宏平均不是按实例数加权的总体指标，也不是类别宏平均准确率。
- 当前仅NCG_H1有用户提供的人工对照计数（总体准确率83.18%）。其余13配置没有逐实例准确率，故未绘制14配置准确率图。
- 图中不使用模拟示例数据，不表示已经实现的画质、显存、内存或帧率收益。

PNG按300 dpi导出；SVG为矢量图。复现入口：`plot_fourteen_configurations.py`。
'''
(HERE / '十四配置插图与正文使用说明.md').write_text(notes, encoding='utf-8')
manifest = {
    'metric': 'TOTAL_VARIATION_DISTANCE_PERCENTAGE_POINTS_NOT_ACCURACY',
    'reference_provenance': data['reference_provenance'],
    'source': str(SOURCE.relative_to(ROOT)),
    'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'scene_count': 13, 'configuration_count': 14,
    'variant_order': variants, 'scene_order': [model['id'] for model in models],
    'macro_tvd_H0_pp': h0.tolist(), 'macro_tvd_H1_pp': h1.tolist(),
    'tree_distance_reduction_pp': gains.tolist(), 'matrix_tvd_pp': matrix.tolist(),
    'strict_best_scene_counts': counts, 'font': font,
    'figures': [{'file': str(path.relative_to(ROOT)),
                 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in saved],
    'synthetic_demo_used': False, 'accuracy_imputed': False,
}
(HERE / 'figure_data_and_provenance.json').write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': 'generated', 'figures': len(saved), 'comparisons': int(matrix.size),
                  'best_NCG_H1': counts['NCG_H1'], 'best_NG_H1': counts['NG_H1'],
                  'font': font, 'source_sha256': manifest['source_sha256']}))
