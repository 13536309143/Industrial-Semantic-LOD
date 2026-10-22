"""Reference-centred scientific figures; no inferred instance accuracy."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
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
reference = np.array([model['normalized_reference_percent'] for model in models])
prediction = np.array([[model['predictions_percent'][variant] for variant in variants]
                       for model in models])
signed = prediction - reference[:, None, :]
mae = np.abs(signed).mean(axis=0)
tvd = np.abs(signed).sum(axis=2) / 2
macro_tvd = tvd.mean(axis=0)
assert signed.shape == (13, 14, 10)
assert np.allclose(reference.sum(axis=1), 100)
assert np.allclose(prediction.sum(axis=2), 100)
assert np.allclose(macro_tvd, mae.sum(axis=1) / 2)
for j, variant in enumerate(variants):
    assert np.isclose(macro_tvd[j], data['summary'][variant]['macro_tvd_pp'])
    for i, model in enumerate(models):
        assert np.isclose(tvd[i, j], model['tvd_percentage_points'][variant])

fonts = {font.name for font in font_manager.fontManager.ttflist}
font = next(name for name in ('Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC') if name in fonts)
plt.rcParams.update({'font.family': font, 'axes.unicode_minus': False,
                     'font.size': 10, 'svg.fonttype': 'none', 'savefig.dpi': 300})
INK, TEAL, ORANGE = '#172B42', '#087F79', '#BE6A18'
saved = []


def save(fig, name):
    for ext in ('png', 'svg'):
        path = OUT / f'{name}.{ext}'
        fig.savefig(path, bbox_inches='tight', facecolor='white')
        saved.append(path)
    plt.close(fig)


def small_number(value):
    return '<0.01' if 0 < value < .01 else f'{value:.2f}'


fig = plt.figure(figsize=(12.4, 8.5))
grid = fig.add_gridspec(1, 2, left=.105, right=.975, top=.82, bottom=.205,
                       width_ratios=[2.25, 1], wspace=.16)
ax = fig.add_subplot(grid[0, 0])
rank_ax = fig.add_subplot(grid[0, 1], sharey=ax)
im = ax.imshow(mae, cmap='Blues', vmin=0, vmax=70, aspect='auto')
ax.set(xticks=range(10), xticklabels=[f'P{k}' for k in range(1, 11)],
       yticks=range(14), yticklabels=variants, xlabel='几何保护与简化策略档位')
ax.tick_params(axis='both', length=0, pad=7, labelsize=10)
ax.set_title('（a）各P档位的平均绝对占比偏差', fontsize=12, loc='left', pad=15)
for row in range(14):
    for col in range(10):
        value = mae[row, col]
        ax.text(col, row, small_number(value), ha='center', va='center', fontsize=8.6,
                color='white' if value > 38 else INK)
for boundary in np.arange(1.5, 13, 2):
    ax.axhline(boundary, color='#CCD6DF', linewidth=.8)
for col in np.arange(.5, 10, 1):
    ax.axvline(col, color='white', linewidth=.5)
ax.add_patch(Rectangle((-.48, 12.53), 9.96, .94,
                       fill=False, edgecolor=TEAL, linewidth=1.8))
for spine in ax.spines.values():
    spine.set_visible(False)
cax = fig.add_axes([.107, .128, .54, .018])
colorbar = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[0, 10, 20, 30, 40, 50, 60, 70])
colorbar.set_label('平均绝对偏差 / 百分点（0表示占比一致，越浅越接近参考）', fontsize=9)
colorbar.ax.tick_params(labelsize=8)

for row in range(14):
    if (row // 2) % 2 == 0:
        rank_ax.axhspan(row-.5, row+.5, color='#F3F6F8', zorder=0)
    color = TEAL if row == 13 else ORANGE if row == 9 else '#8E9EAE'
    rank_ax.hlines(row, 0, macro_tvd[row], color=color, linewidth=2.5, zorder=2)
    rank_ax.plot(macro_tvd[row], row, 'o', color=color, markersize=5.5, zorder=3)
    rank_ax.text(macro_tvd[row]+1.2, row, f'{macro_tvd[row]:.2f}', va='center',
                 fontsize=10, fontweight='bold' if row in (9, 13) else 'normal', color=color)
rank_ax.set(xlim=(0, 89), ylim=(13.5, -.5), xlabel='宏平均TVD / 百分点（越低越接近参考）')
rank_ax.tick_params(axis='y', labelleft=False, length=0)
rank_ax.tick_params(axis='x', labelsize=9)
rank_ax.grid(axis='x', color='#DCE3E9', linewidth=.6)
rank_ax.set_axisbelow(True)
rank_ax.set_title('（b）总体分布距离', fontsize=12, loc='left', pad=15)
for side in ('top', 'right', 'left'):
    rank_ax.spines[side].set_visible(False)
fig.suptitle('以参考分布为基准：十四种融合配置的逐档偏差',
             fontsize=16, y=.975, color=INK)
fig.text(.105, .90, '基于M1—M13等权平均；左图逐场景取绝对值后再平均，避免正负偏差相互抵消。',
         fontsize=10, color='#425469')
fig.text(.70, .103, 'NCG_H1：平均距离最低\nNG_H1：平均距离次低',
         fontsize=9, color=TEAL, linespacing=1.6)
fig.text(.105, .025,
         '参考占比由用户提供；分布一致不等于逐实例判断正确。\n'
         'NCG_H1在10个场景中距离最低，M2、M7、M8由NG_H1取得最低距离。',
         fontsize=9, color='#425469', linespacing=1.7)
save(fig, '05_reference_absolute_deviation_and_distance')

# Full detail: every model, method, and priority retains its signed residual.
fig = plt.figure(figsize=(16.3, 16.4))
grid = fig.add_gridspec(4, 4, left=.065, right=.985, top=.925, bottom=.075,
                       hspace=.39, wspace=.34)
best = np.argmin(tvd, axis=1)
normalizer = TwoSlopeNorm(vmin=-85, vcenter=0, vmax=85)
for i, model in enumerate(models):
    panel = fig.add_subplot(grid[i//4, i%4])
    image = panel.imshow(signed[i], cmap='RdBu_r', norm=normalizer, aspect='auto')
    panel.set(xticks=range(10), xticklabels=[str(k) for k in range(1, 11)],
              yticks=range(14), yticklabels=variants)
    panel.tick_params(axis='both', length=0, pad=3, labelsize=6.3)
    for boundary in np.arange(1.5, 13, 2):
        panel.axhline(boundary, color='#D4DCE3', linewidth=.4)
    panel.add_patch(Rectangle((-.49, best[i]-.47), 9.98, .94,
                             fill=False, edgecolor=TEAL, linewidth=1.4))
    panel.set_title(f"{model['id']}  {model['name']}\n最低TVD：{variants[best[i]]}（{tvd[i,best[i]]:.2f}）",
                    loc='left', fontsize=8.5, linespacing=1.6, pad=7)
    panel.set_xlabel('P档位（P1—P10）', fontsize=7)
    for spine in panel.spines.values():
        spine.set_visible(False)
legend = fig.add_subplot(grid[3, 1:])
legend.axis('off')
legend.text(0, .99, '以每个模型自身参考分布为零点', fontsize=16, color=INK, va='top')
inset = legend.inset_axes([.01, .69, .92, .09])
bar = fig.colorbar(image, cax=inset, orientation='horizontal', ticks=[-80, -40, 0, 40, 80])
bar.set_label('算法占比 − 参考占比 / 百分点', fontsize=10)
legend.text(.01, .45,
            '蓝色：该档位分配偏少；红色：分配偏多。\n'
            '白色：占比接近参考，不等于实例逐一正确。\n'
            '绿色边框：该场景总体TVD最小的配置。\n'
            '十三个模型共用线性±85百分点色标，不逐模型缩放。',
            va='top', fontsize=10, linespacing=2.0, color='#425469')
fig.suptitle('十三模型的P1–P10参考中心偏差：全部十四种配置', fontsize=18, y=.987, color=INK)
fig.text(.065, .025, '每个格子保留一个模型、一种配置、一个P档位的占比差；参考来自用户提供的汇总，非准确率图。',
         fontsize=10, color='#425469')
save(fig, '06_reference_signed_deviation_all_models')

manifest = {
    'source': str(SOURCE.relative_to(ROOT)),
    'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'reference_provenance': data['reference_provenance'],
    'normalization': 'Same normalized report percentages as Figure 5-1 and Table 5-5',
    'prediction_precision': 'Original reports rounded to 2 decimals; <0.01 replaced by midpoint 0.005 before normalization',
    'reference_original_percent': [model['user_reference_original_percent'] for model in models],
    'reference_normalized_percent': reference.tolist(),
    'scene_order': [model['id'] for model in models], 'variant_order': variants,
    'signed_error_pp': signed.tolist(), 'mean_absolute_error_per_priority_pp': mae.tolist(),
    'tvd_per_scene_pp': tvd.tolist(), 'macro_tvd_pp': macro_tvd.tolist(),
    'detail_color_limits_pp': [-85, 85], 'main_color_limits_pp': [0, 70],
    'best_variants': [variants[i] for i in best],
    'synthetic_demo_used': False, 'accuracy_computed': False,
    'figures': [{'file': str(path.relative_to(ROOT)),
                 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in saved],
}
(HERE / 'reference_deviation_data_and_provenance.json').write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'status': 'generated', 'figures': len(saved),
                  'residuals': int(signed.size), 'NCG_H1_macro_TVD': float(macro_tvd[-1]),
                  'NCG_H1_best_scenes': int(sum(best==13)),
                  'NG_H1_best_scenes': int(sum(best==9))}))
