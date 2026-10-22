"""Create paired fusion-analysis figures from verified V5 all-node policies.

Uses matplotlib for standalone PNG/SVG figures and no external web services.
No inference, policy changes, deployment or rendering benchmarks are performed.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'analysis_reports/adaptive_p10_v5_20261021'
POLICIES = ROOT / 'lod_analysis_outputs/adaptive_p10_v5_20261021'
OUT = SOURCE / 'fusion_charts'
SCENES = list('abcdef')
MODES = ['N', 'C', 'NG', 'CG', 'NC', 'NCG']
MAIN = [f'{m}_H{h}' for m in MODES for h in (0, 1)]
VARIANTS = MAIN + ['G_H0', 'G_H1', 'NCG_H1_M1']
OBJECT_STATUSES = {'closed_candidate', 'open_candidate', 'open_class_candidate'}
COLORS = ['#254b7c', '#427bb0', '#74a9cf', '#a8d0e3', '#d9e3e8',
          '#f5d3a5', '#eeac73', '#d87a5c', '#b4474b', '#742b47']
FAMILIES = {'compact_simple': '紧凑简单体', 'compact_curved': '紧凑曲面体',
            'compact_featured': '紧凑细节体', 'plate': '板状', 'rod': '杆状',
            'elongated_sheet': '细长扁平', 'distributed_components': '分散多分量',
            'invalid_geometry': '无效几何'}
FIGURES = []


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def yes(value):
    return str(value).lower() in ('true', '1')


def grade(row):
    return int(row['lod_priority'].split('_')[0][1:])


def ident(row):
    return row['node_index'], row['mesh_index']


def summary(scene, variant, rows):
    n = len(rows)
    f = sum(int(r['face_count']) for r in rows)
    counts = Counter(grade(r) for r in rows)
    faces = Counter()
    for r in rows:
        faces[grade(r)] += int(r['face_count'])
    allowed = OBJECT_STATUSES | ({'metadata_candidate'} if variant.endswith('_M1') else set())
    result = dict(scene=scene, variant=variant, nodes=n, source_faces=f,
        unknown_function_count=sum(r['function_role'] == 'unknown' for r in rows),
        object_unresolved_count=sum(r['resolved_status'] not in allowed for r in rows),
        identity_conflict_count=sum(yes(r['class_conflict']) for r in rows),
        shape_conflict_count=sum(yes(r['type_shape_conflict']) for r in rows),
        review_count=sum(yes(r['needs_review']) for r in rows))
    for p in range(1, 11):
        result[f'P{p}_count'] = counts[p]
        result[f'P{p}_faces'] = faces[p]
    return result


def fractions(r):
    result = dict(r)
    for p in range(1, 11):
        result[f'P{p}_node_fraction'] = r[f'P{p}_count'] / r['nodes']
        result[f'P{p}_face_fraction'] = r[f'P{p}_faces'] / r['source_faces']
    for metric in ('unknown_function', 'object_unresolved', 'identity_conflict', 'shape_conflict', 'review'):
        result[metric + '_fraction'] = r[metric + '_count'] / r['nodes']
    return result


def paired(scene, kind, before, after, old, new):
    if [ident(r) for r in old] != [ident(r) for r in new]:
        raise ValueError(f'Unpaired node identities: {scene}/{before}/{after}')
    n = len(old)
    p_changed = raised = lowered = function = identity = status = controls = 0
    changed_faces = 0
    for a, b in zip(old, new):
        pa, pb = grade(a), grade(b)
        p_changed += pa != pb
        raised += pb > pa
        lowered += pb < pa
        function += a['function_role'] != b['function_role']
        identity += a['resolved_object_class'] != b['resolved_object_class']
        status += a['resolved_status'] != b['resolved_status']
        controls += any(a[k] != b[k] for k in ('feature_lock_budget', 'simplify_error_relative',
                        'screen_error_weight', 'max_axis_shrink', 'target_ratio_far'))
        if pa != pb:
            changed_faces += int(a['face_count'])
    return dict(scene=scene, kind=kind, before=before, after=after, nodes=n,
        source_faces=sum(int(r['face_count']) for r in old),
        priority_changed=p_changed, priority_raised=raised, priority_lowered=lowered,
        function_changed=function, identity_changed=identity, status_changed=status,
        controls_changed=controls, priority_changed_source_faces=changed_faces)


def aggregate(rows, group_fields, numeric_fields):
    groups = {}
    for r in rows:
        k = tuple(r[g] for g in group_fields)
        if k not in groups:
            groups[k] = dict(scene='ALL', **{g: r[g] for g in group_fields},
                             **{f: 0 for f in numeric_fields})
        for f in numeric_fields:
            groups[k][f] += r[f]
    return list(groups.values())


def style():
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 10,
        'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.spines.left': False, 'axes.spines.bottom': False,
        'axes.titleweight': 'bold', 'axes.titlepad': 12,
        'svg.fonttype': 'none', 'figure.facecolor': 'white', 'axes.facecolor': 'white'})


def export_figure(fig, filename, title, note):
    fig.suptitle(title, x=.035, y=.98, ha='left', fontsize=17, fontweight='bold', color='#20334d')
    fig.text(.035, .018, note, fontsize=9, color='#56677c', va='bottom')
    for extension in ('png', 'svg'):
        fig.savefig(OUT / f'{filename}.{extension}', dpi=220, facecolor='white')
    FIGURES.append(dict(file=filename, title=title, note=note))
    plt.close(fig)


def legend(fig, y=.92):
    fig.legend(handles=[Patch(facecolor=c, label=f'P{i+1}') for i, c in enumerate(COLORS)],
               loc='upper center', bbox_to_anchor=(.5, y), ncol=10, frameon=False,
               handlelength=1.5, handleheight=.9, columnspacing=1.4)


def stacks(ax, rows, basis='node', labels=None):
    denominator = 'nodes' if basis == 'node' else 'source_faces'
    suffix = 'count' if basis == 'node' else 'faces'
    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    for p in range(1, 11):
        values = np.array([100 * r[f'P{p}_{suffix}'] / r[denominator] for r in rows])
        ax.barh(y, values, left=left, color=COLORS[p-1], edgecolor='white', linewidth=.35, height=.72)
        for i, value in enumerate(values):
            if value >= 13:
                ax.text(left[i] + value / 2, i, f'{value:.0f}%', ha='center', va='center',
                        fontsize=8, color='white' if p in (1, 2, 9, 10) else '#24334a')
        left += values
    ax.set_yticks(y, labels or [r['variant'].replace('_', ' ') for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ['0%', '25%', '50%', '75%', '100%'])
    ax.tick_params(length=0)
    ax.grid(axis='x', alpha=.15, linewidth=.6)
    ax.set_axisbelow(True)


def heat(ax, matrix, xlabels, ylabels, title, vmax=100, signed=False):
    data = np.asarray(matrix)
    im = ax.imshow(data, aspect='auto', cmap='RdBu_r' if signed else 'Blues',
                   vmin=-vmax if signed else 0, vmax=vmax)
    ax.set_xticks(range(len(xlabels)), xlabels, fontsize=9)
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=10)
    ax.set_title(title, loc='left', fontsize=12)
    ax.tick_params(length=0)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            text = f'{value:+.1f}' if signed else f'{value:.1f}'
            ax.text(j, i, text, ha='center', va='center', fontsize=9,
                    color='white' if abs(value) > vmax * .6 else '#24334a')
    return im


def make_figures(stats, pairs, families, source_uncertainty, cohorts, titles):
    style()
    idx = {(r['scene'], r['variant']): r for r in stats}
    lab = [f'{s}  {titles[s]}' for s in SCENES]
    for basis, number, caption in [('node', '01', '节点'), ('face', '02', '原始三角形实例')]:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.subplots_adjust(top=.85, bottom=.08, left=.075, right=.98, wspace=.28, hspace=.3)
        for s, ax in zip(SCENES, axes.flat):
            stacks(ax, [idx[s, v] for v in MAIN], basis)
            ax.set_title(f'{s}  {titles[s]}\n{idx[s, MAIN[0]]["nodes"]:,} 节点', loc='left', fontsize=10.5)
        legend(fig, .93)
        export_figure(fig, number + '_six_scene_' + basis + '_distribution',
            f'六模型不同融合配置的 P1–P10 分布（{caption}占比）',
            'N=监督，C=开放，G=几何；H0关闭结构树，H1启用局部父组与祖先名称。主对照均不使用叶名称。'
            + (' 三角形占比不等于简化后面数或显存占比。' if basis == 'face' else ' P档位变化不代表识别准确率提升。'))
    for s in SCENES:
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        fig.subplots_adjust(top=.80, bottom=.1, left=.09, right=.98, wspace=.24)
        for ax, basis, title in zip(axes, ('node', 'face'), ('节点占比', '原始三角形实例占比')):
            stacks(ax, [idx[s, v] for v in MAIN], basis)
            ax.set_title(title, loc='left', fontsize=12)
        legend(fig, .92)
        export_figure(fig, f'scene_{s}_fusion', f'{s}  {titles[s]}：十二种融合配置',
                      '左图按节点数，右图按原始三角形实例数；同一配置在两图中的分母不同。H1含祖先名称辅助，非纯拓扑消融。')
    tree = {(r['scene'], r['before'].split('_')[0]): r for r in pairs if r['kind'] == 'tree'}
    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    fig.subplots_adjust(top=.80, bottom=.13, left=.10, right=.98, wspace=.23)
    for ax, field, title in zip(axes, ('priority_changed', 'function_changed'),
                                 ('P档位改变的节点（%）', '功能角色改变的节点（%）')):
        heat(ax, [[100*tree[s, m][field]/tree[s, m]['nodes'] for m in MODES] for s in SCENES], MODES, SCENES, title, vmax=20)
    export_figure(fig, '03_tree_paired_changes', '结构树辅助的逐节点成对变化（H1 对 H0）',
                  '两图共用0–20%色阶。逐节点对齐同一模型；改变不代表正确。a缺乏有效装配树，可作为辅助开关的控制场景。')
    geometry = {(r['scene'], r['before']): r for r in pairs if r['kind'] == 'geometry'}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(top=.80, bottom=.13, left=.08, right=.98, wspace=.22)
    for h, ax in enumerate(axes):
        heat(ax, [[100*geometry[s, f'{m}_H{h}']['priority_changed']/geometry[s, f'{m}_H{h}']['nodes']
                   for m in ('N', 'C', 'NC')] for s in SCENES],
             ['N → NG', 'C → CG', 'NC → NCG'], SCENES, f'H{h}：加入几何后P档位改变（%）')
    export_figure(fig, '04_geometry_branch_changes', '几何分支如何重新分配整体预算',
                  '保持树开关和输入节点相同；G同时参与几何相关融合与约束，图中不是单独一个P分配条件的因果消融。')
    fig, axes = plt.subplots(2, 1, figsize=(17, 10))
    fig.subplots_adjust(top=.87, bottom=.10, left=.08, right=.98, hspace=.37)
    for ax, field, title in zip(axes, ('unknown_function_fraction', 'object_unresolved_fraction'),
         ('功能未明确节点（%）', '无可用对象级输出节点（%）')):
        heat(ax, [[100*idx[s, v][field] for v in MAIN] for s in SCENES],
             [v.replace('_', '\n') for v in MAIN], SCENES, title)
    export_figure(fig, '05_unresolved_evidence', '不同融合配置的未决证据',
                  '功能未明确与没有对象级输出分开统计。后者包含粗类别、拒识和冲突；两者都不是错误率。')
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    fig.subplots_adjust(top=.81, bottom=.14, left=.08, right=.98, wspace=.26)
    stacks(axes[0], [idx['ALL', v] for v in ('NCG_H0', 'NCG_H1', 'NCG_H1_M1')],
           labels=['完整融合 H0', '完整融合 H1', '完整融合 H1+M1'])
    axes[0].set_title('六模型节点汇总（合计后求比例）', loc='left', fontsize=12)
    metadata = {r['scene']: r for r in pairs if r['kind'] == 'metadata'}
    y = np.arange(6)
    axes[1].barh(y-.18, [100*tree[s, 'NCG']['priority_changed']/tree[s, 'NCG']['nodes'] for s in SCENES],
                 height=.32, label='H1 对 H0', color='#427bb0')
    axes[1].barh(y+.18, [100*metadata[s]['priority_changed']/metadata[s]['nodes'] for s in SCENES],
                 height=.32, label='M1 对 H1', color='#d87a5c')
    axes[1].set_yticks(y, SCENES); axes[1].invert_yaxis()
    axes[1].set_xlabel('P档位改变的节点（%）'); axes[1].legend(frameon=False)
    axes[1].set_title('树辅助与叶名称辅助分别比较', loc='left', fontsize=12)
    legend(fig, .93)
    export_figure(fig, '06_full_fusion_and_metadata', '完整融合、结构树与叶名称辅助',
                  'M1使用叶名称，单独作为辅助条件；不能再用同一叶名称给M1计算识别准确率。汇总按节点数加权，非六场景比例的简单平均。')
    fam_names = [f for f in FAMILIES if any(r['family'] == f for r in families)]
    family_idx = {(r['scene'], r['family']): r for r in families}
    fig, axes = plt.subplots(1, 2, figsize=(19, 7))
    fig.subplots_adjust(top=.81, bottom=.13, left=.07, right=.98, wspace=.30)
    totals = {s: sum(r['nodes'] for r in families if r['scene'] == s) for s in SCENES}
    matrix = [[100*family_idx.get((s, f), {'nodes': 0})['nodes']/max(1, totals[s]) for s in SCENES] for f in fam_names]
    heat(axes[0], matrix, SCENES, [FAMILIES[f] for f in fam_names], '未知功能件内部的几何组成（%）')
    matrix = [[100*family_idx['ALL', f][f'P{p}_count']/max(1, family_idx['ALL', f]['nodes'])
               for p in range(1, 11)] for f in fam_names]
    heat(axes[1], matrix, [f'P{p}' for p in range(1, 11)], [FAMILIES[f] for f in fam_names],
         '各几何分型内部的P档位分配（%）')
    export_figure(fig, '07_unknown_geometry_families', '功能未明确件如何利用几何信息细分',
                  '使用NCG H1+M1完整辅助条件。左图每个模型列合计100%，右图每个几何分型行合计100%；分型为形态描述，不是对象识别真值。')
    uidx = {r['scene']: r for r in source_uncertainty}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.subplots_adjust(top=.80, bottom=.15, left=.08, right=.98, wspace=.24)
    for ax, suffix, title in zip(axes, ('node_fraction', 'face_fraction'), ('按节点数', '按原始三角形实例数')):
        x = np.arange(6)
        ax.bar(x-.19, [100*uidx[s]['N_not_reliable_'+suffix] for s in SCENES], width=.36, color='#427bb0', label='N未达到可靠性阈值')
        ax.bar(x+.19, [100*uidx[s]['C_not_object_'+suffix] for s in SCENES], width=.36, color='#d87a5c', label='C无对象级输出')
        ax.set_xticks(x, SCENES); ax.set_ylim(0, 100); ax.set_ylabel('占比（%）'); ax.set_title(title, loc='left')
        ax.legend(frameon=False, fontsize=9); ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    export_figure(fig, '08_branch_uncertainty', '监督与开放分支的原始证据覆盖诊断',
                  'N：Top1<0.70或Top1−Top2<0.08。C：状态非accepted/class_only。判定口径不同，仅作分支诊断，不能作为两模型准确率排名。')
    fig, ax = plt.subplots(figsize=(13, 4.8))
    fig.subplots_adjust(top=.71, bottom=.18, left=.19, right=.98)
    converted = []
    for role in ('unknown', 'standard_fastener'):
        r = next(r for r in cohorts if r['scene'] == 'all' and r['cohort'] == role)
        converted.append({**{f'P{p}_count': int(r[f'P{p}_count']) for p in range(1, 11)},
                          'nodes': int(r['nodes'])})
    stacks(ax, converted, labels=['原P6未知功能件  n=2,508', '原P6普通紧固件  n=2,055'])
    legend(fig, .89)
    export_figure(fig, '09_previous_p6_cohorts', 'V4原P6两批节点在V5中的分配',
                  '固定跟踪同一批节点；不是不同模型的识别准确率比较。降低P6集中度不等于已经证明渲染画质或资源收益。')


def dashboard(stats, pairs, titles):
    template = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>六模型融合分析 V5</title><style>
:root{font-family:"Microsoft YaHei",system-ui,sans-serif;color:#24334a;background:#f3f5f8}body{margin:0}main{max-width:1420px;margin:auto;padding:30px}h1{font-size:27px;margin:0 0 10px}h2{font-size:18px;margin:0 0 14px}p{line-height:1.7}.muted{color:#65758a;font-size:13px}.panel{background:white;padding:23px;border:1px solid #e1e7ef;border-radius:10px;margin:20px 0}.controls{display:flex;gap:22px;flex-wrap:wrap}label{font-size:13px;display:grid;gap:7px}select{padding:9px 13px;border:1px solid #c9d3e0;border-radius:5px;background:white;min-width:210px;font:inherit}.legend{display:flex;gap:17px;flex-wrap:wrap;margin:16px 0;font-size:12px}.swatch{display:inline-block;width:15px;height:11px;margin-right:5px;border-radius:2px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:15px}.card{padding:15px;background:#f8fafc;border-radius:6px}.card strong{font-size:24px;display:block;margin-top:6px}.scroll{overflow:auto}svg{width:100%;min-width:720px;display:block}svg text{font-family:"Microsoft YaHei",sans-serif}table{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}th{background:#20334d;color:white;text-align:right;padding:11px 9px}td{text-align:right;padding:10px 9px;border-bottom:1px solid #e8edf2}th:first-child,td:first-child{text-align:left}tr:hover td{background:#f1f6fc}a{color:#255d96}.figures{display:grid;grid-template-columns:repeat(2,1fr);gap:20px}.figures img{width:100%;border:1px solid #e5eaf1}details{margin:12px 0}@media(max-width:800px){main{padding:16px}.cards{grid-template-columns:repeat(2,1fr)}.figures{grid-template-columns:1fr}}
</style><main><h1>六模型不同融合配置分析</h1><p class="muted">V5全量策略回放。N监督语义，C开放语义，G几何结构。H1含局部父组与祖先名称；M1额外使用叶名称。图中没有实测准确率、帧率或显存结果。</p>
<section class="panel"><div class="controls"><label>模型<select id="scene"><option value="ALL">六模型汇总（按计数加权）</option>__OPTIONS__</select></label><label>配置范围<select id="scope"><option value="main">六组合 × H0/H1</option><option value="H0">六组合 H0</option><option value="H1">六组合 H1</option><option value="full">完整融合与辅助对照</option></select></label><label>占比口径<select id="basis"><option value="node">节点数</option><option value="face">原始三角形实例数</option></select></label></div></section>
<section class="panel"><h2 id="heading"></h2><div class="cards" id="cards"></div><div class="legend" id="legend"></div><div class="scroll" id="bars"></div><p class="muted">颜色表示P档位。鼠标停留可查看精确数量与占比。三角形实例数不等于实际显存；P是整体保护预算，不是类别或识别概率。</p></section>
<section class="panel"><h2>功能未明确与无对象级输出</h2><div class="scroll" id="evidence"></div><p class="muted">两类未决情况分别统计；功能角色可以明确而对象身份仍未确定，也可能已有对象候选但功能未明确。M1辅助单独列示。</p></section>
<section class="panel"><h2>结构树带来的逐节点变化</h2><div class="scroll" id="paired"></div><p class="muted">H1相对H0，节点身份一一对应；升档与降档均计为改变，改变本身不证明正确。</p></section>
<section class="panel"><h2>当前配置精确数据</h2><div class="scroll"><table id="data"></table></div></section>
<section class="panel"><h2>汇报图与分析文件</h2><p><a href="六模型不同融合图表分析.md">阅读完整分析</a>　<a href="configuration_summary.csv">配置汇总CSV</a>　<a href="paired_changes.csv">逐节点配对变化CSV</a></p><div class="figures">__GALLERY__</div></section>
<details class="panel"><summary>统计口径</summary><p>主对照72组均不读取叶名称；纯G和M1为附加条件。汇总先相加计数再求比例，不对场景百分比简单平均。对象级输出状态为closed_candidate/open_candidate/open_class_candidate；M1另计metadata_candidate。未提供人工真值，因此不计算OA、Macro-F1或显著性。六场景均参与过方法开发。</p></details></main>
<script>const DATA=__DATA__, PAIRS=__PAIRS__, TITLES=__TITLES__, COLORS=__COLORS__, MODES=['N','C','NG','CG','NC','NCG'];
const $=id=>document.getElementById(id), pct=v=>(100*v).toFixed(2)+'%', num=v=>v.toLocaleString('zh-CN'), esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function svg(w,h,body){return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}">${body}</svg>`}
function render(){const scene=$('scene').value,scope=$('scope').value,basis=$('basis').value;let rows=DATA.filter(r=>r.scene===scene);rows=rows.filter(r=>scope==='main'?/^(N|C|NG|CG|NC|NCG)_H[01]$/.test(r.variant):scope==='full'?['G_H0','G_H1','NCG_H0','NCG_H1','NCG_H1_M1'].includes(r.variant):r.variant.endsWith('_'+scope)&&MODES.includes(r.variant.split('_')[0]));
$('heading').textContent=(scene==='ALL'?'六模型汇总':scene+'  '+TITLES[scene])+'：P1–P10分布';const ref=DATA.find(r=>r.scene===scene&&r.variant==='NCG_H1_M1');$('cards').innerHTML=[['节点总数',num(ref.nodes)],['原始三角形实例',num(ref.source_faces)],['完整辅助融合的功能未知',pct(ref.unknown_function_fraction)],['完整辅助融合的P6占比',pct(ref.P6_node_fraction)]].map(x=>`<div class="card"><span class="muted">${x[0]}</span><strong>${x[1]}</strong></div>`).join('');$('legend').innerHTML=COLORS.map((c,i)=>`<span><i class="swatch" style="background:${c}"></i>P${i+1}</span>`).join('');
let body='';const left=160,width=1000,dy=42;for(let tick=0;tick<=100;tick+=25){let x=left+width*tick/100;body+=`<line x1="${x}" y1="24" x2="${x}" y2="${rows.length*dy+25}" stroke="#e5eaf0"/><text x="${x}" y="16" text-anchor="middle" fill="#65758a" font-size="12">${tick}%</text>`}rows.forEach((r,i)=>{let x=left,y=36+i*dy,den=basis==='node'?r.nodes:r.source_faces;body+=`<text x="145" y="${y+18}" text-anchor="end" font-size="13">${esc(r.variant)}</text>`;for(let p=1;p<=10;p++){const count=r['P'+p+(basis==='node'?'_count':'_faces')],v=count/den,w=width*v;body+=`<rect x="${x}" y="${y}" width="${w}" height="27" fill="${COLORS[p-1]}" stroke="white" stroke-width=".5"><title>${esc(r.variant)} P${p}: ${num(count)} (${pct(v)})</title></rect>`;if(v>=.10)body+=`<text x="${x+w/2}" y="${y+18}" text-anchor="middle" font-size="12" fill="${[1,2,9,10].includes(p)?'white':'#24334a'}">${pct(v)}</text>`;x+=w}});$('bars').innerHTML=svg(1200,rows.length*dy+60,body);
body='';rows.forEach((r,i)=>{let y=30+i*39;body+=`<text x="145" y="${y+16}" text-anchor="end" font-size="12">${esc(r.variant)}</text>`;for(let j=0;j<2;j++){let value=r[j?'object_unresolved_fraction':'unknown_function_fraction'];body+=`<rect x="${160+j*510}" y="${y}" width="${450*value}" height="24" fill="${j?'#d87a5c':'#427bb0'}"/><text x="${165+j*510+450*value}" y="${y+16}" font-size="12">${pct(value)}</text>`}});body=`<text x="160" y="16" font-size="13">功能未明确</text><text x="670" y="16" font-size="13">无对象级输出</text>`+body;$('evidence').innerHTML=svg(1200,rows.length*39+45,body);
const pr=PAIRS.filter(r=>r.scene===scene&&r.kind==='tree'&&MODES.includes(r.before.split('_')[0]));body='';pr.forEach((r,i)=>{let y=30+i*40;body+=`<text x="145" y="${y+17}" text-anchor="end" font-size="12">${r.before.split('_')[0]}</text>`;for(let j=0;j<2;j++){let v=r[j?'function_changed':'priority_changed']/r.nodes;body+=`<rect x="${160+j*510}" y="${y}" width="${450*v}" height="25" fill="${j?'#d87a5c':'#427bb0'}"/><text x="${165+j*510+450*v}" y="${y+17}" font-size="12">${pct(v)}</text>`}});body=`<text x="160" y="16" font-size="13">P档位改变</text><text x="670" y="16" font-size="13">功能角色改变</text>`+body;$('paired').innerHTML=svg(1200,pr.length*40+45,body);
$('data').innerHTML='<thead><tr><th>配置</th>'+Array.from({length:10},(_,i)=>`<th>P${i+1}</th>`).join('')+'<th>功能未知</th><th>对象未决</th><th>N/C冲突</th></tr></thead><tbody>'+rows.map(r=>'<tr><td>'+r.variant+'</td>'+Array.from({length:10},(_,i)=>'<td>'+pct(r['P'+(i+1)+(basis==='node'?'_node_fraction':'_face_fraction')])+'</td>').join('')+'<td>'+pct(r.unknown_function_fraction)+'</td><td>'+pct(r.object_unresolved_fraction)+'</td><td>'+num(r.identity_conflict_count)+'</td></tr>').join('')+'</tbody>';}
['scene','scope','basis'].forEach(id=>$(id).addEventListener('change',render));render();</script></html>'''
    gallery = ''.join(f'<a href="{f["file"]}.png"><img src="{f["file"]}.png" loading="lazy" alt="{f["title"]}"><p>{f["title"]}</p></a>'
                      for f in FIGURES if not f['file'].startswith('scene_'))
    values = {'__DATA__': json.dumps(stats, ensure_ascii=False), '__PAIRS__': json.dumps(pairs, ensure_ascii=False),
              '__TITLES__': json.dumps(titles, ensure_ascii=False), '__COLORS__': json.dumps(COLORS),
              '__OPTIONS__': ''.join(f'<option value="{s}">{s}  {titles[s]}</option>' for s in SCENES), '__GALLERY__': gallery}
    for key, value in values.items():
        template = template.replace(key, value)
    (OUT / '六模型融合交互图表.html').write_text(template, encoding='utf-8')


def report(stats, pairs, inventory):
    idx = {(r['scene'], r['variant']): r for r in stats}
    tree = [r for r in pairs if r['scene'] == 'ALL' and r['kind'] == 'tree' and r['before'].split('_')[0] in MODES]
    lines = ['# 六模型不同融合配置图表分析（V5）', '',
        '分析对象是V5策略回放，不是V4运行时性能测量。共6模型、11,000节点；主对照为六组合×H0/H1，共72组。'
        '纯G对照12组、M1辅助6组另列，总计90组，所有条件均覆盖场景全部节点。', '',
        'N=监督语义，C=开放语义，G=几何结构。H0关闭树辅助；H1使用局部父组与祖先名称，不能称为纯拓扑消融。'
        'M1额外使用叶名称，不纳入主72组。', '',
        '## 模型与分析规模', '', '| 模型 | 场景 | 节点 | 原始三角形实例 |', '|---|---|---:|---:|']
    for r in inventory:
        lines.append(f"| {r['scene']} | {r['title']} | {int(r['nodes']):,} | {idx[r['scene'], 'NCG_H0']['source_faces']:,} |")
    lines += ['', '三角形按节点实例累计；共享网格可能重复计入，因此不能直接换算实际内存或显存。'
        '汇总一律先相加计数再求比例，避免场景规模不同造成平均比例失真。', '',
        '## 六种融合组合的整体差异', '',
        '| 配置 | 功能未明确 | 无对象级输出 | P1–P3 | P6 | P8–P10 | N/C冲突节点 |', '|---|---:|---:|---:|---:|---:|---:|']
    for v in MAIN:
        r = idx['ALL', v]
        low = sum(r[f'P{p}_node_fraction'] for p in range(1, 4))
        high = sum(r[f'P{p}_node_fraction'] for p in range(8, 11))
        lines.append(f"| {v} | {r['unknown_function_fraction']:.2%} | {r['object_unresolved_fraction']:.2%} | {low:.2%} | {r['P6_node_fraction']:.2%} | {high:.2%} | {r['identity_conflict_count']:,} |")
    for old, new in [('N_H0', 'NG_H0'), ('C_H0', 'CG_H0'), ('NC_H0', 'NCG_H0')]:
        a, b = idx['ALL', old], idx['ALL', new]
        pair = next(r for r in pairs if r['scene'] == 'ALL' and r['before'] == old and r['after'] == new)
        lines += ['', f"- **{old} → {new}**：{pair['priority_changed']:,}节点改变P（{pair['priority_changed']/pair['nodes']:.2%}）；"
            f"功能未明确比例{a['unknown_function_fraction']:.2%}→{b['unknown_function_fraction']:.2%}，"
            f"无对象级输出{a['object_unresolved_fraction']:.2%}→{b['object_unresolved_fraction']:.2%}。"]
    n, c, nc = [idx['ALL', v] for v in ('N_H0', 'C_H0', 'NC_H0')]
    lines += ['', f"监督与开放联合后（NC H0），无对象级输出比例为{nc['object_unresolved_fraction']:.2%}，"
        f"低于仅监督的{n['object_unresolved_fraction']:.2%}和仅开放的{c['object_unresolved_fraction']:.2%}；"
        f"同时显式保留{nc['identity_conflict_count']:,}个N/C分歧节点（{nc['identity_conflict_fraction']:.2%}）。"
        '这表明候选覆盖有互补性，不能由此推导识别更准确。', '',
        'G的直接用途是依据尺度、形态、细节和连接结构分配几何预算。'
        '仅语义配置中的未知功能件使用保守起点P5，启用G后可分散到多个档位；这属于预算分配能力，不能解释成分类准确率增加。'
        'N/C冲突数更多可能表示显式保留了分歧，不能单凭冲突数对方法排名。', '',
        '![节点占比](01_six_scene_node_distribution.png)', '',
        '![三角形实例占比](02_six_scene_face_distribution.png)', '',
        '节点占比突出大量小零件，三角形占比突出高面数节点。两图应并列使用；高P节点较少也可能覆盖大量源几何。', '',
        '## 结构树与名称辅助的作用', '',
        '| 分支组合 | H1改变P | 升档 | 降档 | 改变功能 | 改变对象身份 |', '|---|---:|---:|---:|---:|---:|']
    for r in tree:
        lines.append(f"| {r['before'].split('_')[0]} | {r['priority_changed']:,}（{r['priority_changed']/r['nodes']:.2%}） | {r['priority_raised']:,} | {r['priority_lowered']:,} | {r['function_changed']:,} | {r['identity_changed']:,} |")
    ncg0, ncg1 = idx['ALL', 'NCG_H0'], idx['ALL', 'NCG_H1']
    lines += ['', '此次六组合的H辅助P变化全部为升档，没有降档。当前树分支的作用偏向增加保护，'
        '不能仅凭介入比例高就判定融合更优；后续应检查是否存在不必要的资源开销。', '',
        f"完整融合加入H1后，功能未知比例{ncg0['unknown_function_fraction']:.2%}→{ncg1['unknown_function_fraction']:.2%}，"
        f"无对象级输出{ncg0['object_unresolved_fraction']:.2%}→{ncg1['object_unresolved_fraction']:.2%}，并未减少。"
        '树辅助的收益应优先从局部作用与几何约束是否合理来验证，不能自动解释成识别率提高。']
    a = next(r for r in pairs if r['scene'] == 'a' and r['kind'] == 'tree' and r['before'] == 'NCG_H0')
    m = next(r for r in pairs if r['scene'] == 'ALL' and r['kind'] == 'metadata')
    lines += ['', f"a场景缺乏可用装配树：NCG的H开关改变P为{a['priority_changed']}节点，功能改变为{a['function_changed']}节点。"
        '该控制有助于判断树分支是否在缺少信息时凭空生效。', '',
        f"在已经启用H1的基础上加入M1，另有{m['priority_changed']:,}节点改变P（{m['priority_changed']/m['nodes']:.2%}），"
        f"{m['function_changed']:,}节点改变功能。名称辅助是额外证据，不能与不使用叶名称的主对照混为一组。"
        '纯G不承担对象识别，其全部功能未知只是输入分支的能力范围，不应解释成100%识别失败。', '',
        '![结构树配对](03_tree_paired_changes.png)', '',
        '![几何分支](04_geometry_branch_changes.png)', '',
        '![完整融合与名称](06_full_fusion_and_metadata.png)', '',
        '## 六个模型分别看什么', '',
        '| 模型 | H1完整融合：功能未知 | 加M1后：功能未知 | H1改变P占比 | H1+M1中P8–P10 |',
        '|---|---:|---:|---:|---:|']
    for s in SCENES:
        h, m = idx[s, 'NCG_H1'], idx[s, 'NCG_H1_M1']
        t = next(r for r in pairs if r['scene'] == s and r['kind'] == 'tree' and r['before'] == 'NCG_H0')
        lines.append(f"| {s} | {h['unknown_function_fraction']:.2%} | {m['unknown_function_fraction']:.2%} | {t['priority_changed']/t['nodes']:.2%} | {sum(m[f'P{p}_node_fraction'] for p in range(8,11)):.2%} |")
    strongest = max((r for r in pairs if r['kind'] == 'tree' and r['before'] == 'NCG_H0' and r['scene'] != 'ALL'),
                    key=lambda r: r['priority_changed']/r['nodes'])
    unknown = max(SCENES, key=lambda s: idx[s, 'NCG_H1_M1']['unknown_function_fraction'])
    lines += ['', f"完整融合中，结构树对P分配改变比例最大的场景为{strongest['scene']}（{strongest['priority_changed']/strongest['nodes']:.2%}）；"
        f"名称辅助后功能未明确比例最高的场景为{unknown}（{idx[unknown, 'NCG_H1_M1']['unknown_function_fraction']:.2%}）。"
        '前者适合检查父组作用是否合理，后者适合优先核对未知功能件的形态预算。', '',
        '不能仅凭设备名称断言每个子零件应处于哪一档；“焊接机”“检测线”等标题只说明场景用途。'
        '精密/作业保护比例的差异，应继续追溯到局部部件与装配角色。', '',
        '## 未知功能、分支不确定性与V4原P6', '',
        '![未决证据](05_unresolved_evidence.png)', '',
        '![未知功能几何](07_unknown_geometry_families.png)', '',
        '![分支证据诊断](08_branch_uncertainty.png)', '',
        '![原P6去向](09_previous_p6_cohorts.png)', '',
        '未知功能件的几何分型与P档位分开呈现，避免“不同形状全部一个兜底档位”。'
        'N未达到可靠性阈值与C没有对象级输出采用不同定义，不能据此给两模型的准确率排名。'
        'V4原P6去向是同一批节点的策略变更诊断，属于补充图，不是不同融合方式的识别结果。', '',
        '## 结论与边界', '',
        '现有数据可以支持：比较不同分支的策略分配、结构树/名称的实际介入范围、未决与冲突证据，'
        '以及节点统计和源几何工作量的差异。尚无独立人工真值、V5多视角画质或GPU测量，'
        '因此不报告OA/Macro-F1、显著性、显存降低率或帧率提升。', '',
        '交互入口：[六模型融合交互图表.html](六模型融合交互图表.html)。'
        '图像同时提供PNG与可编辑文字的SVG版本，单模型图以`scene_a_fusion`等命名。', '']
    (OUT / '六模型不同融合图表分析.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    proof = json.loads((SOURCE / 'provenance.json').read_text(encoding='utf-8'))
    inventory_path = ROOT / 'analysis_reports/assembly_context_v3_20261020/scene_inventory.csv'
    inventory = read(inventory_path)
    titles = {r['scene']: r['title'] for r in inventory}
    manifest_path = ROOT / 'lod_analysis_outputs/assembly_context_v3_20261020/scene_sources.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    stats, pairs, family_rows, uncertainties, selected_roles = [], [], [], [], []
    inputs = [SOURCE / 'provenance.json', inventory_path, manifest_path, Path(__file__)]
    keep = ['node_index', 'mesh_index', 'face_count', 'lod_priority', 'function_role',
        'resolved_object_class', 'resolved_status', 'class_conflict', 'type_shape_conflict',
        'needs_review', 'geometry_budget_family', 'feature_lock_budget', 'simplify_error_relative',
        'screen_error_weight', 'max_axis_shrink', 'target_ratio_far']
    for s in SCENES:
        policies = {}
        for v in VARIANTS:
            path = POLICIES / s / (v + '.csv')
            if sha(path) != proof['outputs'].get(str(path.resolve())):
                raise ValueError(f'Unverified policy: {path}')
            with path.open(encoding='utf-8-sig', newline='') as stream:
                rows = [{k: r[k] for k in keep} for r in csv.DictReader(stream)]
            if len(rows) != int(next(r for r in inventory if r['scene'] == s)['nodes']):
                raise ValueError('Incomplete scene')
            if len({ident(r) for r in rows}) != len(rows):
                raise ValueError('Duplicate instance')
            policies[v] = rows
            stats.append(summary(s, v, rows))
            inputs.append(path)
        for mode in (*MODES, 'G'):
            old, new = mode+'_H0', mode+'_H1'
            pairs.append(paired(s, 'tree', old, new, policies[old], policies[new]))
        for h in (0, 1):
            for old_mode, new_mode in [('N', 'NG'), ('C', 'CG'), ('NC', 'NCG')]:
                old, new = f'{old_mode}_H{h}', f'{new_mode}_H{h}'
                pairs.append(paired(s, 'geometry', old, new, policies[old], policies[new]))
        pairs.append(paired(s, 'metadata', 'NCG_H1', 'NCG_H1_M1', policies['NCG_H1'], policies['NCG_H1_M1']))
        selected = policies['NCG_H1_M1']
        unknown = [r for r in selected if r['function_role'] == 'unknown']
        for f in FAMILIES:
            members = [r for r in unknown if r['geometry_budget_family'] == f]
            if members:
                r = summary(s, 'NCG_H1_M1', members)
                r['family'] = f
                family_rows.append(r)
        roles = Counter(r['function_role'] for r in selected)
        selected_roles.extend(dict(scene=s, role=k, nodes=n) for k, n in roles.items())
        pnpath, cpath = Path(manifest[s]['pointnext_csv']), Path(manifest[s]['open_csv'])
        for path, expected in [(pnpath, manifest[s]['pointnext_sha256']), (cpath, manifest[s]['open_sha256'])]:
            if sha(path) != expected:
                raise ValueError('Changed raw inference')
        pn = read(pnpath)
        opened = {ident(r): r for r in read(cpath)}
        if len(opened) != len(pn) or set(opened) != {ident(r) for r in pn}:
            raise ValueError('Unpaired source evidence')
        count_n = count_c = faces_n = faces_c = faces_all = 0
        for r in pn:
            low_n = float(r['confidence']) < .7 or float(r['confidence'])-float(r['second_confidence']) < .08
            low_c = opened[ident(r)]['open_status'] not in ('accepted', 'class_only')
            faces = int(r['face_count']); faces_all += faces
            count_n += low_n; count_c += low_c
            faces_n += faces * low_n; faces_c += faces * low_c
        uncertainties.append(dict(scene=s, nodes=len(pn), source_faces=faces_all,
            N_not_reliable_count=count_n, C_not_object_count=count_c,
            N_not_reliable_faces=faces_n, C_not_object_faces=faces_c))
        inputs.extend([pnpath, cpath])
        print(f'{s}: 15 verified fusion conditions; paired changes and unknown geometry extracted', flush=True)
    count_fields = [k for k in stats[0] if k not in ('scene', 'variant')]
    stats += aggregate(stats, ['variant'], count_fields)
    stats = [fractions(r) for r in stats]
    paired_fields = [k for k in pairs[0] if k not in ('scene', 'kind', 'before', 'after')]
    pairs += aggregate(pairs, ['kind', 'before', 'after'], paired_fields)
    family_fields = [k for k in family_rows[0] if k not in ('scene', 'variant', 'family')]
    family_rows += aggregate(family_rows, ['variant', 'family'], family_fields)
    ufields = [k for k in uncertainties[0] if k != 'scene']
    uncertainties += aggregate(uncertainties, [], ufields)
    for r in uncertainties:
        for prefix in ('N_not_reliable', 'C_not_object'):
            r[prefix+'_node_fraction'] = r[prefix+'_count']/r['nodes']
            r[prefix+'_face_fraction'] = r[prefix+'_faces']/r['source_faces']
    cohorts_path = SOURCE / 'previous_p6_cohorts.csv'
    cohorts = read(cohorts_path); inputs.append(cohorts_path)
    # Reconcile to the independent, previously generated versioned summary.
    expected = {(r['scene'], r['variant']): r for r in read(SOURCE / 'all_profile_comparison.csv')
                if r['version'] == 'adaptive_p10_v5.0'}
    for r in stats:
        if r['scene'] == 'ALL':
            continue
        reference = expected[r['scene'], r['variant']]
        for p in range(1, 11):
            if r[f'P{p}_count'] != int(reference[f'P{p}_count']):
                raise ValueError('Independent grade-count reconciliation failed')
            if abs(r[f'P{p}_face_fraction'] - float(reference[f'P{p}_face_fraction'])) > 1e-12:
                raise ValueError('Independent source-triangle reconciliation failed')
    inputs.append(SOURCE / 'all_profile_comparison.csv')
    write(OUT / 'configuration_summary.csv', stats)
    write(OUT / 'paired_changes.csv', pairs)
    write(OUT / 'unknown_geometry_families.csv', family_rows)
    write(OUT / 'source_uncertainty.csv', uncertainties)
    write(OUT / 'full_fusion_function_roles.csv', selected_roles)
    make_figures(stats, pairs, family_rows, uncertainties, cohorts, titles)
    dashboard(stats, pairs, titles)
    report(stats, pairs, inventory)
    save(OUT / 'workbook_data.json', dict(stats=stats, pairs=pairs, families=family_rows,
        uncertainties=uncertainties, inventory=inventory, colors=COLORS, modes=MODES, variants=VARIANTS))
    save(OUT / 'chart_validation.json', dict(status='passed', verified_policy_files=90,
        source_policy_rows=sum(r['nodes'] for r in stats if r['scene'] != 'ALL'),
        main_conditions=72, supplementary_conditions=18, figure_count=len(FIGURES),
        independent_count_and_triangle_reconciliation=True, manual_accuracy_labels=False,
        runtime_memory_or_fps_measured=False))
    save(OUT / 'chart_manifest.json', dict(figures=FIGURES,
        inputs={str(p.resolve()): sha(p) for p in inputs},
        outputs={p.name: sha(p) for p in OUT.iterdir() if p.suffix in ('.csv', '.png', '.svg', '.html', '.md')},
        policy_version='adaptive_p10_v5.0', aggregation='sum numerator and denominator before division'))
    print(f'Charts and analysis complete: {OUT}', flush=True)


if __name__ == '__main__':
    main()
