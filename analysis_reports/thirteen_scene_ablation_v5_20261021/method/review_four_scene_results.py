"""Evidence-based overall review of four completed industrial semantic tests."""
import argparse
import json
import statistics
from pathlib import Path

from audit_named_scene import ROOT, read_csv, write_csv
from artifact_cache import file_hash
from summarize_named_scenes import expected_known_classes

MODES = [('pointnext', 'N'), ('pointclip', 'C'), ('pointnext_structure', 'N+G'),
         ('pointclip_structure', 'C+G'), ('pointnext_pointclip', 'N+C'),
         ('pointnext_pointclip_structure', 'N+C+G')]
PROFILES = {
    '透气阀产线': ('透气阀生产及相关搬运、检测、排气流程', r'CCD|外观检测|排气|冲裁',
                  '优先核查检测/排气/冲裁工位的局部尺度和功能接口；名称只能说明工位上下文。'),
    '汽车链接器植入-折弯设备': ('连接器植入、冲切与折弯', r'冲切|折弯|摸具|模具',
                  '优先区分端子/产品、工具、夹具和普通装配件，不能把待加工产品按通用紧固件处理。'),
    '手机喇叭检测线': ('喇叭检测及输送、定位、视觉/真空相关作业', r'CCD|真空盒',
                  '优先核查视觉、气动与真空零件，当前14类没有完整覆盖这些功能对象。'),
    'K31自动铆压机': ('铆压及相关送料、锁螺丝、装胶帽作业', r'铆压组|锁螺丝|自动装胶帽',
                  '重点保护经核验的铆压接触部位、定位导向和工件特征；不同送料螺母的功能角色不能只由形状类别决定。'),
}
ALIASES = {'透气阀产线': 'Vent line', '汽车链接器植入-折弯设备': 'Connector machine',
           '手机喇叭检测线': 'Speaker line', 'K31自动铆压机': 'K31 riveter'}


def figures(root, results, table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'pdf.fonttype': 42, 'font.size': 10})
    names = [r['scene'] for r in results]
    by_name = {r['scene']: r for r in table}
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(4)
    ax.bar(x - .18, [by_name[n]['supervised_accepted_percent'] for n in names], .36,
           label='Supervised acceptance coverage', color='#276a9b')
    ax.bar(x + .18, [by_name[n]['open_usable_excluding_micro_percent'] for n in names], .36,
           label='Usable open evidence (current rules)', color='#d7872a')
    ax.set_xticks(x, [ALIASES[n] for n in names])
    ax.set(ylim=(0, 100), ylabel='Mesh nodes (%)', title='Evidence coverage, not classification accuracy')
    ax.legend(frameon=False, loc='upper right')
    fig.tight_layout()
    for suffix in ['png', 'pdf']:
        fig.savefig(root / f'four_scene_evidence_coverage.{suffix}', dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, result in zip(axes.flat, results):
        bottom = np.zeros(6)
        for priority in range(1, 11):
            values = np.array([100 * result['semantics']['six_full'][mode]['priority_counts'][f'P{priority}'] /
                               result['hierarchy']['mesh_nodes'] for mode, label in MODES])
            ax.bar(range(6), values, bottom=bottom, color=plt.cm.viridis((priority - 1) / 9), label=f'P{priority}')
            bottom += values
        ax.set_xticks(range(6), [label for mode, label in MODES])
        ax.set(ylim=(0, 100), title=ALIASES[result['scene']], ylabel='Mesh nodes (%)')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=10, loc='lower center', frameon=False)
    fig.suptitle('Six policy configurations per scene (not strict independent ablations)')
    fig.tight_layout(rect=(0, .05, 1, .96))
    for suffix in ['png', 'pdf']:
        fig.savefig(root / f'four_scene_policy_distribution.{suffix}', dpi=220)
    plt.close(fig)


def main():
    import re
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-root', type=Path, default=ROOT / 'analysis_reports/four_scenes_20261019')
    parser.add_argument('--run-root', type=Path, default=ROOT / 'lod_analysis_outputs/four_scenes_seed2027_20261019')
    args = parser.parse_args()
    root = args.report_root
    source = root / 'four_scene_summary.json'
    results = json.loads(source.read_text(encoding='utf-8'))
    tables = read_csv(root / 'four_scene_summary.csv')
    numeric = ['supervised_accepted_percent', 'open_usable_excluding_micro_percent']
    for row in tables:
        for field in numeric:
            row[field] = float(row[field])
    assert len(results) == 4 and {r['scene'] for r in results} == set(PROFILES)
    total = sum(r['hierarchy']['mesh_nodes'] for r in results)
    aggregate, pairs, function_groups, function_parts, cases, provenance_inputs, input_versions = [], [], [], [], [], [], []
    accepted = sum(r['semantics']['pointnext']['accepted_count'] for r in results)
    open_usable = sum(r['semantics']['open_quality']['usable_excluding_micro'] for r in results)
    for mode, label in MODES:
        low = sum(r['semantics']['six_full'][mode]['low_protection_count'] for r in results)
        high = sum(r['semantics']['six_full'][mode]['high_protection_count'] for r in results)
        aggregate.append({'configuration': label, 'mesh_nodes': total,
            'low_count': low, 'low_percent_pooled': 100 * low / total,
            'high_count': high, 'high_percent_pooled': 100 * high / total,
            'low_percent_scene_macro': statistics.mean(r['semantics']['six_full'][mode]['low_protection_percent'] for r in results),
            'high_percent_scene_macro': statistics.mean(r['semantics']['six_full'][mode]['high_protection_percent'] for r in results)})
    for before, after in [('N', 'N+G'), ('N', 'N+C'), ('N', 'N+C+G'), ('N+C', 'N+C+G')]:
        scene_pairs = [next(p for p in r['semantics']['pairwise_full'] if p['from'] == before and p['to'] == after) for r in results]
        changed = sum(p['lod_changed'] for p in scene_pairs)
        pairs.append({'from': before, 'to': after, 'changed': changed, 'pooled_percent': 100 * changed / total,
                      'scene_macro_percent': statistics.mean(p['lod_changed_share'] * 100 for p in scene_pairs)})
    for result in results:
        name = result['scene']
        folder, run = root / name, args.run_root / name
        state = json.loads((run / 'inference_status.json').read_text(encoding='utf-8'))
        scene_path = Path(result['semantics']['glb'])
        assert result['semantics']['glb_sha256'] == state['glb_sha256'] == file_hash(scene_path)
        old_summary = ROOT / 'analysis_reports/named_scenes_20261019' / name / 'semantic_comparison/analysis_summary.json'
        old = json.loads(old_summary.read_text(encoding='utf-8')) if old_summary.exists() else None
        input_versions.append({'scene': name, 'current_glb_sha256': state['glb_sha256'], 'current_bytes': scene_path.stat().st_size,
            'previous_glb_sha256': old['glb_sha256'] if old else None,
            'same_as_previous_input': old['glb_sha256'] == state['glb_sha256'] if old else None,
            'inference_reused_snapshot': bool(state.get('snapshot_reuse')), 'raw_directory': str(run.resolve()),
            'timing_scope': 'original inference timings' if state.get('snapshot_reuse') else 'fresh inference stage timings'})
        nodes = read_csv(folder / 'mesh_name_context_audit.csv')
        groups = read_csv(folder / 'assembly_groups.csv')
        policy = {r['node_index']: r for r in read_csv(folder / 'semantic_comparison/per_mesh_policy_audit.csv')}
        matched = [g for g in groups if int(g['depth']) > 0 and re.search(PROFILES[name][1], g['name'])]
        for group in matched:
            members = [r for r in nodes if group['node_index'] in r['path_ids'].split('/')[:-1]]
            lows = sum(int(policy[r['node_index']]['N+C+G']) <= 3 for r in members)
            highs = sum(int(policy[r['node_index']]['N+C+G']) >= 8 for r in members)
            function_groups.append({'scene': name, 'assembly_index': group['node_index'], 'assembly_name': group['name'],
                'member_mesh_nodes': len(members), 'full_low_count': lows, 'full_high_count': highs,
                'scope': 'Named process context only. Not every member is a critical functional part; groups may overlap.'})
            for row in members:
                function_parts.append({'scene': name, 'assembly_index': group['node_index'], 'assembly_name': group['name'],
                    'node_index': row['node_index'], 'node_name': row['node_name'], 'name_evidence': row['name_evidence'],
                    'supervised_class': row['supervised_class'], 'confidence': row['supervised_confidence'],
                    'open_label': row['open_label'], 'N_priority': policy[row['node_index']]['N'],
                    'full_priority': policy[row['node_index']]['N+C+G'], 'manual_function_role': '', 'manual_critical_features': ''})
        if name == 'K31自动铆压机':
            selectors = [('明确螺母名称', lambda r: expected_known_classes(r) == {'nuts'}),
                         ('气缸名称', lambda r: r['name_candidate_role'] == 'pneumatic_or_linear_actuator'),
                         ('伺服电机名称', lambda r: '伺服电机' in r['leaf_name'])]
        elif name == '手机喇叭检测线':
            selectors = [('旋转气缸名称', lambda r: '旋转气缸' in r['leaf_name'] and r['name_candidate_role'] != 'mount_or_support'),
                         ('轴承名称', lambda r: r['leaf_name'].startswith('轴承-')),
                         ('光源名称', lambda r: r['leaf_name'].startswith('光源') and r['name_candidate_role'] != 'mount_or_support')]
        elif name == '汽车链接器植入-折弯设备':
            selectors = [('产品/工件名称', lambda r: '产品01' in r['leaf_name']),
                         ('光电感应器名称', lambda r: '光电感应器' in r['leaf_name'])]
        else:
            selectors = [('弹簧名称', lambda r: r['name_candidate_role'] == 'spring'),
                         ('光源名称', lambda r: r['leaf_name'].startswith('光源')),
                         ('温控表名称', lambda r: '温控表' in r['leaf_name'])]
        for description, select in selectors:
            candidates = sorted([r for r in nodes if select(r)], key=lambda r: -float(r['supervised_confidence']))
            if candidates:
                row = candidates[0]
                cases.append({'scene': name, 'evidence_type': description, 'node_index': row['node_index'], 'node_name': row['node_name'],
                    'predicted_class': row['supervised_class'], 'confidence': float(row['supervised_confidence']),
                    'open_label': row['open_label'], 'full_priority': policy[row['node_index']]['N+C+G'],
                    'scope': 'Highest-confidence matching name case selected for review; not an unbiased accuracy sample.'})
        provenance_inputs.extend([folder / 'hierarchy_summary.json', folder / 'mesh_name_context_audit.csv',
                                  folder / 'semantic_comparison/analysis_summary.json', run / 'inference_status.json'])
    for filename, data in [('aggregate_six_policy.csv', aggregate), ('aggregate_policy_changes.csv', pairs),
                           ('device_function_groups.csv', function_groups), ('functional_context_review.csv', function_parts),
                           ('device_case_evidence.csv', cases)]:
        write_csv(root / filename, data)
    (root / 'input_versions.json').write_text(json.dumps(input_versions, ensure_ascii=False, indent=2), encoding='utf-8')
    overall = {'mesh_nodes': total, 'supervised_accepted': accepted, 'supervised_acceptance_pooled_percent': 100 * accepted / total,
        'supervised_acceptance_scene_macro_percent': statistics.mean(t['supervised_accepted_percent'] for t in tables),
        'usable_open_excluding_micro': open_usable, 'open_usable_pooled_percent': 100 * open_usable / total,
        'open_usable_scene_macro_percent': statistics.mean(t['open_usable_excluding_micro_percent'] for t in tables),
        'six_configurations': aggregate, 'policy_changes': pairs, 'input_versions': input_versions,
        'ground_truth_accuracy_available': False, 'actual_lod_render_benchmark_executed': False,
        'device_names_used_for_neural_inference': False}
    nut_rows = [r for r in read_csv(root / 'K31自动铆压机/mesh_name_context_audit.csv') if expected_known_classes(r) == {'nuts'}]
    overall['k31_named_nut_family_diagnostic'] = {
        'named_instances': len(nut_rows),
        'supervised_bearing_family': sum(r['supervised_class'] == 'bearings_bushings_guides' for r in nut_rows),
        'open_nut_family': sum('nut' in r['open_label'].lower() for r in nut_rows),
        'scope': 'Name-based family consistency diagnostic, repeated instances; not verified accuracy.'}
    (root / 'overall_assessment.json').write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding='utf-8')
    figures(root, results, tables)
    tindex = {r['scene']: r for r in tables}
    scene_table = '\n'.join(f"| {r['scene']} | {r['hierarchy']['mesh_nodes']:,} | {r['hierarchy']['assembly_nodes']} | {tindex[r['scene']]['supervised_accepted_percent']:.2f}% | {tindex[r['scene']]['open_usable_excluding_micro_percent']:.2f}% |" for r in results)
    six_table = '\n'.join(f"| {r['configuration']} | {r['low_percent_pooled']:.2f}% | {r['high_percent_pooled']:.2f}% | {r['low_percent_scene_macro']:.2f}% | {r['high_percent_scene_macro']:.2f}% |" for r in aggregate)
    pair_table = '\n'.join(f"| {r['from']}→{r['to']} | {r['changed']:,} | {r['pooled_percent']:.2f}% | {r['scene_macro_percent']:.2f}% |" for r in pairs)
    case_table = '\n'.join(f"| {r['scene']} | {r['node_index']} | {r['node_name']} | {r['predicted_class']} | {r['confidence']*100:.2f}% | {r['open_label']} | P{r['full_priority']} |" for r in cases)
    group_table = '\n'.join(f"| {r['scene']} | {r['assembly_index']} | {r['assembly_name']} | {r['member_mesh_nodes']} | {r['full_low_count']} | {r['full_high_count']} |" for r in function_groups)
    version_table = '\n'.join(f"| {r['scene']} | {r['current_bytes']:,} | {'经核验复用' if r['inference_reused_snapshot'] else '本轮新推断'} | `{r['current_glb_sha256']}` |" for r in input_versions)
    profiles = '\n'.join(f"- **{name}**：设备名提示{task}。{recommendation}" for name, (task, pattern, recommendation) in PROFILES.items())
    links = '\n'.join(f"- [{r['scene']}：完整六配置报告]({r['scene']}/semantic_comparison/analysis_report.md)；[装配树]({r['scene']}/assembly_tree.txt)。" for r in results)
    text = f'''# 四模型全量语义测试与整体评价

日期：2026-10-19。**当前结果支持完整流程覆盖与策略行为分析，但还不足以证明融合识别精度或最终 LOD 质量优于各单分支。** 最需要改进的是工业类别覆盖、未知拒识、开放证据质量，以及物体类别到工艺角色的映射。

## 1. 本轮实际完成什么

四模型共 **{total:,} 个网格节点**，全部覆盖监督 N、全量开放 C、门控开放、结构 G、六配置及完整/门控融合；另提供移除监督不确定性的独立 G、独立 C+G 诊断基线。所有 CSV 逐节点联合键与源网格一一对应。监督模型固定为 PointNeXt-S seed2027 epoch98，验证集 Macro-F1 选择的部署权重，8,192 点；开放固定 ViT-B/16、10视图、8,192点，全量110标签，门控106标签。未改变生产融合公式或训练模型。

| 场景 | 网格节点 | 装配组 | 监督直接接受覆盖率 | 可用于完整融合的开放证据比例 |
|---|---:|---:|---:|---:|
{scene_table}

接受为置信度≥0.70且间隔≥0.08；可用开放证据按现有质量规则且排除micro_uncertain计算。**二者都不是准确率。** 按节点合并，监督接受 {accepted:,}/{total:,}＝{100*accepted/total:.2f}%，开放可用 {open_usable:,}/{total:,}＝{100*open_usable/total:.2f}%；按四场景等权平均，分别为 {overall['supervised_acceptance_scene_macro_percent']:.2f}% 和 {overall['open_usable_scene_macro_percent']:.2f}%。二者并列报告，避免较大产线支配总体比例。

![四场景证据覆盖](four_scene_evidence_coverage.png)

## 2. 输入更新与复用

| 场景 | 当前字节数 | 本轮处理 | 当前SHA256 |
|---|---:|---|---|
{version_table}

手机喇叭检测线、透气阀产线经输入、权重、代码和输出哈希核验后复制独立推断快照，策略与报告重新生成。汽车设备已重新导出，大小从481,674,560变为385,601,804字节，节点顺序/名称实例后缀也变化，必须重跑；不能拿上一轮node_index直接对应本轮。K31为首次全量推断。原三模型报告保留为历史记录。复用快照的耗时属于原始运行，不作为本轮GPU耗时或加速结果。

## 3. 结合模型名和装配名称判断功能是否合理

设备名称只提供设备级任务先验，不能为所有零件提供类别真值。这里再用文件中真实的装配名称确认功能上下文，避免仅凭文件名猜测。

{profiles}

与工艺名称匹配的装配组如下。组内低保护不等于错误：紧固件等可能合理降级，关键接触件则可能需要保护；必须进一步区分成员角色，不能整组统一提到最高等级。

| 设备 | 装配node_index | 装配名称 | 后代网格数 | 完整融合P1–P3 | 完整融合P8–P10 |
|---|---:|---|---:|---:|---:|
{group_table}

这些装配组可能嵌套，计数不能跨组相加。逐成员证据与人工角色填写列见 [functional_context_review.csv](functional_context_review.csv)。

## 4. 名称与预测给出的具体线索

| 场景 | node_index | 零件名称 | 监督预测 | 监督置信度 | 开放标签 | 完整融合 |
|---|---:|---|---|---:|---|---|
{case_table}

案例按每种名称线索中最高监督置信度选取，既包含一致也包含分歧，不是随机误差样本。螺母、气缸、产品、感知器件等名称与预测分歧需要人工核验。伺服电机等一致案例说明监督分支有可用能力；它不支持把全部零件都判为正确。气缸和视觉/温控类别在当前14类中缺乏直接对应，强制闭集预测仍可能很高分。

K31中，保守名称规则识别出{len(nut_rows)}个明确“螺母”名称实例，监督分支有{overall['k31_named_nut_family_diagnostic']['supervised_bearing_family']}个预测为轴承/导向类，开放分支有{overall['k31_named_nut_family_diagnostic']['open_nut_family']}个给出螺母家族标签。这表明开放分支存在有价值的互补线索，不能笼统判断为无效；这些重复实例也不能视作同样数量的独立准确率样本。当前融合主要输出功能角色与LOD策略，不会把监督predicted_class重写成开放类别。因此“开放识别提供线索”和“最终完成类别纠错”必须分开评价。

尤其要区分“是什么”与“在这里做什么”：送料轨道中的螺母可能是工件，铆压接触件可能是定制工具；外观相似不能证明功能相同。只用置信度阈值触发开放分支，会遗漏这类高分的类别覆盖缺口。名称也不能无条件覆盖模型，例如螺母送料盘不是一颗螺母，CCD安装板不是相机。

风险是双向的：名称为光源或轴承却进入P2，存在欠保护的复核需求；名称为螺母、因监督轴承类而进入P10，也存在过度保留的复核需求。真正是否错误取决于该设备中的功能和使用任务，不能把高保护比例当作正确率。两类风险最终都需要在同一几何预算下验证。

## 5. 六种配置的总体行为

N=监督，C=开放，G=结构。下面全部使用相同全量开放输出。P1–P3表示低保护、P8–P10表示高保护；P编号不是几何LOD层级或质量得分。

| 配置 | 合并节点：P1–P3 | 合并节点：P8–P10 | 四场景等权：P1–P3 | 四场景等权：P8–P10 |
|---|---:|---:|---:|---:|
{six_table}

![四场景六配置](four_scene_policy_distribution.png)

| 配置变化 | P等级变化节点 | 合并节点占比 | 四场景等权变化率 |
|---|---:|---:|---:|
{pair_table}

这些结果量化了规则输出的变化，不能当作精度提升。当前完整模式与其他组合的融合权重/后处理不完全一致，原G也读取监督不确定性，原C输出还携带监督置信度。六配置不是严格独立分支消融；独立G/CG补充基线仅用于明确诊断这一差异。

整体判断：监督在部分已知类上出现与名称一致的预测，但真实设备角色超出14类；开放证据只有部分通过质量规则，现有弱开放输出经常停留在通用形状描述，同时也存在螺母家族等互补线索；结构规则对分配有较大影响，但它不是已经学会功能角色的结构网络。完整融合改变分配并不意味着已经纠正类别或保留了真正关键细节。

## 6. 目前不能据此宣称的结果

没有人工类别/功能角色真值，所以没有场景OA、Macro-F1、融合准确率或关键件漏检率；没有实际执行本轮几何简化、重建LOD、相机轨迹和GPU渲染基准，所以没有几何误差、孔洞保持率、实际剔除率或FPS提升。allow_cull是策略元数据，不能当已执行剔除。

全局小节点也不能直接删掉：长产线会把大量正常零件划成相对微小，局部工位尺度、可见性和功能需要分别测量。当前所有节点保留，未把识别未知当成无用，也未证明任意节点永久不可见。

## 7. 对论文和下一步优化的判断

当前适合写作“不同真实工业设备的全量语义推断与策略分配比较”，可以论证流程覆盖、跨设备输入差异和现有瓶颈。要证明多源融合有效，应先修正未知→低重要性的映射，并补齐气动/感知/工艺工具的粗类别；再加入局部装配上下文、全局+局部输入和开放质量校准。不要单纯调大开放权重使策略看起来变化更明显。

名称和层级可以作为明确的元数据M辅助拒识、候选和功能判断，但需另做 N+C+G+M、去名称、去层级等对照。不能同时用名称辅助推断、再把同一名称自动生成的标签当独立真值。对关键工艺组人工核验后，再在相同实际几何预算与轨迹下比较关键细节误差和帧时间，才能支持整体方法优越性的结论。

## 8. 明细、来源与复现

{links}

[名称层级与逐场景配置明细](四模型名称层级与语义明细.md)、[四场景总表](four_scene_summary.csv)、[名称分歧复核表](name_semantic_review.csv)、[总体计算结果](overall_assessment.json)、[输入版本](input_versions.json)。

所有原始结果保存在 `{args.run_root.resolve()}`，报告保存在 `{root.resolve()}`。本报告及表图的输入/输出哈希见 overall_provenance.json；各场景神经阶段来源与状态另见原始目录的provenance和inference_status。完整测试覆盖语义推断与策略配置，未执行训练或实际LOD渲染测评。
'''
    report = root / '四模型全量测试与整体评价.md'
    report.write_text(text, encoding='utf-8')
    outputs = [report, root / 'overall_assessment.json', root / 'input_versions.json', *[root / filename for filename in
        ['aggregate_six_policy.csv', 'aggregate_policy_changes.csv', 'device_function_groups.csv', 'functional_context_review.csv',
         'device_case_evidence.csv', 'four_scene_evidence_coverage.png', 'four_scene_evidence_coverage.pdf',
         'four_scene_policy_distribution.png', 'four_scene_policy_distribution.pdf']]]
    provenance = {'inputs': {str(p.resolve()): file_hash(p) for p in [source, root / 'four_scene_summary.csv', *provenance_inputs]},
                  'source': {str(Path(__file__).resolve()): file_hash(Path(__file__))},
                  'outputs': {str(p.resolve()): file_hash(p) for p in outputs}, 'accuracy_measured': False, 'render_benchmark_executed': False}
    (root / 'overall_provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    latest = {'status': 'four_scenes_full_semantic_test_complete', 'report': str(report.resolve()), 'mesh_nodes': total,
              'scenes': list(PROFILES), 'raw_directory': str(args.run_root.resolve()), 'ground_truth_metrics_available': False,
              'renderer_lod_cache_status': 'not_rebuilt'}
    for p in [ROOT / 'analysis_reports/latest_scene_analysis.json', ROOT / 'lod_analysis_outputs/deployment_status.json']:
        p.write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'overall': overall, 'report': str(report.resolve())}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
