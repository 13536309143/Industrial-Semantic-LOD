"""Summarize independent inference and weak name/tree evidence for named CAD scenes."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from audit_named_scene import ROOT, audit, read_csv, write_csv
from artifact_cache import file_hash

NAMES = ['手机喇叭检测线', '透气阀产线', '汽车链接器植入-折弯设备']
KNOWN = {'motor_or_reducer': {'motors_gearmotors'},
         'bearing_or_linear_guide': {'bearings_bushings_guides'},
         'transmission': {'gears_pulleys_chains'}, 'spring': {'springs'}}


def pct(n, total):
    return 100 * n / total


def expected_known_classes(row):
    """Conservative lexical candidates, not automatically verified labels."""
    role = row['name_candidate_role']
    leaf = row['leaf_name']
    if role == 'mount_or_support':
        return set()
    if role == 'fastener':
        if re.search(r'送料|上料|轨道|支架|固定|安装|座|盘|机构|锁螺丝', leaf):
            return set()
        if '螺母' in leaf:
            return {'nuts'}
        if re.search(r'螺钉|螺栓|螺丝', leaf):
            return {'screws_bolts_studs'}
        if re.search(r'垫圈|挡圈|卡簧', leaf):
            return {'washers_rings_spacers'}
    return KNOWN.get(role, set())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-root', type=Path, default=ROOT / 'analysis_reports/named_scenes_20261019')
    parser.add_argument('--run-root', type=Path, default=ROOT / 'lod_analysis_outputs/named_scenes_seed2027_20261019')
    parser.add_argument('--models', nargs='+', default=NAMES)
    parser.add_argument('--summary-prefix', default='three_scene')
    parser.add_argument('--report-filename', default='三场景名称层级与语义分析.md')
    args = parser.parse_args()
    names = args.models
    prefix = args.summary_prefix
    results, tables, review, evidence_inputs = [], [], [], []
    for name in names:
        folder, run = args.report_root / name, args.run_root / name
        glb = ROOT / '_downloaded_resources' / (name + '.glb')
        hierarchy = audit(glb, folder, run)
        semantic = json.loads((folder / 'semantic_comparison/analysis_summary.json').read_text(encoding='utf-8'))
        state = json.loads((run / 'experiment_status.json').read_text(encoding='utf-8'))
        if state['status'] != 'complete_scene_inference_and_policy_comparison':
            raise ValueError('Incomplete scene: ' + name)
        assert hierarchy['glb_sha256'] == semantic['glb_sha256'] == state['glb_sha256']
        assert hierarchy['mesh_nodes'] == semantic['coverage']['supervised'] == semantic['coverage']['full_open']
        rows = read_csv(folder / 'mesh_name_context_audit.csv')
        policies = {r['node_index']: r for r in read_csv(folder / 'semantic_comparison/per_mesh_policy_audit.csv')}
        direct_conflicts = []
        for row in rows:
            hits = json.loads(row['name_evidence'])
            roles = {role for role, term in hits}
            expected = expected_known_classes(row)
            reason = []
            if expected and row['supervised_class'] not in expected:
                reason.append('名称提示与监督类别不一致，待人工核验')
                direct_conflicts.append(row)
            if row['name_candidate_role'] in {'pneumatic_or_linear_actuator', 'sensing_or_inspection'}:
                reason.append('名称提示气动/感知类别，现有14类缺少直接对应')
            policy = policies[str(row['node_index'])]
            if 'product_or_workpiece' in roles and ('fastener' in roles or int(policy['N+C+G']) <= 3):
                reason.append('产品/工件名称与低保护或紧固件角色并存，需核对工艺角色')
            if int(row['source_triangles']) >= 10000 and row['open_label'] == 'small visual fragment':
                reason.append('高面数对象被开放分支匹配为通用碎片')
            if reason:
                review.append({'scene': name, 'node_index': row['node_index'], 'mesh_index': row['mesh_index'],
                    'node_name': row['node_name'], 'parent_name': row['parent_name'], 'path_names': row['path_names'],
                    'name_evidence': row['name_evidence'], 'predicted_class': row['supervised_class'],
                    'confidence': row['supervised_confidence'], 'open_label': row['open_label'],
                    'open_score': row['open_score'], 'N_priority': policy['N'], 'full_priority': policy['N+C+G'],
                    'source_triangles': row['source_triangles'], 'review_reason': ';'.join(reason),
                    'manual_class': '', 'manual_function_role': '', 'manual_critical_features': '',
                    'scope': 'Targeted review candidates, not confirmed errors or an unbiased accuracy sample.'})
        n = hierarchy['mesh_nodes']
        small = [b for b in hierarchy['size_bins'] if b['max_inclusive'] != 'inf' and b['max_inclusive'] <= .01]
        pn = semantic['pointnext']
        table = {'scene': name, 'mesh_nodes': n, 'unique_meshes': hierarchy['unique_meshes'],
            'assembly_nodes': hierarchy['assembly_nodes'], 'max_depth_edges': hierarchy['max_depth'],
            'unique_mesh_triangles': hierarchy['unique_mesh_triangles'], 'instance_triangles': hierarchy['instance_triangles'],
            'chinese_leaf_names': hierarchy['chinese_leaf_names'],
            'basic_dictionary_hint_rows': hierarchy['name_evidence_status_counts'].get('weak_lexical_hint', 0),
            'nonroot_process_context_rows': hierarchy['with_nonroot_process_context'],
            'supervised_mean_confidence_percent': pn['mean_confidence'] * 100,
            'supervised_accepted_percent': pct(pn['accepted_count'], n),
            'open_usable_excluding_micro_percent': pct(semantic['open_quality']['usable_excluding_micro'], n),
            'size_le_1pct_count': sum(b['count'] for b in small),
            'size_le_1pct_record_percent': sum(b['record_percent'] for b in small),
            'size_le_1pct_triangle_percent': sum(b['triangle_percent'] for b in small),
            'global_small_but_local_ge10pct': hierarchy['global_le_1pct_but_local_ge_10pct'],
            'name_known_class_conflict_candidates': len(direct_conflicts),
            'name_known_class_conflict_high_confidence': sum(float(r['supervised_confidence']) >= .7 for r in direct_conflicts),
            'full_open_mean_basic_weight_percent': semantic['full_fusion_mean_normalized_evidence_weights']['C'] * 100,
            'same_mesh_prediction_disagreement_groups': hierarchy['repeated_mesh_groups_with_class_disagreement'],
            'same_mesh_repeated_groups': hierarchy['repeated_mesh_groups'],
            'first_primitive_alpha_blend_nodes': hierarchy['first_primitive_alpha_blend_nodes'],
            'glb_sha256': hierarchy['glb_sha256'], 'accuracy_available': False}
        tables.append(table)
        results.append({'scene': name, 'hierarchy': hierarchy, 'semantics': semantic})
        evidence_inputs.extend([folder / 'hierarchy_summary.json', folder / 'mesh_name_context_audit.csv',
                                folder / 'semantic_comparison/analysis_summary.json', run / 'experiment_status.json'])
    write_csv(args.report_root / f'{prefix}_summary.csv', tables)
    review.sort(key=lambda r: (r['scene'], -float(r['confidence']), -int(r['source_triangles'])))
    write_csv(args.report_root / 'name_semantic_review.csv', review)
    (args.report_root / f'{prefix}_summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    table = '\n'.join(f"| {r['scene']} | {r['mesh_nodes']:,} | {r['unique_meshes']:,} | {r['assembly_nodes']} | {r['max_depth_edges']} | {r['instance_triangles']:,} |" for r in tables)
    evidence_table = '\n'.join(f"| {r['scene']} | {r['chinese_leaf_names']}（{pct(r['chinese_leaf_names'],r['mesh_nodes']):.2f}%） | {r['basic_dictionary_hint_rows']} | {r['nonroot_process_context_rows']} | {r['name_known_class_conflict_candidates']} / {r['name_known_class_conflict_high_confidence']} |" for r in tables)
    neural_table = '\n'.join(f"| {r['scene']} | {r['supervised_mean_confidence_percent']:.2f}% | {r['supervised_accepted_percent']:.2f}% | {r['open_usable_excluding_micro_percent']:.2f}% | {r['full_open_mean_basic_weight_percent']:.2f}% |" for r in tables)
    size_table = '\n'.join(f"| {r['scene']} | {r['size_le_1pct_count']} | {r['size_le_1pct_record_percent']:.2f}% | {r['size_le_1pct_triangle_percent']:.2f}% | {r['global_small_but_local_ge10pct']} |" for r in tables)
    six_rows = []
    for item in results:
        six_rows.append('### ' + item['scene'])
        six_rows.extend(['', '| 配置 | P1–P3 | P8–P10 |', '|---|---:|---:|'])
        for mode, label in [('pointnext','N'),('pointclip','C'),('pointnext_structure','N+G'),('pointclip_structure','C+G'),('pointnext_pointclip','N+C'),('pointnext_pointclip_structure','N+C+G')]:
            v = item['semantics']['six_full'][mode]
            six_rows.append(f"| {label} | {v['low_protection_percent']:.2f}% | {v['high_protection_percent']:.2f}% |")
        six_rows.append('')
    sample_lines = []
    for name in names:
        # Prefer varied source meshes; repeated instances are not independent cases.
        selected, meshes_seen = [], set()
        cases = [r for r in review if r['scene'] == name]
        for r in cases:
            if r['mesh_index'] in meshes_seen:
                continue
            meshes_seen.add(r['mesh_index'])
            selected.append(r)
            if len(selected) == 5:
                break
        sample_lines.extend(f"| {name} | {r['node_index']} | {r['node_name']} | {r['predicted_class']}（{float(r['confidence'])*100:.2f}%） | {r['open_label']} | {r['review_reason']} |" for r in selected)
    links = '\n'.join(f"- [{name}：全量语义与六配置报告]({name}/semantic_comparison/analysis_report.md)；[装配树]({name}/assembly_tree.txt)；[逐节点名称与上下文]({name}/mesh_name_context_audit.csv)。" for name in names)
    repeat_table = '\n'.join(f"| {r['scene']} | {r['same_mesh_repeated_groups']} | {r['same_mesh_prediction_disagreement_groups']} | {r['first_primitive_alpha_blend_nodes']} |" for r in tables)
    text = f'''# {len(names)} 个具名工业场景：名称、装配层级与语义分支分析

日期：2026-10-19。监督权重为已部署 PointNeXt-S seed2027 epoch98；{len(names)} 个场景共 {sum(r['mesh_nodes'] for r in tables):,} 个带网格节点，本次均覆盖监督、全量开放、候选开放、结构规则、多源融合及六种策略配置。未改变的推断可经内容核验复用，具体见各场景运行状态。旧 a.glb 和上一轮三模型报告未覆盖。

本轮神经推断仍只使用几何输入；名称和装配树作为独立辅助审计，没有暗中加入模型输入。没有人工真值，不能把置信度、名称一致率、策略分布当成场景准确率。

## 1. 输入与层级确实存在

| 场景 | 网格节点/实例 | 唯一 mesh 定义 | 带子节点的装配节点 | 最大树深度（边） | 实例三角形 |
|---|---:|---:|---:|---:|---:|
{table}

默认场景覆盖情况见逐场景 hierarchy_summary.json。重复 mesh 的不同节点实例分别统计，节点名称不作为唯一键；装配路径以 node_index 保留，重名装配不会互相覆盖。实例三角形与唯一 mesh 三角形口径不同，详见 {prefix}_summary.csv。

手机喇叭检测线包含框架、皮带、升降、移栽、NG 及 CCD/真空相关装配；汽车设备包含冲切、摆盘、模具、夹爪、螺母送料等；透气阀产线包含冲裁、搬运、外观检测、放收料、排气和摆盘工位。这些是文件中的名称证据，不是由文件名猜测的内部构成。

## 2. 名称有用，但“有名字”不等于“有类别标签”

| 场景 | 含中文的叶节点名 | 基础词典能提示角色 | 有非根节点工艺/机构上下文 | 已知类别名称冲突候选 / 其中高置信度 |
|---|---:|---:|---:|---:|
{evidence_table}

含中文也可能只是“镜向+图号”，所以不是语义覆盖率。词典覆盖只是本次公开规则的覆盖量，不是全部名称的信息上限。大量 JX、IBC 等图号和型号需经 BOM/制造商目录/人工释义确认；本次没有臆造型号含义。

局部名称与父装配名称分开解析。例如 CCD 支架是支架，不因父组或前缀含 CCD 就成为相机；装在冲切工位中的螺钉也不能直接继承“冲切工具”类别。根节点的全机名称不算局部工艺上下文。

“产品（螺母）”尤其值得注意：物体类别可仍是螺母，但它在工艺中可能是被检测/加工的工件。形状类别和装配功能角色必须分别建模，不能按重复紧固件统一低保护。

## 3. 当前分支的实际表现

| 场景 | 监督平均置信度 | 监督直接接受覆盖率 | 完整融合可用开放证据比例 | 开放基础归一权重均值 |
|---|---:|---:|---:|---:|
{neural_table}

监督接受规则为 top1≥0.70 且 top1–top2≥0.08；开放采用既有质量规则并排除 micro_uncertain。开放权重统计在后处理加分前，不是因果贡献。它们均不代表分类正确率。

手机检测线中可定位到：node 85“旋转气缸”被预测为 motors_gearmotors，置信度约99.03%；node 93“轴承”被预测为 washers_rings_spacers，约71.57%；node 29“光源”被预测为 washers_rings_spacers，约76.46%。这些是基于名称的复核线索，未直接确认为错误真值。气缸、感知和工艺专用零件缺少直接对应的监督类别，闭集高分不足以说明类内归属成立；仅按低置信度触发开放补充会漏掉这类问题。

| 场景 | node_index | 节点名称 | 监督预测 | 开放标签 | 复核原因 |
|---|---:|---|---|---|---|
{chr(10).join(sample_lines)}

完整定向复核表为 [name_semantic_review.csv](name_semantic_review.csv)，保留人工类别、功能角色和关键细节填写列。它按冲突定向筛选，不能作为随机样本估算准确率。

## 4. “小节点”必须结合工位尺度

| 场景 | 全局对角线1%以下节点 | 节点占比 | 实例三角形占比 | 全局≤1%但局部机构≥10%的节点 |
|---|---:|---:|---:|---:|
{size_table}

尺寸是世界坐标包围盒对角线之比，不是毫米阈值或屏幕像素。局部比例以最近一个具有工艺名称、至少含两个网格后代的非根装配组包围盒作参照，仅用于诊断尺度敏感性，尚未校准重要性阈值。

透气阀产线全局小节点承载超过一半实例三角形，不支持把它们整体当作无用碎片。对长产线应同时保留全局、工位、零件三层尺度；内部件和微小关键件还要按剖切、拆解、近距离观察任务单独评价。

## 5. 其他影响识别与“看不到”的因素

| 场景 | 重用同一 mesh 的组数 | 组内监督类别不一致的组数 | 首图元材质 BLEND 的节点 |
|---|---:|---:|---:|
{repeat_table}

同一源 mesh 的实例出现类别差异，提示姿态和采样稳定性需要检查；它们不一定具有相同工艺角色，不能直接复制全部功能标签。逐组证据保存在 repeated_mesh_evidence.csv。

当前 C++ 场景导入在首图元材质为 BLEND 时跳过该实例（src/scene/scene_gltf.cpp）。这一事实可能解释部分对象在应用中缺失，不表示几何无用；部分混合材质节点还需逐图元处理。未执行固定相机轨迹的遮挡、屏幕覆盖或剖切可见性测试，因此不能认定任意节点永久不可见。

离线脚本已修复为父到子的累计世界变换；各场景继承旋转/反射的节点数见 hierarchy_summary.json。监督和开放读取器使用相同父变换处理。三角形审计区分唯一 mesh 与实例数量。旧 a.glb 为平坦树，没有据此重写其历史结果。

## 6. 六策略输出分布

N=监督，C=开放，G=结构。P等级为规则分配，不是已测得的简化质量；各组使用同一份全量开放输出。现有完整融合与其他组合的融合路径仍有差异，G也可能读入监督不确定性，因此这些是配置比较，尚非严格独立消融。

{chr(10).join(six_rows)}

各场景另提供生产门控对照、同词表门控对照及移除监督不确定性的 G/CG 诊断基线。没有重建渲染 LOD、测量最终三角形保留率或 FPS，未修改生产语义保护公式。

## 7. 针对这些场景的优化顺序

1. 先把“未知/不可靠”从“低重要性”分离。名称提示传感、气动、工件等覆盖缺口时，即使监督分数很高也进入拒识或复核；名称不直接覆盖模型分类。
2. 手机检测线优先补齐气动执行、视觉检测、真空和传动部件的粗类别，并检查支架/安装件与功能本体的区分。
3. 透气阀产线优先利用工位树建立局部尺度和上下文；图号居多的叶节点通过 BOM 做可追溯映射，不能依赖全局尺寸筛除。
4. 汽车设备优先区分产品/工件、工具、机构零件和环境对象；同形螺母在不同角色下采用不同关键特征保护。环境地面只作独立对象审计，不据此直接删除。
5. 多尺度开放推断使用粗到细候选、有效视角和局部细节。名称先经可审计词典转成类别描述，图号保持不透明，不直接把中文长图号喂给当前英文提示编码后宣称有效。
6. 融合保留 N/C/G 的独立证据与冲突原因。名称和装配树若正式加入推断，应明确记为额外元数据 M，并增设 N+C+G+M、去名称、去层级、遮蔽型号等对照，不能算作原有三分支免费获得的提升。

## 8. 论文与评估边界

CAD 名称可生成弱标签和人工复核候选，但不能一边用名称辅助推断，一边把相同名称自动生成的标签当独立真值。应人工确认类别、功能角色、关键细节，在独立设备/装配来源上验证，并按几何家族隔离重复件。当前场景已经用于诊断，若继续据此改规则应作为开发场景，不再无条件称为未见测试集。

建议先分层核验这些场景中有代表性的已知、未知、小件、大件、重复件和工件；报告覆盖率、拒识表现、角色 Macro-F1、关键件漏保护，再在同预算同轨迹下验证几何误差和帧时间。仅报告名称一致或策略变化不能证明最终效果。

## 9. 文件与复现

{links}

总表：[{prefix}_summary.csv]({prefix}_summary.csv)、[{prefix}_summary.json]({prefix}_summary.json)。每场景另有 assembly_tree.csv、assembly_groups.csv、world_size_bins.csv、global_vs_assembly_scale.csv、repeated_mesh_evidence.csv。所有节点保留，visibility_status=unmeasured，allow_geometry_drop=False。

原始推断目录为 `{args.run_root.resolve()}`。分析与输入哈希见 provenance.json。推断入口为 tools/pointclipv2/run_scene_semantic_comparison.py；单场景策略分析入口为 analyze_scene_semantics.py；名称层级审计为 audit_named_scene.py；三场景汇总为 summarize_named_scenes.py。本次不重新训练编码器。
'''
    report = args.report_root / args.report_filename
    report.write_text(text, encoding='utf-8')
    outputs = [report, args.report_root / f'{prefix}_summary.csv', args.report_root / f'{prefix}_summary.json', args.report_root / 'name_semantic_review.csv']
    provenance = {'inputs_sha256': {str(p.resolve()): file_hash(p) for p in evidence_inputs},
                  'source_sha256': {str(p.resolve()): file_hash(p) for p in [Path(__file__), ROOT / 'tools/pointclipv2/audit_named_scene.py', ROOT / 'tools/pointnext_lod/gltf_hierarchy.py']},
                  'outputs_sha256': {str(p.resolve()): file_hash(p) for p in outputs},
                  'scenes': names, 'all_mesh_nodes_preserved': True, 'neural_inference_used_name_metadata': False,
                  'ground_truth_accuracy_measured': False, 'production_fusion_modified': False}
    (args.report_root / 'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    latest = {'status': 'named_scenes_inference_and_context_audit_complete', 'scenes': names,
              'report': str(report.resolve()), 'run_directory': str(args.run_root.resolve()),
              'ground_truth_metrics_available': False, 'renderer_lod_cache_status': 'not_rebuilt'}
    (ROOT / 'analysis_reports/latest_scene_analysis.json').write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding='utf-8')
    (ROOT / 'lod_analysis_outputs/deployment_status.json').write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'summary': tables, 'review_candidates': len(review), 'report': str(report.resolve())}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
