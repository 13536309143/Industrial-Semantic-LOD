"""Replay V5 budget design on six complete frozen scenes, without deployment.

Outputs describe policy allocations, not measured memory, FPS or accuracy.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

from adaptive_p10_v5_policy import infer, VERSION, PROFILES, PARAMETERS, yes
from artifact_cache import file_hash
from assembly_context_features import build_contexts, geometry_descriptors
from audit_named_scene import read_csv, write_csv
from experiment_assembly_context import key
from factorial_ablation_policy import global_contexts, MAIN_MODES
from fuse_open_semantic_v2 import builder
from run_pointclipv2_zeroshot_glb import read_glb_chunks

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = Path(__file__).parent
DEFAULT_OUTPUT = ROOT / 'lod_analysis_outputs/adaptive_p10_v5_20261021'
DEFAULT_REPORT = ROOT / 'analysis_reports/adaptive_p10_v5_20261021'


def priority(row):
    return int(row['lod_priority'].split('_')[0][1:])


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def table(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    write_csv(path, [{k: row.get(k, '') for k in columns} for row in rows])


def summary(scene, variant, rows, version):
    counts = Counter(priority(r) for r in rows)
    faces = Counter()
    for r in rows:
        faces[priority(r)] += int(r['face_count'])
    result = dict(scene=scene, variant=variant, version=version, nodes=len(rows),
                  source_instance_faces=sum(faces.values()))
    for p in range(1, 11):
        result.update({f'P{p}_count': counts[p], f'P{p}_node_fraction': counts[p] / max(1, len(rows)),
                       f'P{p}_face_fraction': faces[p] / max(1, sum(faces.values()))})
    return result


def sensitivity(rows):
    """Coefficient sensitivity, not accuracy calibration or a confidence interval."""
    output = []
    changes = [('all', .8), ('all', 1.2)] + [(k, factor) for k in
        ('size', 'shape', 'detail', 'topology', 'boundary') for factor in (.8, 1.2)]
    for role in ('unknown', 'standard_fastener'):
        members = [r for r in rows if r['function_role'] == role]
        for term, factor in changes:
            before, after = [], []
            for r in members:
                terms = json.loads(r['geometry_budget_terms_json'])
                if 'origin' not in terms:
                    continue
                adjusted = sum(v * factor if k == term or (term == 'all' and k not in ('origin', 'conflict')) else v
                               for k, v in terms.items())
                p = max(2, min(7, int(math.floor(adjusted + .5))))
                if role == 'standard_fastener':
                    if priority(r) == 1:
                        p = 1
                    elif yes(r['ordinary_fastener_discount']):
                        p = max(2, p - 1)
                before.append(priority(r))
                after.append(p)
            counts = Counter(after)
            differences = [abs(a - b) for a, b in zip(before, after)]
            result = dict(cohort=role, term=term, multiplier=factor, nodes=len(after),
                changed_nodes=sum(v > 0 for v in differences),
                changed_fraction=sum(v > 0 for v in differences) / max(1, len(after)),
                max_tier_shift=max(differences, default=0),
                mean_absolute_tier_shift=sum(differences) / max(1, len(after)))
            result.update({f'P{p}_count': counts[p] for p in range(1, 11)})
            output.append(result)
    return output


def generate_report(report, scene_stats, cohorts, families, validations, perturbations):
    selected = [r for r in scene_stats if r['variant'] == 'NCG_H1_M1']
    lines = ['# P1–P10 V5：整体预算与局部几何保护分离', '',
        '本次完成规则设计、可执行实现与六模型全量策略回放。复用固定神经推断，未重新训练；'
        '没有修改当前部署或V4历史结果。分布变化不是识别准确率、画质、显存或帧率提升的证明。', '',
        '## 六模型完整融合分布', '',
        'NCG_H1_M1为监督+开放+几何+结构树+叶名称辅助。名称是弱参考，不是真值。', '',
        '| 场景 | 节点 | ' + ' | '.join(f'P{p}' for p in range(1, 11)) + ' |',
        '|---|---:|' + '---:|' * 10]
    for r in selected:
        lines.append(f"| {r['scene']} | {r['nodes']} | " + ' | '.join(f"{100*r[f'P{p}_node_fraction']:.2f}%" for p in range(1, 11)) + ' |')
    total_nodes = sum(r['nodes'] for r in selected)
    lines.append(f'| 合计 | {total_nodes} | ' + ' | '.join(f"{100*sum(r[f'P{p}_count'] for r in selected)/total_nodes:.2f}%" for p in range(1, 11)) + ' |')
    lines += ['', '## 原P6两类零件的逐项去向', '',
        '固定跟踪V4原P6节点，避免用新分母掩盖变化。功能未明确不等于完全没有对象类别预测。', '',
        '| 原始队列 | 节点 | ' + ' | '.join(f'新P{p}' for p in range(1, 11)) + ' |',
        '|---|---:|' + '---:|' * 10]
    for r in cohorts:
        if r['scene'] == 'all':
            lines.append(f"| {r['cohort']} | {r['nodes']} | " + ' | '.join(str(r[f'P{p}_count']) for p in range(1, 11)) + ' |')
    lines += ['', '## 功能未明确件的几何分型', '',
        '分型是形态描述，不会将未知零件伪称为机架、气缸或其他对象。P可相同，但局部保护参数仍可不同。', '',
        '| 几何分型 | 节点 | ' + ' | '.join(f'P{p}' for p in range(1, 11)) + ' |',
        '|---|---:|' + '---:|' * 10]
    for r in families:
        lines.append(f"| {r['family']} | {r['nodes']} | " + ' | '.join(str(r[f'P{p}_count']) for p in range(1, 11)) + ' |')
    lines += ['', '## 系数敏感性诊断', '',
        '分别将尺寸、形态、细节、分量、边界项系数上下调整20%，并增加全部项同时变化的压力测试。'
        '固定对象、功能与几何输入，不重调配额；它不是准确率验证或统计置信区间。', '',
        '| 队列 | 所有几何项倍率 | 发生档位变化 | 最大跨档 |', '|---|---:|---:|---:|']
    for r in perturbations:
        if r['term'] == 'all':
            lines.append(f"| {r['cohort']} | {r['multiplier']:.1f} | {100*r['changed_fraction']:.2f}% | {r['max_tier_shift']} |")
    lines += ['', '完整单项敏感性见`coefficient_sensitivity.csv`；处于相邻档边界的节点会变化，'
        '档位标签需要结合连续分数及局部约束解释，不能把一个整数档位当作客观真值。']
    lines += ['', '## 设计与约束', '',
        '1. P1微小普通重复件；P2低贡献紧凑件；P3简洁体块与普通连接；P4一般表面与形状；'
        'P5曲面与分布细节；P6显著路径与轮廓；P7耦合几何与功能接口；P8作业功能；'
        'P9精密与工艺功能；P10关键功能且几何敏感。它们是整体预算档位，适用对象是例子而非固定类别表。',
        '2. 未知功能依赖全局/局部尺度、杆/板/紧凑形态、饱和的边缘细节指标、分量数量及面积分布分配P2–P7；'
        '没有复杂度二元阈值强制P6，也没有按场景分位数填满档位。',
        '3. 普通紧固件的小尺寸折扣只适用于无冲突、以单一主体为主的节点；大件、复合件、身份冲突不享受折扣。'
        '精密装配紧固件独立走功能保护先验，父节点名称不直接复制为子件身份。',
        '4. 截面敏感度独立收紧特征锁、特征软权重、误差与轴范围收缩限值；薄板或螺钉即使处于低P也不取消这些约束。'
        '保留连通分量存在性，禁止删除节点；这些检查仍不构成孔洞、厚度或完整拓扑证明。',
        '5. 近/中/远模板沿用V4，以隔离档位分配变化；屏幕误差公式也沿用V4。'
        '比率不是最终面数，几何局部约束已发生变化，不能继承V4的运行时质量/性能结论。', '',
        '## 完整性检查', '',
        f"- 状态：{validations['status']}；{validations['policy_files']}组策略，{validations['policy_rows']}行；"
        f"完整融合{validations['selected_nodes']}节点。",
        '- 对象身份、功能角色与拒识结果保持；每个变体与V4同条件逐节点核对。',
        '- 所有比例有序、有限正误差预算、禁止剔除；G关闭时几何预算不参与，H0不读父组尺度。', '',
        '## 结论边界与后续验证', '',
        '本报告能够证明规则实现一致、原P6队列获得形态和尺度区分；不能证明新档位一定正确。'
        '系数与阈值是显式工程先验，没有独立人工标注，也未通过六场景配额拟合。'
        '仍需真实层次构建、多视角局部画质及串行GPU测量，才能决定是否替换当前部署。'
        '三角形占比按节点实例统计，共享网格的实际存储须以渲染器资源计数为准。', '',
        '设计说明：`tools/pointclipv2/ADAPTIVE_P10_V5.md`。',
        '详细表：`all_profile_comparison.csv`、`previous_p6_cohorts.csv`、'
        '`unresolved_geometry_families.csv`、`selected_node_changes.csv`。', '']
    (report / 'P1-P10_V5设计与六模型分配验证.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=list('abcdef'), default=list('abcdef'))
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--report-dir', type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models):
        parser.error('Models must not be repeated')
    if set(args.models) != set('abcdef'):
        subset = 'subset_' + ''.join(args.models)
        if args.output_root == DEFAULT_OUTPUT:
            args.output_root /= subset
        if args.report_dir == DEFAULT_REPORT:
            args.report_dir /= subset
    args.report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / 'lod_analysis_outputs/assembly_context_v3_20261020/scene_sources.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    taxonomy_path = DIRECTORY / 'industrial_open_vocab_v2.json'
    config_path = DIRECTORY / 'assembly_context_config.json'
    taxonomy = json.loads(taxonomy_path.read_text(encoding='utf-8'))
    config = json.loads(config_path.read_text(encoding='utf-8'))
    baseline_root = ROOT / 'lod_analysis_outputs/adaptive_p10_v4_20261020'
    baseline_proof = json.loads((ROOT / 'analysis_reports/adaptive_p10_v4_20261020/provenance.json').read_text(encoding='utf-8'))
    inputs = [manifest_path, taxonomy_path, config_path, Path(__file__), DIRECTORY / 'adaptive_p10_v5_policy.py',
              DIRECTORY / 'adaptive_p10_policy.py', DIRECTORY / 'assembly_context_features.py',
              DIRECTORY / 'assembly_context_policy.py', DIRECTORY / 'factorial_ablation_policy.py',
              DIRECTORY / 'fuse_open_semantic_v2.py', DIRECTORY / 'ADAPTIVE_P10_V5.md',
              DIRECTORY / 'tests/test_adaptive_p10_v5.py', DIRECTORY / 'audit_named_scene.py',
              DIRECTORY / 'experiment_assembly_context.py', DIRECTORY / 'run_pointclipv2_zeroshot_glb.py',
              Path(builder.__file__), ROOT / 'run_p10_v5_design.bat']
    stats, changes, cohort_entries, unresolved, selected_rows = [], [], [], [], []
    policy_files = []
    policy_rows = 0
    specs = [(f'{mode}_H{int(tree == "on")}', mode, tree, False)
             for mode in (*MAIN_MODES, 'G') for tree in ('off', 'on')]
    specs.append(('NCG_H1_M1', 'NCG', 'on', True))
    for scene in args.models:
        source = manifest[scene]
        glb, pn_path, c_path = [Path(source[k]) for k in ('glb', 'pointnext_csv', 'open_csv')]
        for path, expected in ((glb, source['glb_sha256']), (pn_path, source['pointnext_sha256']), (c_path, source['open_sha256'])):
            if file_hash(path) != expected:
                raise ValueError(f'Changed frozen evidence: {path}')
        pn = read_csv(pn_path)
        opened = {key(r): r for r in read_csv(c_path)}
        if len(opened) != len(pn) or set(opened) != {key(r) for r in pn}:
            raise ValueError('Incomplete or duplicate frozen rows')
        data, _, _ = read_glb_chunks(glb)
        expected = {(str(i), str(n['mesh'])) for i, n in enumerate(data['nodes']) if 'mesh' in n}
        if {(r['node_index'], r['mesh_index']) for r in pn} != expected or len(pn) != len(expected):
            raise ValueError('Frozen evidence is not a full scene')
        descriptor_path = ROOT / 'lod_analysis_outputs/assembly_context_v3_20261020' / scene / 'geometry_descriptors.json'
        descriptors = geometry_descriptors(glb, pn, descriptor_path)
        contexts = {'off': global_contexts(pn), 'on': build_contexts(data['nodes'], pn, config)}
        repeats = Counter(r['mesh_index'] for r in pn)
        thresholds = builder.make_thresholds(pn)
        folder = args.output_root / scene
        folder.mkdir(parents=True, exist_ok=True)
        for variant, mode, tree, leaf in specs:
            rows = [infer(r, opened[key(r)], taxonomy, thresholds, descriptors[str(r['node_index'])],
                          contexts[tree][str(r['node_index'])], config, mode, tree, leaf, repeats[r['mesh_index']]) for r in pn]
            old_path = baseline_root / scene / (variant + '.csv')
            expected_hash = baseline_proof['outputs'].get(str(old_path.resolve()))
            if not expected_hash or file_hash(old_path) != expected_hash:
                raise ValueError(f'Unverified V4 comparison: {old_path}')
            old_index = {key(r): r for r in read_csv(old_path)}
            if set(old_index) != {key(r) for r in rows}:
                raise ValueError('Unpaired previous policy')
            old_rows = [old_index[key(r)] for r in rows]
            for r, old in zip(rows, old_rows):
                for field in ('function_role', 'resolved_object_class', 'resolved_status', 'resolved_source',
                              'class_conflict', 'type_shape_conflict', 'name_function_type_conflict'):
                    if str(r.get(field, '')).lower() != str(old.get(field, '')).lower():
                        raise ValueError(f'Changed semantic decision {scene}/{variant}/{key(r)}/{field}')
                near, mid, far = [float(r['target_ratio_' + k]) for k in ('near', 'mid', 'far')]
                if not 0 < far <= mid <= near <= 1 or yes(r['allow_cull']) or yes(r['allow_geometry_drop']):
                    raise ValueError('Invalid schedule or node removal')
                if float(r['simplify_error_relative']) <= 0 or not r['preserve_components']:
                    raise ValueError('Missing geometry guard')
            out = folder / (variant + '.csv')
            write_csv(out, rows)
            policy_files.append(out)
            policy_rows += len(rows)
            stats.extend([summary(scene, variant, rows, VERSION), summary(scene, variant, old_rows, 'adaptive_p10_v4.3')])
            inputs.append(old_path)
            if variant == 'NCG_H1_M1':
                selected_rows.extend(rows)
                for r, old in zip(rows, old_rows):
                    e = json.loads(r['geometry_evidence_json'])
                    record = dict(scene=scene, node_index=r['node_index'], mesh_index=r['mesh_index'],
                        node_name=r['node_name'], face_count=r['face_count'], function_role=r['function_role'],
                        resolved_object_class=r['resolved_object_class'], old_priority=priority(old),
                        new_priority=priority(r), geometry_family=r['geometry_budget_family'],
                        morphology=r['budget_morphology'], geometry_budget_score=r['geometry_budget_score'],
                        scene_size_ratio=e['scene_ratio'], local_size_ratio=e['local_ratio'],
                        size_score=e['size_score'], detail_score=e['detail_score'],
                        shape_demand=e['shape_demand'], topology_score=e['topology_score'],
                        ordinary_fastener_discount=r['ordinary_fastener_discount'],
                        feature_lock_budget=r['feature_lock_budget'], max_axis_shrink=r['max_axis_shrink'],
                        simplify_error_relative=r['simplify_error_relative'])
                    changes.append(record)
                    if r['function_role'] == 'unknown':
                        unresolved.append(r)
                    if priority(old) == 6 and r['function_role'] in ('unknown', 'standard_fastener'):
                        cohort_entries.append((scene, r['function_role'], r))
        inputs.extend([glb, pn_path, c_path, descriptor_path])
        print(f'{scene}: {len(pn)} nodes x {len(specs)} paired policy variants; semantic decisions unchanged', flush=True)
    cohorts = []
    for scene in [*args.models, 'all']:
        for role in ('unknown', 'standard_fastener'):
            members = [r for s, k, r in cohort_entries if k == role and (scene == 'all' or scene == s)]
            result = summary(scene, 'old_P6_cohort', members, VERSION)
            result['cohort'] = role
            cohorts.append(result)
    families = []
    for family in sorted({r['geometry_budget_family'] for r in unresolved}):
        members = [r for r in unresolved if r['geometry_budget_family'] == family]
        result = summary('all', 'unresolved_function_geometry_family', members, VERSION)
        result['family'] = family
        families.append(result)
    # Only current V5 rows in the human report; CSV retains both versions.
    table(args.report_dir / 'all_profile_comparison.csv', stats)
    table(args.report_dir / 'previous_p6_cohorts.csv', cohorts)
    table(args.report_dir / 'unresolved_geometry_families.csv', families)
    table(args.report_dir / 'selected_node_changes.csv', changes)
    perturbations = sensitivity(selected_rows)
    table(args.report_dir / 'coefficient_sensitivity.csv', perturbations)
    validation = dict(status='passed', policy_files=len(policy_files), policy_rows=policy_rows,
        selected_nodes=len(changes), semantic_decisions_unchanged=True,
        no_geometry_removal=True, deployed=False, runtime_quality_measured=False,
        memory_or_frame_time_measured=False, coefficients='explicit engineering priors, not accuracy calibrated')
    save(args.report_dir / 'validation_summary.json', validation)
    save(args.report_dir / 'profile_definitions.json', {'version': VERSION, 'parameters': PARAMETERS, 'profiles': PROFILES})
    generate_report(args.report_dir, [r for r in stats if r['version'] == VERSION], cohorts, families, validation, perturbations)
    outputs = [*policy_files, *args.report_dir.glob('*.csv'), args.report_dir / 'validation_summary.json',
               args.report_dir / 'profile_definitions.json', args.report_dir / 'P1-P10_V5设计与六模型分配验证.md']
    save(args.report_dir / 'provenance.json', dict(version=VERSION, models=args.models,
        inputs={str(p.resolve()): file_hash(p) for p in inputs},
        outputs={str(p.resolve()): file_hash(p) for p in outputs},
        scope='all-node paired policy replay; no deployment or runtime performance assertion'))
    print(f'V5 design complete: {args.report_dir}', flush=True)


if __name__ == '__main__':
    main()
