"""All-node V5 factorial replay: seven branch subsets x H0/HT/H1 + M controls.

This measures evidence/decision coverage and budget allocation, not accuracy,
rendering quality or GPU-memory/frame-time benefits. No policy is deployed.
"""
import argparse
import csv
import gc
import gzip
import hashlib
import json
import math
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

from adaptive_p10_v5_policy import infer, VERSION, PARAMETERS, PROFILES
from assembly_context_features import build_contexts, geometry_descriptors
from audit_named_scene import read_csv
from experiment_assembly_context import key
from factorial_ablation_policy import global_contexts
from fuse_open_semantic_v2 import builder
from run_pointclipv2_zeroshot_glb import read_glb_chunks
from run_six_scene_factorial_ablation import summarize, pooled, source_diagnostic, uncertainty_tables, priority, yes
from run_thirteen_scene_ablation import ROOT, DEFAULT_OUTPUT, DEFAULT_REPORT, digest, valid_stamp, save
from thirteen_scene_coverage import align_mesh_population

HERE = Path(__file__).parent
MODES = ('N', 'C', 'G', 'NC', 'NG', 'CG', 'NCG')
ACTION_FIELDS = ('lod_priority', 'target_ratio_near', 'target_ratio_mid', 'target_ratio_far',
    'allow_cull', 'allow_geometry_drop', 'feature_lock_floor', 'feature_soft_floor',
    'feature_lock_budget', 'feature_soft_budget', 'simplify_error_relative', 'max_axis_shrink',
    'preserve_components')
COMPACT_FIELDS = (*ACTION_FIELDS, 'order', 'node_index', 'mesh_index', 'node_name', 'face_count',
                  'function_role', 'resolved_object_class', 'resolved_status', 'assembly_context_json')


def table(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    opener = gzip.open if path.suffix == '.gz' else open
    temp = path.with_name(path.name + '.tmp')
    with opener(temp, 'wt', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def pair(scene, mode, left, right, before, after, kind):
    assert [key(r) for r in left] == [key(r) for r in right]
    matrix, faces, changes = Counter(), Counter(), []
    for a, b in zip(left, right):
        pa, pb = priority(a), priority(b)
        matrix[pa, pb] += 1
        faces[pa, pb] += int(a['face_count'])
        action = any(str(a.get(k, '')) != str(b.get(k, '')) for k in ACTION_FIELDS)
        if action or a['function_role'] != b['function_role'] or a['resolved_object_class'] != b['resolved_object_class']:
            changes.append(dict(scene=scene, mode=mode, reference=before, comparison=after,
                node_index=a['node_index'], mesh_index=a['mesh_index'], node_name=a['node_name'],
                source_faces=int(a['face_count']), before_priority=pa, after_priority=pb,
                delta=pb-pa, before_function=a['function_role'], after_function=b['function_role'],
                before_identity=a['resolved_object_class'], after_identity=b['resolved_object_class'],
                before_far=float(a['target_ratio_far']), after_far=float(b['target_ratio_far']),
                action_changed=action, ancestor_context=b.get('assembly_context_json', '')))
    n = len(left)
    r = dict(scene=scene, mode=mode, reference=before, comparison=after, kind=kind, nodes=n,
        priority_changed=sum(v for (a, b), v in matrix.items() if a != b),
        priority_raised=sum(v for (a, b), v in matrix.items() if b > a),
        priority_lowered=sum(v for (a, b), v in matrix.items() if b < a),
        function_changed=sum(a['function_role'] != b['function_role'] for a, b in zip(left, right)),
        identity_changed=sum(a['resolved_object_class'] != b['resolved_object_class'] for a, b in zip(left, right)),
        action_changed=sum(c['action_changed'] for c in changes),
        mean_priority_delta=sum((b-a)*v for (a, b), v in matrix.items())/n)
    for k in ('priority_changed', 'priority_raised', 'priority_lowered', 'function_changed', 'identity_changed', 'action_changed'):
        r[k+'_fraction'] = r[k]/n
    transitions = [dict(scene=scene, mode=mode, reference=before, comparison=after,
        from_P=a, to_P=b, count=matrix[a, b], node_fraction=matrix[a, b]/n, source_faces=faces[a, b])
        for a in range(1, 11) for b in range(1, 11)]
    return r, transitions, changes


def descriptor_source(glb_hash, expected, candidates):
    for stamp in candidates:
        try:
            proof = json.loads(stamp.read_text(encoding='utf-8'))
            target = stamp.with_name(stamp.name.replace('.provenance.json', '.json'))
            if glb_hash not in [v for k, v in proof['inputs'].items() if k.lower().endswith('.glb')]:
                continue
            if digest(target) != proof['output_sha256']:
                continue
            if any(not k.lower().endswith('.glb') and digest(Path(k)) != v for k, v in proof['inputs'].items()):
                continue
            values = json.loads(target.read_text(encoding='utf-8'))
            if set(values) != {a for a, b in expected}:
                continue
            return target, values
        except (OSError, KeyError, ValueError, TypeError):
            continue
    return None, None


def render_report(report, summaries, pairs, inventory, profiles, lowtypes, families):
    def pc(v): return '<0.01%' if 0<v<.0001 else f'{100*v:.2f}%'
    total = sum(r['nodes'] for r in inventory)
    empty_count=sum(r.get('empty_mesh_references',0) for r in inventory)
    lines = ['# 十三个工业场景全量融合与装配上下文消融分析', '',
        f'固定V5策略、监督权重、开放词库及阈值；共{len(inventory)}个场景、{total:,}个网格实例。'
        '主实验为7种非空分支组合×H0/H1；HT仅保留拓扑与局部尺度，M额外使用叶名称。'
        '没有重训、调参或替换当前渲染策略；本报告不将输出覆盖或P变化解释为识别准确率、画质、显存或帧率提升。', '',
        'N=监督语义；C=开放视觉语义；G=显式几何；H1=局部装配尺度与受范围限制的祖先名称。'
        'H0保留相同世界变换及输入几何，屏蔽父组尺度和上下文；M主实验关闭。', '',
        '为保证纯C及各组合使用同一份独立证据，本轮对所有节点执行开放推断，'
        '不以本轮耗时评价生产流程的不确定性门控节省；门控计算收益需要单独对照。', '',
        f'原始GLB共{total+empty_count:,}个网格引用，其中{empty_count}个没有POSITION顶点数据，单列为输入空网格。'
        f'主对照覆盖全部{total:,}个有顶点的几何实例；空网格不填造监督预测，也不当作真实零件误识别。'
        '完整记录见excluded_empty_meshes.csv；C的空几何拒识记录保留在原始证据中。', '',
        '## 场景规模与覆盖', '', '| 场景 | 名称 | 网格实例 | 原始三角形实例 | 最大层级 |', '|---|---|---:|---:|---:|']
    for r in inventory:
        lines.append(f"| {r['scene']} | {r['title']} | {r['nodes']:,} | {r['source_faces']:,} | {r['max_depth']} |")
    lines += ['', '每个条件逐一验证(node_index, mesh_index)完整覆盖。相同几何的重复实例仍计入节点统计，'
        '三角形按原始实例展开累计，不等于实际存储、可见三角形或显存。未将13场景自动视为独立外部测试集。', '',
        '## 主要组合与未解决证据', '', '| 配置 | 无对象级输出 | 功能未知 | N/C分歧 | 需复核 | 请求远档面数比例 |', '|---|---:|---:|---:|---:|---:|']
    allrows = [r for r in summaries if r['scene'] == 'ALL' and r['tree'] != 'topology' and not r['leaf_names']]
    for r in allrows:
        lines.append(f"| {r['variant']} | {pc(r['object_unresolved_fraction'])} | {pc(r['unknown_function_fraction'])} | {pc(r['identity_conflict_fraction'])} | {pc(r['review_fraction'])} | {pc(r['requested_far_triangle_fraction'])} |")
    lines += ['', '请求面数比例由策略保留率加权得到，实际简化可能受局部约束和停止条件影响，不能当作已经实现的压缩比。', '',
        '## P1至P10节点与源三角形分布', '']
    for basis, label in [('node', '节点'), ('face', '原始三角形实例')]:
        lines += [f'### 按{label}统计', '', '| 配置 | '+' | '.join(f'P{i}' for i in range(1,11))+' |', '|---|'+'---:|'*10]
        for r in allrows:
            lines.append('| '+r['variant']+' | '+' | '.join(pc(r[f'P{i}_{basis}_fraction']) for i in range(1,11))+' |')
        lines.append('')
    lines += ['## 装配上下文的成对影响', '', '| 组合 | 改变P | 升档 | 降档 | 功能改变 | 对象身份改变 |', '|---|---:|---:|---:|---:|---:|']
    for mode in MODES:
        group = [r for r in pairs if r['mode'] == mode and r['reference']=='H0' and r['comparison']=='H1']
        counts = [sum(r[k] for r in group) for k in ('priority_changed','priority_raised','priority_lowered','function_changed','identity_changed')]
        lines.append('| '+mode+' | '+' | '.join(f'{n:,}（{pc(n/total)}）' for n in counts)+' |')
    lines += ['', '### 完整融合在各场景的变化', '', '| 场景 | 改变P | 升档 | 降档 | 功能改变 | 身份改变 |', '|---|---:|---:|---:|---:|---:|']
    for r in pairs:
        if r['mode']=='NCG' and r['reference']=='H0' and r['comparison']=='H1':
            lines.append(f"| {r['scene']} | {r['priority_changed']}（{pc(r['priority_changed_fraction'])}） | {r['priority_raised']} | {r['priority_lowered']} | {r['function_changed']} | {r['identity_changed']} |")
    lines += ['', 'H0→HT用于观察拓扑及局部尺度变化，HT→H1用于观察加入祖先名称后的变化；'
        '不能将H1的全部效果称为纯拓扑贡献。各因素改变的节点可重叠，计数不可相加作为独立因果效应。'
        '完整成对表见tree_pairs.csv，10×10迁移矩阵包含未改变项。', '',
        '## 低置信与开放拒识', '', '| 场景 | N未达接受条件 | C对象级输出 | C仅粗类别 | C拒识 |', '|---|---:|---:|---:|---:|']
    for r in profiles:
        lines.append(f"| {r['scene']} | {r['N_not_reliable']}（{pc(r['N_not_reliable_fraction'])}） | {r['C_object_usable']}（{pc(r['C_object_fraction'])}） | {r['C_coarse']} | {r['C_abstain']} |")
    lines += ['', 'N采用既有概率/间隔门槛；C按对象级、粗类别与拒识状态统计，质量分数未校准。'
        '两个分支不能按这些比例直接比较准确率。概率≥0.70时，间隔≥0.08不再提供额外筛选，不能据此宣称双阈值的独立收益。', '',
        '### 合并低置信候选类型', '', '| 分支 | 候选类型 | 未达对象级接受条件 | 占该分支未接受记录 |', '|---|---|---:|---:|']
    for branch in ('N','C'):
        selected = sorted([r for r in lowtypes if r['scene']=='ALL' and r['branch']==branch], key=lambda r:r['low_confidence_candidates'], reverse=True)[:10]
        for r in selected:
            share = pc(r['share_of_low_pool']) if r['share_of_low_pool'] is not None else '—'
            lines.append(f"| {branch} | {r['candidate_type'] or '无候选'} | {r['low_confidence_candidates']:,} | {share} |")
    lines += ['', '这些名称是模型候选类别，不是已核验真实类型。逐场景明细见low_confidence_candidate_types.csv；'
        '未知功能件几何分型见unknown_geometry_families.csv。', '',
        '## 结果解释与论文使用', '',
        '1. N/C互补首先看对象候选覆盖及分歧，不能以未解决比例下降证明纠错。',
        '2. G首先看未知形态和整体/局部预算是否得到区分；纯G不承担对象识别，没有类别输出不等于识别失败。',
        '3. H先看功能、身份撤回和保护变化；仅升档可能增加保护与资源开销，不能自动视为收益。',
        '4. 叶名称M作为单列辅助；启用M的条件不使用同一名称计算一致率。其他名称一致率也仅为弱参考，不是OA或功能真值。',
        '5. 本轮全量仅指全部节点及全部条件覆盖；没有运行实际cluster LOD构建、多视角图像或GPU渲染基准。', '',
        'CSV保留逐场景和合并结果；合并比例先合计计数再求比例，另提供场景等权平均。'
        '逐节点CSV使用gzip压缩，避免重复证据和解释字段大量占用磁盘；可解压后在表格软件中查看。', '']
    (report/'十三模型全量消融与结构树对照分析.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=DEFAULT_OUTPUT/'scene_sources.json')
    p.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--report-dir', type=Path, default=DEFAULT_REPORT)
    p.add_argument('--models', nargs='+', default=[f'M{i}' for i in range(1,14)])
    p.add_argument('--wait-for-evidence', action='store_true')
    args = p.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    status = dict(status='running', models=args.models, complete=[], failed=[], policy_version=VERSION,
        main_conditions=len(args.models)*14, total_conditions=len(args.models)*23)
    status_path = args.report_dir/'analysis_status.json'
    def progress(scene, stage):
        status.update(current_scene=scene, current_stage=stage)
        save(status_path, status)
        print(f'{scene}: {stage}', flush=True)
    taxonomy_path = HERE/'industrial_open_vocab_v2.json'
    config_path = HERE/'assembly_context_config.json'
    taxonomy = json.loads(taxonomy_path.read_text(encoding='utf-8'))
    config = json.loads(config_path.read_text(encoding='utf-8'))
    methods = [HERE/name for name in ('adaptive_p10_v5_policy.py','adaptive_p10_policy.py',
        'factorial_ablation_policy.py','assembly_context_policy.py','assembly_context_features.py',
        'fuse_open_semantic_v2.py','run_six_scene_factorial_ablation.py','experiment_assembly_context.py',
        'audit_named_scene.py','summarize_named_scenes.py','run_pointclipv2_zeroshot_glb.py')]
    # Node-policy identity is separate from report implementation: changing a
    # report must not invalidate otherwise identical frozen policy outputs.
    method_hashes = {str(path.resolve()):digest(path) for path in [*methods, taxonomy_path, config_path]}
    analysis_hash = digest(Path(__file__))
    save(args.report_dir/'policy_freeze.json', dict(version=VERSION, files=method_hashes, analysis_sha256=analysis_hash, parameters=PARAMETERS, profiles=PROFILES))
    for path in [*methods, Path(__file__)]:
        target=args.report_dir/'method'/path.name
        target.parent.mkdir(exist_ok=True)
        if not target.exists() or (path==Path(__file__) and digest(target)!=digest(path)):
            shutil.copy2(path,target)
    descriptor_candidates = list((ROOT/'lod_analysis_outputs').rglob('geometry_descriptors.provenance.json'))
    for scene in args.models:
        try:
            progress(scene, 'waiting_for_verified_evidence')
            while True:
                manifest = json.loads(args.manifest.read_text(encoding='utf-8')) if args.manifest.exists() else {}
                if scene in manifest: break
                if not args.wait_for_evidence: raise ValueError('No full-scene neural evidence')
                acquisition = json.loads((args.report_dir/'run_status.json').read_text(encoding='utf-8'))
                if acquisition['status'] != 'running': raise ValueError('Evidence acquisition ended before this scene completed')
                time.sleep(10)
            source = manifest[scene]
            glb,pnpath,cpath = [Path(source[k]) for k in ('glb','pointnext_csv','open_csv')]
            for path,k in ((glb,'glb_sha256'),(pnpath,'pointnext_sha256'),(cpath,'open_sha256')):
                if digest(path) != source[k]: raise ValueError(f'Changed frozen evidence: {path}')
            fingerprint = hashlib.sha256(json.dumps([source, method_hashes],sort_keys=True).encode()).hexdigest()
            folder = args.output_root/scene/'policies'
            folder.mkdir(parents=True, exist_ok=True)
            done = folder/'completion.json'
            if done.exists():
                proof = json.loads(done.read_text(encoding='utf-8'))
                if proof['fingerprint']==fingerprint and all(digest(Path(k))==v for k,v in proof['outputs'].items()):
                    status['complete'].append(scene);progress(scene,'REUSE verified full policy matrix');continue
            progress(scene, 'geometry_and_context')
            pn = read_csv(pnpath)
            crows = read_csv(cpath)
            data,_,_ = read_glb_chunks(glb)
            pn,crows,excluded=align_mesh_population(data,pn,crows)
            opened = {key(r):r for r in crows}
            empty_keys={(str(r['node_index']),str(r['mesh_index'])) for r in excluded}
            expected={(str(i),str(n['mesh'])) for i,n in enumerate(data['nodes']) if 'mesh' in n}-empty_keys
            assert len(pn)==len(opened)==len(crows)==len(expected)
            assert {key(r) for r in pn}==set(opened)
            assert {(str(r['node_index']),str(r['mesh_index'])) for r in pn}==expected
            descriptor_path,descriptors = descriptor_source(source['glb_sha256'], expected, descriptor_candidates)
            if descriptors is None:
                descriptor_path = args.output_root/scene/'geometry_descriptors.json'
                descriptors = geometry_descriptors(glb,pn,descriptor_path)
            contexts={'off':global_contexts(pn), 'topology':build_contexts(data['nodes'],pn,config,use_names=False),
                      'on':build_contexts(data['nodes'],pn,config,use_names=True)}
            thresholds=builder.make_thresholds(pn); repeats=Counter(r['mesh_index'] for r in pn)
            diagnostics=[source_diagnostic(scene,r,opened[key(r)],taxonomy) for r in pn]
            summary_rows=[];compact={};family_counts=Counter();outputs=[]
            specs=[(f'{mode}_H{label}',mode,tree,False,'main' if tree!='topology' else 'topology_control')
                for mode in MODES for label,tree in [('0','off'),('T','topology'),('1','on')]]
            specs += [('NCG_H0_M1','NCG','off',True,'metadata_control'),('NCG_H1_M1','NCG','on',True,'metadata_control')]
            for variant,mode,tree,leaf,kind in specs:
                progress(scene, variant)
                rows=[infer(r,opened[key(r)],taxonomy,thresholds,descriptors[str(r['node_index'])],
                    contexts[tree][str(r['node_index'])],config,mode,tree,leaf,repeats[r['mesh_index']]) for r in pn]
                assert [key(r) for r in rows]==[key(r) for r in pn]
                for row in rows:
                    assert yes(row['allow_cull']) == row['lod_priority'].startswith('P1_')
                    assert not yes(row['allow_geometry_drop'])
                    ratios=[float(row['target_ratio_'+a]) for a in ('near','mid','far')]
                    assert 0<ratios[2]<=ratios[1]<=ratios[0]<=1
                    assert float(row['simplify_error_relative'])>0 and math.isfinite(float(row['simplify_error_relative']))
                path=folder/(variant+'.csv.gz');table(path,rows);outputs.append(path)
                summary_rows.append(summarize(scene,variant,mode,tree,leaf,rows,diagnostics,taxonomy,kind))
                compact[variant]=[{k:r.get(k,'') for k in COMPACT_FIELDS} for r in rows]
                for r in rows:
                    if r['function_role']=='unknown':
                        family_counts[(variant,r.get('geometry_budget_family','unavailable'),priority(r))]+=1
                del rows
            pairs=[];transitions=[];changes=[]
            for mode in MODES:
                for left,right,kind in [('H0','H1','main'),('H0','HT','topology_control'),('HT','H1','ancestor_name_increment')]:
                    r,t,c=pair(scene,mode,compact[f'{mode}_{left}'],compact[f'{mode}_{right}'],left,right,kind)
                    pairs.append(r);transitions.extend(t);changes.extend(c)
            for left,right,kind in [('NCG_H0','NCG_H0_M1','leaf_name_control'),('NCG_H1','NCG_H1_M1','leaf_name_control')]:
                r,t,c=pair(scene,'NCG',compact[left],compact[right],left,right,kind)
                pairs.append(r);transitions.extend(t);changes.extend(c)
            # Also pair enabled geometry/semantic branches, not merely hierarchy switches.
            for left,right in [('N_H0','NC_H0'),('C_H0','NC_H0'),('N_H0','NG_H0'),('C_H0','CG_H0'),('NC_H0','NCG_H0')]:
                r,t,c=pair(scene,'branch',compact[left],compact[right],left,right,'branch_control')
                pairs.append(r);transitions.extend(t)
            family_rows=[dict(scene=scene,variant=v,family=f,priority=p,count=n) for (v,f,p),n in sorted(family_counts.items())]
            max_depth=max(c['depth'] for c in contexts['on'].values())+1
            inventory=dict(scene=scene,title=source['title'],nodes=len(pn),source_faces=sum(int(r['face_count']) for r in pn),
                source_mesh_references=len(expected)+len(excluded),empty_mesh_references=len(excluded),
                assembly_nodes=sum(bool(n.get('children')) for n in data['nodes']),max_depth=max_depth,
                descriptor_invalid=sum(not bool(v.get('geometry_valid')) for v in descriptors.values()),
                source_descriptor=str(descriptor_path),glb_sha256=source['glb_sha256'])
            paths=[('summary.csv',summary_rows),('pairs.csv',pairs),('transitions.csv',transitions),
                   ('source_diagnostics.csv.gz',diagnostics),('unknown_families.csv',family_rows)]
            for name,rows in paths:
                path=folder/name;table(path,rows);outputs.append(path)
            if changes:
                path=folder/'tree_changed_nodes.csv.gz';table(path,changes);outputs.append(path)
                examples=sorted(changes,key=lambda r:(r['mode']=='NCG' and r['comparison']=='H1',r['source_faces']),reverse=True)[:150]
                path=folder/'changed_examples.csv';table(path,examples);outputs.append(path)
            save(folder/'inventory.json',inventory);outputs.append(folder/'inventory.json')
            if excluded:
                table(folder/'excluded_empty_meshes.csv',excluded);outputs.append(folder/'excluded_empty_meshes.csv')
                save(folder/'coverage_audit.json',dict(source_mesh_references=len(expected)+len(excluded),
                    analyzable_geometry_nodes=len(pn),excluded_empty_meshes=excluded,
                    adapter_sha256=digest(HERE/'thirteen_scene_coverage.py'),no_neural_predictions_fabricated=True))
                outputs.append(folder/'coverage_audit.json')
            save(done,dict(fingerprint=fingerprint,version=VERSION,nodes=len(pn),conditions=len(specs),
                policy_rows=len(pn)*len(specs),outputs={str(p.resolve()):digest(p) for p in outputs},
                source=source,descriptor_sha256=digest(descriptor_path)))
            status['complete'].append(scene);progress(scene,'COMPLETE 23 full-node conditions')
            del compact,pn,opened,crows,diagnostics,contexts,descriptors,data,changes
            gc.collect()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            status['failed'].append(dict(scene=scene,error=str(exc)));save(status_path,status)
    # Aggregate only explicitly completed scenes; never label a subset as all thirteen.
    summaries=[];pairs=[];transitions=[];diagnostics=[];inventory=[];families=[];empty_meshes=[]
    for scene in status['complete']:
        folder=args.output_root/scene/'policies'
        summaries+=read_csv(folder/'summary.csv')
        # Preserve numeric and boolean types via deterministic CSV conversion.
        for target,name in [(pairs,'pairs.csv'),(transitions,'transitions.csv'),(families,'unknown_families.csv')]:
            target.extend(read_csv(folder/name))
        with gzip.open(folder/'source_diagnostics.csv.gz','rt',encoding='utf-8-sig',newline='') as stream:
            diagnostics.extend(csv.DictReader(stream))
        item=json.loads((folder/'inventory.json').read_text(encoding='utf-8'))
        item.setdefault('empty_mesh_references',0);item.setdefault('source_mesh_references',item['nodes'])
        inventory.append(item)
        if (folder/'excluded_empty_meshes.csv').exists():
            empty_meshes.extend([dict(scene=scene,**r) for r in read_csv(folder/'excluded_empty_meshes.csv')])
    def typed(rows):
        text_fields={'scene','variant','mode','tree','kind','reference','comparison','family',
            'node_index','mesh_index','node_name','N_candidate','C_candidate','C_family','C_status',
            'C_reason','name_reference_classes','name_reference_role','name_reference_group'}
        for row in rows:
            for k,v in list(row.items()):
                if k in text_fields:continue
                if v=='':row[k]=None
                elif v in ('True','False'):row[k]=v=='True'
                else:
                    try:row[k]=int(v)
                    except (ValueError,TypeError):
                        try:row[k]=float(v)
                        except (ValueError,TypeError):pass
        return rows
    for rows in (summaries,pairs,transitions,diagnostics,families):typed(rows)
    if summaries:
        summaries+=pooled(summaries)
        # No same-name self-evaluation when M is an input.
        for r in summaries:
            r['name_reference_is_accuracy']=False
            if r['leaf_names']:
                for k in list(r):
                    if k.startswith(('name_consistency','name_group_macro','name_object_coverage')):
                        r[k]=None
        profiles,lowtypes=uncertainty_tables(diagnostics,taxonomy)
        table(args.report_dir/'all_conditions_summary.csv',summaries)
        table(args.report_dir/'main_182_comparison.csv',[r for r in summaries if r['kind']=='main'])
        table(args.report_dir/'tree_pairs.csv',pairs)
        table(args.report_dir/'priority_transitions.csv',transitions)
        table(args.report_dir/'source_confidence_summary.csv',profiles)
        table(args.report_dir/'low_confidence_candidate_types.csv',lowtypes)
        table(args.report_dir/'scene_inventory.csv',inventory)
        table(args.report_dir/'excluded_empty_meshes.csv',empty_meshes)
        table(args.report_dir/'unknown_geometry_families.csv',families)
        macros=[]
        for variant in sorted({r['variant'] for r in summaries}):
            rows=[r for r in summaries if r['variant']==variant and r['scene']!='ALL']
            r=dict(variant=variant,scenes=len(rows))
            for k in ('object_unresolved_fraction','unknown_function_fraction','mean_priority',*[f'P{i}_node_fraction' for i in range(1,11)]):
                r[k]=sum(x[k] for x in rows)/len(rows)
            macros.append(r)
        table(args.report_dir/'scene_macro_comparison.csv',macros)
        render_report(args.report_dir,summaries,pairs,inventory,profiles,lowtypes,families)
    status.update(status='partial_failure' if status['failed'] else 'complete',
        scenes_completed=len(inventory),unique_nodes=sum(r['nodes'] for r in inventory),
        actual_policy_files=len(inventory)*23,actual_policy_rows=sum(r['nodes'] for r in inventory)*23,
        source_mesh_references=sum(r['source_mesh_references'] for r in inventory),excluded_empty_meshes=len(empty_meshes),
        classification_accuracy_measured=False,renderer_quality_or_performance_measured=False,deployed=False)
    save(status_path,status)
    print(json.dumps(status,ensure_ascii=False,indent=2),flush=True)
    return bool(status['failed'])


if __name__=='__main__':
    raise SystemExit(main())
