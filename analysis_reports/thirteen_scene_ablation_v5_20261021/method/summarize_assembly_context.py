"""Evidence-based Chinese report of the six-scene assembly/geometry policy experiments."""
import argparse
import json
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np
from audit_named_scene import read_csv,write_csv
from artifact_cache import file_hash
from run_pointclipv2_zeroshot_glb import read_glb_chunks

ROOT=Path(__file__).resolve().parents[2]
TITLES={'a':'无可用名称/装配树','b':'K31自动铆压机','c':'汽车链接器植入-折弯设备',
        'd':'手机喇叭检测线','e':'透气阀产线','f':'圆形电容穿心焊接机'}


def truth(value):return str(value).lower()=='true'
def pnum(row):return int(row['lod_priority'].split('_')[0][1:])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--report-dir',type=Path,required=True);p.add_argument('--models',nargs='+',default=list('abcdef'))
    args=p.parse_args();report=args.report_dir
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    summaries=json.loads((report/'final_guarded/summary.json').read_text(encoding='utf-8'))
    sindex={(r['scene'],r['variant']):r for r in summaries}
    feedback=read_csv(report/'final_guarded/geometry_feedback.csv')
    fidx=defaultdict(list)
    for r in feedback:fidx[(r['scene'],r['variant'])].append(r)
    inventory=[];overview=[];effects=[];cases=[];metrics=[];inputs=[args.manifest,report/'final_guarded/provenance.json']
    for name in args.models:
        glb=Path(manifest[name]['glb']);data,_,_=read_glb_chunks(glb)
        folder=args.output_root/name/'final_guarded'
        rows={v:read_csv(folder/(v+'.csv'))for v in ['v2_NCG','v3_NCG','v3_NCGH','v3_NCGM','v3_NCGHM']}
        base,blind,full=[sindex[(name,v)]for v in ('v2_NCG','v3_NCGH','v3_NCGHM')]
        inv=dict(scene=name,title=TITLES.get(name,name),nodes=len(rows['v2_NCG']),
            assembly_nodes=sum(bool(n.get('children'))for n in data['nodes']),
            unique_meshes=len({r['mesh_index']for r in rows['v2_NCG']}),
            name_reference_nodes=base['named_proxy_nodes'],name_reference_groups=base['name_groups'],
            glb_sha256=manifest[name]['glb_sha256'],previously_tested_identical_scene=name in 'bcde',
            frozen_confirmation=False)
        inventory.append(inv)
        altered=0
        for old,new in zip(rows['v3_NCGM'],rows['v3_NCGHM']):
            if old['node_index']!=new['node_index']:raise ValueError('Counterfactual node mismatch')
            if old['function_role']!=new['function_role']:
                altered+=1
                effects.append(dict(scene=name,node_index=new['node_index'],node_name=new['node_name'],
                    object_identity=new['resolved_object_class'],leaf_only_role=old['function_role'],tree_role=new['function_role'],
                    leaf_only_priority=pnum(old),tree_priority=pnum(new),context_json=new['assembly_context_json']))
        selected=fidx[(name,'v3_NCGHM')]
        total_faces=sum(int(r['source_faces'])for r in selected)
        for variant in ('v3_NCG','v3_NCGH','v3_NCGHM'):
            chosen=fidx[(name,variant)]
            weight=np.array([int(r['source_faces'])for r in chosen]);before=np.array([float(r['incumbent_error'])for r in chosen]);after=np.array([float(r['selected_error'])for r in chosen])
            metrics.append(dict(scene=name,variant=variant,nodes=len(chosen),source_faces=int(weight.sum()),
                incumbent_triangle_ratio=sum(int(r['incumbent_actual_faces'])for r in chosen)/weight.sum(),
                final_triangle_ratio=sum(int(r['selected_actual_faces'])for r in chosen)/weight.sum(),
                incumbent_mean_error=float(before.mean()),final_mean_error=float(after.mean()),
                incumbent_face_weighted_error=float(np.average(before,weights=weight)),
                final_face_weighted_error=float(np.average(after,weights=weight)),
                incumbent_p95_error=float(np.quantile(before,.95)),final_p95_error=float(np.quantile(after,.95)),
                incumbent_axis_shrink_nodes=sum(float(r['incumbent_axis_shrink'])>.1 for r in chosen),
                final_axis_shrink_nodes=sum(float(r['selected_axis_shrink'])>.1 for r in chosen),
                incumbent_component_loss_nodes=sum(int(r['incumbent_component_loss'])>0 for r in chosen),
                final_component_loss_nodes=sum(int(r['selected_component_loss'])>0 for r in chosen),
                guard_applied=sum(truth(r['guard_applied'])for r in chosen),
                residual_review=sum(bool(r['residual_regression'])for r in chosen),
                ground_truth_accuracy=False,cluster_lod_benchmark=False))
        m=metrics[-1]
        overview.append(dict(scene=name,nodes=inv['nodes'],reference_nodes=base['named_proxy_nodes'],
            v2_blind_identity_consistent=base['named_object_consistent'],v2_blind_identity_covered=base['named_object_coverage'],
            v3_blind_identity_consistent=blind['named_object_consistent'],v3_blind_identity_covered=blind['named_object_coverage'],
            baseline_named_critical_low_priority=base['named_critical_low_priority'],
            blind_named_critical_low_priority=blind['named_critical_low_priority'],
            metadata_named_critical_low_priority=full['named_critical_low_priority'],
            tree_function_changes_over_leaf_only=altered,tree_relations_recorded=full['tree_context_applied'],
            guard_adjustments=m['guard_applied'],final_surface_component_loss_nodes=m['final_component_loss_nodes'],
            baseline_actual_triangle_ratio=m['incumbent_triangle_ratio'],final_actual_triangle_ratio=m['final_triangle_ratio'],
            baseline_proxy_error=m['incumbent_mean_error'],final_proxy_error=m['final_mean_error'],
            auxiliary_candidates=full['auxiliary_candidates']))
        selected_cases=[r for r in effects if r['scene']==name]
        seen=set()
        for r in selected_cases:
            signature=(r['leaf_only_role'],r['tree_role'])
            if signature not in seen:
                seen.add(signature);cases.append(r)
            if len(seen)>=3:break
        inputs += [glb,*[folder/(v+'.csv')for v in rows],args.output_root/name/'r4_final_policy/geometry_proxy.provenance.json']
    write_csv(report/'scene_inventory.csv',inventory)
    write_csv(report/'overall_comparison.csv',overview)
    write_csv(report/'geometry_final_comparison.csv',metrics)
    if effects:write_csv(report/'tree_function_changes.csv',effects)
    total=sum(r['nodes']for r in inventory);ref=sum(r['reference_nodes']for r in overview)
    oldmatch=sum(r['v2_blind_identity_consistent']for r in overview);newmatch=sum(r['v3_blind_identity_consistent']for r in overview)
    baseline_low=sum(r['baseline_named_critical_low_priority']for r in overview)
    blind_low=sum(r['blind_named_critical_low_priority']for r in overview)
    final_low=sum(r['metadata_named_critical_low_priority']for r in overview)
    revisions=[]
    for stage in ('r1_geometry_context','r2_scoped_context'):
        path=report/stage/'summary.json'
        if path.exists():
            data=json.loads(path.read_text(encoding='utf-8'))
            revisions.append(dict(round=stage,development_scenes='b,c,d,e',
                tree_relations=sum(r['tree_context_applied']for r in data if r['variant']=='v3_NCGH'),
                low_named_critical=sum(r['named_critical_low_priority']for r in data if r['variant']=='v3_NCGH')))
    summary=dict(scenes=inventory,total_nodes=total,weak_reference_nodes=ref,blind_baseline_matches=oldmatch,
        blind_new_matches=newmatch,named_critical_low_baseline=baseline_low,named_critical_low_blind=blind_low,
        named_critical_low_with_metadata=final_low,total_tree_function_changes=len(effects),
        geometry_guard_adjustments=sum(r['guard_adjustments']for r in overview),
        unresolved_geometry_regressions=sum(r['residual_review']for r in metrics if r['variant']=='v3_NCGHM'),
        config_sha256=file_hash(ROOT/'tools/pointclipv2/assembly_context_config.json'),
        development_rounds=revisions,real_accuracy_measured=False,renderer_lod_measured=False)
    (report/'overall_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    # Standalone scientific plots, with English labels for portable font rendering.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen=[r for r in metrics if r['variant']=='v3_NCGHM']
    fig,axes=plt.subplots(1,3,figsize=(13,3.7));x=np.arange(len(chosen));width=.35
    for ax,(a,b,title)in zip(axes,[('incumbent_triangle_ratio','final_triangle_ratio','Actual triangle ratio (proxy)'),
                                  ('incumbent_mean_error','final_mean_error','Mean normalized QEM error'),
                                  ('incumbent_axis_shrink_nodes','final_axis_shrink_nodes','Nodes with >10% axis shrink')]):
        ax.bar(x-width/2,[r[a]for r in chosen],width,label='Incumbent v2')
        ax.bar(x+width/2,[r[b]for r in chosen],width,label='Assembly + CAD + guard')
        ax.set_xticks(x,[r['scene']for r in chosen]);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(report/'geometry_comparison.png',dpi=180);fig.savefig(report/'geometry_comparison.svg');plt.close(fig)
    lines=['# 六模型装配上下文与几何反馈策略实验',
        '',f'六个GLB共 **{total:,} 个网格节点**，已完成全量监督/开放推断、表面几何描述、策略消融和单网格简化诊断。没有人工类别或功能真值，名称只能作为弱参考；没有执行Vulkan固定轨迹或完整cluster LOD质量/FPS基准。',
        '', '## 结论', '',
        '本轮优化的重点是把“物体类型、装配作用、几何保真要求”分开决策。结构树补充关系与工位上下文，不把父类别复制给子零件；几何分支可以在类别未知时提出截面、轮廓和分离部件保护要求。',
        '',f'在可保守映射到监督类别的 {ref} 个名称弱参考节点中，盲推断最终身份与名称一致数为旧版 {oldmatch}、新版加树 {newmatch}。这不是全部节点的OA，也不是CLIP候选Top-1统计；元数据辅助身份没有计入。**此次改造不能宣称视觉分类准确率显著提高。**',
        '',f'按照名称提供的功能弱参考，关键类别却落入P1–P3的节点数由 {baseline_low} 降为盲配置 {blind_low}、名称辅助配置 {final_low}。这是策略一致性检查，依赖既定功能重要性假设，不能当作分类准确率或真实功能保护率。',
        '', '## 数据与验证范围', '',
        '| 文件 | 场景 | 网格节点 | 装配节点 | 名称参考节点/名称组 |', '|---|---|---:|---:|---:|']
    for r in inventory:lines.append(f"| {r['scene']}.glb | {r['title']} | {r['nodes']} | {r['assembly_nodes']} | {r['name_reference_nodes']} / {r['name_reference_groups']} |")
    lines += ['', 'b、c、d、e与此前四场景文件逐字节相同，不能算四个新独立测试集。a没有可用装配树或名称映射，采用几何与视觉回退；不虚构父结构。b–e用于前期开发；第三轮冻结后对f确认，发现“折弯头”名称规则缺口，第四轮补齐并全量回放。因此六场景均属于开发/诊断数据，本轮没有保留严格独立的最终测试集。重复零件不独立，另给出名称组宏平均；不能以重复实例放大统计显著性。',
        '', '## 反复实验如何改变算法', '',
        '1. **R0：现有v2。** 保留各证据配置，作为迁移参照。',
        '2. **R1：几何与工位上下文初版。** 在b–e运行，发现大工位向普通紧固件扩散“精密装配”作用；测试还检出了轴对齐包围盒归一化的旋转依赖。历史输入、源代码和结果已保存。',
        '3. **R2：局部装配约束。** 改用旋转不变的RMS半径归一化，排除覆盖整个设备的父节点；精密紧固关系限定局部装配数量和相对尺寸。但全量QEM诊断发现部分场景实际误差/分离部件丢失变差，说明“保护下限达标”不能证明简化变好。',
        '4. **R3＋实测反馈：冻结部署候选。** 边缘复杂度触发从14收紧到24；有名称与平面几何共同支持的场景参考件单列。对NCG/NCGH/NCGHM额外执行迁移检查，若QEM误差超过旧NCG的1.05倍加0.002、明显增加轴向收缩或增加显著部件丢失，则提高该节点的保留比例，必要时检查更高比例。身份与功能角色保持独立，不通过篡改类型来解释几何风险。',
        '5. **R4：名称覆盖修正后全量复跑。** f中的“折弯头”仍被分到普通紧固件，补充通用工艺头名称。若功能名称与紧固件身份冲突，输出身份待核验、功能候选及保护策略，不凭名字虚构精确工具型号；所有数值阈值保持第三轮一致。',
        '', '这些数值是冻结的工程规则，不是概率校准。迁移检查把现有NCG当几何参照，并不把旧类别当真值，也不保证绝对几何质量合格。该附加阶段只用于含N/C/G的完整配置；纯C/G消融保持独立，不能把完整系统收益解释成G的单因素因果贡献。',
        '', '## 结构树的实际作用', '',
        f'相对于“同样使用叶名称、但关闭父结构”的配置，父结构改变了 **{len(effects)} 个节点的功能角色**。记录上下文不一定改变P级；薄壁/细长等几何下限仍可能主导最终动作。',
        '', '| 场景 | 名称参考节点 | 旧盲身份一致/覆盖 | 新盲身份一致/覆盖 | 父结构额外改变功能数 | 实测反馈调整数 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in overview:lines.append(f"| {r['scene']} | {r['reference_nodes']} | {r['v2_blind_identity_consistent']}/{r['v2_blind_identity_covered']} | {r['v3_blind_identity_consistent']}/{r['v3_blind_identity_covered']} | {r['tree_function_changes_over_leaf_only']} | {r['guard_adjustments']} |")
    lines += ['', '分子为与名称弱参考一致，分母为该参考集合中输出可映射对象身份的节点数；参考节点总数另列。不能只看已覆盖子集的一致比例。名称辅助配置直接使用名称，因此不参加这张盲身份评分。',
        '', '| 场景 | 零件名称 | 仅叶名称的作用 | 加父结构的作用 | 父结构 |', '|---|---|---|---|---|']
    for r in cases[:15]:
        ctx=json.loads(r['context_json'])
        lines.append(f"| {r['scene']} | {r['node_name'].replace('|','/')} | {r['leaf_only_role']} | {r['tree_role']} | {ctx['name'].replace('|','/')} |")
    lines += ['', '送料工位里的螺母只有在物料名称、对象类型和局部尺度有共同支持时才标作候选工件；不会把送料机构所有螺钉都当产品。CCD安装板等复合名称先认安装件，再根据检测装配赋予安装作用。没有接触/运动/可见性测量，仍不能自动证明某个零件就是工艺关键件。',
        '', '## 全量单网格简化诊断', '',
        '| 场景 | 旧实际三角形比例 | 新实际三角形比例 | 旧平均QEM误差 | 新平均QEM误差 | 旧/新显著部件丢失节点 | 残余回归复核 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in chosen:lines.append(f"| {r['scene']} | {r['incumbent_triangle_ratio']:.2%} | {r['final_triangle_ratio']:.2%} | {r['incumbent_mean_error']:.5f} | {r['final_mean_error']:.5f} | {r['incumbent_component_loss_nodes']} / {r['final_component_loss_nodes']} | {r['residual_review']} |")
    lines += ['', 'QEM误差以两倍RMS半径归一化，是meshoptimizer报告的误差代理，并非Hausdorff距离或人工可见质量。部件丢失指焊接后表面积占比至少0.1%的分离表面分量消失；不代表孔洞保持率。轴向收缩按源几何主轴包围范围检查。所有节点参与，重复网格/相同比例复用数值结果。',
        '', '各策略实际达到的三角形预算不同，上表不是同预算优越性证明。R2/R3还保存了相同请求预算的对照，但实际达成预算不同，不能据此声称公平预算下全面领先。最终反馈阶段优先避免相对旧策略退步，保真改善需要与增加的几何成本一起评估。独立探针没有应用前端的特征软约束/硬锁、cluster层次或运行时剔除；新增特征下限已接入前端，但必须重建LOD后另做实际测试。',
        '', 'f中旧策略已有的11个显著表面分量丢失节点在最终策略中仍存在；迁移检查未增加这项丢失，不等于已消除绝对几何质量问题。名称及层次也不足以证明实际运动、装配接触或运行时可见性。',
        '', '![几何诊断对照](geometry_comparison.png)',
        '', '## 接入与复现', '',
        '运行项目根目录 `run_assembly_context_comparison.bat`，使用gpt-pointnext与PointCLIPV2_LOD。支持 `--policies-only` 复用已核验推断，`--no-export` 仅分析。每个场景失败会记录并继续，存在失败时不发布完整策略。原始输出在 `lod_analysis_outputs/assembly_context_v3_20261020`。',
        '', '六个既有配置继续使用各自证据；第七项加载装配树＋叶名称＋几何反馈的完整策略。第六项保持无名称/无父结构。导出前核验来源和完整节点集合，旧CSV先备份。需重新打开场景、选择第七项并重建LOD。对于共用网格，节点作用可不同，但网格几何仍按最严格实例合并，不能声称已生成独立的每实例几何。',
        '', '[逐场景总表](overall_comparison.csv)、[结构树作用变化](tree_function_changes.csv)、[最终几何诊断](geometry_final_comparison.csv)、[盲名称核验与名称辅助标记](final_guarded/name_weak_reference.csv)、[节点级反馈原因](final_guarded/geometry_feedback.csv)。',
        '', '论文中可表述为“装配上下文与实测几何反馈驱动的策略优化”；目前不能表述为“已验证开放识别准确率、工艺关键件识别准确率或FPS显著提升”。']
    filename=report/'六模型结构树辅助与策略优化实验报告.md'
    filename.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    inputs+=[ROOT/'tools/pointclipv2/assembly_context_config.json',Path(__file__)]
    proof=dict(inputs={str(q.resolve()):file_hash(q)for q in inputs},
        outputs={str(q.resolve()):file_hash(q)for q in report.iterdir()if q.is_file()and q.name!='provenance.json'},
        real_accuracy_measured=False,renderer_lod_measured=False)
    (report/'provenance.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Report: '+str(filename),flush=True)


if __name__=='__main__':main()
