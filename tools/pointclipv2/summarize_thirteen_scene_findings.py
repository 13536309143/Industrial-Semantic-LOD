"""Write numerical conclusions without treating coverage or policy tiers as truth."""
import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from visualize_thirteen_scene_ablation import read
from run_thirteen_scene_ablation import DEFAULT_REPORT, DEFAULT_OUTPUT, save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-dir',type=Path,default=DEFAULT_REPORT)
    p.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT)
    args=p.parse_args();folder=args.report_dir
    rows=read(folder/'all_conditions_summary.csv');pairs=read(folder/'tree_pairs.csv')
    inventory=read(folder/'scene_inventory.csv');index={(r['scene'],r['variant']):r for r in rows}
    total=sum(int(r['nodes']) for r in inventory)
    labels={r['scene']:r['title'] for r in inventory}
    n,c,nc,g0,g1=[index['ALL',v] for v in ['N_H0','C_H0','NC_H0','NCG_H0','NCG_H1']]
    hp=[r for r in pairs if r['mode']=='NCG' and r['reference']=='H0' and r['comparison']=='H1']
    trees={k:int(sum(r[k] for r in hp)) for k in ['priority_changed','priority_raised','priority_lowered','function_changed','identity_changed','action_changed']}
    tree_rank=sorted(hp,key=lambda r:r['priority_changed_fraction'],reverse=True)
    geom=[]
    for left,right in [('N_H0','NG_H0'),('C_H0','CG_H0'),('NC_H0','NCG_H0')]:
        group=[r for r in pairs if r['reference']==left and r['comparison']==right]
        count=sum(int(r['priority_changed']) for r in group)
        geom.append(dict(reference=left,comparison=right,priority_changed=count,priority_changed_fraction=count/total))
    findings=dict(scenes=len(inventory),nodes=total,source_triangle_instances=int(sum(r['source_faces'] for r in inventory)),
        source_mesh_references=int(sum(r.get('source_mesh_references',r['nodes']) for r in inventory)),
        excluded_empty_meshes=int(sum(r.get('empty_mesh_references',0) for r in inventory)),
        N_unresolved=n['object_unresolved_fraction'],C_unresolved=c['object_unresolved_fraction'],
        NC_unresolved=nc['object_unresolved_fraction'],NC_conflicts=int(nc['identity_conflict_count']),
        NC_conflict_fraction=nc['identity_conflict_fraction'],NCG_H0_unknown_function=g0['unknown_function_fraction'],
        NCG_H1_unknown_function=g1['unknown_function_fraction'],tree=trees,
        tree_priority_changed_fraction=trees['priority_changed']/total,geometry_pairs=geom,
        NCG_H0_requested_far_fraction=g0['requested_far_triangle_fraction'],NCG_H1_requested_far_fraction=g1['requested_far_triangle_fraction'],
        largest_tree_fraction=[dict(scene=r['scene'],title=labels[r['scene']],fraction=r['priority_changed_fraction'],count=int(r['priority_changed'])) for r in tree_rank[:3]],
        no_accuracy_or_rendering_claim=True)
    decomposition=[]
    for mode in ['N','C','G','NC','NG','CG','NCG']:
        item=dict(mode=mode)
        for label,left,right in [('topology','H0','HT'),('ancestor_names','HT','H1')]:
            group=[r for r in pairs if r['mode']==mode and r['reference']==left and r['comparison']==right]
            assert len(group)==len(inventory)
            for field in ['priority_changed','function_changed','identity_changed','action_changed']:
                item[label+'_'+field]=int(sum(r[field] for r in group))
        decomposition.append(item)
    findings['tree_decomposition']=decomposition
    tiers=[dict(priority=f'P{i}',nodes=int(g1[f'P{i}_count']),
        node_fraction=g1[f'P{i}_node_fraction'],source_triangle_fraction=g1[f'P{i}_face_fraction']) for i in range(1,11)]
    findings['NCG_H1_priorities']=tiers
    save(folder/'numerical_findings.json',findings)
    pct=lambda x:'<0.01%' if 0<x<.0001 else f'{100*x:.2f}%'
    lines=['## 数值结果的综合分析','',
        f"1. **监督与开放候选存在覆盖互补，同时保留分歧。** 关闭上下文时，N、C的无对象级输出比例分别为{pct(n['object_unresolved_fraction'])}、{pct(c['object_unresolved_fraction'])}，"
        f"联合NC后为{pct(nc['object_unresolved_fraction'])}，相对N下降{100*(n['object_unresolved_fraction']-nc['object_unresolved_fraction']):.2f}个百分点。"
        f"NC仍有{int(nc['identity_conflict_count']):,}个分歧节点（{pct(nc['identity_conflict_fraction'])}）。这说明候选覆盖发生变化，不证明这些新增候选正确。",'']
    lines += ['2. **几何分支主要影响保护预算。** 下列成对结果使用相同节点与神经证据。', '']
    for r in geom:
        a=index['ALL',r['reference']];b=index['ALL',r['comparison']]
        lines.append(f"**几何对照{r['reference']}→{r['comparison']}。** {r['priority_changed']:,}个节点改变P档位（{pct(r['priority_changed_fraction'])}）；"
            f"功能未知由{pct(a['unknown_function_fraction'])}变为{pct(b['unknown_function_fraction'])}，"
            f"无对象级输出由{pct(a['object_unresolved_fraction'])}变为{pct(b['object_unresolved_fraction'])}。"
            '几何分支主要提供形态与保真证据，预算改变和对象识别应分开评价。\n')
    lines.extend([f"3. **装配上下文在完整融合中的介入范围。** H0→H1改变{trees['priority_changed']:,}个节点的P档位（{pct(trees['priority_changed']/total)}），"
        f"其中升档{trees['priority_raised']:,}、降档{trees['priority_lowered']:,}；功能判断改变{trees['function_changed']:,}，对象身份改变{trees['identity_changed']:,}。"
        f"功能未知比例由{pct(g0['unknown_function_fraction'])}变为{pct(g1['unknown_function_fraction'])}。"
        '身份改变可能包括保守撤回，不能都解释为纠错。','',
        f"4. **保护与请求预算的代价。** 完整融合请求远档面数比例由{pct(g0['requested_far_triangle_fraction'])}变为{pct(g1['requested_far_triangle_fraction'])}。"
        '这一数值是原始实例面数加权的请求值；若升档增加预算，后续应核验其是否改善关键结构，不能仅以介入节点多判断更优。','',
        '5. **树辅助的场景差异。** 按改变P的节点比例，介入较多的场景为：'+
        '；'.join(f"{r['scene']} {labels[r['scene']]} {pct(r['priority_changed_fraction'])}（{int(r['priority_changed']):,}节点）" for r in tree_rank[:3])+
        '。绝对数量与场景规模有关，应同时查看比例、源三角形占比及逐节点原因。','',
        '上述结论适用于固定规则下的完整场景诊断。没有独立人工真值或实际LOD/渲染测量，不能转换成分类准确率、关键件识别率、显存节省或帧率增益。',''])
    lines += ['### 拓扑与祖先名称的贡献拆分', '',
        '| 组合 | H0→HT改变P | HT→H1改变P | H0→HT功能改变 | HT→H1功能改变 |',
        '|---|---:|---:|---:|---:|']
    for item in decomposition:
        fields=['topology_priority_changed','ancestor_names_priority_changed',
            'topology_function_changed','ancestor_names_function_changed']
        lines.append('| '+item['mode']+' | '+' | '.join(f"{item[k]:,}（{pct(item[k]/total)}）" for k in fields)+' |')
    lines += ['', 'H0→HT固定名称线索，仅启用父组拓扑及尺度；HT→H1固定拓扑，再加入受范围限制的祖先名称。'
        '两段变化可能发生在相同节点，并可能相互抵消，不能相加替代H0→H1的净变化。'
        '若主要变化来自第二段，论文应表述为装配上下文或名称语义辅助，而不能全部归因于结构树拓扑。', '']
    largest=max(tiers,key=lambda r:r['nodes']);costly=max(tiers,key=lambda r:r['source_triangle_fraction'])
    empty=[r['priority'] for r in tiers if not r['nodes']]
    lines += ['### 完整融合的档位集中与几何负担', '',
        f"NCG_H1按节点最多的档位为{largest['priority']}，共{largest['nodes']:,}个（{pct(largest['node_fraction'])}）；"
        f"按源三角形实例占比最高的是{costly['priority']}（{pct(costly['source_triangle_fraction'])}）。"
        '节点占比反映分配频度，源三角形占比反映原始几何负担，两者均不能替代运行时可见性或资源测量。',
        ('本轮完整融合中未触发的档位为'+ '、'.join(empty)+'。不应为填满十档而人为重分配；应依据其触发条件判断是否合理。')
        if empty else '完整融合触发了全部十档；各档占比不需要均匀，是否得当仍需结合对应节点的结构与视觉验证。', '']
    lines += ['| 档位 | 节点数 | 节点占比 | 源三角形实例占比 |','|---|---:|---:|---:|']
    for item in tiers:
        lines.append(f"| {item['priority']} | {item['nodes']:,} | {pct(item['node_fraction'])} | {pct(item['source_triangle_fraction'])} |")
    lines += ['', '小于0.01%的非零比例单独显示，避免将极少量档位误读为未触发。', '']
    # Inspect unknown and routine-fastener cohorts from original full rows,
    # retaining distinct shapes rather than a single unknown fallback category.
    cohort=defaultdict(Counter);family=defaultdict(Counter);role_changes=defaultdict(Counter)
    for item in inventory:
        scene=item['scene']
        before_roles={}
        for variant in ['NCG_H0','NCG_H1']:
            path=args.output_root/scene/'policies'/(variant+'.csv.gz')
            with gzip.open(path,'rt',encoding='utf-8-sig',newline='') as stream:
                for row in csv.DictReader(stream):
                    tier=int(row['lod_priority'].split('_')[0][1:]);role=row['function_role']
                    key=(row['node_index'],row['mesh_index'])
                    if variant=='NCG_H0':before_roles[key]=role
                    else:
                        for s in [scene,'ALL']:role_changes[s][(before_roles[key],role)]+=1
                    for s in [scene,'ALL']:
                        cohort[(s,variant,role)][tier]+=1
                        if role=='unknown':family[(s,variant,row['geometry_budget_family'])][tier]+=1
    out=[]
    for (scene,variant,role),hist in sorted(cohort.items()):
        r=dict(scene=scene,variant=variant,function_role=role,nodes=sum(hist.values()))
        r.update({f'P{i}_count':hist[i] for i in range(1,11)});out.append(r)
    if out:
        with (folder/'function_role_priority_distribution.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(out[0]));writer.writeheader();writer.writerows(out)
    shape_labels={'compact_simple':'简单紧凑形态','compact_curved':'曲面紧凑形态','compact_featured':'局部特征紧凑形态',
        'plate':'板状形态','rod':'杆状形态','elongated_sheet':'细长片状形态','distributed_components':'分布复合形态','invalid_geometry':'几何无效'}
    shape_rows=[]
    lines += ['', '### 功能未知件的几何拆分', '',
        '| 几何分型 | H0未知节点 | H1未知节点 | H1中P2—P3 | H1中P4—P5 | H1中P6—P7 |',
        '|---|---:|---:|---:|---:|---:|']
    for f in sorted({k[2] for k in family}):
        a=family.get(('ALL','NCG_H0',f),Counter());b=family.get(('ALL','NCG_H1',f),Counter())
        shape_rows.append(dict(family=f,H0_count=sum(a.values()),H1_count=sum(b.values())))
        lines.append('| '+shape_labels.get(f,f)+' | '+' | '.join(str(v) for v in [sum(a.values()),sum(b.values()),b[2]+b[3],b[4]+b[5],b[6]+b[7]])+' |')
    unknown_count=sum(sum(v.values()) for k,v in family.items() if k[:2]==('ALL','NCG_H1'))
    assert unknown_count==int(g1['unknown_function_count'])
    fasteners=cohort.get(('ALL','NCG_H1','standard_fastener'),Counter())
    lines += ['',f"完整融合H1共有{unknown_count:,}个功能未知节点，按几何分型与P档位分别统计，避免将未知功能误作同一种形状。"
        f"普通紧固功能节点为{sum(fasteners.values()):,}个，其P1—P3、P4—P5、P6—P10数量分别为"
        f"{sum(fasteners[i] for i in range(1,4)):,}、{sum(fasteners[i] for i in range(4,6)):,}、{sum(fasteners[i] for i in range(6,11)):,}。"
        '这些是规则给出的功能候选与预算，不是人工确认的零件数量。完整角色分布见function_role_priority_distribution.csv。','']
    findings['unknown_geometry_families']=shape_rows
    findings['NCG_H1_standard_fastener_nodes']=sum(fasteners.values())
    findings['NCG_H1_unknown_function_nodes']=unknown_count
    transitions=[dict(scene=s,before_function=a,after_function=b,nodes=count)
        for s,hist in sorted(role_changes.items()) for (a,b),count in sorted(hist.items())]
    with (folder/'function_role_transitions.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(transitions[0]));writer.writeheader();writer.writerows(transitions)
    role_matrix=role_changes['ALL'];assert sum(role_matrix.values())==total
    withdrawn=sum(count for (a,b),count in role_matrix.items() if a!='unknown' and b=='unknown')
    resolved=sum(count for (a,b),count in role_matrix.items() if a=='unknown' and b!='unknown')
    assert withdrawn-resolved==int(g1['unknown_function_count'])-int(g0['unknown_function_count'])
    assert sum(count for (a,b),count in role_matrix.items() if a!=b)==trees['function_changed']
    findings['tree_function_withdrawn_to_unknown']=withdrawn
    findings['tree_function_resolved_from_unknown']=resolved
    lines += [f"H0→H1中，{withdrawn:,}个节点的功能候选撤回为未知，{resolved:,}个节点由未知得到功能候选。"
        '两项分别统计，并不预先认定新增或撤回正确；完整功能角色迁移见function_role_transitions.csv。', '']
    topology=next(r for r in decomposition if r['mode']=='NCG')
    heavy=[r for r in tiers if r['priority'] in {'P6','P8'}]
    lines += ['### 对当前融合机制的判断', '',
        f"本轮完整融合中，拓扑与父组尺度改变{topology['topology_priority_changed']:,}个节点的档位，"
        f"随后加入祖先名称改变{topology['ancestor_names_priority_changed']:,}个节点。"
        '主要介入来源是局部尺度与保护预算调整；祖先名称在较小范围内细化已有功能候选。', '',
        f"功能未知仍有{unknown_count:,}个（{pct(g1['unknown_function_fraction'])}），且本轮树辅助没有使未知节点获得新的功能候选。"
        '因此，当前结果适合论证未知功能下的几何分型与保守预算分配，实际语义识别效果仍需要独立标注检验。', '',
        f"P6与P8合计仅占{pct(sum(r['node_fraction'] for r in heavy))}的节点，却对应"
        f"{pct(sum(r['source_triangle_fraction'] for r in heavy))}的原始三角形实例。"
        '后续实际LOD与渲染对照应重点检查这些高面数构件的简化结果，同时验证升档增加的保护预算是否换来了结构或画质收益。', '']
    save(folder/'numerical_findings.json',findings)
    text='\n'.join(lines)
    target=folder/'十三模型全量消融与结构树对照分析.md'
    original=target.read_text(encoding='utf-8')
    if '\n## 数值结果的综合分析' in original:
        before,rest=original.split('\n## 数值结果的综合分析',1)
        if '\n## ' in rest:
            original=before+'\n## '+rest.split('\n## ',1)[1]
        else:original=before
    insert='\n## 图表与逐模型报告'
    if insert in original:
        original=original.replace(insert,'\n'+text+insert,1)
    else:original+='\n'+text
    target.write_text(original,encoding='utf-8')
    print(json.dumps(findings,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
