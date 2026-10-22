"""Write numerical conclusions without treating coverage or policy tiers as truth."""
import argparse
import json
from pathlib import Path
from visualize_thirteen_scene_ablation import read
from run_thirteen_scene_ablation import DEFAULT_REPORT, save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-dir',type=Path,default=DEFAULT_REPORT)
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
        N_unresolved=n['object_unresolved_fraction'],C_unresolved=c['object_unresolved_fraction'],
        NC_unresolved=nc['object_unresolved_fraction'],NC_conflicts=int(nc['identity_conflict_count']),
        NC_conflict_fraction=nc['identity_conflict_fraction'],NCG_H0_unknown_function=g0['unknown_function_fraction'],
        NCG_H1_unknown_function=g1['unknown_function_fraction'],tree=trees,
        tree_priority_changed_fraction=trees['priority_changed']/total,geometry_pairs=geom,
        NCG_H0_requested_far_fraction=g0['requested_far_triangle_fraction'],NCG_H1_requested_far_fraction=g1['requested_far_triangle_fraction'],
        largest_tree_fraction=[dict(scene=r['scene'],title=labels[r['scene']],fraction=r['priority_changed_fraction'],count=int(r['priority_changed'])) for r in tree_rank[:3]],
        no_accuracy_or_rendering_claim=True)
    save(folder/'numerical_findings.json',findings)
    pct=lambda x:f'{100*x:.2f}%'
    lines=['## 数值结果的综合分析','',
        f"1. **监督与开放候选存在覆盖互补，同时保留分歧。** 关闭上下文时，N、C的无对象级输出比例分别为{pct(n['object_unresolved_fraction'])}、{pct(c['object_unresolved_fraction'])}，"
        f"联合NC后为{pct(nc['object_unresolved_fraction'])}，相对N下降{100*(n['object_unresolved_fraction']-nc['object_unresolved_fraction']):.2f}个百分点。"
        f"NC仍有{int(nc['identity_conflict_count']):,}个分歧节点（{pct(nc['identity_conflict_fraction'])}）。这说明候选覆盖发生变化，不证明这些新增候选正确。",'']
    for r in geom:
        a=index['ALL',r['reference']];b=index['ALL',r['comparison']]
        lines.append(f"2. **{r['reference']}→{r['comparison']}的几何预算变化。** {r['priority_changed']:,}个节点改变P档位（{pct(r['priority_changed_fraction'])}）；"
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
    # Use separate numbered labels for the three geometry pair paragraphs.
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
