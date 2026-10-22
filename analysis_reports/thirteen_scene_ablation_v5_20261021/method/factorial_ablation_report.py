"""Complete paired tables and research interpretation for the frozen six-scene ablation."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from audit_named_scene import write_csv
from artifact_cache import file_hash
from factorial_ablation_policy import MAIN_MODES

ROOT=Path(__file__).resolve().parents[2]


def percent(value):return '—'if value is None else f'{value:.2%}'
def aggregate_pairs(pairs):
    out=[]
    for mode in MAIN_MODES:
        rows=[r for r in pairs if r['mode']==mode and r['kind']=='main']
        n=sum(r['nodes']for r in rows)
        result=dict(scene='ALL',mode=mode,reference='H0',comparison='H1',kind='main',nodes=n)
        for key in ('priority_changed','priority_raised','priority_lowered','function_changed','identity_changed','action_changed'):
            result[key]=sum(r[key]for r in rows)
        result['mean_priority_delta']=sum(r['mean_priority_delta']*r['nodes']for r in rows)/n
        for p in range(1,11):result[f'P{p}_delta_pp']=sum(r[f'P{p}_delta_pp']*r['nodes']for r in rows)/n
        out.append(result)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir',type=Path,default=ROOT/'analysis_reports/six_scene_factorial_ablation_20261020')
    args=parser.parse_args();folder=args.report_dir
    data=json.loads((folder/'comparison_data.json').read_text(encoding='utf-8'))
    main_rows=data['main'];index={(r['scene'],r['variant']):r for r in main_rows}
    totals=[index[('ALL',f'{mode}_H{h}')]for mode in MAIN_MODES for h in (0,1)]
    pairs=[r for r in data['pairs']if r['kind']=='main']+aggregate_pairs(data['pairs'])
    pooled_pairs={r['mode']:r for r in pairs if r['scene']=='ALL'}
    macros=[];grades=[]
    for r in main_rows:
        for p in range(1,11):grades.append(dict(scene=r['scene'],variant=r['variant'],P=p,
            count=r[f'P{p}_count'],node_fraction=r[f'P{p}_node_fraction'],source_faces=r[f'P{p}_faces'],
            face_fraction=r[f'P{p}_face_fraction'],source_area=r[f'P{p}_area'],area_fraction=r[f'P{p}_area_fraction']))
    for mode in MAIN_MODES:
        for h in (0,1):
            rows=[r for r in main_rows if r['scene']!='ALL'and r['variant']==f'{mode}_H{h}']
            macros.append(dict(mode=mode,tree='on'if h else 'off',scenes=len(rows),
                **{f'P{p}_node_fraction_macro':float(np.mean([r[f'P{p}_node_fraction']for r in rows]))for p in range(1,11)},
                object_unresolved_fraction_macro=float(np.mean([r['object_unresolved_fraction']for r in rows]))))
    write_csv(folder/'priority_distribution_long.csv',grades)
    write_csv(folder/'scene_macro_comparison.csv',macros)
    write_csv(folder/'tree_pairs_with_total.csv',pairs)
    public_conf=[]
    for r in main_rows:public_conf.append({k:r[k]for k in ['scene','variant','mode','tree','nodes',
        'N_not_reliable_count','N_not_reliable_fraction','C_not_object_count','C_not_object_fraction',
        'C_abstain_count','C_coarse_count','object_unresolved_count','object_unresolved_fraction',
        'unknown_function_count','unknown_function_fraction','identity_conflict_count','shape_conflict_count',
        'review_count','name_reference_count','name_object_coverage_count','name_consistent_count',
        'name_consistency_all_reference','name_object_coverage_fraction','name_group_macro_consistency']})
    write_csv(folder/'main_confidence_and_conflicts.csv',public_conf)
    named=[{k:r[k]for k in ['scene','variant','name_reference_count','name_groups','name_object_coverage_count',
        'name_consistent_count','name_consistency_all_reference','name_consistency_covered','name_object_coverage_fraction','name_group_macro_consistency']}
        for r in main_rows]
    write_csv(folder/'blind_name_reference_comparison.csv',named)
    # Primary results have no M and no migration guard. Ancillary name-aided outputs are not scored against names.
    aux=[{k:v for k,v in r.items()if not k.startswith('name_')}for r in data['summaries']if r['kind']!='main']
    write_csv(folder/'supplementary_controls.csv',aux)
    geometry={(r['scene'],r['variant']):r for r in data['geometry']}
    lines=['# 六模型六组合与结构树开关全量消融', '',
        '本次固定神经网络权重、词库和策略阈值，复用逐文件哈希一致的全量推断；对11,000个节点重新执行融合与策略分配。主实验为六模型×六组合×树开关，共72组、132,000条节点策略记录。另有纯G、仅拓扑和叶名称辅助对照，共138组、253,000条记录。没有重训网络或改写已部署CSV。',
        '', '## 实验定义与可比性', '',
        '| 符号 | 含义 |', '|---|---|',
        '| N | 工业监督编码器候选及置信度 |','| C | 开放视觉语义候选、对象级输出/粗类别/拒识 |',
        '| G | 表面几何风险、形状兼容性和保真约束 |','| H0 | 不使用装配关系、父组尺度或祖先名称；世界变换和输入几何保持不变 |',
        '| H1 | 使用局部装配尺度与受范围限制的祖先名称上下文 |','| HT | 使用拓扑及父组尺度，不读取祖先名称，作为分解H效应的附加对照 |',
        '| M | 直接使用叶名称，仅放附表，不进入主实验 |',
        '', '六种主组合沿用现有配置：N、C、NG、CG、NC、NCG。纯G是补充参照。H1包含祖先名称，因此不能把H1的全部效果归因于纯拓扑；HT附表用于区分这两个因素。所有主配置M=0、旧NCG几何回退=0，防止回退机制把N/C证据带入单分支消融。',
        '', '原生产函数中部分上下文检查会读取原始N/C字段。本次在实验边界显式遮蔽被关闭的分支；H0也关闭先前可能隐含使用的父组相对尺寸。因此本表与上一轮“局部关闭功能上下文”的结果并非同一实验口径，不应直接混表比较。',
        '', '## P1–P10总分布', '',
        '下表按11,000个网格节点合并统计，分母是节点总数，不是六个场景占比的简单平均。每个P等级均保留，0表示确实没有分到该档。各等级的计数、三角形数和面积占比见Excel与[长表](priority_distribution_long.csv)；场景等权均值另见[宏平均](scene_macro_comparison.csv)。',
        '', '| 配置 | '+' | '.join(f'P{p}'for p in range(1,11))+' |', '|---|'+'---:|'*10]
    for r in totals:lines.append('| '+r['variant']+' | '+' | '.join(percent(r[f'P{p}_node_fraction'])for p in range(1,11))+' |')
    lines += ['', '主实验P1/P3/P4/P10均为0。这反映当前规则实际只激活部分档位：未知默认至少P5，大体量部件没有单凭尺寸进入P3，主实验关闭叶名称使场景参考件P4不激活，功能优先级表最高为P9。不能为填满十档而人工分配；论文应说明保留档位，或在后续独立版本重新定义分级。',
        '', '![各配置的P等级节点与三角形分配](priority_allocation.png)',
        '', '## 节点占比与三角形占比', '',
        '以下分母是原始节点实例引用的三角形总数，同一网格的多次实例会重复计入。它衡量几何工作量分布，不等于实际显存、屏幕贡献或运行时可见性。',
        '', '| 配置 | '+' | '.join(f'P{p}'for p in range(1,11))+' |', '|---|'+'---:|'*10]
    for r in totals:lines.append('| '+r['variant']+' | '+' | '.join(percent(r[f'P{p}_face_fraction'])for p in range(1,11))+' |')
    ncg0=index[('ALL','NCG_H0')];ncg1=index[('ALL','NCG_H1')]
    lines += ['',f"例如NCG_H1的P2占节点{percent(ncg1['P2_node_fraction'])}，占原始三角形{percent(ncg1['P2_face_fraction'])}；两种口径应同时看，不能以小零件数量多就认定它们支配几何预算。P5/P6仍可能是未知类型的保守保护，并不代表识别出了具体器件。",
        '', '## 结构树成对变化', '',
        '| 组合 | P级改变 | 升级 | 降级 | 功能候选改变 | 对象身份改变 | 平均P变化 |', '|---|---:|---:|---:|---:|---:|---:|']
    for mode in MAIN_MODES:
        r=pooled_pairs[mode];lines.append(f"| {mode} | {r['priority_changed']} | {r['priority_raised']} | {r['priority_lowered']} | {r['function_changed']} | {r['identity_changed']} | {r['mean_priority_delta']:+.4f} |")
    lines += ['', 'a没有可用装配树，H0/H1各组合的决策应完全一致，是负对照。其他场景里，H可把安装件区分为检测/运动支撑，把局部精密装配的紧固件提高保护。含G时，父组尺度也可能触发形状与“标准紧固件”身份不兼容，撤回身份并增加未知；这属于保守回退，不是新增识别正确数。',
        '', '| 场景（NCG） | P级改变 | 功能候选改变 | 身份撤回/改变 |', '|---|---:|---:|---:|']
    for r in [r for r in pairs if r['mode']=='NCG'and r['scene']!='ALL']:
        lines.append(f"| {r['scene']} | {r['priority_changed']} | {r['function_changed']} | {r['identity_changed']} |")
    lines += ['', 'f的局部装配作用变化最多；e的P级变化多于功能变化，说明父组几何尺度也参与了动作。d有26个身份变化，主要是含G后的兼容性回退，不能算识别出了26个新类型。',
        '', '| 组合 | H0→HT的P级改变 | HT→H1的P级改变 | H0→HT功能改变 | HT→H1功能改变 |', '|---|---:|---:|---:|---:|']
    for mode in MAIN_MODES:
        topology=[r for r in data['pairs']if r['mode']==mode and r['kind']=='topology_control']
        lexical=[r for r in data['pairs']if r['mode']==mode and r['kind']=='ancestor_name_increment']
        lines.append(f"| {mode} | {sum(r['priority_changed']for r in topology)} | {sum(r['priority_changed']for r in lexical)} | {sum(r['function_changed']for r in topology)} | {sum(r['function_changed']for r in lexical)} |")
    lines += ['', '上表将父组尺度与祖先语义分开对照；改变节点集合可能重叠，计数不能直接相加当成可加的因果贡献。',
        '', '[完整成对表](tree_pairs_with_total.csv)、[10×10迁移矩阵](priority_transitions.csv)、[改变节点与原因](changed_nodes.csv)保留了不变、升级和降级，不能只展示升级案例。',
        '', '## 置信度与未解决类型', '',
        'N可靠条件为概率≥0.70且第一、二候选差值≥0.08；“非常低”另指概率<0.40，间隔不足可能与概率分组重叠。C不按同一概率阈值判定：accepted/class_only为对象级，family_only仅粗类别，abstain为拒识。C质量分数未校准，不能与N概率直接比较。对象级输出也不等于判断正确。',
        '', '| 场景 | 节点 | N未达可靠条件 | C对象级输出 | C仅粗类别 | C拒识 |', '|---|---:|---:|---:|---:|---:|']
    for r in data['confidence']:lines.append(f"| {r['scene']} | {r['nodes']} | {r['N_not_reliable']}（{percent(r['N_not_reliable_fraction'])}） | {r['C_object_usable']}（{percent(r['C_object_fraction'])}） | {r['C_coarse']} | {r['C_abstain']} |")
    source_total=next(r for r in data['confidence']if r['scene']=='ALL')
    lines += ['',f"按原始三角形加权，N未可靠节点占{percent(source_total['N_low_face_fraction'])}，C没有对象级输出的节点占{percent(source_total['C_not_object_face_fraction'])}。因此开放分支的不足并非仅由微小零件数量多造成，大量几何工作量也处在粗类别/拒识状态。"]
    lines += ['', '| 配置 | 无对象级输出 | 功能未知 | N/C冲突 | 几何/类型冲突 | 需复核 |', '|---|---:|---:|---:|---:|---:|']
    for r in totals:lines.append(f"| {r['variant']} | {r['object_unresolved_count']}（{percent(r['object_unresolved_fraction'])}） | {r['unknown_function_count']} | {r['identity_conflict_count']} | {r['shape_conflict_count']} | {r['review_count']} |")
    lines += ['', '“需复核”还包括上下文候选，不等于低置信度数量；功能未知与没有对象级身份也是不同指标。G不承担类别识别，不应把它没有分类概率解释成100%识别失败。原始候选、置信度和名称弱提示保存在[source_uncertainty_nodes.csv](source_uncertainty_nodes.csv)。',
        '', '### 低置信候选类别', '',
        '以下按低置信候选数量排序。“类别”来自预测候选，不是真实标签；一类数量多，可能因为原本就常被预测到，必须同时看类内低置信比例。C表包含被拒识的候选，不能把这些候选当已采用的识别结果。']
    for branch in ('N','C'):
        lines += ['',f'**{branch}分支，低置信候选最多的前10类**', '', '| 候选类型 | 候选总数 | 低置信数 | 类内比例 | 占该分支低置信池 |', '|---|---:|---:|---:|---:|']
        rows=sorted([r for r in data['low_types']if r['scene']=='ALL'and r['branch']==branch],key=lambda r:-r['low_confidence_candidates'])[:10]
        for r in rows:lines.append(f"| {r['candidate_type']} | {r['total_candidates']} | {r['low_confidence_candidates']} | {percent(r['within_type_low_fraction'])} | {percent(r['share_of_low_pool'])} |")
    lines += ['', '[全部类别、全部场景](low_confidence_candidate_types.csv)同时提供计数、类内比例、低置信池占比和关联三角形数。',
        '', '## 几何简化诊断与成本', '',
        '主72组都对所有节点做了单网格QEM诊断，按相同源几何、相同简化器，分别执行各策略的远景请求比例。相同网格/比例复用结果；没有应用前端特征锁、cluster层次或剔除。',
        '', '| 组合 | H0实际保留比例 | H1实际保留比例 | H0平均QEM误差 | H1平均QEM误差 | H0/H1轴向收缩>10% | H0/H1显著分量丢失节点 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    if geometry:
        for mode in MAIN_MODES:
            a,b=[geometry[('ALL',f'{mode}_H{h}')]for h in (0,1)]
            lines.append(f"| {mode} | {a['actual_triangle_fraction']:.4%} | {b['actual_triangle_fraction']:.4%} | {a['mean_relative_error']:.5f} | {b['mean_relative_error']:.5f} | {a['axis_shrink_nodes']} / {b['axis_shrink_nodes']} | {a['component_loss_nodes']} / {b['component_loss_nodes']} |")
        a,b=[geometry[('ALL',f'NCG_H{h}')]for h in (0,1)]
        lines += ['',f"NCG加H后，实际简化结果只增加{b['actual_faces']-a['actual_faces']:,}个三角形，占源三角形的{(b['actual_triangle_fraction']-a['actual_triangle_fraction'])*100:.6f}个百分点；平均QEM误差相对降低{(a['mean_relative_error']-b['mean_relative_error'])/a['mean_relative_error']:.2%}。这是较小的全局变化。",
            '',f"NCG_H1请求保留{b['requested_triangle_fraction']:.4%}，实际保留{b['actual_triangle_fraction']:.4%}，说明请求比例没有等量转化为实际简化。可能受到网格拓扑、边界及简化器约束限制；需要在实际LOD链路中定位，不能只继续调语义权重。所有主组合在该探针下均有21个显著分量丢失节点，H没有解决这项风险。"]
    lines += ['', 'QEM误差相对两倍RMS半径归一化，非Hausdorff距离；分量丢失检查的是源表面积至少0.1%的分离表面分量，非孔洞保持率。不同策略实际预算不同，误差降低时必须同时看增加的三角形。这里没有真实LOD、图像质量或FPS结论。',
        '', '## 名称弱参考与分析结论', '',
        '| 配置 | 名称参考节点 | 对象身份覆盖 | 与名称一致 | 所有参考中一致率 | 名称组宏平均 |', '|---|---:|---:|---:|---:|---:|']
    for r in totals:lines.append(f"| {r['variant']} | {r['name_reference_count']} | {r['name_object_coverage_count']} | {r['name_consistent_count']} | {percent(r['name_consistency_all_reference'])} | {percent(r['name_group_macro_consistency'])} |")
    lines += ['', '仅评分能保守映射到14类的名称，655个实例包含重复件，不等于655个独立样本；a没有这类参考。M辅助输出排除在该评价之外。b–e与此前模型相同，六场景都参与过方法开发，本轮不是独立测试集，也没有计算真实OA/Macro-F1或统计显著性。',
        '',f"NCG中N/C冲突为{ncg0['identity_conflict_count']}个节点（{percent(ncg0['identity_conflict_fraction'])}）。加入G/H的价值应首先看形状风险是否被保护、冲突是否被显式保留，不能把更高P等级或更少剔除解释成更高语义准确率。",
        '', '从N→NG、C→CG、NC→NCG的对比看，G主要重新分配P5/P6及几何保护要求；它不会凭几何证明装配功能。H主要改变局部装配作用与约束，未重算视觉候选。N与C的融合既可补充对象证据，也会保留高分分歧；是否有收益需要覆盖、弱参考一致性、几何预算与误差一起判断。',
        '',f"**融合尚未显示盲识别优势。** N_H0在名称弱参考中覆盖{index[('ALL','N_H0')]['name_object_coverage_count']}、一致{index[('ALL','N_H0')]['name_consistent_count']}；NCG_H1覆盖{ncg1['name_object_coverage_count']}、一致{ncg1['name_consistent_count']}。融合撤回了部分冲突身份，覆盖下降，不能只报已覆盖子集的一致比例来宣称改善。",
        '',f"**结构树的全局几何影响较小。** NCG中P级改变{pooled_pairs['NCG']['priority_changed']}个节点，占{pooled_pairs['NCG']['priority_changed']/ncg1['nodes']:.2%}，作用候选改变{pooled_pairs['NCG']['function_changed']}个。它主要影响局部关系；不能从这些计数推断全场景显著收益。",
        '', '**开放分支仍偏向标准件与通用形状。** C_H0的P2与P5合计超过99%，对象级输出覆盖有限；大量块状、模具、安装板等候选未能采用。C+G将很多节点转入P6，说明几何保护补充了作用，但不能证明开放语义已经识别了这些非标件。',
        '', '论文可将本实验定位为“固定编码器下的分支与装配上下文策略消融”。当前结果不支持将P1–P10当成已全部充分使用的十类，也不支持声称结构树显著提高了盲分类准确率。',
        '', '## 各模型P1–P10节点占比完整表']
    for scene in data['models']:
        title=next(r['title']for r in data['inventory']if r['scene']==scene)
        lines += ['',f'### {scene}.glb：{title}', '', '| 配置 | '+' | '.join(f'P{p}'for p in range(1,11))+' |', '|---|'+'---:|'*10]
        for mode in MAIN_MODES:
            for h in (0,1):
                r=index[(scene,f'{mode}_H{h}')];lines.append('| '+r['variant']+' | '+' | '.join(percent(r[f'P{p}_node_fraction'])for p in range(1,11))+' |')
    lines += ['', '## 文件', '',
        '- `六模型六组合与结构树消融完整对比.xlsx`：可筛选的汇总、各P节点/三角形比例、树开关差异、低置信类型、几何诊断和原始统计计数。',
        '- [主72组原始统计（另含12行总计）](main_72_comparison.csv)、[主配置置信度与冲突](main_confidence_and_conflicts.csv)、[名称弱参考](blind_name_reference_comparison.csv)。',
        '- [HT、纯G、M辅助附表](supplementary_controls.csv)；M辅助不做名称一致率评分。',
        '- `provenance.json`记录源文件和策略结果哈希；节点策略和几何明细在`lod_analysis_outputs/six_scene_factorial_ablation_20261020`。',
        '', '主实验固定M=0与迁移回退=0，与当前第七项部署策略不是同一配置。此次只生成分析结果，没有替换渲染器策略。']
    (folder/'六模型六组合与结构树消融分析.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Export-ready, typed data for the workbook; no score is rounded in these inputs.
    data['paired_main']=pairs;data['totals_main']=totals;data['confidence_main']=public_conf
    savepath=folder/'workbook_data.json';savepath.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    arrays=[np.array([[r[f'P{p}_{measure}_fraction']*100 for p in range(1,11)]for r in totals])for measure in ('node','face')]
    for ax,values,title in zip(axes,arrays,['Node allocation (%)','Source triangle allocation (%)']):
        img=ax.imshow(values,cmap='Blues',vmin=0,vmax=100,aspect='auto');ax.set_title(title)
        ax.set_xticks(range(10),[f'P{i}'for i in range(1,11)]);ax.set_yticks(range(12),[r['variant']for r in totals])
        for i in range(12):
            for j in range(10):ax.text(j,i,f'{values[i,j]:.1f}',ha='center',va='center',fontsize=7,color='white'if values[i,j]>55 else '#202020')
    fig.colorbar(img,ax=axes.ravel().tolist(),fraction=.02,pad=.02);fig.subplots_adjust(wspace=.35,right=.92)
    fig.savefig(folder/'priority_allocation.png',dpi=190);fig.savefig(folder/'priority_allocation.svg');plt.close(fig)
    final_proof=dict(inputs={str((folder/'comparison_data.json').resolve()):file_hash(folder/'comparison_data.json'),str(Path(__file__).resolve()):file_hash(Path(__file__))},
        outputs={str(p.resolve()):file_hash(p)for p in folder.iterdir()if p.is_file()and p.name not in {'provenance.json','report_provenance.json'}})
    (folder/'report_provenance.json').write_text(json.dumps(final_proof,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Complete tables and report: '+str(folder),flush=True)


if __name__=='__main__':main()
