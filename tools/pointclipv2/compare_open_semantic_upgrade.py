"""Compare legacy and upgraded evidence; name checks are explicitly weak proxies."""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
import numpy as np

from audit_named_scene import ROOT,read_csv,write_csv,name_hints
from summarize_named_scenes import expected_known_classes
from open_semantic_evidence import metadata_evidence
from run_pointclipv2_zeroshot_glb import read_glb_chunks
from audit_named_scene import hierarchy_paths
sys.path.insert(0,str(ROOT/'tools/pointnext_lod'))
from artifact_cache import file_hash
import build_lod_constraints_from_analysis as legacy


def old_known(label):
    if label == 'seal_gasket_interface':
        return 'washers_rings_spacers'
    for pattern,target in [('nut','nuts'),('washer|retaining_ring|spacer','washers_rings_spacers'),
        ('bearing|bushing|linear_guide','bearings_bushings_guides'),('screw|bolt|stud','screws_bolts_studs'),
        ('spring','springs'),('gear|pulley|sprocket|chain','gears_pulleys_chains'),('motor|reducer','motors_gearmotors'),
        ('pin|rivet|shaft_key','pins_rivets_keys')]:
        if re.search(pattern,label):return target
    return ''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',required=True)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--report-dir',type=Path,required=True)
    args=parser.parse_args()
    args.report_dir.mkdir(parents=True,exist_ok=True)
    directory=Path(__file__).parent
    taxonomy=json.loads((directory/'industrial_open_vocab_v2.json').read_text(encoding='utf-8'))
    label_map={label['id']:label for label in taxonomy['labels']}
    table=[];reviews=[];proxy=[];policies=[];inputs=[];examples=[]
    key=lambda row:tuple(str(row[k])for k in ['order','node_index','mesh_index'])
    totals=Counter()
    for name in args.models:
        run=args.run_root/name;base=args.baseline_root/name
        glb=ROOT/'_downloaded_resources'/(name+'.glb')
        state=json.loads((run/'status.json').read_text(encoding='utf-8'))
        if state['partial_smoke_test']or state['glb_sha256']!=file_hash(glb):raise ValueError('Partial or stale new scene: '+name)
        pn_path=base/(name+'_pointnext_analysis.csv')
        if not pn_path.exists():pn_path=run/'closed'/(name+'_pointnext_analysis.csv')
        pn=read_csv(pn_path)
        old_path=base/(name+'_pointclipv2_full.csv')
        old=read_csv(old_path)if old_path.exists()else []
        new=read_csv(run/'open_semantic_combined.csv')
        depth=read_csv(run/'open_semantic_depth_only.csv');surface=read_csv(run/'open_semantic_surface_only.csv')
        all_sets=[{key(r)for r in rows}for rows in [pn,new,depth,surface]]
        if any(len(rows)!=len(keys)for rows,keys in zip([pn,new,depth,surface],all_sets))or any(keys!=all_sets[0]for keys in all_sets):
            raise ValueError('Coverage/key mismatch: '+name)
        if old and {key(r)for r in old}!=all_sets[0]:raise ValueError('Legacy scene keys differ: '+name)
        old_index={key(r):r for r in old};new_index={key(r):r for r in new}
        raw_fusion=read_csv(run/'fusion/lod_constraints_NCG.csv');meta_fusion=read_csv(run/'fusion/lod_constraints_NCGM.csv')
        fused={key(r):r for r in raw_fusion};meta={key(r):r for r in meta_fusion}
        g,_,_=read_glb_chunks(glb);paths=hierarchy_paths(g['nodes'])
        named_total=old_all_match=new_all_match=old_fine_covered=old_fine_match=new_fine_covered=new_fine_match=0
        for row in pn:
            identity=key(row);current=new_index[identity];previous=old_index.get(identity,{})
            leaf,hits=name_hints(row['node_name'])
            expected=expected_known_classes({'leaf_name':leaf,'name_candidate_role':hits[0][0]if hits else ''})
            diagnostic=json.loads(current['open_diagnostics_json'])
            candidate=diagnostic.get('candidate_id','')
            ancestors=[g['nodes'][i].get('name','')for i in paths[int(row['node_index'])][:-1]]
            hint=metadata_evidence(row['node_name'],ancestors,taxonomy)
            if expected:
                named_total+=1
                old_label=old_known(previous.get('pointclip_top1_id',''))
                new_label=label_map.get(candidate,{}).get('known_class')
                old_all_match+=old_label in expected
                new_all_match+=new_label in expected
                old_covered=bool(previous and legacy.pointclip_role_scores(previous,False))
                new_covered=current['open_status']in {'accepted','class_only'}
                accepted_known=current.get('open_group_known_class') or label_map.get(current.get('open_object_id',''),{}).get('known_class')
                old_fine_covered+=old_covered;old_fine_match+=old_covered and old_label in expected
                new_fine_covered+=new_covered;new_fine_match+=new_covered and accepted_known in expected
                proxy.append({'scene':name,'node_index':row['node_index'],'node_name':row['node_name'],
                    'weak_expected_known_classes':','.join(sorted(expected)),'legacy_candidate':previous.get('pointclip_top1_name',''),
                    'new_candidate':candidate,'new_status':current['open_status'],'legacy_candidate_name_consistent':old_label in expected,
                    'new_candidate_name_consistent':new_label in expected,'scope':'CAD-name weak proxy, repeated instances; not accuracy. M-assisted identities are excluded.'})
            old_priority=''
            legacy_policy=base/'six_full'/(name+'_lod_constraints_6_pointnext_pointclip_structure.csv')
            # Load outside this loop below; only select semantic conflicts here.
            reason=[]
            if diagnostic.get('render_agreement')=='conflict':reason.append('depth_surface_conflict')
            if fused[identity]['class_conflict']=='True':reason.append('closed_open_conflict')
            if hint.get('object_id')and current['open_object_id']and hint['object_id']!=current['open_object_id']:reason.append('cad_visual_identity_difference')
            if reason:
                reviews.append({'scene':name,'node_index':row['node_index'],'node_name':row['node_name'],
                    'supervised_class':row['predicted_class'],'supervised_confidence':row['confidence'],
                    'legacy_open':previous.get('pointclip_top1_name',''),'depth_candidate':diagnostic.get('depth_family',''),
                    'surface_candidate':diagnostic.get('surface_family',''),'new_status':current['open_status'],
                    'new_candidate':candidate,'new_resolved':fused[identity]['resolved_object_class'],
                    'metadata_candidate':meta[identity]['resolved_object_class'],'review_reason':';'.join(reason),
                    'manual_class':'','manual_role':'','manual_critical_features':''})
            if expected and (old_label!=new_label or row['predicted_class']not in expected)and len([e for e in examples if e['scene']==name])<5:
                examples.append({'scene':name,'node_index':row['node_index'],'node_name':row['node_name'],
                    'closed':row['predicted_class'],'old':previous.get('pointclip_top1_name',''),
                    'new':candidate,'new_status':current['open_status'],'resolved':fused[identity]['resolved_object_class'],
                    'metadata_resolved':meta[identity]['resolved_object_class']})
        accepted=sum(r['open_status']!='abstain'for r in new)
        fine=sum(r['open_status']=='accepted'for r in new)
        old_usable=sum(bool(legacy.pointclip_role_scores(r,False))for r in old)
        weights=[json.loads(r['evidence_weights_json']).get('C',0)for r in raw_fusion]
        scene={'scene':name,'nodes':len(pn),'legacy_usable':old_usable,'legacy_usable_percent':100*old_usable/len(pn),
            'depth_usable':sum(r['open_status']!='abstain'for r in depth),'surface_usable':sum(r['open_status']!='abstain'for r in surface),
            'new_dual_usable':accepted,'new_dual_usable_percent':100*accepted/len(pn),'new_fine_candidates':fine,
            'new_class_group_candidates':sum(r['open_status']=='class_only'for r in new),
            'new_abstained':len(pn)-accepted,'mean_C_weight_new':float(np.mean(weights)),
            'named_proxy_nodes':named_total,'legacy_candidate_proxy_matches':old_all_match,'new_candidate_proxy_matches':new_all_match,
            'legacy_reliable_proxy_coverage':old_fine_covered,'legacy_reliable_proxy_matches':old_fine_match,
            'new_object_proxy_coverage':new_fine_covered,'new_object_proxy_matches':new_fine_match,
            'NCG_conflicts':sum(r['class_conflict']=='True'for r in raw_fusion),
            'NCGM_name_candidates':sum(r['resolved_source']=='cad_name_weak_hint'for r in meta_fusion),
            'glb_sha256':file_hash(glb),'accuracy_measured':False}
        table.append(scene)
        for field in ['nodes','legacy_usable','new_dual_usable','new_fine_candidates','new_class_group_candidates','new_abstained','named_proxy_nodes','legacy_candidate_proxy_matches','new_candidate_proxy_matches','new_object_proxy_coverage','new_object_proxy_matches','NCG_conflicts','NCGM_name_candidates']:
            totals[field]+=scene[field]
        summary=json.loads((run/'fusion/summary.json').read_text(encoding='utf-8'))
        for mode,values in summary.items():policies.append({'scene':name,'mode':mode,'rows':values['rows'],'low_percent':values['low_percent'],'high_percent':values['high_percent'],'class_conflicts':values['class_conflicts']})
        inputs.extend([glb,pn_path,run/'open_semantic_combined.csv',run/'combined_summary.json',run/'fusion/summary.json',run/'fusion/lod_constraints_NCG.csv',run/'fusion/lod_constraints_NCGM.csv'])
        if old:inputs.append(old_path)
    for filename,rows in [('scene_comparison.csv',table),('name_proxy_checks.csv',proxy),('semantic_conflict_review.csv',reviews),('unified_policy_comparison.csv',policies),('case_comparison.csv',examples)]:
        if rows:write_csv(args.report_dir/filename,rows)
    (args.report_dir/'comparison_summary.json').write_text(json.dumps({'scenes':table,'totals':dict(totals),
        'scope':'Coverage/behavior and CAD-name proxy analysis; no verified accuracy or actual LOD/render benchmark.'},ensure_ascii=False,indent=2),encoding='utf-8')
    scene_table='\n'.join(f"| {r['scene']} | {r['nodes']} | {r['legacy_usable_percent']:.2f}% | {r['new_dual_usable_percent']:.2f}% | {r['new_fine_candidates']} | {r['new_abstained']} | {r['NCG_conflicts']} | {r['NCGM_name_candidates']} |"for r in table)
    proxy_table='\n'.join(f"| {r['scene']} | {r['named_proxy_nodes']} | {r['legacy_candidate_proxy_matches']} | {r['new_candidate_proxy_matches']} | {r['new_object_proxy_coverage']} / {r['new_object_proxy_matches']} |"for r in table)
    case_table='\n'.join(f"| {r['scene']} | {r['node_index']} | {r['node_name']} | {r['closed']} | {r['old']} | {r['new']}（{r['new_status']}） | {r['resolved']} | {r['metadata_resolved']} |"for r in examples)
    policy_table='\n'.join(f"| {r['scene']} | {r['mode']} | {r['low_percent']:.2f}% | {r['high_percent']:.2f}% | {r['class_conflicts']} |"for r in policies)
    cweights='、'.join(f"{r['scene']} {100*r['mean_C_weight_new']:.2f}%"for r in table)
    named_total=sum(r['named_proxy_nodes']for r in table)
    old_named=sum(r['legacy_candidate_proxy_matches']for r in table)
    new_named=sum(r['new_candidate_proxy_matches']for r in table)
    object_covered=sum(r['new_object_proxy_coverage']for r in table)
    object_matches=sum(r['new_object_proxy_matches']for r in table)
    text=f'''# 开放语义分支升级与对照分析

本轮完成了词库、几何输入、视觉汇总、拒识、语义身份输出、统一融合和渲染CSV导出接口的改造；四场景共{totals['nodes']:,}个节点。**这是一套已运行的工程升级，尚未证明真实识别准确率提高。** 原版结果保留在four_scenes_seed2027_20261019，不重写历史对照。

## 改了什么

- 原110标签词库换为66物体类别、12上层类别，包含气动、视觉、工艺工具和送料机构等；中文别名只供单独的CAD弱证据使用。每类平衡的名词、外观和CAD描述替代大量同义词重复竞争。未知、微小和不重要不再混为类别。
- 深度输入使用原姿态与确定性PCA主轴姿态，按信息量选择视图；复杂/细长对象增加有限局部视图，局部证据只能在与全局类别一致时补充。
- 增加16,384表面点的法线着色正交视图，补充原低层深度投影难表达的曲面、孔和台阶线索。它是表面点渲染，不是具有完整材质/光照的照片，也不是完整三角形光栅化。
- 按“上层类别—对象大类—精确类型”判别；例如两路都支持螺母、但不能区分盖形/六角时，输出class_only的螺母大类，不假装精确类型已确认。粗文本只能提供有界辅助，不能压过具体视觉证据。绝对相似度、间隔、跨视图一致性及空背景参照共同参与拒识，排名softmax不是正确率。
- 深度与法线着色分别判别：同类才形成共识；细类别分歧只输出上层类别；上层冲突明确拒识；仅一路有支持时限制为粗类别。证据利用率不作为精度提升的替代指标。
- 新融合N/C/G共享同一个函数，G不读监督不确定性，C-only不借监督置信度；额外提供G-only和NCGM。输出resolved_object_class、resolved_source、resolved_status和冲突原因，原监督类别另行保留。高分冲突不再静默覆盖。
- 未知默认中性保护；结构复杂度、冲突与工件上下文可以设置保护下限。未知不直接P1，暂时看不到不表示可以删掉。

实现参考了[PointCLIP V2官方逐视图投影与加权接口](https://github.com/yangyangyang127/PointCLIP_V2/blob/main/zeroshot_cls/trainers/zeroshot.py)以及[CLIP图像预处理](https://github.com/openai/CLIP/blob/main/clip/clip.py)。新增几何着色、多视图汇总和拒识规则属于本项目工程方法，未声称复现官方最优权重或论文精度。原版PointCLIP深度分支本身没有采用标准RGB归一化，新版RGB处理属于方法变化，不能把原版简单说成遗漏预处理的bug。

## 本轮全量结果

| 场景 | 节点 | 旧规则可用开放证据 | 新双渲染非拒识 | 新细类别候选数 | 新拒识数 | N/C上层冲突数 | +M明确名称候选数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{scene_table}

旧规则可用与新非拒识采用不同质量定义，只用于说明行为和工作量，不能把差值叫准确率提升。拒识的节点仍完整保留，未知或冲突明确进入后续保守处理。

新版NCG的基础归一C权重均值为：{cweights}。这与原固定0.18系数不同，尚未经人工开发集校准；不等于因果贡献。是否真正更合理需要看功能标注和同预算几何误差，不能以权重变大作为成功标准。

## 名称辅助核验：有覆盖率才有解释

名称只产生弱参考，不等于类别真值。下表只取保守词典能映射到已知类别的叶名称；重复实例不独立，支架/安装件与机构本体已做基本区分。候选一致统计含被拒识的视觉候选，避免仅留下容易样本。+M不参加这项评价，因为用名称推断再对照同一名称会循环自证。

| 场景 | 弱名称参考节点 | 旧候选与名称一致数 | 新候选与名称一致数 | 新对象类别覆盖数 / 其中名称一致数 |
|---|---:|---:|---:|---:|
{proxy_table}

词表与输入均有改变，因此这不是单因素词库消融，更不是正式OA。缓存中的cached_feature_controls.csv另提供相同新词库下的原姿态均值、原姿态层次、主轴/局部、单独着色与双渲染决策，供后续人工对照。

合并这{named_total}个名称弱参考实例，旧候选一致{old_named}个，新候选一致{new_named}个；新版输出对象类别的覆盖为{object_covered}/{named_total}，其中与名称一致{object_matches}个。候选与最终输出要分开看，不能把被拒识的好候选当成已经正确采用，也不能只保留容易样本。该参考主要覆盖词典能明确映射的标准零件，并未验证全部66类，更没有验证气动、感知和工艺工具的真实准确率。

## 可核验案例

| 场景 | node_index | 名称 | 监督类别 | 旧开放 | 新视觉候选/状态 | NCG语义身份 | NCGM弱名称身份 |
|---|---:|---|---|---|---|---|---|
{case_table}

这些是定向复核案例，不是随机精度样本。含“产品”、送料工件、气动/检测对象可能超出现有监督14类。新版可以表达开放身份、粗类别和冲突，但不会把任何一个高分猜测自动变成真值。

## 统一配置与CAD辅助版本

| 场景 | 配置 | P1–P3 | P8–P10 | N/C冲突 |
|---|---|---:|---:|---:|
{policy_table}

NCG为纯视觉开放+监督+几何，NCGM额外使用可审计的叶名称与工艺上下文。CAD名称不会进入纯视觉特征或词库排名。父组只能给上下文，不能给所有子零件复制同一个类别；螺母送料盘、CCD安装板等复合名称有单独防误判规则。M输出标为metadata_candidate，仍需要人工确认。

相比旧版，词库、输入、可靠性定义与融合规则均变化，因此旧/新最终P分配不是某一个模块的因果收益。新版各证据开关共享公式；G-only只能支持几何形状保护，不能声称仅凭包围盒识别全部功能。

## 如何使用

项目根目录运行run_open_semantic_v2.bat。它激活gpt-pointnext，自动找到PointCLIPV2_LOD，使用隔离的DLL路径；模型失败会记录并继续后续场景，最后返回失败状态，避免把部分结果汇总成完整结果。成功阶段按输入/输出哈希复用。词库或判别规则变更可用缓存重新评分，无需全部重做视觉编码。

默认导出渲染器既有六配置文件名，模式6对应纯NCG。NCGM另存为显式metadata CSV，不暗中混入六配置。已有不同内容的导出会先备份。需重新打开场景并重建LOD后才会消费新策略；本轮没有重建LOD或测量FPS。可用--no-export仅生成分析结果。

编译后的应用另有第七项“PointNeXt + open semantics + structure + CAD context”，读取NCGM专用文件；该项缺少文件时不会悄悄回退成普通融合。第六项仍保持纯视觉NCG。带可信CAD名称的模型可选择第七项，人工确认名称冲突后再判断功能保护是否合理。

![K31真实几何的表面着色输入](surface_input_audit.png)

上图分别取K31的三个节点，展示的是孤立零件输入，不是场景可见性测试。孔洞、轮廓、台阶等可作为视觉线索，仍不能仅凭投影认定装配功能。

原始输出：{args.run_root.resolve()}。新词库：tools/pointclipv2/industrial_open_vocab_v2.json；固定参数：open_semantic_v2_config.json。候选与证据分数没有人工校准，当前四场景应视为开发诊断场景；最终论文需人工类别/功能标签和独立设备评估。

[逐场景对照](scene_comparison.csv)、[名称弱参考](name_proxy_checks.csv)、[冲突复核清单](semantic_conflict_review.csv)、[统一策略分布](unified_policy_comparison.csv)。细节与复现说明见项目中的OPEN_SEMANTIC_V2.md。未删除节点，未宣称真实准确率、关键细节保持或FPS提升。
'''
    report=args.report_dir/'开放语义分支升级与对照分析.md'
    report.write_text(text,encoding='utf-8')
    outputs=[p for p in args.report_dir.iterdir()if p.is_file()and p.suffix in {'.csv','.md','.json'}and p.name!='provenance.json']
    sources=[Path(__file__),directory/'open_semantic_evidence.py',directory/'open_semantic_shading.py',directory/'fuse_open_semantic_v2.py',directory/'industrial_open_vocab_v2.json',directory/'open_semantic_v2_config.json',directory/'combine_open_renderings.py']
    (args.report_dir/'provenance.json').write_text(json.dumps({'inputs':{str(p.resolve()):file_hash(p)for p in inputs},
        'sources':{str(p.resolve()):file_hash(p)for p in sources},'outputs':{str(p.resolve()):file_hash(p)for p in outputs},
        'accuracy_measured':False,'name_assisted_mode_excluded_from_name_consistency_evaluation':True},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'scenes':table,'totals':dict(totals),'report':str(report.resolve())},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
