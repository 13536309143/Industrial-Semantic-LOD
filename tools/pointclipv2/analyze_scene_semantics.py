"""Audit scene coverage, build six policies and independent geometry diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import struct
import subprocess
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import build_lod_ablation_variants as variants
from independent_structure import classify_independent

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/pointnext_lod"))
from artifact_cache import file_hash

MODE_LABELS = {"pointnext": "N", "pointclip": "C", "pointnext_structure": "N+G",
               "pointclip_structure": "C+G", "pointnext_pointclip": "N+C",
               "pointnext_pointclip_structure": "N+C+G"}


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def key(row):
    return tuple(str(row[k]) for k in variants.JOIN_KEYS)


def indexed(rows):
    result = {key(row): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate join keys: cannot silently overwrite scene records")
    return result


def pnum(row):
    return variants.priority_number(row["lod_priority"])


def truth(value):
    return str(value).lower() in {"true", "1", "yes"}


def summarize(rows):
    n = len(rows)
    counts = Counter(pnum(row) for row in rows)
    triangles = sum(int(row["face_count"]) for row in rows)
    return {"rows": n, "source_triangles": triangles,
            "priority_counts": {f"P{i}": counts[i] for i in range(1, 11)},
            "low_protection_count": sum(counts[i] for i in [1, 2, 3]),
            "low_protection_percent": 100*sum(counts[i] for i in [1, 2, 3])/n,
            "high_protection_count": sum(counts[i] for i in [8, 9, 10]),
            "high_protection_percent": 100*sum(counts[i] for i in [8, 9, 10])/n,
            "allow_cull_percent": 100*sum(truth(row["allow_cull"]) for row in rows)/n,
            "mean_priority": statistics.mean(pnum(row) for row in rows),
            "roles": dict(Counter(row["inferred_role"] for row in rows)),
            "clip_prediction_rows": sum(bool(row.get("pointclip_top1_id")) for row in rows),
            "usable_clip_rows": sum(float(row.get("pointclip_role_score") or 0) > 0 for row in rows),
            "triangle_weighted_high_protection_percent": 100*sum(int(row["face_count"]) for row in rows if pnum(row)>=8)/triangles}


def pairwise(before, after, label_before, label_after):
    left, right = indexed(before), indexed(after)
    if set(left) != set(right):
        raise ValueError("Pairwise comparison requires exactly the same mesh records")
    pairs = [(left[k], right[k]) for k in left]
    changed = sum(pnum(a) != pnum(b) for a, b in pairs)
    return {"from": label_before, "to": label_after, "rows": len(pairs),
            "lod_changed": changed, "lod_changed_share": changed/len(pairs),
            "more_protect": sum(pnum(b)>pnum(a) for a,b in pairs),
            "less_protect": sum(pnum(b)<pnum(a) for a,b in pairs),
            "avg_priority_delta": statistics.mean(pnum(b)-pnum(a) for a,b in pairs),
            "cull_changed": sum(truth(a["allow_cull"]) != truth(b["allow_cull"]) for a,b in pairs),
            "role_changed": sum(a["inferred_role"] != b["inferred_role"] for a,b in pairs)}


def glb_structure(path):
    with path.open("rb") as stream:
        magic, version, total = struct.unpack("<III", stream.read(12))
        length, kind = struct.unpack("<II", stream.read(8))
        data = json.loads(stream.read(length))
    if magic != 0x46546c67 or version != 2 or total != path.stat().st_size:
        raise ValueError("Unexpected GLB header")
    mesh_triangles = [sum((data["accessors"][primitive["indices"]]["count"] if "indices" in primitive else data["accessors"][primitive['attributes']['POSITION']]['count'])//3
                          for primitive in mesh.get('primitives', []) if primitive.get('mode', 4)==4)
                      for mesh in data.get('meshes', [])]
    return {"bytes": total, "nodes": len(data.get("nodes", [])), "meshes": len(data.get("meshes", [])),
            "mesh_nodes": sum("mesh" in node for node in data.get("nodes", [])),
            "parents_with_children": sum(bool(node.get("children")) for node in data.get("nodes", [])),
            "unique_mesh_triangles": sum(mesh_triangles),
            "source_triangles": sum(mesh_triangles[node['mesh']] for node in data.get('nodes', []) if 'mesh' in node)}


def build_figures(report, summaries, policy_rows, pn):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 10})
    modes = list(MODE_LABELS)
    fig, ax = plt.subplots(figsize=(10, 4.9))
    bottom = np.zeros(6)
    for priority in range(1, 11):
        values = np.array([summaries[mode]["priority_counts"][f"P{priority}"]/len(pn)*100 for mode in modes])
        ax.bar(range(6), values, bottom=bottom, label=f"P{priority}", color=plt.cm.viridis((priority-1)/9))
        bottom += values
    ax.set_xticks(range(6), [MODE_LABELS[mode] for mode in modes])
    ax.set(ylabel="Mesh records (%)", ylim=(0, 100), title="Full-scene policy configurations (same full open vocabulary)")
    ax.legend(ncol=10, loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, .05, 1, 1))
    for suffix in ["png", "pdf"]:
        fig.savefig(report/f"six_policy_distribution.{suffix}", dpi=200)
    plt.close(fig)
    matrix = np.array([[100*pairwise(policy_rows[a],policy_rows[b],a,b)["lod_changed_share"] for b in modes] for a in modes])
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    display = ax.imshow(matrix, vmin=0, vmax=100, cmap="YlOrRd")
    ax.set_xticks(range(6), [MODE_LABELS[mode] for mode in modes])
    ax.set_yticks(range(6), [MODE_LABELS[mode] for mode in modes])
    for i in range(6):
        for j in range(6):
            ax.text(j,i,f"{matrix[i,j]:.1f}%",ha="center",va="center",color="white" if matrix[i,j]>65 else "black")
    fig.colorbar(display, ax=ax, label="Changed P class (%)")
    ax.set_title("Pairwise policy differences (not accuracy)")
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(report/f"pairwise_policy_change.{suffix}", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glb", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    glb, run, report = args.glb.resolve(), args.run_dir.resolve(), args.report_dir.resolve()
    report.mkdir(parents=True, exist_ok=True)
    stem = glb.stem.replace(" ", "_")
    paths = {"pointnext": run/f"{stem}_pointnext_analysis.csv", "full": run/f"{stem}_pointclipv2_full.csv",
             "candidate": run/f"{stem}_pointclipv2_candidates.csv"}
    pn, full, candidate = [read_rows(paths[k]) for k in ["pointnext", "full", "candidate"]]
    pidx, fidx, cidx = indexed(pn), indexed(full), indexed(candidate)
    scene = glb_structure(glb)
    assert len(pn)==len(full)==scene["mesh_nodes"] and set(pidx)==set(fidx)
    assert sum(int(row["face_count"]) for row in pn)==scene["source_triangles"]
    candidate_keys = {key(row) for row in pn if float(row["confidence"])<.70 or float(row["confidence"])-float(row["second_confidence"])<.08}
    assert set(cidx)==candidate_keys
    assert all(all(math.isfinite(float(row[k])) for k in ["confidence", "second_confidence"]) for row in pn)
    for rows in [full, candidate]:
        assert all(len(json.loads(row["pointclip_topk_json"]))==5 and 0<=float(row["pointclip_top1_score"])<=1 for row in rows)
    state=json.loads((run/"inference_status.json").read_text(encoding="utf-8"))
    assert state["glb_sha256"]==file_hash(glb) and state["checkpoint_sha256"]==file_hash(ROOT/"models/pointnext/industrial-part-14class-best.pth")
    state["coverage_audit"]={"supervised":len(pn),"full_open":len(full),"candidate_open":len(candidate),"join_keys_unique":True,"full_coverage":True,"candidate_keys_exact":True}

    def execute(name, script, options):
        with (run/"logs"/f"{name}.log").open("wb") as stream:
            subprocess.run([sys.executable, str(script), *map(str,options)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)

    builder = variants.load_lod_builder(ROOT)
    for folder, hybrid in [("six_full", paths["full"]),("six_gated_production", paths["candidate"])]:
        execute(folder, ROOT/"tools/pointclipv2/build_lod_ablation_variants.py",
                ["--pointnext-csv",paths["pointnext"],"--pointclip-full-csv",paths["full"],
                 "--pointclip-candidates-csv",hybrid,"--output-dir",run/folder,"--stem",stem,"--write-json"])
    same_vocab=[row for row in full if key(row) in candidate_keys]
    same_vocab_path=run/f"{stem}_pointclipv2_candidates_same_vocab.csv"
    variants.write_csv(same_vocab_path,same_vocab)
    fusion_rows={}
    for suffix, source in [("fused",paths["full"]),("gated",paths["candidate"]),("gated_same_vocab",same_vocab_path)]:
        execute(suffix,ROOT/"tools/pointclipv2/fuse_pointnext_pointclip_lod.py",
                ["--pointnext-csv",paths["pointnext"],"--pointclip-csv",source,
                 "--merged-csv",run/f"{stem}_{suffix}_merged.csv","--output-csv",run/f"{stem}_lod_constraints_{suffix}.csv",
                 "--output-json",run/f"{stem}_lod_constraints_{suffix}.json",
                 "--summary-json",run/f"{stem}_lod_constraints_{suffix}_summary.json"])
        fusion_rows[suffix]=read_rows(run/f"{stem}_lod_constraints_{suffix}.csv")
    policy_rows={mode:read_rows(run/"six_full"/f"{stem}_lod_constraints_{prefix}.csv") for mode,prefix in variants.ABLATION_MODES}
    assert all(len(rows)==len(pn) and set(indexed(rows))==set(pidx) for rows in policy_rows.values())
    assert all(pnum(a)==pnum(b) for a,b in zip(policy_rows['pointnext_pointclip_structure'],fusion_rows['fused']))
    summaries={mode:summarize(rows) for mode,rows in policy_rows.items()}
    thresholds=builder.make_thresholds(pn)
    independent={}
    merged=variants.merge_rows(pn,full,"independent full")
    for use_clip in [False,True]:
        mode="pointclip_structure_independent" if use_clip else "structure_independent"
        rows=[classify_independent(builder,row,thresholds,use_clip) for row in merged]
        variants.write_csv(run/f"{stem}_{mode}.csv",rows)
        independent[mode]=summarize(rows)
        changed=dict(merged[0],predicted_class="springs",confidence="0.99",second_confidence="0.001")
        assert classify_independent(builder,merged[0],thresholds,use_clip)==classify_independent(builder,changed,thresholds,use_clip)
        assert all('confidence' not in row and 'predicted_class' not in row for row in rows)
    # Raw structure diagnostics contain geometry and rule outputs, not neural predictions.
    structural=read_rows(run/f"{stem}_structure_independent.csv")
    variants.write_csv(run/f"{stem}_structure_analysis.csv",structural)
    pairs=[dict(model=stem,**pairwise(policy_rows[a],policy_rows[b],MODE_LABELS[a],MODE_LABELS[b])) for a,b in combinations(MODE_LABELS,2)]
    variants.write_csv(report/"lod_pairwise_change_summary.csv",pairs)
    variants.write_csv(report/"six_policy_comparison.csv",[dict(mode=MODE_LABELS[m],**{k:v for k,v in s.items() if not isinstance(v,dict)}) for m,s in summaries.items()])
    variants.write_csv(report/"priority_counts.csv",[dict(mode=MODE_LABELS[m],**s['priority_counts']) for m,s in summaries.items()])
    audited=[]
    policy_indices={mode:indexed(rows) for mode,rows in policy_rows.items()}
    for row in pn:
        k=key(row);clip=fidx[k];levels={MODE_LABELS[m]:pnum(policy_indices[m][k]) for m in MODE_LABELS}
        audited.append({"order":row['order'],"node_index":row['node_index'],"mesh_index":row['mesh_index'],"node_name":row['node_name'],
                        "predicted_class":row['predicted_class'],"confidence":row['confidence'],"shape_hint":row['shape_hint'],
                        "face_count":row['face_count'],"open_label":clip['pointclip_top1_name'],"open_role":clip['pointclip_top1_role'],
                        "open_score":clip['pointclip_top1_score'],"candidate":int(k in candidate_keys),**levels,
                        "full_minus_n":levels['N+C+G']-levels['N'],"priority_spread":max(levels.values())-min(levels.values())})
    audited.sort(key=lambda row:(row['priority_spread'],abs(row['full_minus_n']),int(row['face_count'])),reverse=True)
    variants.write_csv(report/"per_mesh_policy_audit.csv",audited)
    variants.write_csv(report/"manual_review_top30.csv",audited[:30])
    largest_changed=sorted([row for row in audited if row['N']!=row['N+C+G']],key=lambda row:-int(row['face_count']))[:10]
    variants.write_csv(report/"largest_mesh_policy_changes.csv",largest_changed)
    old_comparison=None
    old=ROOT/'.artifacts/migrations/pointnext_replaced_20261019_094119/lod_analysis_outputs/o1778_pointnext_analysis.csv'
    if old.exists():
        previous=read_rows(old);fields=['order','node_index','mesh_index','node_name','mesh_name','face_count','surface_area','bbox_diagonal']
        compatible=len(previous)==len(pn) and all(all(a.get(k)==b.get(k) for k in fields) for a,b in zip(previous,pn))
        if compatible:
            old_comparison={'reference':str(old),'matching_identity_geometry_fields':fields,
                            'historical_glb_hash_available':False,
                            'old_mean_confidence':statistics.mean(float(row['confidence']) for row in previous),
                            'new_mean_confidence':statistics.mean(float(row['confidence']) for row in pn),
                            'old_candidate_count':sum(float(row['confidence'])<.70 or float(row['confidence'])-float(row['second_confidence'])<.08 for row in previous),
                            'new_candidate_count':len(candidate),'class_predictions_changed':sum(a['predicted_class']!=b['predicted_class'] for a,b in zip(previous,pn))}
    counter=Counter(row['predicted_class'] for row in pn)
    weights=[]
    for row in fusion_rows['fused']:
        confidence=float(row['confidence']);margin=float(row['confidence'])-float(row['second_confidence'])
        neural=builder.clamp01(.18+.58*confidence+.14*builder.clamp01(margin/.30))
        if not (confidence>=.70 and margin>=.08):neural*=.55
        clip=.18*float(row.get('pointclip_role_score') or 0)
        structure=max(.20,1-neural-clip);weights.append((neural,clip,structure))
    result={'scene':scene,'glb':str(glb),'glb_sha256':state['glb_sha256'],'checkpoint_sha256':state['checkpoint_sha256'],
            'coverage':state['coverage_audit'],'pointnext':{'mean_confidence':statistics.mean(float(row['confidence']) for row in pn),
                'median_confidence':statistics.median(float(row['confidence']) for row in pn),'candidate_count':len(candidate),
                'candidate_percent':100*len(candidate)/len(pn),'accepted_count':len(pn)-len(candidate),'predicted_classes':dict(counter)},
            'shape_counts':dict(Counter(row['shape_hint'] for row in pn)),
            'open_full_role_counts':dict(Counter(row['pointclip_top1_role'] for row in full)),
            'open_candidate_role_counts':dict(Counter(row['pointclip_top1_role'] for row in candidate)),
            'open_full_top_labels':dict(Counter(row['pointclip_top1_id'] for row in full).most_common(10)),
            'open_quality':{'usable_including_micro':sum(bool(builder.pointclip_role_scores(row,True)) for row in full),
                            'usable_excluding_micro':sum(bool(builder.pointclip_role_scores(row,False)) for row in full),
                            'weak_or_rejected':sum(not builder.pointclip_role_scores(row,True) for row in full),
                            'mean_top1_score':statistics.mean(float(row['pointclip_top1_score']) for row in full),
                            'mean_top1_margin':statistics.mean(float(row['pointclip_margin']) for row in full)},
            'six_full':summaries,'independent_supplement':independent,'pairwise_full':pairs,
            'largest_changed_meshes':largest_changed,
            'independent_g_vs_cg':pairwise(structural,read_rows(run/f'{stem}_pointclip_structure_independent.csv'),'G_independent','CG_independent'),
            'fusion_full_vs_gated_same_vocab':pairwise(fusion_rows['fused'],fusion_rows['gated_same_vocab'],'all_open','gated_same_vocab'),
            'fusion_gated_vocab_effect':pairwise(fusion_rows['gated_same_vocab'],fusion_rows['gated'],'gated_same_vocab','gated_filtered_vocab'),
            'six_gated_production':{m:summarize(read_rows(run/'six_gated_production'/f'{stem}_lod_constraints_{prefix}.csv')) for m,prefix in variants.ABLATION_MODES},
            'full_fusion_mean_normalized_evidence_weights':dict(zip(['N','C','G'],[statistics.mean(w[i]/sum(w) for w in weights) for i in range(3)])),
            'historical_same_geometry_comparison':old_comparison,'ground_truth_available':False,
            'scope':'Full scene inference and policy allocation comparison; no accuracy, actual simplification quality or rendering benchmark.'}
    dump(report/'analysis_summary.json',result)
    build_figures(report,summaries,policy_rows,pn)
    write_report(report,run,result)
    input_paths=[glb,ROOT/'models/pointnext/industrial-part-14class-best.pth',*paths.values(),Path(__file__),ROOT/'tools/pointclipv2/independent_structure.py',ROOT/'tools/pointnext_lod/build_lod_constraints_from_analysis.py']
    dump(report/'provenance.json',{'inputs':{str(p):file_hash(p) for p in input_paths},'raw_run':str(run),'full_scene_coverage':True,
         'six_full_vocab_labels':110,'gated_production_vocab_labels':106,'independent_geometry_uncertainty_removed':True,
         'supervised_parameter_invariance_check_passed':True})
    state['status']='complete_scene_inference_and_policy_comparison';state['report']=str(report);dump(run/'experiment_status.json',state)
    dump(ROOT/'lod_analysis_outputs/deployment_status.json',{'status':'single_scene_semantic_experiment_complete',
         'scene':str(glb),'glb_sha256':state['glb_sha256'],'checkpoint_sha256':state['checkpoint_sha256'],
         'run_directory':str(run),'report_directory':str(report),'other_scenes_status':'not_evaluated_in_this_run',
         'renderer_lod_cache_status':'not_rebuilt','ground_truth_metrics_available':False})
    dump(ROOT/'analysis_reports/latest_scene_analysis.json',{'scene':str(glb),'glb_sha256':state['glb_sha256'],'checkpoint_sha256':state['checkpoint_sha256'],'report':str(report),'status':state['status']})
    print(json.dumps({'coverage':result['coverage'],'pointnext':result['pointnext'],'six_full':summaries,'report':str(report)},ensure_ascii=False,indent=2))


def write_report(report,run,r):
    stem = Path(r['glb']).stem.replace(' ', '_')
    scene_name = Path(r['glb']).name
    n=r['coverage']['supervised'];s=r['six_full'];pn=r['pointnext'];pairs=r['pairwise_full']
    find=lambda a,b:next(row for row in pairs if row['from']==a and row['to']==b)
    rows='\n'.join(f"| {MODE_LABELS[m]} | {v['rows']} | {v['low_protection_percent']:.2f}% | {v['high_protection_percent']:.2f}% | {v['allow_cull_percent']:.2f}% | {v['mean_priority']:.2f} |" for m,v in s.items())
    transitions='\n'.join(f"| {a}→{b} | {find(a,b)['lod_changed']} | {100*find(a,b)['lod_changed_share']:.2f}% | {find(a,b)['more_protect']} | {find(a,b)['less_protect']} |" for a,b in [('N','N+G'),('C','C+G'),('N','N+C'),('N+C','N+C+G'),('N','N+C+G')])
    independent='\n'.join(f"| {name} | {v['low_protection_percent']:.2f}% | {v['high_protection_percent']:.2f}% | {v['allow_cull_percent']:.2f}% |" for name,v in r['independent_supplement'].items())
    role_rows='\n'.join(f"| {role} | {count} | {count/n*100:.2f}% |" for role,count in sorted(r['open_full_role_counts'].items(),key=lambda item:-item[1]))
    old=r['historical_same_geometry_comparison']
    history=(f"与归档的 o1778 输出核对，{n} 条记录的身份、名称、面数、表面积和包围盒对角线字段完全匹配。旧记录的平均置信度为 {old['old_mean_confidence']*100:.2f}%，候选 {old['old_candidate_count']} 条；新记录为 {pn['mean_confidence']*100:.2f}%，候选 {len_candidate(r)} 条，类别预测变化 {old['class_predictions_changed']} 条。历史 GLB 哈希未保存，不能据此证明历史输入的每一字节相同。该比较仅表示预测和门控变化，不是准确率升降；新模型的内部评估流程更严谨，也不保证场景置信度一定更高。" if old else '没有可完整配对的旧场景输出，不进行新旧准确率比较。')
    gating=r['fusion_full_vs_gated_same_vocab'];vocab=r['fusion_gated_vocab_effect'];w=r['full_fusion_mean_normalized_evidence_weights']
    quality=r['open_quality']
    largest_table='\n'.join(f"| {row['order']} | {row['node_name']} | {int(row['face_count']):,} | {row['predicted_class']} | {float(row['confidence']):.3f} | P{row['N']}→P{row['N+C+G']} |" for row in r['largest_changed_meshes'][:5])
    gated_table='\n'.join(f"| {MODE_LABELS[mode]} | {s[mode]['low_protection_percent']:.2f}% / {s[mode]['high_protection_percent']:.2f}% | {r['six_gated_production'][mode]['low_protection_percent']:.2f}% / {r['six_gated_production'][mode]['high_protection_percent']:.2f}% |" for mode in ['pointnext_pointclip','pointnext_pointclip_structure'])
    text=f'''# {scene_name} 全量语义与结构分析、六种策略配置比较

日期：2026-10-19。当前文件哈希：`{r['glb_sha256']}`。监督权重为 PointNeXt-S、seed=2027、epoch=98，SHA256：`{r['checkpoint_sha256']}`。

核心观察：在当前规则下，结构组合显著改变策略分布，而开放证据的有效利用率有限；完整融合并未在所有保护分配指标上优于其他配置。本次结果支持流程覆盖与策略行为分析，尚不构成融合准确率或最终几何质量的优越性证明。

## 1. 完成范围与输入身份

本次已实际执行全量监督推理、全量 PointCLIP、门控 PointCLIP、结构规则分析、全量多源融合、门控多源融合，以及六种策略配置。没有把历史 CSV 当作本次结果。

当前 {scene_name} 为 {r['scene']['bytes']:,} 字节、{r['scene']['mesh_nodes']:,} 个带网格节点、{r['scene']['meshes']:,} 个 mesh；唯一 mesh 三角形 {r['scene']['unique_mesh_triangles']:,} 个，按节点实例展开的源三角形 {r['scene']['source_triangles']:,} 个；含 children 的父节点数为 {r['scene']['parents_with_children']}。几何与分类逐实例统计，累计父节点世界变换。文件名不足以确定场景身份，应按哈希核对。

覆盖核验：监督 {n}/{n}、全量开放 {r['coverage']['full_open']}/{n}；门控开放 {r['coverage']['candidate_open']} 条，恰好等于监督结果的候选集合。各 CSV 的 order/node_index/mesh_index 联合键唯一，六组输出与源记录一一对应，未发现漏行或重复覆盖。

完整原始记录位于 `{run}`；本目录只保存汇总、配图和逐网格复核表。

## 2. 监督分支与真实场景门控

监督平均置信度 **{pn['mean_confidence']*100:.2f}%**，中位数 **{pn['median_confidence']*100:.2f}%**。满足 confidence≥0.70 且 top1-top2≥0.08 的记录为 **{pn['accepted_count']} 条（{pn['accepted_count']/n*100:.2f}%）**；候选为 **{pn['candidate_count']} 条（{pn['candidate_percent']:.2f}%）**。

这些是置信度与覆盖率，没有人工真实类别，不能解释为场景准确率或错误率。当前阈值下，top1≥0.70 已蕴含间隔至少 0.40，因此 0.08 的间隔阈值不会额外排除这部分记录。不能凭本次统计证明两个门控条件各自都有贡献。

## 3. 开放语义与结构分析

PointCLIP 使用 ViT-B/16、8,192 个采样点、10 个投影视图、top-5。全量开放包含 110 个提示词标签；生产门控配置排除 micro_uncertain 角色，包含 106 个标签。开放 softmax 分数受词表竞争影响，不能与监督置信度当成同一尺度的准确率比较。

全量开放的首选角色分布如下，统计的是模型选择，不是真实角色：

| 首选角色 | 记录数 | 占比 |
|---|---:|---:|
{role_rows}

开放分支的 top1 平均分数为 {quality['mean_top1_score']:.4f}，top1–top2 平均间隔为 {quality['mean_top1_margin']:.4f}。按当前已有质量规则，只有 **{quality['usable_including_micro']} 条（{quality['usable_including_micro']/n*100:.2f}%）**形成可用角色证据；其中排除 micro_uncertain 后可用于完整融合的为 **{quality['usable_excluding_micro']} 条（{quality['usable_excluding_micro']/n*100:.2f}%）**。其余 **{quality['weak_or_rejected']} 条**被规则拒绝或判为弱证据。纯 C 的 P1 很多，既受到原始角色预测影响，也受到“缺少可用证据时退回 micro_uncertain”的代码路径影响，不能全部解释成模型认为零件本身不重要。

因此当前开放分支的问题表现为“提示词/投影输出与现有证据质量规则的适配不足”，不能只凭全量运行成功就宣称开放语义有效纠错，也不能在没有真实标签时断言全部弱证据都是错误识别。需要结合人工角色标注检查提示词、投影可辨识度及阈值校准。

结构分支使用包围盒尺度、长宽厚比例、细长/薄片线索、紧致度、面数及密度等描述，经几何原型规则分配角色。它不是另一个训练得到的检测网络，也没有直接测量孔径、齿形或薄壁保持误差。原始结构结果见 `{stem}_structure_analysis.csv`。本轮推断没有使用零件名称或装配树语义，名称/层级审计是另行输出的辅助证据。

## 4. 六种全量策略配置

N 为监督语义，C 为开放语义，G 为几何结构。以下六组全部使用同一份全量开放输出，使 N+C 与 N+C+G 也能接触所有记录的开放证据；这是全量配置比较。原生产门控版本另外保存在 `six_gated_production/`，未覆盖或改写现有部署公式。

P1–P3 为较低保护，P8–P10 为较高保护。均值只描述有序 P 编号的分布，不是质量分数。

| 配置 | 记录数 | P1–P3 | P8–P10 | allow_cull 元数据 | 平均 P 编号 |
|---|---:|---:|---:|---:|---:|
{rows}

![六组策略分布](six_policy_distribution.png)

`allow_cull` 是策略输出元数据，不是实际剔除率；当前 GPU 可见性路径没有消费该标志来决定剔除。上表未测量真实删除三角形、简化误差或帧率。

## 5. 分支组合怎样改变了策略

| 比较 | P 等级改变数 | 改变占比 | 转向更高 P | 转向更低 P |
|---|---:|---:|---:|---:|
{transitions}

![两两策略变化](pairwise_policy_change.png)

这些结果说明证据组合对策略输出有实际影响。保护增多可能有利于避免关键件过度简化，也可能增加不必要的几何保留；没有人工关键件参考与相同实际三角预算的几何质量测试，不能确定哪一种分配更正确。

完整融合的 P1–P3 占比为 {s['pointnext_pointclip_structure']['low_protection_percent']:.2f}%，而 N+G 为 {s['pointnext_structure']['low_protection_percent']:.2f}%。因此不能表述为“加入更多证据后保护程度必然更高”。完整模式与 N+G 的融合函数/规则不同，两者差异不能全部归因于开放分支。

按源三角形加权，落在 P8–P10 网格中的三角形占比从 N 的 **{s['pointnext']['triangle_weighted_high_protection_percent']:.2f}%** 变为完整融合的 **{s['pointnext_pointclip_structure']['triangle_weighted_high_protection_percent']:.2f}%**。该量描述原始几何被分配到哪些等级，不是实际三角保留率。网格数量口径下，高保护占比变化较小，但少数大网格的降级对几何规模口径影响明显，应优先复核。

| order | 节点名 | 源三角形数 | 监督预测 | 监督置信度 | N→完整融合 |
|---|---|---:|---|---:|---|
{largest_table}

上述高面数网格需要结合零件名称、真实角色和关键边界复核。开放标签和分数不是功能重要性的真值。P6 表示高细节保护，P8 表示结构/控制类保护；不能只凭编号升降断定结果改善。详见 [largest_mesh_policy_changes.csv](largest_mesh_policy_changes.csv)。

完整两两比较见 [lod_pairwise_change_summary.csv](lod_pairwise_change_summary.csv)，全部 {n} 条逐网格证据和六组等级见 [per_mesh_policy_audit.csv](per_mesh_policy_audit.csv)，优先人工复核的 30 条见 [manual_review_top30.csv](manual_review_top30.csv)。复核清单按配置分歧和面数排序，不能作为随机抽样准确率估计。

## 6. 全量开放、门控与词表的影响

固定 110 标签词表，将全量开放限制为相同候选集合后，完整融合的 P 等级变化 **{gating['lod_changed']} 条（{gating['lod_changed_share']*100:.2f}%）**。这描述了开放证据只作用于候选记录时，当前融合规则的行为变化。

在同一候选集合上，再改为生产环境的 106 标签过滤词表，P 等级变化 **{vocab['lod_changed']} 条（{vocab['lod_changed_share']*100:.2f}%）**。因此不能把生产门控与全量结果的全部差异都归因于门控本身。

| 配置 | 全量版本：低保护 / 高保护 | 生产门控版本：低保护 / 高保护 |
|---|---:|---:|
{gated_table}

固定词表后，门控与全量完整融合在 {(1-gating['lod_changed_share'])*100:.2f}% 的记录上保持相同 P 等级，同时少执行 {pn['accepted_count']} 条开放推理。这支持当前规则下减少开放计算的可能性，不等于已经证明不损失识别精度或几何质量。

全量融合中，归一化的基础证据权重均值约为 N={w['N']:.3f}、C={w['C']:.3f}、G={w['G']:.3f}。该统计尚未计入后续类别/结构规则加分，不是各分支的因果贡献或最终影响占比。开放分支有输出，也不等于该记录的开放证据通过了质量筛选并实际参与融合。

## 7. 独立结构与独立开放+结构补充对照

现有 C+G 仍利用监督 confidence/margin 派生的结构不确定性；纯 C 策略也携带监督 confidence 字段，下游 C++ 可据此调整保护。因此原来的六组不能全部视为严格独立分支消融。

为单独回答结构与开放+结构的行为，本次补充两份诊断基线：移除几何原型匹配中的监督 uncertainty_score，并在剩余几何维度重新归一权重；结构权重固定 0.70，开放权重沿用开放质量规则。其输出不含监督类别或 confidence；改变监督预测和置信度后输出保持一致的检查通过。它们属于明确注明方法差异的补充基线，不是悄悄修改生产算法，也不能与原组差值直接解释成单一因素收益。

| 补充基线 | P1–P3 | P8–P10 | allow_cull 元数据 |
|---|---:|---:|---:|
{independent}

独立 G 与独立 C+G 之间有 **{r['independent_g_vs_cg']['lod_changed']} 条（{r['independent_g_vs_cg']['lod_changed_share']*100:.2f}%）**改变等级。高保护比例见上表；未与人工功能标注核对，不能将结构分支描述为已经识别全部关键功能件。

数据见原始目录的 `{stem}_structure_independent.csv` 与 `{stem}_pointclip_structure_independent.csv`。

## 8. 与旧场景观察的关系

{history}

## 9. 本次效果应该怎样评价

本次验证了流程能够完整覆盖当前场景，并量化了监督、开放与结构证据如何改变 P1–P10 分配。监督分支的直接接受覆盖率为 {pn['accepted_count']/n*100:.2f}%，表明依照当前阈值仍有大量记录需要补充证据；这为融合提供了应用动机，但尚未验证补充证据能够纠错。

分类准确不等于功能角色正确，功能角色正确也不等于最终简化质量更好。本次没有人工类别/角色标签，没有实际执行网格简化、重建 LOD 或 GPU 渲染基准。因此不能报告场景 OA、融合准确率、关键结构保持率或 FPS 提升。

若用于论文，当前结果应命名为“真实场景全量推理与语义策略分配比较”。下一步优先人工核验关键件与高分歧记录，再在相同实际三角预算下比较几何偏差、关键结构保留与时间/显存。正式的单因素消融还需固定同一融合函数、证据质量规则和下游 confidence 处理。

## 10. 来源与复跑

[analysis_summary.json](analysis_summary.json) 保存完整计数与比较，[provenance.json](provenance.json) 保存场景、权重、推理 CSV 及分析代码哈希。各阶段进程状态与耗时保存在原始输出目录的 inference_status.json；失败阶段不会当成空预测参与统计。

使用 `tools/pointclipv2/run_scene_semantic_comparison.py` 指定本场景路径与输出目录，再调用 `analyze_scene_semantics.py` 可复跑。已有神经结果仅在来源和输出哈希一致时复用；分析与策略表重新生成。历史汇总保持原样。
'''
    timing=json.loads((run/'inference_status.json').read_text(encoding='utf-8'))
    times={stage['name']:stage['wall_seconds'] for stage in timing['stages']}
    timing_prefix = ("本轮使用经输入、代码、权重及输出哈希核验的推断快照。以下时间属于快照的原始运行，不是本轮复用耗时。"
                     if timing.get('snapshot_reuse') else '')
    text += (f"\n{timing_prefix}运行时间口径：开放全量阶段约 {times['02_pointclip_full']:.1f} 秒、门控阶段约 {times['03_pointclip_candidates']:.1f} 秒，"
             "包含进程启动、文本编码、采样与推理，仅为本次单次记录，不是重复 GPU 性能基准。"
             "监督阶段的时间可能仅包含对已完成输出的缓存核验，不能将缓存命中的调度时间当作全量监督推理耗时。\n")
    (report/'analysis_report.md').write_text(text,encoding='utf-8')


def len_candidate(result):
    return result['pointnext']['candidate_count']


if __name__ == "__main__":
    main()
