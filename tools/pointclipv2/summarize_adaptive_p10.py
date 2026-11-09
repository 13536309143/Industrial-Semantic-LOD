"""Create an auditable ten-profile definition and full-node/runtime comparison report."""
import json
import math
import re
import shutil
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np
from audit_named_scene import read_csv,write_csv
from artifact_cache import file_hash
from adaptive_p10_policy import PROFILES,VERSION
from benchmark_p10_runtime import accepted_benchmark_hashes

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'analysis_reports/adaptive_p10_v4_20261020'
POLICIES=ROOT/'lod_analysis_outputs/adaptive_p10_v4_20261020'
DEFINITIONS={
 1:('微小重复件','普通标准件身份、极小相对尺寸、同网格重复且无功能冲突；不用于丢弃未知件'),
 2:('普通重复标准件','普通紧固、标准连接等；精密装配中的紧固件可升档'),
 3:('简洁静态体块','形状简单的大体块、机架实体；没有仅凭大尺寸假定安全'),
 4:('普通表面与支撑','普通安装板、支撑、外罩及简单表面；薄壁风险仍可提高保护'),
 5:('未决与均衡保护','身份不明确、证据冲突或几何无效时的保守起点'),
 6:('细长薄壁与轮廓','管、杆、薄板、异形轮廓和复杂几何；不凭形状编造对象身份'),
 7:('功能接口与局部安装','流体连接、检测安装、精密紧固及送料候选等'),
 8:('驱动控制与作业对象','执行、检测控制、工件候选等有功能依据的部件'),
 9:('精密运动与工艺定位','导向、传动、工装和工艺工具的功能保护'),
10:('敏感关键工艺细节','精密/工艺功能假设与细长、薄壁或复杂几何共同成立；高置信度本身不够'),
}


def pct(x):return f'{float(x):.2%}'
def load(path):return json.loads(path.read_text(encoding='utf-8'))
def save(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def runtime_table():
    result=[];trials=[]
    for variant in ('v3','v4_profile','v4_selected','v4_selected_strict','v4_compact','v4_selected_gpu_candidate'):
        path=REPORT/f'runtime_{variant}.json'
        if not path.exists():continue
        data=load(path)
        source_root=ROOT/'lod_analysis_outputs/assembly_context_v3_20261020'if variant=='v3'else POLICIES
        source_name='final_guarded/v3_NCGHM.csv'if variant=='v3'else 'NCG_H1_M1.csv'if variant=='v4_profile'else 'NCG_H1_M1_selected.csv'
        permitted_executables=accepted_benchmark_hashes(REPORT)
        for scene in 'abcdef':
            group=[r for r in data if r['scene']==scene]
            if not group:continue
            expected_policy=file_hash(source_root/scene/source_name)
            if any(r['policy_sha256']!=expected_policy or r['exe_sha256']not in permitted_executables for r in group):continue
            build=next(r for r in group if r['stage']=='preprocess')
            render=[r for r in group if r['stage']=='render']
            if not all(r['success']for r in group)or not build.get('expected_policy_loaded'):raise ValueError('Failed/incorrect runtime input')
            measured=[]
            for r in render:
                text=Path(r['log']).read_text(encoding='utf-8',errors='replace')
                frame=re.findall(r'Timer "Frame"; GPU; avg (\d+);.*?CPU; avg (\d+);.*?samples (\d+);',text)
                if not frame or int(frame[-1][2])<100 or not r.get('sequence_wait_for_scene'):raise ValueError('Unloaded/insufficient render measurement')
                viewport=re.findall(r'framebuffer: (\d+) x (\d+) \(target\)',text)[-1]
                gpu,cpu,samples=map(int,frame[-1]);rr=dict(r,gpu_frame_us=gpu,cpu_frame_us=cpu,samples=samples,
                    render_width=int(viewport[0]),render_height=int(viewport[1]))
                measured.append(rr);trials.append(rr)
            if not measured:raise ValueError('No actual rendered trials')
            result.append(dict(scene=scene,variant=variant,trials=len(measured),cache_bytes=build['cache_bytes'],
                build_seconds=build['seconds'],build_peak_working_set_bytes=build['peak_working_set_bytes'],
                render_peak_working_set_mean_bytes=float(np.mean([r['peak_working_set_bytes']for r in measured])),
                render_peak_private_mean_bytes=float(np.mean([r['peak_private_bytes']for r in measured])),
                geometry_gpu_bytes=measured[0]['geometry_gpu_actual_bytes'],
                gpu_frame_us_mean=float(np.mean([r['gpu_frame_us']for r in measured])),
                gpu_frame_us_min=min(r['gpu_frame_us']for r in measured),gpu_frame_us_max=max(r['gpu_frame_us']for r in measured),
                cpu_frame_us_mean=float(np.mean([r['cpu_frame_us']for r in measured])),
                GPU_time_equivalent_fps=1e6/float(np.mean([r['gpu_frame_us']for r in measured])),
                render_width=measured[0]['render_width'],render_height=measured[0]['render_height'],screen_error=measured[0].get('screen_error',2.),
                compressed=bool(measured[0].get('compressed',False)),
                ext_mesh=measured[0].get('ext_mesh'),unified_traversal=bool(measured[0].get('unified_traversal',False)),
                scope='scene-ready fixed-view offscreen, screen error 2, streaming off, shadows off; GPU time is not interactive FPS'))
    if result:write_csv(REPORT/'runtime_comparison.csv',result)
    if trials:
        columns=list(dict.fromkeys(k for r in trials for k in r))
        write_csv(REPORT/'runtime_trials.csv',[{k:r.get(k,'')for k in columns}for r in trials])
    return result


def main():
    summaries=read_csv(REPORT/'all_profile_comparison.csv')
    runtime=runtime_table();index={(r['scene'],r['variant']):r for r in runtime}
    metrics=read_csv(REPORT/'single_mesh_geometry_nodes.csv')
    grouped=defaultdict(list)
    for r in metrics:grouped[r['variant']].append(r)
    pooled=[]
    for variant,rows in grouped.items():
        faces=sum(int(r['source_faces'])for r in rows)
        pooled.append(dict(variant=variant,nodes=len(rows),source_faces=faces,
            actual_faces=sum(int(r['actual_faces'])for r in rows),
            actual_triangle_fraction=sum(int(r['actual_faces'])for r in rows)/faces,
            mean_QEM=float(np.mean([float(r['relative_surface_error'])for r in rows])),
            face_weighted_QEM=sum(float(r['relative_surface_error'])*int(r['source_faces'])for r in rows)/faces,
            axis_shrink_over_10pct=sum(float(r['max_axis_shrink'])>.1 for r in rows),
            component_loss_nodes=sum(int(r['missing_significant_components'])>0 for r in rows)))
    write_csv(REPORT/'geometry_pooled.csv',pooled)
    image_path=REPORT/'fixed_view_image_quality_extended.csv';images=read_csv(image_path)if image_path.exists()else read_csv(REPORT/'fixed_view_image_quality.csv')
    images=[r for r in images if (r['scene'],r['variant'])in index]
    lines=['# P1–P10自适应几何保护策略：规则升级与六模型验证','',
        f'当前策略版本：`{VERSION}`。本次不重训N/C，也不把名称弱参考当作准确率。优化对象是融合输出后的保护分配、层次构建约束和运行时细化选择。',
        '', '## 十档定义与适用对象','',
        'P是有序的保护配置，不是物体类别；同类零件因功能上下文、几何敏感性和尺度不同可以进入不同档位。语义未知不意味着可删除，几何复杂也不自动意味着关键工艺件。',
        '', '| 档位 | 定义 | 适用证据与边界 | 模板近/中/远 | 硬锁预算 | 相对增量QEM上限 |',
        '|---|---|---|---|---:|---:|']
    for p,(_,near,mid,far,weight,lock,error)in PROFILES.items():
        label,description=DEFINITIONS[p]
        lines.append(f'| P{p} | {label} | {description} | {near:.2f}/{mid:.2f}/{far:.2f} | {lock:.1%} | {error:.3f} |')
    lines += ['', '模板比例是每次cluster合并简化的请求，不是最终三角形、内存或显存保留率。连续尺度与几何敏感性会小幅调整比例及屏幕误差权重；实测搜索还能修正远层请求。P档位不为填满直方图而分配。硬锁预算不覆盖不可松动的非流形点；表中比例不是精确锁定点数量。',
        '', '## 从融合证据到实际执行','',
        '1. 保留v3对象身份、拒识与冲突；H只推断局部装配作用，不把父节点名称复制成子件类型。N/C消融继续显式遮蔽关闭的分支，H0不读取父组尺度，M单独开关。',
        '2. 将功能意图与形状保护分开。大而简单的未知形状可进入P3/P4，但不会被宣称识别为机架；薄壁、细长、复杂形状形成保护下限；P10要求功能和几何共同支持。',
        '3. 不再由旧P档位隐式追加10%–30%的硬锁，再叠加一套几何下限。schema=4显式指定锁预算、软权重与保护阈值，低语义置信度不降低几何安全约束。非流形保护仍保留。',
        '4. 启用G且几何有效时，先在最高精度层做极小误差的冗余清理：请求min(0.90,0.45+0.045P)，误差上限0.0008/(1+0.18P)。保留原有特征锁、分量检查和0.5%包围范围收缩上限；初始误差写入层次，不能伪装成零误差。后续cluster简化使用有限增量QEM预算及组范围/分量检查；违规回退，禁用sloppy和误差截小。',
        '5. 全量单网格探针搜索可行请求；只改变简化配置，不修改对象身份或功能。相同实际面数时优先较温和的请求，无法满足时回退到原始/近乎无误差的几何。该探针与真实cluster递归不是同一评价尺度。',
        '6. 同网格不同实例仍合并最严格构建约束；运行时误差权重保留各实例的作用差异。没有复制每实例几何，也没有根据小尺寸或名称自动删除节点。',
        '', '## 实验范围与可复现边界','',
        '六场景a–f共11,000个节点。生成六组合×H0/H1的72组主策略，另有G对照和M辅助；模板90组加实测修正6组，共96个策略CSV。所有神经推断按内容哈希复用，保留旧报告与旧策略备份。单网格几何对比覆盖旧部署、v4模板、v4实测修正的全部33,000条节点结果。',
        '', '真实程序重建与GPU测量只比较完整NCGHM配置的上述三版，不把它冒充72组渲染消融。使用RTX 3060；固定视角、1920×1080、supersample=1、关闭阴影/流式、屏幕误差2，每配置两次进程。明确等待异步加载与渲染就绪后再预热和计时；守卫加入前的诊断不进入最终结论。',
        '', '测量的精确可执行文件保存在`method/benchmarked_t1.exe`，并记录全部渲染/几何源码哈希。测量后只更新UI十档图例文字，当前Release重新构建通过；该文字变更没有混写到原始测量的二进制哈希中。',
        '', '## 原始输入的远层单网格预算诊断','',
        '此表保留对原始输入的受控测量，不含后续新增的最高层清理和前端特征锁；它是比例搜索依据，不是最终递归几何的整体质量证明。最高层清理的最终效果另由真实程序、参考图和原生构建测试核验。', '',
        '| 版本 | 实际三角形保留 | 平均归一化QEM | 面数加权QEM | 主轴收缩>10%的节点 | 显著分量丢失节点 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in pooled:lines.append(f'| {r["variant"]} | {pct(r["actual_triangle_fraction"])} | {r["mean_QEM"]:.6f} | {r["face_weighted_QEM"]:.6f} | {r["axis_shrink_over_10pct"]} | {r["component_loss_nodes"]} |')
    lines += ['', 'QEM按两倍RMS半径归一化，非Hausdorff距离；主轴范围与分量存在性不能证明孔洞、厚度和全部拓扑保持。这里的结果不能直接换算为真实显存或帧率，因此另外测量实际程序。',
        '', '## 真实程序：旧部署与最终候选','',
        '| 场景 | 几何GPU内存 MiB 旧→新 | 变化 | 渲染工作集 MiB 旧→新 | GPU帧耗时 ms 旧→新 | 新版两次范围 ms |',
        '|---|---:|---:|---:|---:|---:|']
    for scene in 'abcdef':
        if (scene,'v3')not in index or (scene,'v4_selected')not in index:continue
        a=index[(scene,'v3')];b=index[(scene,'v4_selected')]
        lines.append(f'| {scene} | {a["geometry_gpu_bytes"]/2**20:.2f}→{b["geometry_gpu_bytes"]/2**20:.2f} | {(b["geometry_gpu_bytes"]/a["geometry_gpu_bytes"]-1):+.2%} | {a["render_peak_working_set_mean_bytes"]/2**20:.2f}→{b["render_peak_working_set_mean_bytes"]/2**20:.2f} | {a["gpu_frame_us_mean"]/1000:.3f}→{b["gpu_frame_us_mean"]/1000:.3f} | {b["gpu_frame_us_min"]/1000:.3f}–{b["gpu_frame_us_max"]/1000:.3f} |')
    lines += ['', '几何GPU字节来自渲染器资源计数，不是整进程总显存；工作集按0.1秒轮询，受文件映射/系统缓存影响，不是几何有效载荷的同义词。GPU帧时间来自timestamp profiler；其倒数只代表GPU时间对应的吞吐，不是含窗口呈现、用户交互和CPU约束的实测FPS。两次进程只能作为工程诊断，不用于显著性结论。',
        '', '![实际内存与GPU帧时间](runtime_comparison.png)',
        '', '## 固定视角图像核验','',
        '参考图使用相同渲染器、相同相机、屏幕误差0.0001；普通图为2。JPEG质量100，图像指标每隔一个像素取样。全图SSIM会被背景稀释，因此同时报告参考边缘区域SSIM及2像素容差边缘召回。空白参考会直接判为失败。该视角仍不能覆盖被遮挡内部或替代多视角/人工功能标注。',
        '', '| 场景 | 版本 | PSNR dB | 全图SSIM | 边缘区域SSIM | 边缘召回 |',
        '|---|---|---:|---:|---:|---:|']
    for r in images:lines.append(f'| {r["scene"]} | {r["variant"]} | {float(r["PSNR_dB"]):.2f} | {float(r["SSIM_mean"]):.5f} | {float(r["reference_edge_region_SSIM"]):.5f} | {pct(r["reference_edges_recall_2px"])} |')
    lines += ['', '原图在[images](images/)；本表是质量与成本权衡，不能将PSNR降低解释为识别能力变差。若重要区域图像退步超过可接受范围，应调整屏幕误差权重/选择更细层，保留物体身份和P等级的独立性。',
        '', '## 画质与性能的操作点','',
        '最终版本额外测试屏幕误差0.5（strict），复用同一几何缓存，只改变运行时细化。压缩几何（compact，屏幕误差2）只在v4.2开发轮执行，属于已有压缩功能的配置对照，不混入本版主表，也不作为新算法贡献。',
        '', '| 场景 | 旧版 GPU ms | 新版均衡 GPU ms | 新版严格 GPU ms | 严格配置边缘SSIM |',
        '|---|---:|---:|---:|---:|']
    imindex={(r['scene'],r['variant']):r for r in images}
    for scene in 'abcdef':
        required=[(scene,v)for v in ('v3','v4_selected','v4_selected_strict')]
        if not all(k in index for k in required):continue
        values=[index[k]['gpu_frame_us_mean']/1000 for k in required]
        quality=float(imindex[(scene,'v4_selected_strict')]['reference_edge_region_SSIM'])
        lines.append('| '+scene+' | '+' | '.join(f'{v:.3f}'for v in values)+f' | {quality:.5f} |')
    if all((s,v)in index for s in 'abcdef'for v in ('v3','v4_selected')):
        old=[index[(s,'v3')]for s in 'abcdef'];new=[index[(s,'v4_selected')]for s in 'abcdef']
        mem_change=sum(r['geometry_gpu_bytes']for r in new)/sum(r['geometry_gpu_bytes']for r in old)-1
        faster=sum(b['gpu_frame_us_mean']<a['gpu_frame_us_mean']for a,b in zip(old,new))
        gpuchange=sum(r['gpu_frame_us_mean']for r in new)/sum(r['gpu_frame_us_mean']for r in old)-1
        slower=[b['scene']for a,b in zip(old,new)if b['gpu_frame_us_mean']>=a['gpu_frame_us_mean']]
        lines += ['', f'六次独立测量的几何GPU分配字节合计变化为{mem_change:+.2%}；这不是同时加载六场景的显存测量。均衡配置{faster}/6场景GPU时间下降，六场景GPU时间之和变化{gpuchange:+.2%}；这不等于桌面交互FPS结论。未下降场景：{", ".join(slower)or "无"}。需要保留中间层粒度与有效绘制簇开销的分析，不能宣称所有场景全面领先。',
            '', '严格配置会付出更多绘制时间，不能把它与均衡配置混为同一个性能结果。旧v4.2开发轮的压缩对照未显示稳定的几何GPU字节收益，因此最终不默认开启；旧版本压缩数据不混入当前版本表格。',
            '', '建议常规查看采用屏幕误差2，检查关键机构采用0.5；默认GUI的1处于二者之间，但本表未单独测量1。需要同画质FPS领先时，应继续优化层次粒度/有效绘制簇数量，不能再靠放宽保护阈值宣称成功。']
    if all((s,'v4_selected_gpu_candidate')in index for s in 'abcdef'):
        lines += ['', '另保持同一策略/缓存测试EXT mesh shader与统一遍历组合（`v4_selected_gpu_candidate`）。该候选未降低总体GPU时间，未加入默认启动配置；原始分场景数据保留在[runtime_comparison.csv](runtime_comparison.csv)。这是已有渲染配置的试验，不计为十档规则本身的收益。']
    lines += ['', '## 十档实际分布','', '| 场景 | '+' | '.join(f'P{p}'for p in range(1,11))+' |', '|---|'+'---:|'*10]
    histogram=Counter();n=0
    for scene in 'abcdef':
        rows=read_csv(POLICIES/scene/'NCG_H1_M1_selected.csv');h=Counter(int(r['lod_priority'].split('_')[0][1:])for r in rows)
        histogram.update(h);n+=len(rows)
        lines.append('| '+scene+' | '+' | '.join(pct(h[p]/len(rows))for p in range(1,11))+' |')
    lines.append('| 合并 | '+' | '.join(pct(histogram[p]/n)for p in range(1,11))+' |')
    lines += ['', '完整节点/三角形占比以及六组合树开关结果见[all_profile_comparison.csv](all_profile_comparison.csv)。各P适用模型类型是证据示例，不是按类别强制查表；同一对象身份可以因局部功能和几何风险改变保护配置。',
        '', '## 使用与局限','',
        '运行根目录`run_adaptive_p10_comparison.bat`重新生成并验证后导出。额外运行`run_adaptive_p10_runtime_validation.bat`完成真实离屏对比及图像核验。启动程序选择第七项（命令行索引6）并重建LOD使用NCGHM实测配置；第六项仍是关闭树/名称的NCG对照。已有旧CSV备份在`.artifacts/migrations`，旧实验目录不覆盖。',
        '', '缓存需要随策略重建；CSV全行哈希参与缓存身份。继续采用已有共享网格去重与流式预算机制；本次没有改变全分辨率原始几何，因此无法保证所有场景、所有视角的内存、显存和FPS同时改善。六场景均参与开发，不能声称全局最优或独立测试泛化结论。']
    (REPORT/'P1-P10策略重构与六模型验证.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if all((s,v)in index for s in 'abcdef'for v in ('v3','v4_selected')):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,3.7),layout='constrained')
        positions=np.arange(6)
        for variant,offset,color in [('v3',-.18,'#61758B'),('v4_selected',.18,'#2F857C')]:
            rows=[index[(s,variant)]for s in 'abcdef']
            axes[0].bar(positions+offset,[r['geometry_gpu_bytes']/2**20 for r in rows],width=.34,label=variant,color=color)
            means=np.array([r['gpu_frame_us_mean']/1000 for r in rows]);lo=[r['gpu_frame_us_min']/1000 for r in rows];hi=[r['gpu_frame_us_max']/1000 for r in rows]
            axes[1].errorbar(positions+offset,means,yerr=[means-lo,np.array(hi)-means],fmt='o',color=color,capsize=3,label=variant)
        for ax in axes:ax.set_xticks(positions,list('abcdef'));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(frameon=False)
        axes[0].set_ylabel('Allocated geometry (MiB)');axes[1].set_ylabel('GPU frame time (ms; two trials)')
        fig.savefig(REPORT/'runtime_comparison.png',dpi=180);fig.savefig(REPORT/'runtime_comparison.svg');plt.close(fig)
    sources=[Path(__file__),ROOT/'tools/pointclipv2/adaptive_p10_policy.py',ROOT/'tools/pointclipv2/run_adaptive_p10.py',
        ROOT/'tools/pointclipv2/validate_adaptive_p10_artifacts.py',ROOT/'tools/pointclipv2/validate_adaptive_p10_runtime.py',
        ROOT/'tools/pointclipv2/native/semantic_geometry_probe.cpp',ROOT/'CMakeLists.txt',
        ROOT/'tools/pointclipv2/tests/test_adaptive_p10.py',ROOT/'tools/pointclipv2/ADAPTIVE_P10_V4.md',
        ROOT/'run_adaptive_p10_comparison.bat',ROOT/'run_adaptive_p10_runtime_validation.bat',ROOT/'run_adaptive_p10_viewer.bat',
        ROOT/'tools/pointclipv2/benchmark_p10_runtime.py',ROOT/'tools/pointclipv2/evaluate_p10_images.py',
        ROOT/'src/scene/scene_semantic_lod.cpp',ROOT/'src/scene/scene.hpp',ROOT/'src/meshlod/meshlod_build.h',
        ROOT/'src/meshlod/meshlod_simplify.h',ROOT/'src/meshlod/meshlod_types.h',ROOT/'src/app/main.cpp',ROOT/'src/app/lodclusters.hpp',ROOT/'src/app/lodclusters_ui.cpp']
    method=REPORT/'method';method.mkdir(exist_ok=True)
    for path in sources:shutil.copyfile(path,method/path.name)
    outputs=[p for p in REPORT.glob('*')if p.is_file()and p.name not in {'delivery_manifest.json'}]
    save(REPORT/'delivery_manifest.json',dict(version=VERSION,sources={str(p.resolve()):file_hash(p)for p in sources},
        outputs={str(p.resolve()):file_hash(p)for p in outputs},runtime_experiment_conditions=len(runtime),
        all_nodes=11000,ground_truth_accuracy_measured=False,independent_test_set=False))
    print('Adaptive P1-P10 report generated',flush=True)


if __name__=='__main__':main()
