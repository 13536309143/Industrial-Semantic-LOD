"""Build scientific figures and an offline interactive reader from verified tables."""
import argparse
import csv
import html
import json
import time
from pathlib import Path

from run_thirteen_scene_ablation import DEFAULT_REPORT, digest, save


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows=list(csv.DictReader(stream))
    for row in rows:
        for k,v in list(row.items()):
            if v in ('True','False'):row[k]=v=='True'
            elif v=='':row[k]=None
            else:
                try:row[k]=float(v)
                except ValueError:pass
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-dir',type=Path,default=DEFAULT_REPORT)
    p.add_argument('--wait',action='store_true')
    args=p.parse_args();report=args.report_dir
    while args.wait:
        status=json.loads((report/'analysis_status.json').read_text(encoding='utf-8'))
        if status['status']=='complete':break
        if status['status']!='running':raise RuntimeError('Analysis incomplete; do not publish a complete dashboard')
        time.sleep(15)
    rows=read(report/'all_conditions_summary.csv')
    pairs=read(report/'tree_pairs.csv')
    inventory=read(report/'scene_inventory.csv')
    profiles=read(report/'source_confidence_summary.csv')
    models=sorted([r['scene'] for r in inventory],key=lambda s:int(s[1:]))
    modes=['N','C','G','NC','NG','CG','NCG']
    index={(r['scene'],r['variant']):r for r in rows}
    for r in rows:
        assert sum(int(r[f'P{i}_count']) for i in range(1,11))==int(r['nodes'])
        assert abs(sum(r[f'P{i}_node_fraction'] for i in range(1,11))-1)<1e-9
        assert abs(sum(r[f'P{i}_face_fraction'] for i in range(1,11))-1)<1e-9
    figdir=report/'figures';figdir.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,
        'savefig.dpi':180,'font.size':10,'svg.fonttype':'none'})
    colors=['#d7edf2','#b8dce8','#8dc4d6','#5eacc7','#3292b9','#3977ab','#535f9d','#74498c','#9d3c75','#be3556']
    saved=[]
    def finish(fig,name):
        fig.tight_layout()
        for ext in ('png','svg'):
            path=figdir/(name+'.'+ext);fig.savefig(path,bbox_inches='tight');saved.append(path)
        plt.close(fig)
    variants=[f'{m}_H{h}' for m in modes for h in (0,1)]
    matrix=np.array([[100*index['ALL',v][f'P{i}_node_fraction'] for i in range(1,11)] for v in variants])
    fig,ax=plt.subplots(figsize=(11,7));im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=max(50,matrix.max()))
    ax.set(xticks=range(10),xticklabels=[f'P{i}' for i in range(1,11)],yticks=range(len(variants)),yticklabels=variants,
        title='七种分支组合与装配上下文开关：P1—P10节点占比')
    for y,row in enumerate(matrix):
        for x,v in enumerate(row):ax.text(x,y,f'{v:.1f}',ha='center',va='center',color='white' if v>35 else '#172b4d',fontsize=8)
    fig.colorbar(im,ax=ax,label='节点占比 / %');finish(fig,'01_overall_priority_heatmap')
    values=np.array([[next(r for r in pairs if r['scene']==s and r['mode']==m and r['reference']=='H0' and r['comparison']=='H1')['priority_changed_fraction']*100 for m in modes] for s in models])
    fig,ax=plt.subplots(figsize=(10,8));im=ax.imshow(values,cmap='YlOrRd',vmin=0,vmax=max(20,values.max()))
    ax.set(xticks=range(7),xticklabels=modes,yticks=range(len(models)),yticklabels=models,title='各场景启用装配上下文后改变P档位的节点比例')
    for y,row in enumerate(values):
        for x,v in enumerate(row):ax.text(x,y,f'{v:.1f}%',ha='center',va='center',fontsize=8,color='white' if v>values.max()*.65 else '#172b4d')
    fig.colorbar(im,ax=ax,label='改变P的节点 / %');finish(fig,'02_tree_change_heatmap')
    for basis,label in [('node','节点'),('face','原始三角形实例')]:
        fig,ax=plt.subplots(figsize=(13,8));bottom=np.zeros(len(models)*2)
        chartrows=[index[s,f'NCG_H{h}'] for s in models for h in (0,1)]
        for i,color in enumerate(colors,1):
            v=np.array([r[f'P{i}_{basis}_fraction']*100 for r in chartrows]);ax.bar(range(len(v)),v,bottom=bottom,color=color,label=f'P{i}',width=.85);bottom+=v
        ax.set(xticks=range(len(chartrows)),xticklabels=[f'{s}\nH{h}' for s in models for h in (0,1)],ylim=(0,100),
            ylabel=f'{label}占比 / %',title=f'完整融合NCG：装配上下文开关前后的P分配（按{label}）')
        ax.tick_params(axis='x',labelsize=8);ax.legend(ncol=10,loc='upper center',bbox_to_anchor=(.5,-.12),frameon=False)
        finish(fig,'03_full_fusion_'+basis+'_distribution')
    fig,ax=plt.subplots(figsize=(11,5));x=np.arange(7);width=.36
    for shift,h,color in [(-width/2,0,'#456e9d'),(width/2,1,'#d07848')]:
        ax.bar(x+shift,[index['ALL',f'{m}_H{h}']['object_unresolved_fraction']*100 for m in modes],width,label=f'H{h}',color=color)
    ax.set(xticks=x,xticklabels=modes,ylabel='无对象级输出 / %',ylim=(0,105),title='对象级输出覆盖诊断（不是识别准确率）');ax.legend();finish(fig,'04_unresolved_evidence')
    # Per-scene prose tables retain the complete main matrix, rather than selected outcomes.
    modeldir=report/'models';modeldir.mkdir(exist_ok=True)
    for scene in models:
        item=next(r for r in inventory if r['scene']==scene)
        lines=[f"# {scene} {item['title']} 全量融合消融",'',
            f"共{int(item['nodes']):,}个网格实例，{int(item['source_faces']):,}个原始三角形实例。",'',
            '| 配置 | 无对象输出 | 功能未知 | N/C分歧 | '+' | '.join(f'P{i}' for i in range(1,11))+' |',
            '|---|'+'---:|'*13]
        for v in variants:
            r=index[scene,v]
            fields=[r['object_unresolved_fraction'],r['unknown_function_fraction'],r['identity_conflict_fraction'],*[r[f'P{i}_node_fraction'] for i in range(1,11)]]
            lines.append('| '+v+' | '+' | '.join(f'{100*a:.2f}%' for a in fields)+' |')
        lines+=['','## H0到H1的成对变化','','| 组合 | 改变P | 升档 | 降档 | 功能改变 | 身份改变 |','|---|---:|---:|---:|---:|---:|']
        for m in modes:
            r=next(r for r in pairs if r['scene']==scene and r['mode']==m and r['reference']=='H0' and r['comparison']=='H1')
            lines.append('| '+m+' | '+' | '.join(str(int(r[k])) for k in ('priority_changed','priority_raised','priority_lowered','function_changed','identity_changed'))+' |')
        r=index[scene,'NCG_H1'];before=index[scene,'NCG_H0']
        change=next(r for r in pairs if r['scene']==scene and r['mode']=='NCG' and r['reference']=='H0' and r['comparison']=='H1')
        lines+=['',f"完整融合中，无对象级输出为{100*r['object_unresolved_fraction']:.2f}%，功能未知为{100*r['unknown_function_fraction']:.2f}%。"
            f"H1改变{int(change['priority_changed']):,}个节点的P档位，其中升档{int(change['priority_raised']):,}、降档{int(change['priority_lowered']):,}。"
            f"请求远档面数比例由{100*before['requested_far_triangle_fraction']:.2f}%变为{100*r['requested_far_triangle_fraction']:.2f}%。"
            '这说明分配行为发生变化，不证明实际几何压缩或视觉改善。', '',
            'HT纯拓扑/尺度对照、叶名称M辅助、源三角形占比及完整字段均保存在总表；原始逐节点文件位于对应模型policies目录。', '']
        (modeldir/(scene+'.md')).write_text('\n'.join(lines),encoding='utf-8')
    payload=json.dumps(dict(rows=rows,pairs=pairs,inventory=inventory,profiles=profiles),ensure_ascii=False).replace('</','<\\/')
    dashboard='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>13模型融合与结构树消融</title>
<style>body{font:15px/1.65 "Microsoft YaHei",sans-serif;max-width:1350px;margin:35px auto;padding:0 24px;color:#17304a;background:#fafbfd}h1,h2{color:#14334a}select{padding:8px;margin:6px 20px 15px 5px}table{border-collapse:collapse;width:100%;background:white;margin:20px 0}th,td{border:1px solid #dce4ec;padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}.bar{height:27px;display:flex;background:#eee;margin:8px 0}.bar span{display:block;overflow:hidden;white-space:nowrap;font-size:11px;text-align:center;color:#122536}.row{display:grid;grid-template-columns:100px 1fr;align-items:center}.legend{display:flex;gap:13px;flex-wrap:wrap}.legend i{display:inline-block;width:14px;height:14px;margin-right:4px}p{color:#425b70}.note{font-size:13px}</style>
<h1>13模型全量融合与装配上下文消融</h1><p>V5固定规则；主实验7种分支组合×H0/H1，另有HT纯拓扑/尺度与叶名称M对照。覆盖与策略分配不是准确率、画质或渲染性能。</p>
<label>模型 <select id="scene"></select></label><label>统计口径 <select id="basis"><option value="node">节点占比</option><option value="face">原始三角形实例占比</option></select></label><label>上下文 <select id="tree"><option value="H0">H0 关闭</option><option value="H1">H1 装配上下文</option><option value="HT">HT 仅拓扑与尺度</option></select></label>
<h2 id="name"></h2><div id="legend" class="legend"></div><div id="bars"></div><div id="summary"></div><h2>完整融合：H0 与 H1 配对</h2><div id="pair"></div><p class="note">H1含祖先名称；M主实验关闭。源三角形按实例展开，不能换算实际显存；请求保留率也不是已实现压缩比。</p>
<script id="data" type="application/json">PAYLOAD</script><script>
const d=JSON.parse(document.querySelector('#data').textContent), colors=COLORS, modes=['N','C','G','NC','NG','CG','NCG'];
const ss=document.querySelector('#scene');for(const [id,title] of [['ALL','合并统计'],...d.inventory.map(r=>[r.scene,r.title])]){let o=document.createElement('option');o.value=id;o.textContent=id+' '+title;ss.append(o)}
document.querySelector('#legend').innerHTML=colors.map((c,i)=>'<span><i style="background:'+c+'"></i>P'+(i+1)+'</span>').join('');
const pct=v=>(100*v).toFixed(2)+'%';function update(){const scene=ss.value,basis=document.querySelector('#basis').value,tree=document.querySelector('#tree').value;const rows=modes.map(m=>d.rows.find(r=>r.scene===scene&&r.variant===m+'_'+tree));document.querySelector('#name').textContent=ss.selectedOptions[0].textContent;
document.querySelector('#bars').innerHTML=rows.map(r=>'<div class="row"><b>'+r.variant+'</b><div class="bar">'+colors.map((c,i)=>{let v=r['P'+(i+1)+'_'+basis+'_fraction'];return '<span style="background:'+c+';width:'+(100*v)+'%" title="P'+(i+1)+' '+pct(v)+'">'+(v>.06?'P'+(i+1):'')+'</span>'}).join('')+'</div></div>').join('');
document.querySelector('#summary').innerHTML='<table><tr><th>配置</th><th>节点</th><th>无对象级输出</th><th>功能未知</th><th>N/C分歧</th><th>请求远档面数比例</th></tr>'+rows.map(r=>'<tr><td>'+r.variant+'</td><td>'+r.nodes.toLocaleString()+'</td><td>'+pct(r.object_unresolved_fraction)+'</td><td>'+pct(r.unknown_function_fraction)+'</td><td>'+pct(r.identity_conflict_fraction)+'</td><td>'+pct(r.requested_far_triangle_fraction)+'</td></tr>').join('')+'</table>';
let ps=d.pairs.filter(r=>r.mode==='NCG'&&r.reference==='H0'&&r.comparison==='H1'&&(scene==='ALL'||r.scene===scene));document.querySelector('#pair').innerHTML='<table><tr><th>场景</th><th>改变P</th><th>升档</th><th>降档</th><th>功能改变</th><th>身份改变</th></tr>'+ps.map(r=>'<tr><td>'+r.scene+'</td>'+['priority_changed','priority_raised','priority_lowered','function_changed','identity_changed'].map(k=>'<td>'+r[k].toLocaleString()+'</td>').join('')+'</tr>').join('')+'</table>';}
for(let id of ['scene','basis','tree'])document.querySelector('#'+id).addEventListener('change',update);update();
</script></html>'''.replace('PAYLOAD',payload).replace('COLORS',json.dumps(colors))
    (report/'十三模型融合消融交互图表.html').write_text(dashboard,encoding='utf-8')
    mainreport=report/'十三模型全量消融与结构树对照分析.md'
    extra=['','## 图表与逐模型报告','', '[离线交互图表](十三模型融合消融交互图表.html)','']
    for name in ['01_overall_priority_heatmap','02_tree_change_heatmap','03_full_fusion_node_distribution','03_full_fusion_face_distribution','04_unresolved_evidence']:
        extra.append(f'![{name}](figures/{name}.png)\n')
    extra.extend(f"- [{s} {next(r['title'] for r in inventory if r['scene']==s)}](models/{s}.md)" for s in models)
    text=mainreport.read_text(encoding='utf-8').split('\n## 图表与逐模型报告')[0]
    mainreport.write_text(text+'\n'.join(extra)+'\n',encoding='utf-8')
    save(report/'chart_validation.json',dict(status='passed_numeric_checks',scenes=len(models),
        all_priority_counts_reconcile=True,all_node_and_face_fractions_sum_to_one=True,
        figures=[str(p) for p in saved],interactive_html=True))
    print(f'Figures and dashboard complete: {report}',flush=True)


if __name__=='__main__':main()
