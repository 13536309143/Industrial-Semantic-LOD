import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';

const reportDir=path.resolve(process.argv[2]);
const root=path.resolve(process.argv[3]||'.');
const outputDir=path.join(root,'outputs','six_scene_factorial_ablation_20261020');
const previewDir=path.join(root,'.artifacts','factorial_workbook','previews');
await fs.mkdir(outputDir,{recursive:true});await fs.mkdir(previewDir,{recursive:true});
const d=JSON.parse(await fs.readFile(path.join(reportDir,'workbook_data.json'),'utf8'));
const taxonomy=JSON.parse(await fs.readFile(path.join(root,'tools/pointclipv2/industrial_open_vocab_v2.json'),'utf8'));
const wb=Workbook.create();
const names=['总览','P等级节点占比','P等级三角形占比','树开关变化','置信度与冲突','低置信候选类型','几何诊断','统计计数','方法与来源'];
const sheets=Object.fromEntries(names.map(n=>[n,wb.worksheets.add(n)]));
const font='Microsoft YaHei';const blue='#23395B';const text='#202938';
const modes=['N','C','NG','CG','NC','NCG'];
const bodyStart=5;
function col(n){let s='';for(n++;n;n=Math.floor((n-1)/26))s=String.fromCharCode(65+(n-1)%26)+s;return s;}
function frame(sheet,title,note,columns,rows,{tableName,formats={},widths={},freeze=3}={}){
  const end=4+rows.length;const last=col(columns.length-1);
  const area=sheet.getRange(`A1:${last}${Math.max(end,5)}`);
  area.format.font={name:font,size:10,color:text};area.format.rowHeight=23;
  area.format.verticalAlignment='center';sheet.showGridLines=false;
  sheet.getRange('A2').values=[[title]];sheet.getRange('A2').format.font={name:font,size:15,bold:true,color:blue};
  sheet.getRange(`A2:${last}2`).format.rowHeight=32;
  sheet.getRange('A3').values=[[note]];sheet.getRange('A3').format.font={name:font,size:9,color:'#667085'};
  sheet.getRange(`A4:${last}4`).values=[columns];
  if(rows.length)sheet.getRange(`A5:${last}${end}`).values=rows;
  if(tableName){const t=sheet.tables.add(`A4:${last}${end}`,true,tableName);t.showFilterButton=true;}
  sheet.getRange(`A4:${last}4`).format={fill:blue,font:{name:font,size:10,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',verticalAlignment:'center',rowHeight:34,wrapText:true};
  sheet.getRange(`A5:${last}${end}`).format.horizontalAlignment='right';
  for(let i=0;i<columns.length;i++){
    sheet.getRange(`${col(i)}4:${col(i)}${end}`).format.columnWidth=widths[i]??14;
    if(formats[i])sheet.getRange(`${col(i)}5:${col(i)}${end}`).setNumberFormat(formats[i]);
  }
  sheet.getRange(`A5:${col(Math.min(2,columns.length-1))}${end}`).format.horizontalAlignment='left';
  if(rows.length>15){sheet.freezePanes.freezeRows(4);if(freeze)sheet.freezePanes.freezeColumns(freeze);}
  sheet.tabColor=blue;
  return {start:5,end,last};
}
const raw=d.main.filter(r=>r.scene!=='ALL');
const statsKeys=['scene','variant','nodes','source_faces',...Array.from({length:10},(_,i)=>`P${i+1}_count`),
  ...Array.from({length:10},(_,i)=>`P${i+1}_faces`),'object_unresolved_count','unknown_function_count',
  'identity_conflict_count','shape_conflict_count','review_count','N_not_reliable_count','C_not_object_count',
  'name_reference_count','name_object_coverage_count','name_consistent_count'];
const statsHeaders=['模型','配置','节点数','源三角形数',...Array.from({length:10},(_,i)=>`P${i+1}节点数`),
  ...Array.from({length:10},(_,i)=>`P${i+1}源三角形数`),'无对象级输出','功能未知','N/C冲突','几何类型冲突','需复核','N未可靠','C无对象输出','名称参考数','参考中覆盖数','参考中一致数'];
const source=sheets['统计计数'];
frame(source,'主72组统计计数','来源：main_72_comparison.csv。分支关闭时的诊断计数为不适用，不填零。',statsHeaders,
 raw.map(r=>statsKeys.map(k=>r[k]??'n.a.')),{tableName:'SourceCounts',widths:{0:9,1:15},formats:Object.fromEntries(statsKeys.map((k,i)=>[i,i>1?'#,##0':'@']))});
source.tabColor='#A9B4C2';
const sourceEnd=4+raw.length;
const keyCol=Object.fromEntries(statsKeys.map((k,i)=>[k,col(i)]));
function sumFormula(key,scene,variant){
  const range=`'统计计数'!$${keyCol[key]}$5:$${keyCol[key]}$${sourceEnd}`;
  const criteria=`'统计计数'!$B$5:$B$${sourceEnd},"${variant}"`;
  const sceneCriteria=scene==='ALL'?'':`,'统计计数'!$A$5:$A$${sourceEnd},"${scene}"`;
  return `SUMIFS(${range},${criteria}${sceneCriteria})`;
}
function sceneOrder(a,b){return (a.scene==='ALL'?-1:d.models.indexOf(a.scene))-(b.scene==='ALL'?-1:d.models.indexOf(b.scene));}
function conditionOrder(a,b){return sceneOrder(a,b)||modes.indexOf(a.mode)-modes.indexOf(b.mode)||(a.tree==='off'?0:1)-(b.tree==='off'?0:1);}
const ordered=[...d.main].sort(conditionOrder);
const viewRow=new Map(ordered.map((r,i)=>[`${r.scene}|${r.variant}`,i+5]));
for(const [name,measure,denominator]of [['P等级节点占比','count','nodes'],['P等级三角形占比','faces','source_faces']]){
  const sh=sheets[name];
  frame(sh,name,measure==='count'?'H0完全关闭树辅助；H1使用局部父组及祖先名称。主配置M=0、迁移回退=0。':'分母为源三角形实例数；不等于实际简化后数量或显存。',
    ['模型','组合','树辅助',measure==='count'?'节点数':'源三角形数',...Array.from({length:10},(_,i)=>`P${i+1}`)],
    ordered.map(r=>[r.scene,r.mode,r.tree==='off'?'H0':'H1',null,...Array(10).fill(null)]),
    {tableName:measure==='count'?'NodeAllocation':'FaceAllocation',widths:{0:9,1:11,2:11,3:17},formats:Object.fromEntries(Array.from({length:14},(_,i)=>[i,i===3?'#,##0':i>3?'0.00%':'@']))});
  const matrix=ordered.map((r,i)=>['='+sumFormula(denominator,r.scene,r.variant),...Array.from({length:10},(_,p)=>`=${sumFormula(`P${p+1}_${measure}`,r.scene,r.variant)}/$D${i+5}`)]);
  sh.getRange(`D5:N${ordered.length+4}`).formulas=matrix;
  sh.getRange(`E5:N${ordered.length+4}`).conditionalFormats.add('colorScale',{colors:['#FFFFFF','#DCE7F5','#4676B7'],thresholds:['min','50%','max']});
}
const pairData=d.paired_main.sort((a,b)=>sceneOrder(a,b)||modes.indexOf(a.mode)-modes.indexOf(b.mode));
const pairMap=new Map(pairData.map((r,i)=>[`${r.scene}|${r.mode}`,i+5]));
frame(sheets['树开关变化'],'结构树成对变化','H1减H0；P级变化与功能候选变化分开统计，包含全部升级、降级和不变。',
 ['模型','组合','节点数','P级改变','改变占比','升级','降级','功能改变','身份改变','动作改变','平均P变化',...Array.from({length:10},(_,i)=>`P${i+1}变化(pp)`)],
 pairData.map(r=>[r.scene,r.mode,r.nodes,r.priority_changed,null,r.priority_raised,r.priority_lowered,r.function_changed,r.identity_changed,r.action_changed,r.mean_priority_delta,...Array.from({length:10},(_,i)=>r[`P${i+1}_delta_pp`])]),
 {tableName:'TreePairs',widths:{0:9,1:11},formats:{2:'#,##0',3:'#,##0',4:'0.00%',5:'#,##0',6:'#,##0',7:'#,##0',8:'#,##0',9:'#,##0',10:'0.0000',...Object.fromEntries(Array.from({length:10},(_,i)=>[i+11,'0.00']))}});
sheets['树开关变化'].getRange(`E5:E${pairData.length+4}`).formulas=pairData.map((r,i)=>[`=D${i+5}/C${i+5}`]);
const confidenceRows=ordered.map(r=>[r.scene,r.mode,r.tree==='off'?'H0':'H1',r.nodes,
 r.N_not_reliable_count??'n.a.',r.C_not_object_count??'n.a.',r.object_unresolved_count,null,r.unknown_function_count,null,
 r.identity_conflict_count,r.shape_conflict_count,r.review_count,r.name_reference_count,r.name_object_coverage_count,r.name_consistent_count,null]);
frame(sheets['置信度与冲突'],'置信度、冲突和名称弱参考','N未可靠：p<0.70或间隔<0.08。C无对象输出：粗类别或拒识。名称一致不是准确率。',
 ['模型','组合','树辅助','节点数','N未可靠数','C无对象数','无对象级输出','无对象比例','功能未知数','未知功能比例','N/C冲突数','形状冲突数','需复核数','名称参考数','参考中覆盖数','参考中一致数','全部参考中一致'],
 confidenceRows,{tableName:'ConfidenceConflicts',widths:{0:9,1:11,2:11,6:18,7:16,9:17,13:16,14:18,15:18,16:21},formats:{3:'#,##0',4:'#,##0',5:'#,##0',6:'#,##0',7:'0.00%',8:'#,##0',9:'0.00%',10:'#,##0',11:'#,##0',12:'#,##0',13:'#,##0',14:'#,##0',15:'#,##0',16:'0.00%'}});
for(let i=0;i<ordered.length;i++){let r=i+5;const sh=sheets['置信度与冲突'];sh.getRange(`H${r}`).formulas=[[`=G${r}/D${r}`]];sh.getRange(`J${r}`).formulas=[[`=I${r}/D${r}`]];sh.getRange(`Q${r}`).formulas=[[`=IF(N${r}=0,"n.a.",P${r}/N${r})`]];}
const nlabels={pins_rivets_keys:'销、铆钉、键',bearings_bushings_guides:'轴承、衬套、导向件',handles_controls:'把手与控制件',joints_clamps_structural_connectors:'连接与夹紧件',plates_discs_shapes:'板、盘与型材',pipe_fittings_valves_nozzles:'管接头、阀、喷嘴',washers_rings_spacers:'垫圈、环与隔套',screws_bolts_studs:'螺钉、螺栓、螺柱',rotating_fluid_machinery:'旋转流体机械',nuts:'螺母',springs:'弹簧',motors_gearmotors:'电机与减速电机',gears_pulleys_chains:'齿轮、带轮、链条',wheels_castors:'轮与脚轮'};
const labelMap=Object.fromEntries(taxonomy.labels.map(r=>[r.id,r.zh_aliases[0]||r.name]));
const low=[...d.low_types].sort((a,b)=>sceneOrder(a,b)||a.branch.localeCompare(b.branch)||b.low_confidence_candidates-a.low_confidence_candidates);
frame(sheets['低置信候选类型'],'低置信候选类型','类别来自候选预测，不是真实标签。N与C采用不同判定；类内比例和低置信池占比均列出。',
 ['模型','分支','候选类型ID','中文提示','候选总数','低置信数','类内低置信比例','占低置信池','低置信源三角形'],
 low.map(r=>[r.scene,r.branch,r.candidate_type,(r.branch==='N'?nlabels:labelMap)[r.candidate_type]||'',r.total_candidates,r.low_confidence_candidates,null,r.share_of_low_pool??'n.a.',r.low_faces]),
 {tableName:'LowConfidenceTypes',widths:{0:9,1:9,2:46,3:27,6:22,7:19,8:24},formats:{4:'#,##0',5:'#,##0',6:'0.00%',7:'0.00%',8:'#,##0'}});
sheets['低置信候选类型'].getRange(`G5:G${low.length+4}`).formulas=low.map((r,i)=>[`=F${i+5}/E${i+5}`]);
const geom=[...d.geometry].sort(conditionOrder);
frame(sheets['几何诊断'],'单网格简化诊断','QEM误差按两倍RMS半径归一化。不同实际预算，不能据此声称真实LOD或FPS提升。',
 ['模型','组合','树辅助','节点数','源三角形数','请求三角形数','实际三角形数','请求保留比例','实际保留比例','平均QEM误差','面数加权误差','轴收缩>10%节点','显著分量丢失节点','空几何节点'],
 geom.map(r=>[r.scene,r.mode,r.tree==='off'?'H0':'H1',r.nodes,r.source_faces,r.requested_faces,r.actual_faces,null,null,r.mean_relative_error,r.face_weighted_relative_error,r.axis_shrink_nodes,r.component_loss_nodes,r.empty_geometry_nodes]),
 {tableName:'GeometryDiagnostics',widths:{0:9,1:10,2:11,4:20,5:21,6:21,7:18,8:18,9:19,10:21,11:23,12:25,13:18},formats:{3:'#,##0',4:'#,##0',5:'#,##0',6:'#,##0',7:'0.0000%',8:'0.0000%',9:'0.00000',10:'0.00000',11:'#,##0',12:'#,##0',13:'#,##0'}});
sheets['几何诊断'].getRange(`H5:I${geom.length+4}`).formulas=geom.map((r,i)=>[`=F${i+5}/E${i+5}`,`=G${i+5}/E${i+5}`]);
const overview=sheets['总览'];const totals=d.totals_main;
frame(overview,'六模型六组合与结构树消融','六模型11,000节点；72组主实验。H0/H1配对，M与迁移回退均关闭。',
 ['配置','节点数','P1–P3','P5','P6','P8–P10','无对象级输出','功能未知','N/C冲突数','名称参考一致','对象级覆盖','QEM平均误差','实际三角形比例'],
 totals.map(r=>[r.variant,r.nodes,null,null,null,null,null,null,r.identity_conflict_count,null,null,null,null]),
 {tableName:'OverviewMain',freeze:1,widths:{0:16,1:16,2:15,3:14,4:14,5:17,6:20,7:17,8:18,9:21,10:20,11:22,12:24},formats:{1:'#,##0',2:'0.00%',3:'0.00%',4:'0.00%',5:'0.00%',6:'0.00%',7:'0.00%',8:'#,##0',9:'0.00%',10:'0.00%',11:'0.00000',12:'0.0000%'}});
for(let i=0;i<totals.length;i++){
 const r=totals[i],rr=i+5,j=viewRow.get(`ALL|${r.variant}`),g=geom.findIndex(x=>x.scene==='ALL'&&x.variant===r.variant)+5;
 overview.getRange(`C${rr}:H${rr}`).formulas=[[`=SUM('P等级节点占比'!E${j}:G${j})`,`='P等级节点占比'!I${j}`,`='P等级节点占比'!J${j}`,`=SUM('P等级节点占比'!L${j}:N${j})`,`='置信度与冲突'!H${j}`,`='置信度与冲突'!J${j}`]];
 overview.getRange(`J${rr}:M${rr}`).formulas=[[`='置信度与冲突'!Q${j}`,`='置信度与冲突'!O${j}/'置信度与冲突'!N${j}`,`='几何诊断'!J${g}`,`='几何诊断'!I${g}`]];
}
overview.getRange('A19').values=[['结构树改变哪些节点']];overview.getRange('A19').format.font={name:font,size:12,bold:true,color:blue};
overview.getRange('A21:C21').values=[['组合','P级改变占比','功能改变占比']];
overview.getRange('A21:C21').format={fill:blue,font:{name:font,size:10,color:'#FFFFFF',bold:true}};
for(let i=0;i<modes.length;i++){const r=i+22,j=pairMap.get(`ALL|${modes[i]}`);overview.getRange(`A${r}:C${r}`).values=[[modes[i],null,null]];overview.getRange(`B${r}:C${r}`).formulas=[[`='树开关变化'!E${j}`,`='树开关变化'!H${j}/'树开关变化'!C${j}`]];}
overview.getRange('B22:C27').setNumberFormat('0.00%');
const chart=overview.charts.add('bar',overview.getRange('A21:C27'));chart.setPosition('A30','H45');
chart.title='结构树改变节点占比';chart.titleTextStyle.fontSize=13;chart.titleTextStyle.typeface=font;
chart.legend={position:'top',textStyle:{typeface:font,fontSize:11}};
chart.xAxis={axisType:'textAxis',textStyle:{typeface:font,fontSize:11}};
chart.yAxis={numberFormatCode:'0.0%',numberFormatSourceLinked:false,textStyle:{typeface:font,fontSize:11}};
chart.series.items[0].fill='#345A8A';chart.series.items[1].fill='#B79A60';
const notes=sheets['方法与来源'];
const rules=[['主组合','N、C、NG、CG、NC、NCG；纯G在附表。'],
 ['H0','完全关闭装配辅助，包含父组相对尺寸；保持输入几何及世界坐标一致。'],
 ['H1','局部父组尺度与祖先名称上下文；不是纯拓扑效果。'],
 ['HT','仅拓扑和局部尺度，不用祖先名称；详见supplementary_controls.csv。'],
 ['固定开关','主72组均M=0、旧策略几何回退=0；禁用分支输入已遮蔽。'],
 ['N未可靠','p<0.70或Top1−Top2<0.08。不是错误率。'],
 ['C无对象输出','family_only或abstain；证据质量未校准，不与N概率直接比较。'],
 ['名称参考','仅655个可映射到14类的名称实例，重复件不独立；M辅助不做名称评分。'],
 ['P档位','当前主实验P1/P3/P4/P10为0；P6可能只是几何保护，不表示类别识别成功。'],
 ['几何指标','单网格QEM代理；非Hausdorff、孔洞保持率、真实cluster LOD质量或FPS。'],
 ['来源','main_72_comparison.csv、tree_pairs_with_total.csv、low_confidence_candidate_types.csv、geometry_comparison.csv。'],
 ['版本与哈希','见provenance.json及method目录；所有场景参与过开发，不是独立测试集。']];
frame(notes,'实验口径与来源','统计计数供追溯，其他工作表的比例由计数公式计算。', ['项目','定义'],rules,{widths:{0:23,1:115},freeze:0});
notes.getRange('B5:B16').format.wrapText=true;notes.getRange('A5:B16').format.rowHeight=28;
notes.getRange('B5:B16').format.horizontalAlignment='left';
notes.getRange('A19:E19').values=[['模型','场景名称','节点数','装配节点数','文件SHA256']];
notes.getRange('A19:E19').format={fill:blue,font:{name:font,size:10,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',rowHeight:26};
notes.getRange('A20:E25').values=d.inventory.map(r=>[r.scene,r.title,r.nodes,r.assembly_nodes,r.glb_sha256]);
notes.getRange('A19:A25').format.columnWidth=23;notes.getRange('B19:B25').format.columnWidth=115;notes.getRange('C19:D25').format.columnWidth=17;notes.getRange('E19:E25').format.columnWidth=80;
notes.getRange('A20:E25').format.font={name:font,size:10};notes.getRange('A20:E25').format.rowHeight=25;
notes.tabColor='#98A4AE';
// In-memory edit/restore verifies that derived proportions respond to counts rather than cached snapshots.
const probeAddress=`${keyCol.P2_count}5`;
const probeOriginal=source.getRange(probeAddress).values[0][0];
const totalNodeRow=viewRow.get('ALL|N_H0');
const beforeProbe=sheets['P等级节点占比'].getRange(`F${totalNodeRow}`).values[0][0];
source.getRange(probeAddress).values=[[probeOriginal+1]];
wb.recalculate();
const afterProbe=sheets['P等级节点占比'].getRange(`F${totalNodeRow}`).values[0][0];
if(Math.abs(afterProbe-beforeProbe-1/11000)>1e-10)throw new Error('Source edit did not recalculate pooled node fractions');
source.getRange(probeAddress).values=[[probeOriginal]];
wb.recalculate();
// Independent numeric checks reconcile all 84 formula rows to the exact backend counts.
let checked=0;
for(const name of ['P等级节点占比','P等级三角形占比']){
 const measure=name==='P等级节点占比'?'node':'face';const values=sheets[name].getRange(`E5:N${ordered.length+4}`).values;
 for(let i=0;i<values.length;i++){let sum=0;for(let p=0;p<10;p++){const v=values[i][p];if(typeof v!=='number'||Math.abs(v-ordered[i][`P${p+1}_${measure}_fraction`])>1e-10)throw new Error(`Formula reconciliation failed ${name} ${i} ${p}: ${v}`);sum+=v;checked++;}if(Math.abs(sum-1)>1e-10)throw new Error('P fractions do not sum to one');}
}
const inspection=await wb.inspect({kind:'table',range:'总览!A4:M8',include:'values,formulas',tableMaxRows:5,tableMaxCols:13,maxChars:2200});
await fs.writeFile(path.join(previewDir,'inspection.ndjson'),inspection.ndjson);
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},maxChars:2500});
await fs.writeFile(path.join(previewDir,'formula_errors.ndjson'),errors.ndjson);
for(const name of names){const endCol=name==='方法与来源'?'B':name==='低置信候选类型'?'I':name.startsWith('P等级')?'N':'M';const endRow=name==='总览'||name.startsWith('P等级')?16:14;const blob=await wb.render({sheetName:name,range:`A1:${endCol}${endRow}`,scale:1.3,format:'png'});await fs.writeFile(path.join(previewDir,`${name}.png`),new Uint8Array(await blob.arrayBuffer()));}
const chartPreview=await wb.render({sheetName:'总览',range:'A19:H46',scale:1.3,format:'png'});await fs.writeFile(path.join(previewDir,'overview_chart.png'),new Uint8Array(await chartPreview.arrayBuffer()));
const filename='六模型六组合与结构树消融完整对比.xlsx';const file=await SpreadsheetFile.exportXlsx(wb);await file.save(path.join(outputDir,filename));
await fs.copyFile(path.join(outputDir,filename),path.join(reportDir,filename));
await fs.writeFile(path.join(previewDir,'validation.json'),JSON.stringify({numeric_cells_reconciled:checked,input_change_recalculation_verified:true,sheets:names,output:path.join(reportDir,filename),error_scan:errors.ndjson},null,2));
console.log(JSON.stringify({sheets:names.length,numeric_cells_reconciled:checked,output:path.join(reportDir,filename)}));
