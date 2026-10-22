"""Create fictional format examples, never human annotation or evaluation truth."""
import hashlib
import json
import random
from pathlib import Path

FOLDER=Path(__file__).resolve().parent
ROOT=FOLDER.parents[1]
SEED=20261022
sources=json.loads((ROOT/'lod_analysis_outputs/thirteen_scene_ablation_v5_20261021/scene_sources.json').read_text(encoding='utf-8'))
# All items below are invented examples; no actual node names/IDs/predictions are read.
common=[
    ('模拟螺钉','紧固件','普通连接','紧凑形态',['P1','P2','P3']),
    ('模拟螺母','紧固件','普通连接','紧凑形态',['P2','P3']),
    ('模拟机架横梁','机架构件','支撑','杆状形态',['P4','P5','P6']),
    ('模拟底板','板件','支撑','板状形态',['P3','P4','P5']),
    ('模拟护罩','防护罩','防护','薄壳形态',['P3','P4','P5']),
    ('模拟电机壳体','电机','驱动','曲面紧凑形态',['P7','P8']),
    ('模拟导向杆','导向件','精密导向','杆状形态',['P6','P7','P9']),
    ('模拟软管','管线','流体输送','细长形态',['P4','P5','P6']),
    ('模拟连接块','连接件','装配连接','局部特征紧凑形态',['P3','P4','P5']),
    ('模拟不明异形件','无法判定','无法判定','分布复合形态',['P4','P5','P6','P7']),
]
special={
    'M1':('模拟检测相机支架','检测部件','检测辅助','板状形态',['P7','P8','P9']),
    'M2':('模拟焊接电极座','焊接部件','工艺执行','局部特征紧凑形态',['P8','P9','P10']),
    'M3':('模拟铆压头','压装部件','工艺执行','曲面紧凑形态',['P8','P9','P10']),
    'M4':('模拟极耳裁切座','裁切部件','工艺执行','局部特征紧凑形态',['P8','P9']),
    'M5':('模拟折弯夹具','夹具','定位夹持','局部特征紧凑形态',['P7','P8','P9']),
    'M6':('模拟阀体定位座','夹具','定位夹持','紧凑形态',['P7','P8','P9']),
    'M7':('模拟过渡输送滚轮','输送部件','输送','曲面紧凑形态',['P6','P7','P8']),
    'M8':('模拟转子夹持座','夹具','定位夹持','曲面紧凑形态',['P7','P8','P9']),
    'M9':('模拟点胶喷嘴座','点胶部件','工艺执行','细长形态',['P7','P8','P9']),
    'M10':('模拟刷架定位夹具','夹具','定位夹持','局部特征紧凑形态',['P7','P8','P9']),
    'M11':('模拟分选导料槽','送料部件','送料','细长片状形态',['P5','P6','P7']),
    'M12':('模拟继电器工装座','夹具','定位夹持','局部特征紧凑形态',['P7','P8','P9']),
    'M13':('模拟电池模组托盘','载具','承载定位','板状形态',['P6','P7','P8']),
}
rng=random.Random(SEED)
rows=[]
for scene in sorted(sources,key=lambda s:int(s[1:])):
    choices=common+[special[scene]]*3
    for i in range(1,41):
        label,category,role,shape,priorities=rng.choice(choices)
        rows.append(dict(dataset_kind='SYNTHETIC_FORMAT_DEMO_NOT_GROUND_TRUTH',
            is_synthetic=True,is_human_annotated=False,review_status='NOT_HUMAN_REVIEWED',
            scene_id=scene,scene_name_reference=sources[scene]['title'],
            synthetic_object_id=f'SIM_ONLY_{scene}_{i:04d}',actual_glb_node_index=None,
            actual_glb_mesh_index=None,fictional_part_description=label,
            mock_semantic_category=category,mock_function_role=role,mock_geometry_family=shape,
            hypothetical_priority=rng.choice(priorities),
            allowed_use='annotation_format_and_software_pipeline_demo_only'))
target=FOLDER/'虚构模拟标注_非人工真值.jsonl'
target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
loaded=[json.loads(line) for line in target.read_text(encoding='utf-8').splitlines()]
assert len(loaded)==520 and len({r['synthetic_object_id'] for r in loaded})==520
assert all(r['is_synthetic'] and not r['is_human_annotated'] and r['actual_glb_node_index'] is None for r in loaded)
(FOLDER/'模拟数据说明.json').write_text(json.dumps(dict(dataset_kind='SYNTHETIC_FORMAT_DEMO_NOT_GROUND_TRUTH',
    seed=SEED,scenes=13,fictional_objects=520,objects_per_scene=40,
    contains_human_annotations=False,contains_real_node_mapping=False,
    contains_algorithm_predictions=False,contains_empirical_accuracy_results=False,
    data_sha256=hashlib.sha256(target.read_bytes()).hexdigest()),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(scenes=13,fictional_objects=520,output=str(target),validation='passed'),ensure_ascii=False))
