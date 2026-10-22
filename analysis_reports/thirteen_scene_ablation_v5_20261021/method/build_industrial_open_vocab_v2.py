"""Author the versioned industrial visual taxonomy (no scene-specific model codes)."""
import json
from pathlib import Path

# id | English noun | Chinese name/metadata aliases | family | visible shape description | optional 14-class mapping
CATALOG = '''hex_nut|hexagonal nut|螺母,六角螺母|fastening|a six-sided body with a circular threaded hole|nuts
cap_nut|domed cap nut|盖形螺母|fastening|a hexagonal base and a rounded closed end|nuts
slotted_nut|slotted nut|开槽螺母|fastening|a hexagonal nut with slots cut around its end|nuts
socket_screw|socket head screw|内六角螺钉|fastening|a cylindrical head with a hexagonal recess and a threaded shaft|screws_bolts_studs
hex_bolt|hexagonal head bolt|螺栓,螺钉,螺丝|fastening|a six-sided head and a long threaded cylindrical shaft|screws_bolts_studs
threaded_stud|threaded stud|螺柱,螺杆|fastening|a straight threaded rod without a large head|screws_bolts_studs
flat_washer|flat washer|平垫圈,垫圈|fastening|a thin flat circular ring with a central hole|washers_rings_spacers
retaining_ring|retaining ring|卡簧,挡圈|fastening|an open circular clip with a gap and narrow ring profile|washers_rings_spacers
dowel_pin|dowel pin|定位销,圆柱销|fastening|a solid smooth cylindrical pin with flat or rounded ends|pins_rivets_keys
shaft_key|shaft key|平键,键条|fastening|a short rectangular bar with rounded or flat ends|pins_rivets_keys
rolling_bearing|rolling bearing|轴承|guidance|concentric inner and outer rings enclosing rolling elements|bearings_bushings_guides
linear_rail|linear guide rail|直线导轨,导轨,滑轨|guidance|a long straight profiled rail with repeated mounting holes|bearings_bushings_guides
linear_slider|linear guide carriage|滑块,直线滑块|guidance|a compact rectangular carriage with an open rail channel|bearings_bushings_guides
guide_bushing|guide bushing|导套|guidance|a hollow cylindrical sleeve with a smooth central bore|bearings_bushings_guides
guide_shaft|guide shaft|导柱,导向轴|guidance|a long smooth cylindrical shaft with a precise straight axis|bearings_bushings_guides
spur_gear|spur gear|齿轮|transmission|a circular wheel with repeated teeth around its rim|gears_pulleys_chains
timing_pulley|timing pulley|同步轮,皮带轮|transmission|a circular pulley with rim grooves and a central shaft hole|gears_pulleys_chains
timing_belt|timing belt|同步带,皮带|transmission|a long flexible loop with regularly spaced teeth or ribs|gears_pulleys_chains
sprocket|chain sprocket|链轮|transmission|a circular wheel with widely spaced pointed chain teeth|gears_pulleys_chains
lead_screw|lead screw|丝杆,丝杠|transmission|a long shaft with continuous helical threads|gears_pulleys_chains
shaft_coupling|shaft coupling|联轴器|transmission|a cylindrical connector with two aligned shaft bores|joints_clamps_structural_connectors
coil_spring|coil spring|弹簧|transmission|a wire wound into a repeated open helical coil|springs
electric_motor|electric motor|伺服电机,电机,马达|actuation|a motor housing with an output shaft and mounting flange|motors_gearmotors
gear_reducer|gear reducer|减速机,减速器|actuation|a compact gearbox housing with shaft and mounting faces|motors_gearmotors
pneumatic_cylinder|pneumatic cylinder|气缸,气动缸,无杆缸|actuation|an actuator body with a sliding piston rod and end fittings|
rotary_actuator|rotary pneumatic actuator|旋转气缸|actuation|a compact actuator with a circular output table on a block body|
linear_stage|linear positioning stage|滑台,直线模组|actuation|a moving carriage on a rectangular linear guide base|
parallel_gripper|parallel gripper|气爪,夹爪|actuation|two opposed gripping jaws on a compact actuator body|
industrial_camera|industrial camera|工业相机,相机,摄像头|sensing|a box-shaped camera housing with a round lens mount|
camera_lens|camera lens|镜头|sensing|a cylindrical optical barrel with a circular glass opening|
photoelectric_sensor|photoelectric sensor|光电感应器,光电开关,光电传感器|sensing|a compact sensor housing with an optical window and cable|
proximity_sensor|proximity sensor|接近开关,接近传感器|sensing|a threaded cylindrical sensor with a flat sensing face|
inspection_light|machine vision light|光源,环形灯|sensing|a ring or rectangular light housing surrounding an illuminated opening|
temperature_controller|temperature controller|温控表,温度控制器|sensing|a rectangular panel instrument with a display face|
suction_cup|vacuum suction cup|吸盘,吸盘头|fluid|a hollow bell-shaped cup with a flexible circular rim|
pipe_fitting|pipe fitting|接头,管接头|fluid|a hollow angled or straight connector with round tube ports|pipe_fittings_valves_nozzles
industrial_valve|industrial valve|电磁阀,气阀,阀体|fluid|a block or cylindrical valve body with multiple fluid ports|pipe_fittings_valves_nozzles
manifold|pneumatic manifold|阀岛,集气块|fluid|a rectangular block with repeated ports and fittings|
nozzle|industrial nozzle|气嘴,喷嘴|fluid|a tapered hollow outlet with a narrow opening|pipe_fittings_valves_nozzles
flexible_hose|flexible hose|气管,软管|fluid|a curved hollow tube following a flexible path|pipe_fittings_valves_nozzles
seal_ring|seal ring|密封圈,O型圈|fluid|a continuous circular sealing ring with a rounded cross section|washers_rings_spacers
pump|industrial pump|真空泵,水泵|fluid|a pump casing with an impeller chamber and pipe connections|rotating_fluid_machinery
vacuum_chamber|vacuum chamber|真空盒,真空箱|fluid|a sealed rectangular chamber with an open cavity or cover|
press_tool|pressing tool|铆头,压头,铆压头|tooling|a rigid tool with a narrow contact tip and mounting shank|
punch_tool|punch tool|冲头,冲针|tooling|a long rigid punch with a shaped working tip|
die_block|forming die|模具,摸具,冲模|tooling|a rigid die block with precise shaped cavities or slots|
fixture|workholding fixture|治具,夹具,载具|tooling|a fixture body with locating holes stops or clamping surfaces|
feeder_bowl|vibratory feeder bowl|螺母送料盘,振动盘|handling|a circular bowl with an internal spiral feeding track|
feed_track|part feeding track|送料轨道,上料轨道,转运轨道|handling|a long narrow profiled track guiding individual parts|
conveyor_roller|conveyor roller|滚筒,包胶滚筒|handling|a cylindrical roller with an axial shaft at both ends|
conveyor_belt|conveyor belt|输送带|handling|a wide flat continuous loop supported by rollers|
aluminum_extrusion|aluminum extrusion|铝型材,型材|structure|a long structural beam with longitudinal slots and a profiled cross section|
frame_beam|machine frame beam|横梁,立柱,机架|structure|a long rigid rectangular structural member|
mounting_bracket|mounting bracket|支架,角码,斜撑|structure|an angled or bent support with flat mounting faces|joints_clamps_structural_connectors
mounting_plate|mounting plate|安装板,固定板,固定件,固定座,固定块,垫板,安装件,底板,支座|structure|a flat rigid plate or block with mounting holes and contact faces|plates_discs_shapes
machine_base|machine base|底座,底架|structure|a broad rigid support base carrying machine assemblies|
machine_cover|machine cover panel|盖板,门板,护罩,防护罩,外壳,箱盖,盒盖|enclosure|a sheet-like cover or housing panel with an outer silhouette|
cabinet|machine cabinet|机箱,电柜|enclosure|a hollow box enclosure with walls and access openings|
handle|machine handle|把手,手柄|enclosure|a curved or straight grip joined to mounting ends|handles_controls
adjustable_foot|adjustable machine foot|地脚,脚杯,轮脚|structure|a circular support pad joined to a threaded stem|
connector|electrical connector|连接器,链接器,端子|electrical|a compact housing with repeated pins contacts or sockets|
cooling_fan|cooling fan|风扇|electrical|a circular rotor with radial blades in a frame|
cable|electrical cable|电缆,线束|electrical|one or several long thin flexible insulated wires|
feature_plate|unidentified perforated plate|孔板|shape|a flat plate containing holes or cutouts without an identified function|plates_discs_shapes
plain_shaft|unidentified cylindrical shaft|圆轴|shape|a solid elongated cylinder without visible functional detail|
plain_block|unidentified solid block|方块|shape|a rectangular solid without a recognizable mechanism|plates_discs_shapes'''

FAMILIES = {
    'fastening': ('fastener', 'bolts nuts washers retaining rings and locating pins', {'repeated_fastener': .85, 'balanced_visible': .15}),
    'guidance': ('precision guide component', 'bearings linear rails carriages and guide shafts', {'motion_precision': .85, 'critical_preserve': .15}),
    'transmission': ('mechanical transmission component', 'gears pulleys belts couplings and helical springs', {'motion_precision': .55, 'critical_preserve': .45}),
    'actuation': ('industrial actuator', 'electric motors pneumatic cylinders positioning stages and grippers', {'motion_precision': .65, 'critical_preserve': .35}),
    'sensing': ('industrial sensing or inspection component', 'cameras optical sensors machine vision lights and measurement instruments', {'structural_control': .8, 'balanced_visible': .2}),
    'fluid': ('fluid or vacuum component', 'valves pipe connectors nozzles suction cups and sealed chambers', {'interface_fluid': .9, 'balanced_visible': .1}),
    'tooling': ('manufacturing tool or fixture', 'press tools forming dies punches and workholding fixtures', {'motion_precision': .55, 'structural_control': .45}),
    'handling': ('material handling component', 'feeding bowls part tracks rollers and conveyors', {'structural_control': .6, 'balanced_visible': .4}),
    'structure': ('machine structural component', 'beams extrusions support brackets and mounting plates', {'structural_control': .55, 'balanced_visible': .45}),
    'enclosure': ('machine enclosure component', 'covers cabinets door panels and handles', {'balanced_visible': .75, 'ordinary_low_detail': .25}),
    'electrical': ('electrical machine component', 'electrical connectors terminals cooling fans and cables', {'structural_control': .65, 'balanced_visible': .35}),
    'shape': ('unidentified geometric component', 'a plate cylinder or block whose function is not identifiable', {'balanced_visible': 1.0}),
}


def build():
    labels = []
    for line in CATALOG.splitlines():
        label_id, name, chinese, family, description, known = line.split('|')
        labels.append({'id': label_id, 'name': name, 'zh_aliases': chinese.split(','), 'family': family,
                       'description': description, 'known_class': known or None,
                       'generic_shape': family == 'shape', 'role_priors': FAMILIES[family][2],
                       'prompts': [f'a rendered industrial {name}.', f'a CAD model of {name}, {description}.',
                                   f'a grayscale view of a mechanical {name}.']})
    return {'schema': 2, 'name': 'industrial_open_v2', 'scope': 'Visual taxonomy; CAD names are separate optional weak evidence.',
            'unknown_is_rejection_not_label': True,
            'families': [{'id': key, 'name': name, 'description': description,
                          'prompts': [f'a CAD rendering of a {name}.', f'industrial components including {description}.']}
                         for key, (name, description, roles) in FAMILIES.items()],
            'negative_prompts': ['a blank white image with no visible object.', 'an unrecognizable shape without identifiable detail.',
                                 'an empty grayscale background.'], 'labels': labels}


if __name__ == '__main__':
    path = Path(__file__).with_name('industrial_open_vocab_v2.json')
    data = build()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {path}: {len(data["labels"])} object labels, {len(data["families"])} families')
