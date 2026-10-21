"""Separate object identity, assembly function and geometry-preservation constraints."""
import json
import re

from fuse_open_semantic_v2 import fuse, builder, normalize
from open_semantic_evidence import metadata_evidence
from assembly_context_features import geometry_risk, number


FUNCTION_PRIORITY = {
    'unknown':5,'standard_fastener':2,'mounting_support':5,'structural_frame':5,
    'enclosure_panel':5,'scene_reference_candidate':4,'inspection_mount':7,'motion_mount':6,
    'process_fixture':8,'process_tool':9,'sensing_inspection':8,'actuation':8,
    'precision_guidance':9,'transmission':9,'fluid_connection':7,
    'workpiece_candidate':8,'feeding_material_candidate':7,'precision_fastener':6,
}
CLASS_FUNCTION = {
    **{c:'standard_fastener'for c in builder.FASTENER_TYPES},
    'bearings_bushings_guides':'precision_guidance','gears_pulleys_chains':'transmission',
    'motors_gearmotors':'actuation','rotating_fluid_machinery':'actuation',
    'springs':'precision_guidance','wheels_castors':'motion_mount',
    'pipe_fittings_valves_nozzles':'fluid_connection',
    'joints_clamps_structural_connectors':'mounting_support',
    'plates_discs_shapes':'unknown','handles_controls':'unknown',
}


def leaf_text(name):
    return name.replace('\\','/').split('/')[-1].split('^')[0]


def leaf_function(name):
    """Leaf functional hints are separate from ancestor context; compound supports take precedence."""
    leaf=leaf_text(name)
    rules=[
        ('mounting_support',r'支架|安装板|安装件|固定板|固定件|固定块|固定座|支撑|支座|压板|垫板|轴承座|轴承块|导轨座|(?:相机|CCD|光源|电机|马达|气缸|传感器|轴承).*(?:板|座|块|罩)'),
        ('structural_frame',r'机架|框架|铝型材|型材|立柱|横梁|底架|底座'),
        ('enclosure_panel',r'防护罩|防护门|门板|外壳|盖板|盒盖|护罩|侧板|顶板'),
        ('scene_reference_candidate',r'^(?:地面|地板|floor|ground|参考面|坐标系|标注)(?:$|[-_.\d])|场景参考|辅助坐标'),
        ('workpiece_candidate',r'产品|工件|端子料带'),
        ('sensing_inspection',r'传感器|光电|相机|镜头|光源|CCD|温控表|检测器'),
        ('actuation',r'气缸|电缸|滑台|电机|马达|减速机|减速器'),
        ('precision_guidance',r'轴承|导轨|滑块|导套|导柱|弹簧'),
        ('transmission',r'齿轮|同步轮|皮带轮|链轮|丝杆|丝杠|同步带|齿条'),
        ('process_fixture',r'夹具|治具|夹爪|工装|定位销|定位块'),
        ('process_tool',r'冲头|折弯刀|冲切刀|切刀|模具|摸具|铆头|压头|焊头|焊针'),
        ('fluid_connection',r'接头|气管|气嘴|吸盘|真空|气阀|电磁阀'),
        ('standard_fastener',r'螺钉|螺栓|螺母|螺丝|垫圈|挡圈|卡簧'),
    ]
    for role,pattern in rules:
        if re.search(pattern,leaf,re.I):return role
    return ''


def object_group(result, taxonomy):
    identity=result.get('resolved_object_class','')
    if identity.startswith(('known:','closed:')):return identity.split(':',1)[1]
    labels={x['id']:x for x in taxonomy['labels']}
    return labels.get(identity,{}).get('known_class','')


def apply_policy(row, opened, taxonomy, thresholds, descriptor, context, config,
                 mode='NCG', use_tree=False, use_leaf=False, refinement='scoped'):
    """Semantic evidence proposes identity/function; geometry and tree constrain the action."""
    use_g='G'in mode
    base_mode=mode.replace('G','') or 'C'
    # G cannot read either semantic branch. Its output begins with an unknown identity.
    visual=opened if 'C'in mode else {}
    base=fuse(row,visual,taxonomy,thresholds,base_mode)
    before_priority=int(base['lod_priority'].split('_')[0][1:])
    risk=geometry_risk(row,descriptor,context,config) if use_g else dict(flags=[],priority_floor=0,lock_floor=0.,soft_floor=0.,large=False,slender=False,thin=False)
    metadata=metadata_evidence(leaf_text(row.get('node_name','')),[],taxonomy)if use_leaf else {}
    leaf_role=leaf_function(row.get('node_name',''))if use_leaf else ''
    if metadata.get('object_id'):
        # Keep v2 generic-name/object-group distinction, without inheriting a parent's identity.
        base=fuse(row,visual,taxonomy,thresholds,base_mode,metadata)
    identity_before=base.get('resolved_object_class','')
    group=object_group(base,taxonomy)
    function=CLASS_FUNCTION.get(group,'unknown')
    label=next((label for label in taxonomy['labels']if label['id']==identity_before),None)
    if label and not group:
        family_function={'actuation':'actuation','sensing':'sensing_inspection','fluid':'fluid_connection',
            'tooling':'process_tool','structure':'mounting_support','enclosure':'enclosure_panel',
            'transmission':'transmission','guidance':'precision_guidance'}
        function=family_function.get(label['family'],'unknown')
    reasons=[]
    if leaf_role:
        function=leaf_role
        reasons.append('direct_leaf_function_weak_hint')
    # Geometry can reject an implausible standard-part interpretation, but never invent a motor/tool identity.
    shape_conflict=bool(use_g and not leaf_role and group in builder.FASTENER_TYPES and
        ((risk['slender'] and context['scene_size_ratio']>.025 and group in {'nuts','washers_rings_spacers'}) or
         (context['scene_size_ratio']>.20 and context['group_size_ratio']>.60)))
    if shape_conflict:
        function='unknown'
        base.update(resolved_object_class='',resolved_status='shape_identity_conflict',resolved_source='geometry_incompatibility_review',needs_review=True)
        reasons.append('large_or_slender_geometry_incompatible_with_standard_part_prior')
    chosen=context.get('context')if use_tree else None
    category=chosen['categories'][0]if chosen else ''
    strength=(config['context_strength']*config['scope_decay']**(chosen['distance']-1))if chosen else 0.
    context_applied=False
    if chosen and strength>=.2:
        # Function is relational; the ancestor is never copied into resolved_object_class.
        if function=='mounting_support':
            if category=='inspection':function='inspection_mount';context_applied=True
            elif category in {'actuation','handling','fixture','tooling'}:function='motion_mount';context_applied=True
        elif function=='standard_fastener' and category in {'inspection','fixture','tooling'}:
            if refinement=='broad' or (context['group_size_ratio']<=config['small_local_ratio'] and
                                      chosen['descendant_count']<=32):
                function='precision_fastener';context_applied=True
        elif function=='standard_fastener' and category=='feeding':
            material_match=bool(group=='nuts'and re.search(r'螺母.*(?:送料|上料)|(?:送料|上料).*螺母',chosen['name']))
            # Distinguish a possible feeding nut from ordinary fasteners mounted on a feeder.
            independent_type_support=leaf_role=='standard_fastener' or (
                opened.get('open_group_known_class')==group and opened.get('open_status')in {'accepted','class_only'}
                and number(row,'confidence')>=.7 and row.get('predicted_class')==group)
            if material_match and context['group_size_ratio']<=config['small_local_ratio'] and independent_type_support:
                function='feeding_material_candidate';context_applied=True
        elif function in {'precision_guidance','transmission','actuation','process_fixture','process_tool'}:
            # Record the assembly relation without raising every sibling's priority.
            context_applied=category in {'fixture','tooling','actuation','handling','feeding','inspection'}
    priority=FUNCTION_PRIORITY[function]
    # Coarse visual roles cannot establish a precise function. Retain their existing conservative policy.
    if function=='unknown':priority=max(5,min(before_priority,7))
    if base.get('resolved_status')=='conflict':priority=max(priority,7)
    priority=max(priority,risk['priority_floor'])
    if function=='scene_reference_candidate' and not (risk['thin'] and risk['large']):
        # A suggestive name alone cannot downrank an arbitrary solid object as scenery.
        priority=max(priority,5)
        reasons.append('auxiliary_hint_not_verified_as_large_planar_reference')
    policy=builder.make_p10_policy(priority,'assembly_context_v3_'+function)
    policy['lod_priority']=f'P{priority}_{function}'
    lock_floor=risk['lock_floor'];soft_floor=risk['soft_floor']
    if risk['slender'] or risk['thin']:
        policy['target_ratio_near']=max(policy['target_ratio_near'],.84)
        policy['target_ratio_mid']=max(policy['target_ratio_mid'],.58)
        policy['target_ratio_far']=max(policy['target_ratio_far'],.32)
        reasons.append('cross_section_and_silhouette_constraints')
    if base.get('class_conflict')or shape_conflict or function!='standard_fastener' or risk['flags']:
        policy['allow_cull']=False
    if context_applied:reasons.append('scoped_assembly_function_relation')
    reasons.extend(risk['flags'])
    result={**base,**policy,'evidence_mode':mode+('H'if use_tree else '')+('M'if use_leaf else ''),
        'inferred_role':function,'function_role':function,'object_class_before_context':identity_before,
        'functional_role_source':'leaf_and_assembly_weak_evidence'if use_leaf and context_applied else
                                 'leaf_name_weak_evidence'if leaf_role else 'ancestor_and_visual_relation'if context_applied else
                                 'semantic_identity_prior'if function!='unknown'else 'unresolved_function',
        'assembly_context_category':category,'assembly_context_strength':strength,
        'assembly_context_applied':context_applied,'assembly_context_json':json.dumps(chosen,ensure_ascii=False)if chosen else '',
        'geometry_risk_json':json.dumps(risk),'geometry_descriptor_json':json.dumps(descriptor)if use_g else '',
        'scene_size_ratio':context['scene_size_ratio'],'local_size_ratio':context['group_size_ratio'],
        'local_group_index':context['local_group_index'],'local_group_members':context['local_group_members'],
        'feature_lock_floor':lock_floor,'feature_soft_floor':soft_floor,
        'type_shape_conflict':shape_conflict,'leaf_names_used':use_leaf,'ancestor_names_used':use_tree,
        'scene_auxiliary_candidate':function=='scene_reference_candidate','needs_review':bool(base['needs_review']or context_applied or leaf_role or shape_conflict),
        'inference_reason':';'.join(reasons)or 'identity_function_and_geometry_separated',
        'allow_geometry_drop':False,'visibility_status':'unmeasured',
        'geometry_constraint_quality':1. if descriptor.get('geometry_valid')and use_g else 0.,
        'policy_version':config['version'],'calibrated_function_probability':False}
    return result
