"""V5: assign a whole-part budget separately from local geometry constraints.

The deterministic coefficients are engineering priors, not fitted accuracy or
importance probabilities. No histogram quotas or object-name rules occur here.
The V4 module remains unchanged for paired reproduction.
"""
import json
import math

from adaptive_p10_policy import PROFILES as V4_PROFILES, evidence as v4_evidence
from adaptive_p10_policy import clamp, runtime_weight, yes
from assembly_context_features import number
from factorial_ablation_policy import classify


VERSION = 'adaptive_p10_v5.0'
NAMES = {
    1: 'micro_routine_repetition', 2: 'compact_low_contribution',
    3: 'simple_bulk_or_routine_connection', 4: 'surface_and_general_shape',
    5: 'curvature_and_distributed_detail', 6: 'salient_path_and_contour',
    7: 'coupled_geometry_or_functional_interface', 8: 'operational_function',
    9: 'precision_and_process_function', 10: 'critical_function_sensitive_geometry',
}
# Keep the construction templates fixed while changing their assignment. This
# avoids attributing simultaneous template tuning to a better allocation rule.
PROFILES = {p: (NAMES[p], *v[1:]) for p, v in V4_PROFILES.items()}
ROLE_PRIOR = {
    'standard_fastener': 2, 'structural_frame': 3, 'enclosure_panel': 4,
    'mounting_support': 3, 'scene_reference_candidate': 3, 'unknown': 5,
    'motion_mount': 6, 'inspection_mount': 7, 'precision_fastener': 7,
    'fluid_connection': 7, 'feeding_material_candidate': 7,
    'sensing_inspection': 8, 'actuation': 8, 'workpiece_candidate': 8,
    'process_fixture': 9, 'precision_guidance': 9, 'transmission': 9,
    'process_tool': 9,
}
CRITICAL_ROLES = {'process_tool', 'process_fixture', 'precision_guidance', 'transmission'}
PARAMETERS = {
    'scene_small': .0025, 'scene_large': .16,
    'local_small': .02, 'local_large': .5, 'local_influence': .15,
    'size_coefficient': 2., 'shape_coefficient': 1.25,
    'detail_coefficient': .90, 'topology_coefficient': .75,
    'boundary_coefficient': .35, 'conflict_addition': .35,
    'sharp_scale': 12., 'sharp_saturation': 120.,
    'curve_scale': 24., 'curve_saturation': 256.,
    'fastener_small': .01, 'micro_size': .0025, 'micro_repeats': 4,
}


def log_range(value, low, high):
    return clamp(math.log2(max(value, low) / low) / math.log2(high / low))


def bounded_detail(value, scale, saturation):
    return clamp(math.log1p(max(0., value) / scale) / math.log1p(saturation / scale))


def geometry_evidence(row, descriptor, context, repeats=1, enabled=True):
    if not enabled:
        # Explicit neutral values: disabled G must not read geometry or repeats.
        return dict(valid=False, enabled=False, morphology='unavailable',
            geometry_family='unavailable', scene_ratio=0., local_ratio=0.,
            size_score=0., rod_demand=0., sheet_demand=0., shape_demand=0.,
            section_sensitivity=0., sharp_score=0., curve_score=0.,
            detail_score=0., topology_score=0., boundary_score=0.,
            components=1, largest_component_fraction=1., slender=False,
            thin=False, sharp_density=0., curved_density=0.,
            salience=0., sensitivity=0., repeats=1, local_scale_used=False)
    e = v4_evidence(row, descriptor, context, repeats)
    for obsolete in ('complex_shape', 'simple', 'micro'):
        e.pop(obsolete, None)
    long = max(number(row, 'obb_size_long'), 1e-12)
    mid = max(number(row, 'obb_size_mid'), 1e-12)
    short = max(number(row, 'obb_size_short'), 1e-12)
    aspect = long / mid
    sheet_aspect = mid / short
    rod = log_range(aspect, 2., 32.)
    sheet = log_range(sheet_aspect, 4., 64.)
    morphology = ('elongated_sheet' if aspect >= 4. and sheet_aspect >= 8.
                  else 'rod' if aspect >= 4.
                  else 'plate' if sheet_aspect >= 8. else 'compact')
    area = max(number(descriptor, 'surface_area_normalized', 1.), 1e-8)
    boundary_density = number(descriptor, 'boundary_length_normalized') / math.sqrt(area)
    sharp = bounded_detail(e['sharp_density'], PARAMETERS['sharp_scale'], PARAMETERS['sharp_saturation'])
    curve = bounded_detail(e['curved_density'], PARAMETERS['curve_scale'], PARAMETERS['curve_saturation'])
    boundary = bounded_detail(boundary_density, 4., 64.)
    largest = clamp(number(descriptor, 'largest_component_area_fraction', 1.))
    component_count = log_range(max(1, e['components']), 1., 16.)
    dispersed_area = clamp((1. - largest) / .5)
    topology = .45 * component_count + .55 * dispersed_area
    detail = .55 * sharp + .45 * curve
    global_size = log_range(e['scene_ratio'], PARAMETERS['scene_small'], PARAMETERS['scene_large'])
    local_used = (context.get('local_group_index') is not None
                  and int(number(context, 'local_group_members')) >= 3)
    local_ratio = max(number(context, 'group_size_ratio'), 0.) if local_used else 0.
    local_size = log_range(local_ratio, PARAMETERS['local_small'], PARAMETERS['local_large']) if local_used else 0.
    size = global_size + PARAMETERS['local_influence'] * local_size * (1. - global_size)
    if not e['valid']:
        family = 'invalid_geometry'
    elif topology >= .5:
        family = 'distributed_components'
    elif morphology != 'compact':
        family = morphology
    else:
        family = 'compact_featured' if sharp >= .45 else 'compact_curved' if curve >= .3 else 'compact_simple'
    # Plate thinness is a strong LOCAL constraint, but only a modest whole-part
    # budget signal: a large flat panel is often cheap to represent faithfully.
    e.update(enabled=True, morphology=morphology, geometry_family=family,
        aspect=aspect, sheet_aspect=sheet_aspect, local_ratio=local_ratio,
        local_scale_used=local_used, size_score=size, rod_demand=rod,
        sheet_demand=sheet, shape_demand=max(rod, .35 * sheet),
        section_sensitivity=max(rod, sheet), sharp_score=sharp,
        curve_score=curve, detail_score=detail, topology_score=topology,
        boundary_score=boundary, boundary_density=boundary_density,
        largest_component_fraction=largest)
    return e


def geometry_budget(e, conflict=False):
    if not e['valid']:
        return 5, None, {'invalid_or_disabled_geometry': True}
    terms = dict(origin=2., size=PARAMETERS['size_coefficient'] * e['size_score'],
        shape=PARAMETERS['shape_coefficient'] * e['shape_demand'],
        detail=PARAMETERS['detail_coefficient'] * e['detail_score'],
        topology=PARAMETERS['topology_coefficient'] * e['topology_score'],
        boundary=PARAMETERS['boundary_coefficient'] * e['boundary_score'],
        conflict=PARAMETERS['conflict_addition'] if conflict else 0.)
    score = sum(terms.values())
    return max(2, min(7, int(math.floor(score + .5)))), score, terms


def reprofile(base, row, descriptor, context, repeats=1, geometry_enabled=True):
    result = dict(base)
    role = base.get('function_role', 'unknown')
    conflict = any(yes(base.get(k)) for k in
        ('class_conflict', 'type_shape_conflict', 'name_function_type_conflict'))
    e = geometry_evidence(row, descriptor, context, repeats, geometry_enabled)
    gp, score, terms = geometry_budget(e, conflict)
    prior = ROLE_PRIOR.get(role, 5)
    if role == 'scene_reference_candidate' and not yes(base.get('scene_reference_geometry_supported')):
        prior = 5
    reasons = []
    ordinary_discount = False
    if not geometry_enabled:
        p = prior
        route = 'function_only_geometry_disabled'
    elif not e['valid']:
        p = max(prior, 5)
        route = 'invalid_geometry_conservative'
    elif role == 'unknown':
        p = gp
        route = 'unresolved_function_geometry_budget'
    elif role == 'standard_fastener':
        p = gp
        # Thread/fillet complexity alone does not imply a large whole-part budget.
        # A large or contradictory fastener prediction receives NO discount.
        ordinary_discount = (not conflict and e['scene_ratio'] < PARAMETERS['fastener_small']
            and e['components'] <= 2 and e['largest_component_fraction'] >= .9)
        if ordinary_discount:
            p = max(2, gp - 1)
            reasons.append('small_routine_connection_budget_discount')
        if (ordinary_discount and e['scene_ratio'] < PARAMETERS['micro_size']
                and e['repeats'] >= PARAMETERS['micro_repeats'] and e['aspect'] <= 20.):
            p = 1
            reasons.append('micro_repeated_routine_connection')
        route = 'routine_fastener_geometry_budget'
    elif role == 'scene_reference_candidate' and not yes(base.get('scene_reference_geometry_supported')):
        p = gp
        route = 'unverified_reference_uses_geometry_not_name'
    else:
        p = max(prior, gp)
        route = 'function_prior_and_geometry_budget'
    if (geometry_enabled and e['valid'] and role in CRITICAL_ROLES
            and (e['section_sensitivity'] >= .65
                 or (e['topology_score'] >= .6 and e['detail_score'] >= .6))):
        p = 10
        reasons.append('critical_function_and_section_or_distributed_detail')
    reasons.insert(0, route)
    if conflict:
        reasons.append('conflict_prevents_fastener_discount_and_tightens_error')
    name, near, mid, far, _, lock, tolerance = PROFILES[p]
    if e['valid']:
        section, topology = e['section_sensitivity'], e['topology_score']
        # Local guards survive a low P: plate thickness, rod cross-section,
        # sharp features and disconnected components are not bought by P6.
        lock = min(.16, lock + .012 * section + .010 * e['sharp_score'] + .012 * topology)
        soft = .60 + .045 * p + .15 * section + .10 * topology
        feature_weight = .60 + .07 * p + .12 * section + .08 * e['sharp_score']
        tolerance /= 1. + .7 * section + .4 * topology + .15 * e['boundary_score']
        if conflict:
            tolerance *= .85
        shrink = .05 - .035 * section
        protect = .82 - .06 * section
        adjustment = .04 * e['salience'] + .025 * e['sensitivity']
        # Retain the V4 visual selection formula for this allocation study.
        # Rendering quality and frame time still need a paired measurement.
        weight = runtime_weight(p, e)
        base_ratio = min(.90, .45 + .045 * p)
        base_error = min(.0008 / (1. + .18 * p), tolerance * .05)
    else:
        lock = max(lock, .035)
        soft, feature_weight, tolerance, shrink, protect = 1., 1., min(tolerance, .01), .025, .82
        adjustment = 0.
        weight = PROFILES[p][4]
        base_ratio, base_error = 1., 0.
    near = min(.90, near + adjustment)
    mid = min(near, mid + .8 * adjustment)
    far = min(mid, far + .6 * adjustment)
    result.update(lod_priority=f'P{p}_{name}', lod_strategy=f'{VERSION}_{name}',
        strategy=f'{VERSION}_{name}', policy_version=VERSION, policy_schema=4,
        preservation_profile=name, target_ratio_near=near, target_ratio_mid=mid,
        target_ratio_far=far, feature_lock_budget=lock, feature_lock_floor=lock,
        feature_soft_floor=soft, feature_soft_scale=soft,
        feature_weight_scale=feature_weight, feature_protect_threshold=protect,
        feature_critical_threshold=.98, simplify_error_relative=tolerance,
        max_axis_shrink=shrink, preserve_components=True,
        screen_error_weight=weight, base_simplify_ratio=base_ratio,
        base_error_relative=base_error, allow_cull=False, allow_geometry_drop=False,
        visibility_status='unmeasured', measured_visibility_used=False,
        geometry_evidence_json=json.dumps(e, sort_keys=True, allow_nan=False),
        geometry_budget_priority=gp if geometry_enabled else '',
        geometry_budget_score=score if score is not None else '',
        geometry_budget_terms_json=json.dumps(terms, sort_keys=True, allow_nan=False),
        budget_morphology=e['morphology'], geometry_budget_family=e['geometry_family'],
        budget_route=route, ordinary_fastener_discount=ordinary_discount,
        profile_reason=';'.join(reasons),
        profile_scope='whole-part budget; local section/detail constraints are independent',
        runtime_guard_scope='local feature weights + group extent + component presence + finite QEM',
        runtime_visual_quality_rule='V4 visual formula; new whole-part budget assignment; not a performance claim')
    # Do not carry a previous schedule search or its quality claim into a new rule.
    for field in ('quality_fallback', 'measured_proxy_faces', 'measured_proxy_error',
                  'measured_proxy_axis_shrink', 'measured_proxy_component_loss',
                  'quality_search', 'runtime_scope'):
        result.pop(field, None)
    return result


def infer(row, opened, taxonomy, thresholds, descriptor, context, config,
          mode='NCG', tree='off', leaf=False, repeats=1):
    base = classify(row, opened, taxonomy, thresholds, descriptor, context, config, mode, tree, leaf)
    return reprofile(base, row, descriptor, context, repeats, 'G' in mode)
