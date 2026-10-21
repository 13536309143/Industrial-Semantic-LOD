"""Version 4: relational semantics select intent; geometry constrains simplification.

P is an ordered preservation profile, never a calibrated importance probability.
No policy removes geometry or infers invisibility from size/name/confidence.
"""
import json
import math

from assembly_context_features import number
from factorial_ablation_policy import classify


VERSION = 'adaptive_p10_v4.2'
# name, near/mid/far per-iteration ratios, screen weight, lock budget, relative QEM limit
# Ratios describe hierarchy construction, not final resident triangle percentages.
PROFILES = {
    1: ('micro_repeated_low_impact', .38, .20, .08, .48, .008, .025),
    2: ('ordinary_repeated_parts', .44, .24, .10, .62, .012, .022),
    3: ('simple_static_bulk', .48, .28, .12, .76, .018, .018),
    4: ('ordinary_surface_support', .52, .32, .16, .88, .025, .016),
    5: ('unresolved_balanced', .58, .38, .22, 1.00, .035, .014),
    6: ('thin_slender_silhouette', .60, .42, .26, 1.10, .050, .010),
    7: ('functional_interface', .64, .46, .30, 1.22, .060, .009),
    8: ('functional_control_motion', .66, .50, .34, 1.34, .075, .008),
    9: ('precision_guidance_transmission', .70, .54, .40, 1.50, .090, .006),
    10: ('critical_process_detail', .76, .60, .48, 1.72, .120, .004),
}
ROLE_P = {
    'standard_fastener': 2, 'structural_frame': 3, 'enclosure_panel': 4,
    'mounting_support': 4, 'scene_reference_candidate': 4, 'unknown': 5,
    'motion_mount': 6, 'inspection_mount': 7, 'precision_fastener': 7,
    'fluid_connection': 7, 'feeding_material_candidate': 7,
    'sensing_inspection': 8, 'actuation': 8, 'workpiece_candidate': 8,
    'process_fixture': 9, 'precision_guidance': 9, 'transmission': 9,
    'process_tool': 9,
}
CRITICAL_ROLES = {'process_tool', 'process_fixture', 'precision_guidance', 'transmission'}


def yes(value):
    return str(value).lower() in {'true', '1'}


def clamp(value, low=0., high=1.):
    return max(low, min(high, value))


def runtime_weight(priority, geometry_evidence):
    # Geometry storage and visual selection are separate budgets. Aggressive
    # hierarchy steps need a stricter on-screen tolerance to avoid visible jumps.
    e=geometry_evidence
    return PROFILES[priority][4]*(1.+.10*e['salience'])*(2.+1.5*e['sensitivity']+.5*e['salience'])


def evidence(row, descriptor, context, repeats=1):
    """Geometry-only inputs. Scene size is potential visibility, not occlusion."""
    long = max(number(row, 'obb_size_long'), 1e-12)
    mid = max(number(row, 'obb_size_mid'), 1e-12)
    short = max(number(row, 'obb_size_short'), 0.)
    scale = max(number(context, 'scene_size_ratio'), 0.)
    area = max(number(descriptor, 'surface_area_normalized', 1.), 1e-8)
    # Edge length per sqrt(area) removes size/tessellation-count as a direct proxy.
    sharp = number(descriptor, 'sharp_length_normalized') / math.sqrt(area)
    curved = number(descriptor, 'curved_length_normalized') / math.sqrt(area)
    slender = long / mid >= 6.
    thin = short / long < .025
    components = int(number(descriptor, 'significant_components', 1))
    valid = yes(descriptor.get('geometry_valid', False))
    complex_shape = sharp > 28. or curved > 32. or components >= 4
    simple = valid and components == 1 and sharp < 16. and curved < 16. and not slender
    salience = clamp(math.sqrt(scale / .12))
    sensitivity = max(.75 if slender or thin else 0., clamp(sharp / 55.),
                      clamp(curved / 60.), .8 if components >= 4 else 0.)
    return dict(valid=valid, scene_ratio=scale, slender=slender, thin=thin,
                components=components, sharp_density=sharp, curved_density=curved,
                complex_shape=complex_shape, simple=simple, salience=salience,
                sensitivity=sensitivity, repeats=int(repeats),
                micro=scale < .0025 and repeats >= 4)


def reprofile(base, row, descriptor, context, repeats=1, geometry_enabled=True):
    """Apply only enabled evidence; retain identity/function and uncertainty audit."""
    result = dict(base)
    e = evidence(row, descriptor, context, repeats) if geometry_enabled else dict(
        valid=False, scene_ratio=0., slender=False, thin=False, components=1,
        sharp_density=0., curved_density=0., complex_shape=False, simple=False,
        salience=0., sensitivity=0., repeats=1, micro=False)
    role = base.get('function_role', 'unknown')
    conflict = yes(base.get('class_conflict')) or yes(base.get('type_shape_conflict')) or yes(base.get('name_function_type_conflict'))
    p = ROLE_P.get(role, 5)
    reasons = ['functional_intent_' + role]
    # Unknown shape can qualify for a coarse static profile, never an object identity.
    if geometry_enabled and e['simple'] and role == 'unknown' and not conflict:
        p = 3 if e['scene_ratio'] >= .06 else 4
        reasons.append('simple_geometry_candidate_without_object_claim')
    if role == 'standard_fastener' and e['valid'] and e['micro'] and not e['complex_shape'] and not conflict:
        p = 1
        reasons.append('micro_repeated_standard_part_not_deletion')
    if geometry_enabled and (e['slender'] or e['thin']) and role != 'scene_reference_candidate':
        p = max(p, 6)
        reasons.append('cross_section_and_silhouette_floor')
    if geometry_enabled and e['complex_shape']:
        p = max(p, 6)
        reasons.append('feature_or_component_floor')
    # Highest protection requires a functional hypothesis AND geometric sensitivity.
    if role in CRITICAL_ROLES and e['valid'] and (e['slender'] or e['thin'] or e['complex_shape']):
        p = 10
        reasons.append('critical_function_with_geometric_sensitivity')
    if conflict:
        p = max(p, 5)
        reasons.append('unresolved_conflict_balanced_floor')
    if geometry_enabled and not e['valid']:
        p = max(p, 5)
        reasons.append('invalid_geometry_no_aggressive_profile')
    name, near, mid, far, weight, lock, tolerance = PROFILES[p]
    # Small continuous adjustments within a profile; no scene-specific percentile filling.
    adjustment = .04 * e['salience'] + .025 * e['sensitivity'] if geometry_enabled else 0.
    near = min(.90, near + adjustment)
    mid = min(near, mid + adjustment * .8)
    far = min(mid, far + adjustment * .6)
    weight = runtime_weight(p,e)
    if not geometry_enabled:
        lock = max(lock, .025)
    result.update(lod_priority=f'P{p}_{name}', strategy=VERSION + '_' + name,
        policy_version=VERSION, policy_schema=4, preservation_profile=name,
        target_ratio_near=near, target_ratio_mid=mid, target_ratio_far=far,
        screen_error_weight=weight, feature_lock_floor=lock,
        feature_soft_floor=.60 + .045*p, feature_lock_budget=lock,
        feature_soft_scale=.60 + .045*p, feature_weight_scale=.60 + .07*p,
        feature_protect_threshold=.82, feature_critical_threshold=.98,
        simplify_error_relative=tolerance, max_axis_shrink=.025 if p >= 6 else .06,
        preserve_components=True, allow_cull=False, allow_geometry_drop=False,
        visibility_status='unmeasured', measured_visibility_used=False,
        geometry_evidence_json=json.dumps(e, sort_keys=True),
        profile_reason=';'.join(reasons),
        profile_scope='ordered geometry preservation; not an object classification',
        runtime_guard_scope='per-group AABB extent + welded connected-component presence + incremental QEM')
    return result


def infer(row, opened, taxonomy, thresholds, descriptor, context, config,
          mode='NCG', tree='off', leaf=False, repeats=1):
    base = classify(row, opened, taxonomy, thresholds, descriptor, context, config, mode, tree, leaf)
    return reprofile(base, row, descriptor, context, repeats, 'G' in mode)

