"""Generic weak name evidence shared by scene-analysis tools.

Case-specific reporting and observations are not part of the public source.
"""
import re

KNOWN = {'motor_or_reducer': {'motors_gearmotors'},
         'bearing_or_linear_guide': {'bearings_bushings_guides'},
         'transmission': {'gears_pulleys_chains'}, 'spring': {'springs'}}


def expected_known_classes(row):
    """Conservative lexical candidates, not automatically verified labels."""
    role = row['name_candidate_role']
    leaf = row['leaf_name']
    if role == 'mount_or_support':
        return set()
    if role == 'fastener':
        if re.search(r'送料|上料|轨道|支架|固定|安装|座|盘|机构|锁螺丝', leaf):
            return set()
        if '螺母' in leaf:
            return {'nuts'}
        if re.search(r'螺钉|螺栓|螺丝', leaf):
            return {'screws_bolts_studs'}
        if re.search(r'垫圈|挡圈|卡簧', leaf):
            return {'washers_rings_spacers'}
    return KNOWN.get(role, set())
