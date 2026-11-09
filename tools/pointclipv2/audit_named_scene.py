"""Audit real CAD names/hierarchy as weak evidence, never as ground truth."""
import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/pointnext_lod'))
from artifact_cache import file_hash
from run_pointclipv2_zeroshot_glb import read_glb_chunks, node_matrix

# Local, auditable lexical hints. Manufacturer codes need a verified catalog.
RULES = [
    ('environment', r'地面|地板|floor|ground'),
    ('mount_or_support', r'支架|安装板|固定板|固定座|固定件|固定块|底板|垫板|支撑|支座|安装上板|安装垫块|安装件|(?:马达|电机|气缸|滑台|CCD|光源).*(?:板|座|斜撑|垫块|固定块)'),
    ('sensing_or_inspection', r'传感器|光电|相机|镜头|光源|CCD|温控表|检测器'),
    ('motor_or_reducer', r'电机|马达|减速机|减速器'),
    ('pneumatic_or_linear_actuator', r'气缸|电缸|滑台'),
    ('bearing_or_linear_guide', r'轴承|导轨|滑块|导套|导柱'),
    ('transmission', r'齿轮|同步轮|皮带轮|链轮|丝杆|丝杠|同步带|齿条'),
    ('fastener', r'螺钉|螺栓|螺母|垫圈|挡圈|卡簧'),
    ('spring', r'弹簧'),
    ('fluid_or_vacuum', r'接头|气管|气嘴|吸盘|真空|气阀|电磁阀'),
    ('tooling_or_process', r'夹爪|夹具|治具|冲头|折弯|冲切|模具|植入|铆头|压头|铆压|压装'),
    ('frame_or_structure', r'铝型材|型材|机架|立柱|横梁|框架|底座'),
    ('enclosure', r'防护罩|防护门|门板|外壳|盖板|盒盖|护罩'),
    ('product_or_workpiece', r'产品|喇叭|连接器|链接器|透气阀'),
]
PROCESS_RE = r'冲裁|裁切|冲切|搬运|送料|放料|收料|上料|摆盘|分割盘|变距|升降|移栽|检测|真空|排气|测试|皮带|框架|模具|摸具|铆压|铆接|锁螺丝|装胶帽|下料|移位|焊脚'


def name_hints(name):
    # Exported slash prefixes describe assembly context, not the leaf identity.
    leaf = name.replace('\\', '/').split('/')[-1]
    hits = [(role, match.group()) for role, pattern in RULES
            if (match := re.search(pattern, leaf, flags=re.I))]
    return leaf, hits


def hierarchy_paths(nodes):
    paths = {}
    for index in range(len(nodes)):
        chain, cursor = [], index
        while cursor is not None and cursor not in paths:
            chain.append(cursor)
            cursor = nodes[cursor]['_parent_index']
        prefix = [] if cursor is None else list(paths[cursor])
        for item in reversed(chain):
            prefix = prefix + [item]
            paths[item] = prefix
    return paths


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def triangle_counts(data):
    result = []
    for mesh in data.get('meshes', []):
        total = 0
        for primitive in mesh.get('primitives', []):
            count = data['accessors'][primitive['indices']]['count'] if 'indices' in primitive else data['accessors'][primitive['attributes']['POSITION']]['count']
            mode = primitive.get('mode', 4)
            total += count // 3 if mode == 4 else max(0, count - 2) if mode in (5, 6) else 0
        result.append(total)
    return result


def audit(glb, output, run=None):
    data, _, _ = read_glb_chunks(glb)
    nodes, meshes = data.get('nodes', []), data.get('meshes', [])
    paths = hierarchy_paths(nodes)
    active_roots = data.get('scenes', [{}])[data.get('scene', 0)].get('nodes', [])
    active = set()
    pending = list(active_roots)
    while pending:
        index = pending.pop()
        if index in active:
            continue
        active.add(index)
        pending.extend(nodes[index].get('children', []))
    faces = triangle_counts(data)
    pn_path = run / f'{glb.stem}_pointnext_analysis.csv' if run else None
    pn = {int(row['node_index']): row for row in read_csv(pn_path)} if pn_path and pn_path.exists() else {}
    clip_path = run / f'{glb.stem}_pointclipv2_full.csv' if run else None
    clip = {int(row['node_index']): row for row in read_csv(clip_path)} if clip_path and clip_path.exists() else {}
    rows, tree = [], []
    local_diff, inherited_nonuniform, inherited_rotation = 0, 0, 0
    for index, node in enumerate(nodes):
        name = node.get('name', '')
        path_ids = paths[index]
        path_names = [nodes[i].get('name', f'node_{i}') for i in path_ids]
        leaf, hints = name_hints(name)
        tree.append({'node_index': index, 'parent_index': node['_parent_index'], 'depth': len(path_ids) - 1,
                     'name': name, 'mesh_index': node.get('mesh', ''), 'children': json.dumps(node.get('children', [])),
                     'path_ids': '/'.join(map(str, path_ids)), 'path_names': ' > '.join(path_names), 'active_scene': index in active})
        if 'mesh' not in node:
            continue
        parent_names = path_names[:-1]
        context = [{'node': i, 'name': nodes[i].get('name', ''), 'hints': name_hints(nodes[i].get('name', ''))[1]}
                   for i in path_ids[:-1] if name_hints(nodes[i].get('name', ''))[1]]
        world, local = node['_world_matrix'], node_matrix(node)
        changed = not np.allclose(world, local, rtol=1e-8, atol=1e-7)
        local_diff += changed
        parent = np.eye(4) if node['_parent_index'] is None else nodes[node['_parent_index']]['_world_matrix']
        scales = np.linalg.svd(parent[:3, :3], compute_uv=False)
        nonuniform = bool(max(scales) - min(scales) > max(max(scales), 1e-12) * 1e-6)
        inherited_nonuniform += nonuniform
        column_scales = np.linalg.norm(parent[:3, :3], axis=0)
        rotation_changed = not np.allclose(parent[:3, :3] / np.maximum(column_scales, 1e-12), np.eye(3), atol=1e-6)
        inherited_rotation += rotation_changed
        chinese = bool(re.search(r'[\u4e00-\u9fff]', leaf))
        materials = [data.get('materials', [])[p['material']].get('alphaMode', 'OPAQUE') if 'material' in p else 'OPAQUE'
                     for p in meshes[node['mesh']].get('primitives', [])]
        process_context = [{'node': i, 'name': nodes[i].get('name', '')}
                           for i in path_ids[1:-1] if re.search(PROCESS_RE, nodes[i].get('name', ''))]
        row = {'node_index': index, 'mesh_index': node['mesh'], 'node_name': name, 'leaf_name': leaf,
               'parent_index': node['_parent_index'], 'parent_name': parent_names[-1] if parent_names else '',
               'depth': len(path_ids) - 1, 'path_ids': '/'.join(map(str, path_ids)), 'path_names': ' > '.join(path_names),
               'name_evidence': json.dumps(hints, ensure_ascii=False),
               'name_candidate_role': hints[0][0] if hints else '',
               'name_evidence_status': 'weak_lexical_hint' if hints else 'unmapped_descriptive_name' if chinese else 'code_or_generic_name',
               'ancestor_context': json.dumps(context, ensure_ascii=False),
               'nonroot_process_context': json.dumps(process_context, ensure_ascii=False),
               'context_is_leaf_label': False, 'source_triangles': faces[node['mesh']],
               'active_scene': index in active, 'world_differs_from_local': changed,
               'inherited_nonuniform_scale': nonuniform, 'any_alpha_blend': 'BLEND' in materials,
               'inherited_rotation_or_reflection': rotation_changed,
               'first_primitive_alpha_blend': bool(materials and materials[0] == 'BLEND'),
               'visibility_status': 'unmeasured', 'allow_geometry_drop': False,
               'supervised_class': pn.get(index, {}).get('predicted_class', ''),
               'supervised_confidence': pn.get(index, {}).get('confidence', ''),
               'bbox_diagonal': pn.get(index, {}).get('bbox_diagonal', ''),
               'open_label': clip.get(index, {}).get('pointclip_top1_name', ''),
               'open_role': clip.get(index, {}).get('pointclip_top1_role', ''),
               'open_score': clip.get(index, {}).get('pointclip_top1_score', '')}
        rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'mesh_name_context_audit.csv', rows)
    write_csv(output / 'assembly_tree.csv', tree)
    # Tree display uses indexed paths: duplicate exported names are preserved.
    lines = []
    stack = [(i, 0) for i in reversed([i for i, n in enumerate(nodes) if n['_parent_index'] is None])]
    while stack:
        i, depth = stack.pop()
        node = nodes[i]
        lines.append('  ' * depth + f'[{i}] ' + node.get('name', '') + (f" (mesh={node['mesh']})" if 'mesh' in node else ''))
        stack.extend((child, depth + 1) for child in reversed(node.get('children', [])))
    (output / 'assembly_tree.txt').write_text('\n'.join(lines), encoding='utf-8')
    groups = []
    for index, node in enumerate(nodes):
        if not node.get('children'):
            continue
        members = [row for row in rows if index in paths[row['node_index']][:-1]]
        groups.append({'node_index': index, 'name': node.get('name', ''), 'depth': len(paths[index]) - 1,
                       'descendant_mesh_nodes': len(members), 'instance_triangles': sum(row['source_triangles'] for row in members),
                       'path_names': ' > '.join(nodes[i].get('name', '') for i in paths[index]),
                       'name_hints': json.dumps(name_hints(node.get('name', ''))[1], ensure_ascii=False)})
    write_csv(output / 'assembly_groups.csv', sorted(groups, key=lambda r: (-r['descendant_mesh_nodes'], r['depth'])))
    name_counts = Counter(row['node_name'] for row in rows)
    mesh_refs = Counter(row['mesh_index'] for row in rows)
    summary = {'scene': str(glb.resolve()), 'glb_sha256': file_hash(glb), 'nodes': len(nodes), 'mesh_nodes': len(rows),
               'unique_meshes': len(meshes), 'referenced_meshes': len(mesh_refs), 'assembly_nodes': len(groups),
               'max_depth': max(len(p) - 1 for p in paths.values()), 'active_mesh_nodes': sum(row['active_scene'] for row in rows),
               'unique_mesh_triangles': sum(faces), 'instance_triangles': sum(row['source_triangles'] for row in rows),
               'mesh_reference_reuse_savings': len(rows) - len(mesh_refs),
               'duplicate_node_names': {name: count for name, count in name_counts.items() if count > 1},
               'chinese_leaf_names': sum(bool(re.search(r'[\u4e00-\u9fff]', row['leaf_name'])) for row in rows),
               'name_evidence_status_counts': dict(Counter(row['name_evidence_status'] for row in rows)),
               'name_candidate_role_counts': dict(Counter(row['name_candidate_role'] or 'unmapped' for row in rows)),
               'with_ancestor_lexical_context': sum(bool(json.loads(row['ancestor_context'])) for row in rows),
               'with_nonroot_process_context': sum(bool(json.loads(row['nonroot_process_context'])) for row in rows),
               'world_differs_from_local': int(local_diff), 'inherited_nonuniform_scale': int(inherited_nonuniform),
               'inherited_rotation_or_reflection': int(inherited_rotation),
               'alpha_blend_mesh_nodes': sum(row['any_alpha_blend'] for row in rows),
               'first_primitive_alpha_blend_nodes': sum(row['first_primitive_alpha_blend'] for row in rows),
               'neural_rows_joined': len(pn), 'open_rows_joined': len(clip),
               'ground_truth_available': False, 'visibility_measured': False,
               'scope': 'Name hints are weak metadata evidence, not accuracy labels or deletion criteria.',
               'source_sha256': file_hash(Path(__file__)), 'rules': RULES}
    if pn:
        assert set(pn) == {row['node_index'] for row in rows}
        assert sum(int(r['face_count']) for r in pn.values()) == summary['instance_triangles']
        low = np.min([[float(r[f'bbox_min_{a}']) for a in 'xyz'] for r in pn.values()], axis=0)
        high = np.max([[float(r[f'bbox_max_{a}']) for a in 'xyz'] for r in pn.values()], axis=0)
        diagonal = float(np.linalg.norm(high - low))
        summary['world_scene_diagonal'] = diagonal
        summary['world_scene_bbox_min'] = low.tolist()
        summary['world_scene_bbox_max'] = high.tolist()
        group_bounds = {}
        group_counts = Counter()
        for index, r in pn.items():
            lo = np.array([float(r[f'bbox_min_{a}']) for a in 'xyz'])
            hi = np.array([float(r[f'bbox_max_{a}']) for a in 'xyz'])
            for ancestor in paths[index][:-1]:
                old_lo, old_hi = group_bounds.get(ancestor, (lo, hi))
                group_bounds[ancestor] = (np.minimum(old_lo, lo), np.maximum(old_hi, hi))
                group_counts[ancestor] += 1
        local_rows = []
        for row in rows:
            candidates = [i for i in paths[row['node_index']][1:-1]
                          if group_counts[i] >= 2 and re.search(PROCESS_RE, nodes[i].get('name', ''))]
            if not candidates:
                continue
            group = candidates[-1]
            lo, hi = group_bounds[group]
            group_diagonal = float(np.linalg.norm(hi - lo))
            if group_diagonal <= 0:
                continue
            node_diagonal = float(row['bbox_diagonal'])
            local_rows.append({'node_index': row['node_index'], 'node_name': row['node_name'],
                'process_group_index': group, 'process_group_name': nodes[group].get('name', ''),
                'group_mesh_nodes': group_counts[group], 'group_world_bbox_diagonal': group_diagonal,
                'global_size_ratio': node_diagonal / diagonal, 'local_group_size_ratio': node_diagonal / group_diagonal})
        if local_rows:
            write_csv(output / 'global_vs_assembly_scale.csv', local_rows)
        summary['local_process_scale_rows'] = len(local_rows)
        summary['global_le_1pct_but_local_ge_10pct'] = sum(r['global_size_ratio'] <= .01 and r['local_group_size_ratio'] >= .10 for r in local_rows)
        environment_ids = {row['node_index'] for row in rows if row['name_candidate_role'] == 'environment'}
        equipment = [r for index, r in pn.items() if index not in environment_ids]
        if environment_ids and equipment:
            equip_low = np.min([[float(r[f'bbox_min_{a}']) for a in 'xyz'] for r in equipment], axis=0)
            equip_high = np.max([[float(r[f'bbox_max_{a}']) for a in 'xyz'] for r in equipment], axis=0)
            summary['equipment_bbox_diagonal_excluding_named_environment'] = float(np.linalg.norm(equip_high - equip_low))
            summary['named_environment_node_indices'] = sorted(environment_ids)
            summary['environment_scope'] = 'Alternative scale reference only; no geometry removed or visibility inferred.'
        size_bins = []
        for lower, upper in [(0, .005), (.005, .01), (.01, .02), (.02, .1), (.1, math.inf)]:
            subset = [r for r in pn.values() if lower < float(r['bbox_diagonal']) / diagonal <= upper]
            if not subset:
                continue
            size_bins.append({'min_exclusive': lower, 'max_inclusive': upper if math.isfinite(upper) else 'inf',
                'count': len(subset), 'record_percent': 100 * len(subset) / len(rows),
                'triangle_percent': 100 * sum(int(r['face_count']) for r in subset) / summary['instance_triangles'],
                'mean_supervised_confidence': sum(float(r['confidence']) for r in subset) / len(subset)})
        summary['size_bins'] = size_bins
        write_csv(output / 'world_size_bins.csv', size_bins)
        summary['neural_inputs_sha256'] = {str(pn_path): file_hash(pn_path), **({str(clip_path): file_hash(clip_path)} if clip else {})}
        repeated = []
        for mesh, count in mesh_refs.items():
            if count < 2:
                continue
            members = [r for r in rows if r['mesh_index'] == mesh]
            predictions = Counter(r['supervised_class'] for r in members)
            repeated.append({'mesh_index': mesh, 'instances': count,
                             'node_indices': ','.join(str(r['node_index']) for r in members),
                             'node_names': json.dumps([r['node_name'] for r in members], ensure_ascii=False),
                             'predicted_classes': json.dumps(predictions, ensure_ascii=False),
                             'distinct_predicted_classes': len(predictions),
                             'min_confidence': min(float(r['supervised_confidence']) for r in members),
                             'max_confidence': max(float(r['supervised_confidence']) for r in members),
                             'scope': 'Same source mesh; world orientation and sampling can differ. Contextual roles may differ.'})
        if repeated:
            write_csv(output / 'repeated_mesh_evidence.csv', repeated)
        summary['repeated_mesh_groups'] = len(repeated)
        summary['repeated_mesh_groups_with_class_disagreement'] = sum(r['distinct_predicted_classes'] > 1 for r in repeated)
    (output / 'hierarchy_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--glb', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    result = audit(args.glb, args.output_dir, args.run_dir)
    print(json.dumps({k: v for k, v in result.items() if k not in ('duplicate_node_names', 'rules')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
