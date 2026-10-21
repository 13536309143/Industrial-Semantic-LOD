"""Independent, scale-free geometry descriptors and scoped glTF assembly evidence."""
import json
import mmap
import re
from collections import Counter
from pathlib import Path

import numpy as np

from audit_named_scene import hierarchy_paths, write_csv
from run_pointclipv2_zeroshot_glb import read_glb_chunks, mesh_for_node
from artifact_cache import file_hash


CONTEXT_PATTERNS = [
    ('inspection', r'检测|检验|测试|视觉|CCD|相机|inspection|vision|camera'),
    ('tooling', r'冲切|冲裁|裁切|切断|折弯|铆压|铆接|压装|焊接|焊脚|校直|切脚|模具|摸具|tooling|weld|press'),
    ('fixture', r'夹具|治具|工装|夹持|夹爪|定位|fixture|jig|gripper'),
    ('feeding', r'送料|上料|下料|供料|放料|收料|振动盘|料道|feed|loading'),
    ('handling', r'搬运|移栽|移位|抓取|转运|转盘|分割盘|变距|升降|输送|载具|transfer|conveyor'),
    ('actuation', r'气缸|无杆缸|滑台|电缸|驱动|传动|电机|马达|导轨|导向|actuat|drive|motor|guide'),
    ('fluid', r'真空|气路|气管|管路|水路|冷却|vacuum|fluid|pneumatic'),
    ('frame', r'机架|框架|底架|台面|支撑架|frame|chassis'),
    ('enclosure', r'防护|外罩|安全门|电箱|电柜|enclosure|guard|cabinet'),
]


def number(row, key, default=0.):
    try:
        value = float(row.get(key) or default)
        return value if np.isfinite(value) else default
    except (ValueError, TypeError):
        return default


def surface_descriptor(vertices, faces):
    """Dihedral/edge measures in normalized coordinates; no class/name/confidence input."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if len(v) < 3 or not len(f) or not np.isfinite(v).all():
        raise ValueError('missing_or_nonfinite_surface')
    diagonal = float(np.linalg.norm(v.max(0)-v.min(0)))
    if diagonal <= 1e-12:
        raise ValueError('zero_extent_surface')
    # Weld CAD seam duplicates at a relative tolerance, keeping source indexing outside this calculation.
    normalized = (v-v.mean(0))/diagonal
    _, index, remap = np.unique(np.round(normalized, 6), axis=0, return_index=True, return_inverse=True)
    points = normalized[index]
    triangles = remap[f]
    cross = np.cross(points[triangles[:,1]]-points[triangles[:,0]], points[triangles[:,2]]-points[triangles[:,0]])
    lengths = np.linalg.norm(cross, axis=-1)
    valid = lengths > 1e-13
    triangles, cross, lengths = triangles[valid], cross[valid], lengths[valid]
    if not len(triangles):
        raise ValueError('no_nondegenerate_triangles')
    normals = cross/lengths[:,None]
    edges = np.sort(np.concatenate([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]]), axis=-1)
    owners = np.tile(np.arange(len(triangles)),3)
    unique, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    edge_length = np.linalg.norm(points[unique[:,1]]-points[unique[:,0]],axis=-1)
    ordered = np.argsort(inverse,kind='stable')
    starts = np.cumsum(counts)-counts
    pair = np.flatnonzero(counts==2)
    left, right = owners[ordered[starts[pair]]], owners[ordered[starts[pair]+1]]
    # abs(dot) tolerates inconsistent winding; this is not an orientation/volume proof.
    angles = np.arccos(np.clip(np.abs(np.sum(normals[left]*normals[right],axis=-1)),0,1))
    sharp_length = float(edge_length[pair][angles>np.deg2rad(30)].sum())
    curved_length = float(edge_length[pair][(angles>np.deg2rad(5))&(angles<=np.deg2rad(30))].sum())
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    adjacency = coo_matrix((np.ones(len(unique),dtype=np.int8),(unique[:,0],unique[:,1])),shape=(len(points),len(points))).tocsr()
    _, components = connected_components(adjacency,directed=False)
    areas = lengths*.5
    component_areas = np.bincount(components[triangles[:,0]],weights=areas)
    significant = int(np.sum(component_areas>max(areas.sum()*.001,1e-12)))
    return dict(geometry_valid=True, welded_vertices=len(points), valid_triangles=len(triangles),
        sharp_length_normalized=sharp_length, curved_length_normalized=curved_length,
        boundary_length_normalized=float(edge_length[counts==1].sum()),
        nonmanifold_edge_fraction=float(np.mean(counts>2)),
        significant_components=significant,
        largest_component_area_fraction=float(component_areas.max()/areas.sum()),
        surface_area_normalized=float(areas.sum()), descriptor_error='')


def geometry_descriptors(glb, rows, output):
    stamp = output.with_suffix('.provenance.json')
    inputs = {str(glb.resolve()):file_hash(glb),str(Path(__file__).resolve()):file_hash(Path(__file__))}
    if stamp.exists() and output.exists():
        old = json.loads(stamp.read_text(encoding='utf-8'))
        if old.get('inputs')==inputs and old.get('output_sha256')==file_hash(output):
            return json.loads(output.read_text(encoding='utf-8'))
    data, offset, _ = read_glb_chunks(glb)
    result, cache = {}, {}
    with glb.open('rb') as stream, mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mm:
        for j,row in enumerate(rows):
            i = int(row['node_index']); node = data['nodes'][i]
            transform = node['_world_matrix'][:3,:3]
            gram = transform.T@transform
            gram /= max(float(np.trace(gram)),1e-12)
            key = (node['mesh'],tuple(np.round(gram.ravel(),7)))
            if key not in cache:
                try:
                    vertices,faces = mesh_for_node(data,mm,offset,node)
                    cache[key] = surface_descriptor(vertices,faces)
                except Exception as exc:
                    cache[key] = dict(geometry_valid=False,descriptor_error=str(exc))
            result[str(i)] = cache[key]
            if (j+1)%500==0:
                print(f'Geometry descriptors {glb.stem}: {j+1}/{len(rows)}',flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    stamp.write_text(json.dumps(dict(inputs=inputs,output_sha256=file_hash(output),rows=len(result),
        unique_shape_descriptors=len(cache),units='dimensionless',uses_semantic_predictions=False),indent=2),encoding='utf-8')
    return result


def build_contexts(nodes, rows, config, use_names=True):
    """Only ancestor names and descendant geometry; never reads a sibling/leaf class."""
    paths = hierarchy_paths(nodes)
    by_node = {int(r['node_index']):r for r in rows}
    bounds, counts, references = {},Counter(),{}
    low = np.min([[number(r,'bbox_min_'+a)for a in 'xyz']for r in rows],axis=0)
    high = np.max([[number(r,'bbox_max_'+a)for a in 'xyz']for r in rows],axis=0)
    scene_diagonal = max(float(np.linalg.norm(high-low)),1e-12)
    for i,row in by_node.items():
        lo=np.array([number(row,'bbox_min_'+a)for a in 'xyz'])
        hi=np.array([number(row,'bbox_max_'+a)for a in 'xyz'])
        for ancestor in paths[i][:-1]:
            old_lo,old_hi=bounds.get(ancestor,(lo,hi))
            bounds[ancestor]=(np.minimum(old_lo,lo),np.maximum(old_hi,hi))
            counts[ancestor]+=1
            references.setdefault(ancestor,Counter())[int(row['mesh_index'])]+=1
    contexts={}
    for i,row in by_node.items():
        parents=paths[i][:-1]
        candidates=[]
        for distance,ancestor in enumerate(reversed(parents),1):
            count=counts[ancestor]
            if not 2<=count<=max(2,len(rows)*config['max_context_fraction']):continue
            # Scope by descendant count as well as tree depth: an equipment title is not a workcell label.
            name=nodes[ancestor].get('name','') if use_names else ''
            categories=[category for category,pattern in CONTEXT_PATTERNS if re.search(pattern,name,re.I)]
            if categories:
                candidates.append(dict(node_index=ancestor,name=name,categories=categories,distance=distance,
                    descendant_count=count,diagonal=float(np.linalg.norm(bounds[ancestor][1]-bounds[ancestor][0]))))
        selected=candidates[0] if candidates else None
        local_group=next((a for a in reversed(parents)if counts[a]>=2 and counts[a]<=max(2,len(rows)*config['max_context_fraction'])),None)
        group_diagonal=float(np.linalg.norm(bounds[local_group][1]-bounds[local_group][0])) if local_group is not None else scene_diagonal
        contexts[str(i)]=dict(context_available=bool(selected),context=selected,ancestor_candidates=candidates,
            topology_available=bool(parents),parent_index=nodes[i]['_parent_index'],depth=len(paths[i])-1,
            local_group_index=local_group,local_group_members=counts[local_group]if local_group is not None else 0,
            group_size_ratio=number(row,'bbox_diagonal')/max(group_diagonal,1e-12),
            scene_size_ratio=number(row,'bbox_diagonal')/scene_diagonal,
            local_mesh_repeat_count=references[local_group][int(row['mesh_index'])]if local_group is not None else 1,
            scene_diagonal=scene_diagonal)
    return contexts


def geometry_risk(row, descriptor, context, config):
    long=max(number(row,'obb_size_long'),1e-12)
    mid=number(row,'obb_size_mid');short=number(row,'obb_size_short')
    ratio=context['scene_size_ratio'];local=context['group_size_ratio']
    slender=long/max(mid,1e-12)>=config['slender_aspect']
    thin=short/long<config['thin_ratio']
    feature_rich=descriptor.get('sharp_length_normalized',0)>=config['feature_length']
    curved=descriptor.get('curved_length_normalized',0)>=config['curved_length']
    disconnected=descriptor.get('significant_components',0)>=config['component_count']
    large=ratio>=config['large_scene_ratio'] or (context['local_group_members']>=3 and local>=config['large_local_ratio'])
    flags=[]
    for valid,name in [(slender,'slender_cross_section'),(thin,'thin_surface'),(feature_rich,'feature_rich_edges'),
                       (curved,'curved_surface'),(disconnected,'multiple_surface_components'),(large,'large_extent')]:
        if valid:flags.append(name)
    # Large alone requests silhouette coverage; it is not a precision-function prediction.
    floor=6 if slender or thin or feature_rich or disconnected else 5 if large else 0
    if not descriptor.get('geometry_valid',False):floor=max(floor,5);flags.append('invalid_surface_descriptor')
    return dict(flags=flags,priority_floor=floor,slender=slender,thin=thin,large=large,
        feature_rich=feature_rich,curved=curved,disconnected=disconnected,
        relative_thickness=short/long,scene_size_ratio=ratio,local_size_ratio=local,
        lock_floor=.16 if slender or thin else .13 if feature_rich or disconnected else .08 if large else 0.,
        soft_floor=1.25 if slender or thin or feature_rich or disconnected else 1. if large else 0.)


def write_context_audit(path,rows,contexts,descriptors):
    write_csv(path,[dict(node_index=r['node_index'],mesh_index=r['mesh_index'],node_name=r.get('node_name',''),
        context_json=json.dumps(contexts[str(r['node_index'])],ensure_ascii=False),
        descriptor_json=json.dumps(descriptors[str(r['node_index'])]))for r in rows])
