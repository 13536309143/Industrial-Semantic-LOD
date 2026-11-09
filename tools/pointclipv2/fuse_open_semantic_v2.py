"""Unified N/C/G fusion, explicit open identities, and opt-in CAD metadata M."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from open_semantic_evidence import metadata_evidence
from run_pointclipv2_zeroshot_glb import read_glb_chunks
from audit_named_scene import read_csv, hierarchy_paths, write_csv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/pointnext_lod'))
import build_lod_constraints_from_analysis as builder
from artifact_cache import file_hash

GEOMETRY_FIELDS = {'order', 'node_index', 'mesh_index', 'node_name', 'mesh_name', 'vertex_count', 'face_count',
    'surface_area', 'bbox_diagonal', 'bbox_volume', 'obb_size_long', 'obb_size_mid', 'obb_size_short',
    'elongation', 'flatness', 'compactness', 'shape_hint'}
MODES = {'N': (True,False,False), 'C': (False,True,False), 'G': (False,False,True),
         'NG': (True,False,True), 'CG': (False,True,True), 'NC': (True,True,False), 'NCG': (True,True,True)}


def normalize(scores):
    total = sum(max(0., value) for value in scores.values())
    return {role: max(0., value)/total for role,value in scores.items()} if total else {}


def geometry_evidence(row, thresholds):
    _, size = builder.classify_size(row, thresholds)
    _, detail = builder.detail_level(row, thresholds)
    features = builder.structural_feature_vector(row, size, detail, False, 0., builder.geometry_fallback(row))
    features.pop('uncertainty_score')
    # Geometry supports shape preservation, not a claimed motor/tool identity.
    roles = ['large_static_bulk', 'ordinary_low_detail', 'balanced_visible', 'high_detail_shape']
    scores = {}
    for role in roles:
        expected = {k:v for k,v in builder.ROLE_PROTOTYPES[role]['expected_features'].items() if k != 'uncertainty_score'}
        scores[role] = sum(max(0.,1-abs(features[k]-value))*(.75+value) for k,value in expected.items()) / sum(.75+v for v in expected.values())
    return normalize(scores), features


def fuse(row, opened, taxonomy, thresholds, mode='NCG', metadata=None):
    use_n, use_c, use_g = MODES[mode]
    labels = {label['id']: label for label in taxonomy['labels']}
    nclass = row.get('predicted_class', '') if use_n else ''
    confidence = float(row.get('confidence') or 0) if use_n else 0.
    margin = confidence-float(row.get('second_confidence') or 0) if use_n else 0.
    naccepted = confidence >= .7 and margin >= .08
    nscores = normalize(builder.CLASS_ROLE_PRIORS.get(nclass, {})) if use_n and confidence >= .4 and margin >= .08 else {}
    nw = confidence*(.7+.3*min(max(margin/.3,0),1))*(1. if naccepted else .55) if nscores else 0.
    cscores = normalize(json.loads(opened.get('open_role_scores_json') or '{}')) if use_c and opened.get('open_status') in {'accepted','class_only','family_only'} else {}
    if cscores and opened.get('open_status') == 'family_only':
        # A broad family is not enough evidence for aggressive functional simplification.
        cscores = {role:.45*value for role,value in cscores.items()}
        cscores['balanced_visible'] = cscores.get('balanced_visible',0.)+.55
    cw = float(opened.get('open_reliability') or 0) if cscores else 0.
    gscores, features = geometry_evidence(row, thresholds) if use_g else ({}, {})
    gw = .35 if gscores else 0.
    metadata = metadata or {}
    mlabel = labels.get(metadata.get('object_id'))
    mscores = normalize(mlabel['role_priors']) if mlabel else {}
    mw = float(metadata.get('strength', 0)) if mscores else 0.
    cid = opened.get('open_object_id', '') if use_c and cscores and opened.get('open_status') == 'accepted' else ''
    clabel = labels.get(cid)
    cgroup = opened.get('open_class_group','') if use_c and cscores and opened.get('open_status') in {'accepted','class_only'} else ''
    cknown = opened.get('open_group_known_class','') if cgroup else clabel.get('known_class','') if clabel else ''
    cfam = opened.get('open_family', '') if use_c and cscores else ''
    nfam = {label['family'] for label in labels.values() if label.get('known_class') == nclass} if use_n else set()
    conflict = bool(nw and cw and (cfam not in nfam or
        (cknown and cknown != nclass)))
    reasons = []
    if conflict:
        nw *= .5
        reasons.append('closed_open_identity_or_family_conflict')
    if mlabel and use_n and nclass and (mlabel.get('known_class') != nclass):
        nw *= .25
        reasons.append('cad_name_disagrees_with_closed_identity')
    if mlabel and cw and mlabel['family'] != cfam:
        cw *= .6
        reasons.append('cad_name_disagrees_with_visual_family')
    evidence = [(nscores,nw,'N'),(cscores,cw,'C'),(gscores,gw,'G'),(mscores,mw,'M')]
    total = sum(weight for scores,weight,name in evidence)
    fused = {}
    for scores, weight, name in evidence:
        for role, score in scores.items():
            fused[role] = fused.get(role,0.) + score*weight/max(total,1e-9)
    role = max(fused,key=fused.get) if fused else 'balanced_visible'
    priority = builder.ROLE_PROTOTYPES[role]['priority']
    # Unknown evidence never means disposable; precision conflicts have a conservative floor.
    semantic_unknown = not (nscores or cscores or mscores)
    if semantic_unknown:
        priority = max(priority,5)
        reasons.append('unknown_function_neutral_preservation')
    if conflict and naccepted and not mlabel:
        priority = max(priority,7)
        reasons.append('unresolved_conflict_review_floor')
    if use_g and (float(row.get('face_count') or 0) >= 10000 or float(row.get('flatness') or 1) < .015):
        priority = max(priority,6)
        reasons.append('geometric_detail_or_thin_shape_floor')
    if metadata.get('workpiece_candidate'):
        priority = max(priority,7)
        reasons.append('cad_workpiece_context_preservation')
    resolved, source, resolved_status = '', 'none', 'unknown'
    if mlabel:
        generic_names = {'螺母':'nuts','螺钉':'screws_bolts_studs','螺栓':'screws_bolts_studs','螺丝':'screws_bolts_studs',
                         '轴承':'bearings_bushings_guides','弹簧':'springs','电机':'motors_gearmotors','马达':'motors_gearmotors',
                         '接头':'pipe_fittings_valves_nozzles'}
        generic_class = generic_names.get(metadata.get('matched_alias'))
        resolved = 'known:'+generic_class if generic_class else mlabel['id']
        source, resolved_status = 'cad_name_weak_hint', 'metadata_candidate'
        if generic_class:
            metadata = dict(metadata,identity_granularity='object_group',resolved_identity=resolved)
    elif metadata.get('workpiece_candidate'):
        resolved, source, resolved_status = 'workpiece:unidentified', 'cad_workpiece_context', 'workpiece_candidate'
    elif conflict and naccepted:
        source, resolved_status = 'closed_open_conflict', 'conflict'
    elif clabel:
        resolved, source, resolved_status = clabel['id'], 'open_visual', 'open_candidate'
    elif cgroup:
        resolved, source, resolved_status = cgroup, 'open_class_group', 'open_class_candidate'
    elif use_c and cscores:
        resolved, source, resolved_status = 'family:'+cfam, 'open_family', 'coarse_only'
    elif use_n and naccepted:
        resolved, source, resolved_status = 'closed:'+nclass, 'supervised', 'closed_candidate'
    policy = builder.make_p10_policy(priority, 'industrial_open_v2_'+role)
    if priority != builder.ROLE_PROTOTYPES[role]['priority']:
        policy['lod_priority'] = f'P{priority}_guarded_{role}'
    if semantic_unknown or conflict or metadata.get('workpiece_candidate'):
        policy['allow_cull'] = False
    keep = {k:v for k,v in row.items() if k in GEOMETRY_FIELDS or k.startswith(('bbox_min_','bbox_max_','bbox_size_','centroid_'))}
    semantic_weight = nw+cw+mw
    reliability = (nw*confidence + cw*float(opened.get('open_reliability') or 0) + mw*float(metadata.get('strength',0)))/semantic_weight if semantic_weight else 0.
    result = {**keep, **policy, 'evidence_mode': mode+('M' if metadata else ''),
        'confidence': reliability, 'confidence_scope': 'uncalibrated fused evidence reliability; not classification probability',
        'inferred_role': role, 'function_role': 'workpiece_or_product_candidate' if metadata.get('workpiece_candidate') else builder.ROLE_PROTOTYPES[role]['function_role'],
        'resolved_object_class': resolved, 'resolved_status': resolved_status, 'resolved_source': source,
        'class_conflict': conflict, 'needs_review': conflict or resolved_status in {'unknown','metadata_candidate','workpiece_candidate'},
        'evidence_weights_json': json.dumps({name:weight/max(total,1e-9) for scores,weight,name in evidence}),
        'role_scores_json': json.dumps(fused), 'inference_reason': ';'.join(reasons) or 'consistent_available_evidence',
        'open_status': opened.get('open_status','') if use_c else '',
        'open_object_id': cid, 'open_family': cfam,
        'open_class_group': cgroup,
        'metadata_evidence_json': json.dumps(metadata,ensure_ascii=False) if metadata else '',
        'visibility_status': 'unmeasured', 'allow_geometry_drop': False}
    if use_n:
        result.update(source_pointnext_class=nclass, source_pointnext_confidence=confidence)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--glb',type=Path,required=True)
    parser.add_argument('--pointnext-csv',type=Path,required=True)
    parser.add_argument('--open-csv',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--library',type=Path,default=Path(__file__).with_name('industrial_open_vocab_v2.json'))
    args = parser.parse_args()
    pn, opened = read_csv(args.pointnext_csv), read_csv(args.open_csv)
    key = lambda row:tuple(str(row[k]) for k in ['order','node_index','mesh_index'])
    pidx, cidx = {key(r):r for r in pn}, {key(r):r for r in opened}
    if len(pidx) != len(pn) or len(cidx) != len(opened) or set(pidx) != set(cidx):
        raise ValueError('Fusion requires complete unique identical scene keys')
    gltf,_,_ = read_glb_chunks(args.glb)
    if len(pn) != sum('mesh' in n for n in gltf['nodes']):
        raise ValueError('Partial predictions cannot be fused as a full scene')
    paths = hierarchy_paths(gltf['nodes'])
    taxonomy = json.loads(args.library.read_text(encoding='utf-8'))
    thresholds = builder.make_thresholds(pn)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    summaries, outputs = {}, []
    for mode in [*MODES,'NCGM']:
        rows = []
        for row in pn:
            node_index = int(row['node_index'])
            metadata = None
            if mode == 'NCGM':
                ancestors = [gltf['nodes'][i].get('name','') for i in paths[node_index][:-1]]
                metadata = metadata_evidence(row['node_name'],ancestors,taxonomy)
            rows.append(fuse(row,cidx[key(row)],taxonomy,thresholds,mode[:3] if mode=='NCGM' else mode,metadata))
        path = args.output_dir/f'lod_constraints_{mode}.csv'
        write_csv(path,rows)
        outputs.append(path)
        priorities = Counter(int(r['lod_priority'].split('_',1)[0][1:]) for r in rows)
        summaries[mode] = {'rows':len(rows),'priority_counts':dict(priorities),
            'low_percent':100*sum(priorities[i] for i in (1,2,3))/len(rows),
            'high_percent':100*sum(priorities[i] for i in (8,9,10))/len(rows),
            'identity_status':dict(Counter(r['resolved_status'] for r in rows)),
            'class_conflicts':sum(r['class_conflict'] for r in rows)}
    (args.output_dir/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')
    inputs=[args.glb,args.pointnext_csv,args.open_csv,args.library,Path(__file__),Path(__file__).with_name('open_semantic_evidence.py'),Path(builder.__file__)]
    (args.output_dir/'provenance.json').write_text(json.dumps({'inputs':{str(p.resolve()):file_hash(p) for p in inputs},
        'outputs':{str(p.resolve()):file_hash(p) for p in outputs},'same_fusion_function_all_modes':True,
        'G_has_no_supervised_uncertainty':True,'C_has_no_supervised_confidence':True,'metadata_separate_mode':'NCGM',
        'thresholds_calibrated':False,'geometry_removed':0},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
