"""Hierarchical late fusion of cached views, abstention and separate CAD hints."""
import json
import re
import numpy as np


def softmax(values, axis=-1):
    values = np.asarray(values, dtype=np.float64)
    exp = np.exp(values - np.max(values, axis=axis, keepdims=True))
    return exp / np.maximum(exp.sum(axis=axis, keepdims=True), 1e-12)


def logmeanexp(values, temperature=20., axis=-1):
    scaled = np.asarray(values) * temperature
    high = np.max(scaled, axis=axis, keepdims=True)
    return (np.log(np.exp(scaled-high).mean(axis=axis)) + high.squeeze(axis)) / temperature


def smooth(low, high, value):
    t = np.clip((value-low)/max(high-low, 1e-12), 0., 1.)
    return t*t*(3-2*t)


def label_distribution(similarities, family_similarities, taxonomy, config):
    families = [f['id'] for f in taxonomy['families']]
    members = [[i for i, label in enumerate(taxonomy['labels']) if label['family'] == family] for family in families]
    # Fixed top-two support avoids penalizing a family merely for having more labels.
    grouped = np.stack([np.sort(similarities[..., ids], axis=-1)[..., -min(2, len(ids)):].mean(axis=-1) for ids in members], axis=-1)
    # Broad coarse text is a bounded residual, not a prior that can erase fine visual evidence.
    residual = family_similarities - np.mean(family_similarities, axis=-1, keepdims=True)
    family_scores = grouped + config['family_direct_weight'] * np.clip(residual, -.02, .02)
    family_probs = softmax(family_scores * config['family_temperature'])
    result = np.zeros_like(similarities, dtype=np.float64)
    for family, ids in enumerate(members):
        result[..., ids] = family_probs[..., family, None] * softmax(similarities[..., ids] * config['class_temperature'])
    return result, family_scores, family_probs


def infer_evidence(features, qualities, kinds, text, taxonomy, config):
    labels, families = taxonomy['labels'], taxonomy['families']
    if len(features) == 0 or not np.isfinite(features).all():
        return {'status': 'abstain', 'reason': 'invalid_or_missing_geometry', 'reliability': 0., 'roles': {}, 'topk': []}
    # No feature renormalization after view averaging: scores retain their cosine scale.
    features = np.asarray(features, dtype=np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), 1e-8)
    view_sim = features @ text['labels'].T
    view_family_sim = features @ text['families'].T
    view_null = (features @ text['negative'].T).max(-1)
    global_mask = np.asarray(kinds) < 2
    weights = np.maximum(np.asarray(qualities, dtype=float), 0)
    valid = (weights > .01) & global_mask
    if valid.sum() < config['min_valid_views']:
        return {'status': 'abstain', 'reason': 'insufficient_informative_views', 'reliability': 0., 'roles': {}, 'topk': []}
    # Equal total mass for world and PCA streams; duplicate views cannot dominate.
    global_weights = np.zeros(len(features))
    streams = sorted(set(np.asarray(kinds)[valid]))
    for kind in streams:
        mask = (np.asarray(kinds) == kind) & valid
        w = weights[mask]
        global_weights[mask] = w / w.sum() / len(streams)
    global_sim = global_weights @ view_sim
    global_family = global_weights @ view_family_sim
    _, family_scores, family_probs = label_distribution(global_sim, global_family, taxonomy, config)
    family_index = int(np.argmax(family_scores))
    # Local context is a bounded supplement only when it agrees with the global family.
    local = np.asarray(kinds) >= 2
    local_used = False
    if np.any(local & (weights > .01)):
        local_weights = weights * local
        local_weights /= local_weights.sum()
        local_sim = local_weights @ view_sim
        local_family = local_weights @ view_family_sim
        _, local_fs, _ = label_distribution(local_sim, local_family, taxonomy, config)
        if int(np.argmax(local_fs)) == family_index:
            alpha = config['local_weight']
            global_sim = (1-alpha)*global_sim + alpha*local_sim
            global_family = (1-alpha)*global_family + alpha*local_family
            local_used = True
    distribution, family_scores, family_probs = label_distribution(global_sim, global_family, taxonomy, config)
    family_index = int(np.argmax(family_scores))
    family_order = np.argsort(-family_scores)
    family_margin = float(family_scores[family_order[0]] - family_scores[family_order[1]])
    _, per_view_family, _ = label_distribution(view_sim, view_family_sim, taxonomy, config)
    agreement = float(global_weights @ (np.argmax(per_view_family, axis=-1) == family_index))
    members = [i for i, label in enumerate(labels) if label['family'] == families[family_index]['id']]
    member_order = sorted(members, key=lambda i: -global_sim[i])
    chosen = member_order[0]
    fine_margin = float(global_sim[chosen] - global_sim[member_order[1]]) if len(member_order) > 1 else 1.
    conditional = softmax(global_sim[members] * config['class_temperature'])
    fine_agreement = float(global_weights @ (np.argmax(view_sim[:, members], axis=-1) == members.index(chosen)))
    groups = {}
    for member in members:
        group = ('known:'+labels[member]['known_class']) if labels[member].get('known_class') else ('open:'+labels[member]['id'])
        groups.setdefault(group, []).append(member)
    group_ids = list(groups)
    group_scores = np.array([max(global_sim[i] for i in groups[group]) for group in group_ids])
    group_order = np.argsort(-group_scores)
    group_index = int(group_order[0])
    group_margin = float(group_scores[group_order[0]]-group_scores[group_order[1]]) if len(group_ids)>1 else 1.
    view_groups = np.stack([view_sim[:, groups[group]].max(-1) for group in group_ids], axis=-1)
    group_agreement = float(global_weights @ (np.argmax(view_groups, axis=-1)==group_index))
    similarity = float(global_sim[chosen])
    null_margin = float(similarity - global_weights @ view_null)
    quality = float(np.clip(np.mean(weights[global_mask]), 0, 1))
    reliability = float(.25*smooth(.18, .32, similarity) + .25*smooth(.003, .04, family_margin) +
                        .25*agreement + .15*smooth(-.02, .06, null_margin) + .10*quality)
    failures = []
    for valid_check, reason in [
        (similarity >= config['min_similarity'], 'weak_absolute_similarity'),
        (null_margin >= config['min_null_margin'], 'background_or_ambiguous_match'),
        (family_margin >= config['min_family_margin'], 'ambiguous_family_margin'),
        (agreement >= config['min_family_agreement'], 'view_disagreement')]:
        if not valid_check:
            failures.append(reason)
    accepted = not failures and families[family_index]['id'] != 'shape'
    fine = accepted and fine_margin >= config['min_class_margin'] and fine_agreement >= config['min_class_agreement']
    status = 'accepted' if fine else 'family_only' if accepted else 'abstain'
    group_supported = accepted and group_margin >= config['min_class_margin'] and group_agreement >= config['min_class_agreement']
    group_id = group_ids[group_index] if group_supported else ''
    roles = {}
    if accepted:
        # Family acceptance can support a role without inventing a fine object identity.
        probabilities = softmax(global_sim[members] * config['class_temperature'])
        for p, index in zip(probabilities, members):
            for role, prior in labels[index]['role_priors'].items():
                roles[role] = roles.get(role, 0.) + float(p) * prior
    # Cross-family conditional probabilities have a class-count bias; rank fine candidates by cosine.
    top_indices = [chosen, *[int(i) for i in np.argsort(-global_sim) if i != chosen]][:5]
    rank_scores = softmax(global_sim * config['class_temperature'])
    topk = [{'id': labels[i]['id'], 'name': labels[i]['name'], 'family': labels[i]['family'],
             'score': float(rank_scores[i]), 'similarity': float(global_sim[i])} for i in top_indices]
    return {'status': status, 'reason': ';'.join(failures) if failures else 'generic_shape' if not accepted else 'multi_view_supported',
            'reliability': reliability, 'roles': roles, 'topk': topk,
            'object_id': labels[chosen]['id'] if fine else '', 'candidate_id': labels[chosen]['id'],
            'family': families[family_index]['id'], 'family_margin': family_margin,
            'family_posterior': float(family_probs[family_index]), 'fine_margin': fine_margin,
            'view_agreement': agreement, 'similarity': similarity, 'null_margin': null_margin,
            'valid_views': int(valid.sum()), 'view_quality': quality, 'local_used': local_used,
            'class_view_agreement': fine_agreement,
            'class_group': group_id, 'class_group_known_class': group_id[6:] if group_id.startswith('known:') else '',
            'class_group_margin': group_margin, 'class_group_agreement': group_agreement,
            'conditional_posterior': float(max(conditional))}


def metadata_evidence(name, ancestors, taxonomy):
    """Optional M evidence; never fed to the visual scorer or treated as ground truth."""
    leaf = name.replace('\\', '/').split('/')[-1]
    product = bool(re.search(r'产品|工件|端子料带', leaf))
    context = [text for text in ancestors[1:] if re.search(r'检测|CCD|铆压|冲切|折弯|送料|上料|模具|摸具|真空', text)]
    if re.search(r'螺母送料盘|振动盘', leaf):
        label_id, alias = 'feeder_bowl', '送料盘'
    elif re.search(r'支架|安装板|固定板|固定件|固定块|垫板|支座|安装件|(?:电机|马达|气缸|光源|CCD).*(?:座|板|垫块)', leaf, re.I):
        label_id, alias = 'mounting_bracket' if '支架' in leaf else 'mounting_plate', '安装/支撑零件'
    else:
        hits = [(len(alias), label['id'], alias) for label in taxonomy['labels'] for alias in label['zh_aliases'] if alias in leaf]
        hits.sort(key=lambda h: (-h[0], h[1]))
        label_id, alias = (hits[0][1], hits[0][2]) if hits else ('', '')
    label = next((label for label in taxonomy['labels'] if label['id'] == label_id), None)
    feeding_workpiece = bool(label and label.get('known_class') == 'nuts' and any(re.search(r'送料|上料', text) for text in ancestors[1:]))
    return {'object_id': label_id, 'family': label['family'] if label else '', 'matched_alias': alias,
            'strength': .75 if label else 0., 'workpiece_candidate': product or feeding_workpiece,
            'contexts': context, 'source': 'cad_name_weak_hint' if label else 'no_direct_name_hint',
            'not_ground_truth': True}


def serialize_evidence(base, evidence):
    top = evidence.get('topk', [])
    roles = evidence['roles']
    role = max(roles, key=roles.get) if roles else ''
    first = top[0] if top else {'id': '', 'name': '', 'score': 0.}
    second = top[1] if len(top) > 1 else {'id': '', 'name': '', 'score': 0.}
    row = {**base, 'open_engine': 'industrial_open_v2',
           'open_status': evidence['status'], 'open_reason': evidence['reason'],
           'open_object_id': evidence.get('object_id', ''), 'open_family': evidence.get('family', ''),
           'open_class_group': evidence.get('class_group',''), 'open_group_known_class': evidence.get('class_group_known_class',''),
           'open_reliability': evidence['reliability'], 'open_role_scores_json': json.dumps(roles),
           'open_diagnostics_json': json.dumps({k:v for k,v in evidence.items() if k not in {'topk','roles'}}, ensure_ascii=False),
           'pointclip_top1_id': first['id'], 'pointclip_top1_name': first['name'], 'pointclip_top1_role': role,
           'pointclip_top1_score': first['score'], 'pointclip_top2_id': second['id'], 'pointclip_top2_name': second['name'],
           'pointclip_top2_role': '', 'pointclip_top2_score': second['score'],
           'pointclip_margin': first['score']-second['score'], 'pointclip_topk_json': json.dumps(top),
           'score_scope': 'hierarchical ranking and uncalibrated evidence quality; not classification accuracy'}
    return row
