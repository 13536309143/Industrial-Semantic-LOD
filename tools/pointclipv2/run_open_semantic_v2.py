"""Full-scene industrial open semantics with reusable per-view feature caches."""
import argparse
import json
import mmap
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np

from run_pointclipv2_zeroshot_glb import read_glb_chunks, mesh_for_node, sample_points
from open_semantic_geometry import proposals
from open_semantic_evidence import infer_evidence, serialize_evidence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/pointnext_lod'))
from artifact_cache import file_hash, stage_identity, matches, record


def write_rows(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        import csv
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def image_batch_encode(model, images, batch_size):
    import torch
    result, cursor = [], 0
    size = batch_size
    while cursor < len(images):
        try:
            with torch.inference_mode():
                features = model.encode_image(images[cursor:cursor+size].type(model.dtype)).float()
                features /= features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            result.append(features.cpu().numpy().astype(np.float16))
            cursor += size
        except torch.cuda.OutOfMemoryError:
            if size <= 1:
                raise
            size = max(1, size // 2)
            torch.cuda.empty_cache()
    return np.concatenate(result)


def prepare_images(images, preprocess):
    import torch
    import torch.nn.functional as F
    images = torch.nan_to_num(images.float(), nan=1., posinf=1., neginf=0.).clamp(0, 1)
    grayscale = images[:, 0]
    foreground = grayscale < .97
    area = foreground.float().mean((-1, -2))
    contrast = grayscale.std((-1, -2))
    qualities = torch.sqrt((area / .18).clamp(0, 1)) * (contrast / .08).clamp(0, 1)
    crops = []
    for image, mask in zip(images, foreground):
        coords = torch.nonzero(mask)
        if len(coords) < 4:
            crops.append(F.interpolate(image[None], (224, 224), mode='bicubic', align_corners=False)[0])
            continue
        lo, hi = coords.min(0).values, coords.max(0).values + 1
        y0, x0 = [int(v) for v in lo]
        y1, x1 = [int(v) for v in hi]
        crop = image[:, y0:y1, x0:x1]
        height, width = crop.shape[-2:]
        side = int(max(height, width) * 1.20) + 2
        pad = ((side-width)//2, side-width-(side-width)//2,
               (side-height)//2, side-height-(side-height)//2)
        crop = F.pad(crop, pad, value=1.)
        crops.append(F.interpolate(crop[None], (224, 224), mode='bicubic', align_corners=False)[0])
    output = torch.stack(crops).clamp(0, 1)
    if preprocess == 'clip-rgb':
        mean = output.new_tensor([.48145466, .4578275, .40821073])[None, :, None, None]
        std = output.new_tensor([.26862954, .26130258, .27577711])[None, :, None, None]
        output = (output - mean) / std
    return output, qualities.cpu().numpy()


def text_bank(clip, model, taxonomy, path, identity):
    stamp = path.with_suffix('.provenance.json')
    if matches(stamp, identity, [path]):
        with np.load(path, allow_pickle=False) as bank:
            return {key: bank[key].copy() for key in bank.files}
    import torch
    def encode(items):
        vectors = []
        for item in items:
            prompts = item['prompts']
            tokens = clip.tokenize(prompts, truncate=True).cuda()
            with torch.inference_mode():
                features = model.encode_text(tokens).float()
                features /= features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                vector = features.mean(0)
                vector /= vector.norm().clamp_min(1e-8)
            vectors.append(vector.cpu().numpy())
        return np.stack(vectors)
    bank = {'labels': encode(taxonomy['labels']), 'families': encode(taxonomy['families']),
            'negative': encode([{'prompts': [p]} for p in taxonomy['negative_prompts']])}
    np.savez(path, **bank)
    record(stamp, identity, [path])
    return bank


def build_views(args, config, gltf, offset, nodes, model, projection, cache, identity):
    import torch
    stamp = cache.with_suffix('.provenance.json')
    errors_path = cache.with_suffix('.errors.json')
    if matches(stamp, identity, [cache, errors_path]):
        print('REUSE verified per-view image cache', flush=True)
        return True
    per_features, per_quality, per_kinds = [[] for _ in nodes], [[] for _ in nodes], [[] for _ in nodes]
    pending_points, pending_nodes, pending_kinds, errors = [], [], [], []
    def flush():
        if not pending_points:
            return
        points = torch.from_numpy(np.stack(pending_points)).cuda()
        with torch.inference_mode():
            images = projection.get_img(points)
            images, quality = prepare_images(images, config['preprocess'])
            features = image_batch_encode(model, images, config['image_batch'])
        views = projection.num_views
        for i, (index, kind) in enumerate(zip(pending_nodes, pending_kinds)):
            per_features[index].append(features[i*views:(i+1)*views])
            per_quality[index].extend(quality[i*views:(i+1)*views].tolist())
            per_kinds[index].extend([kind] * views)
        pending_points.clear(); pending_nodes.clear(); pending_kinds.clear()
    with args.glb.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for index, (node_index, node) in enumerate(nodes):
            try:
                vertices, faces = mesh_for_node(gltf, mm, offset, node)
                rng = np.random.default_rng(config['seed'] + int(node['mesh']))
                points = sample_points(vertices, faces, config['num_points'], rng)
                clouds, kinds = proposals(points, len(faces), config['max_local_crops'])
            except Exception as exc:
                errors.append({'node_index': node_index, 'error': str(exc)})
                continue
            for cloud, kind in zip(clouds, kinds):
                pending_points.append(cloud); pending_nodes.append(index); pending_kinds.append(kind)
                if len(pending_points) >= config['cloud_batch']:
                    flush()
            if (index + 1) % 250 == 0:
                print(f'Views encoded {index+1}/{len(nodes)} nodes', flush=True)
        flush()
    offsets = [0]
    flattened = []
    for arrays in per_features:
        array = np.concatenate(arrays) if arrays else np.empty((0, model.visual.output_dim), dtype=np.float16)
        flattened.append(array)
        offsets.append(offsets[-1] + len(array))
    temporary = cache.with_suffix('.tmp.npz')
    np.savez(temporary, features=np.concatenate(flattened), offsets=np.array(offsets),
             qualities=np.array([q for values in per_quality for q in values], dtype=np.float32),
             kinds=np.array([kind for values in per_kinds for kind in values], dtype=np.int8),
             node_indices=np.array([i for i, node in nodes]))
    temporary.replace(cache)
    errors_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding='utf-8')
    record(stamp, identity, [cache, errors_path])
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--glb', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--library', type=Path, default=Path(__file__).with_name('industrial_open_vocab_v2.json'))
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('open_semantic_v2_config.json'))
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required by the PointCLIP projection')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    taxonomy = json.loads(args.library.read_text(encoding='utf-8'))
    ids = [label['id'] for label in taxonomy['labels']]
    if len(set(ids)) != len(ids) or any(label['family'] not in {f['id'] for f in taxonomy['families']} for label in taxonomy['labels']):
        raise ValueError('Invalid taxonomy')
    clip_root = Path(__file__).parent / 'PointCLIP_V2/zeroshot_cls'
    sys.path.insert(0, str(clip_root / 'Dassl3D'))
    sys.path.insert(0, str(clip_root))
    from clip import clip
    from trainers.mv_utils_zs import Realistic_Projection
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / 'cache'
    cache_dir.mkdir(exist_ok=True)
    started = time.perf_counter()
    model, _ = clip.load('ViT-B/16', device='cuda')
    model.eval()
    weight = Path.home() / '.cache/clip/ViT-B-16.pt'
    shared = [weight, clip_root / 'clip', clip_root / 'trainers/mv_utils_zs.py']
    identity = stage_identity([args.glb, Path(__file__), Path(__file__).with_name('open_semantic_geometry.py'),
        Path(__file__).with_name('run_pointclipv2_zeroshot_glb.py'), ROOT / 'tools/pointnext_lod/gltf_hierarchy.py', *shared],
        {key:config[key] for key in ['num_points', 'seed', 'max_local_crops', 'preprocess', 'cloud_batch', 'image_batch']} | {'limit': args.limit})
    gltf, offset, _ = read_glb_chunks(args.glb)
    nodes = [(i, node) for i, node in enumerate(gltf['nodes']) if 'mesh' in node]
    if args.limit is not None:
        nodes = nodes[:args.limit]
    status = {'status': 'running', 'scene': str(args.glb.resolve()), 'glb_sha256': file_hash(args.glb),
              'library_sha256': file_hash(args.library), 'config': config, 'rows_expected': len(nodes), 'partial_smoke_test': args.limit is not None}
    (out / 'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    reused = build_views(args, config, gltf, offset, nodes, model, Realistic_Projection(), cache_dir / 'views.npz', identity)
    text_identity = stage_identity([args.library, weight, Path(__file__), clip_root / 'clip'], {'text_ensemble': 'normalized mean per class'})
    text = text_bank(clip, model, taxonomy, cache_dir / 'text.npz', text_identity)
    with np.load(cache_dir / 'views.npz', allow_pickle=False) as archive:
        bank = {key: archive[key] for key in archive.files}
        rows = []
        for i, (index, node) in enumerate(nodes):
            lo, hi = bank['offsets'][i:i+2]
            evidence = infer_evidence(bank['features'][lo:hi], bank['qualities'][lo:hi], bank['kinds'][lo:hi], text, taxonomy, config)
            mesh = gltf['meshes'][node['mesh']]
            rows.append(serialize_evidence({'order': i+1, 'node_index': index, 'mesh_index': node['mesh'],
                'node_name': node.get('name', ''), 'mesh_name': mesh.get('name', '')}, evidence))
    write_rows(out / 'open_semantic_v2.csv', rows)
    (out / 'open_semantic_v2.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    diagnostics = [json.loads(row['open_diagnostics_json']) for row in rows]
    summary = {'rows': len(rows), 'status_counts': dict(Counter(row['open_status'] for row in rows)),
               'families': dict(Counter(row['open_family'] for row in rows)),
               'reasons': dict(Counter(row['open_reason'] for row in rows)),
               'mean_similarity': float(np.mean([d.get('similarity', 0) for d in diagnostics])),
               'mean_family_agreement': float(np.mean([d.get('view_agreement', 0) for d in diagnostics])),
               'image_cache_reused': reused, 'accuracy_measured': False}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    outputs = [out / 'open_semantic_v2.csv', out / 'open_semantic_v2.json', out / 'summary.json']
    pred_identity = stage_identity([args.glb, args.library, args.config, Path(__file__).with_name('open_semantic_evidence.py'),
                                  cache_dir / 'views.npz', cache_dir / 'text.npz', Path(__file__)], {'limit': args.limit})
    record(out / 'prediction.provenance.json', pred_identity, outputs)
    status.update(status='complete', image_cache_reused=reused, wall_seconds=time.perf_counter()-started,
                  rows=len(rows), summary=summary, scope='No CAD names, hierarchy labels or PointNeXt predictions used by visual scoring.')
    (out / 'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
