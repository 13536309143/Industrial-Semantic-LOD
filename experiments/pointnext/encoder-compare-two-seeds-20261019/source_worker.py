"""One fresh model/seed run. Only validation selects checkpoints; test once at end."""
import argparse
import copy
import csv
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common import ROOT, digest, metrics_from_confusion, model_config, read_json, write_csv, write_json


def build_model(name):
    sys.path.insert(0, str(ROOT / 'PointNeXt'))
    from openpoints.utils import EasyConfig
    from openpoints.models import build_model_from_cfg
    cfg = EasyConfig()
    cfg.update(copy.deepcopy(model_config(name)['model']))
    return build_model_from_cfg(cfg)


def environment():
    return {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
            'numpy': np.__version__, 'cuda_runtime': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(),
            'gpu': torch.cuda.get_device_name(0),
            'gpu_total_memory': torch.cuda.get_device_properties(0).total_memory,
            'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}


def preflight(methods, amp):
    import matplotlib
    # Check the NumPy bridge as well as CUDA; CPU-only wheels must fail early.
    torch.from_numpy(np.zeros(1, dtype=np.float32)).numpy()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA-enabled PyTorch and an NVIDIA GPU are required. See experiments/encoder_compare/README.md')
    for name in methods:
        model = build_model(name).cuda().eval()
        # Extension and ABI checks with a tiny inference, no optimization step.
        xyz = torch.randn(2, 512, 3, device='cuda')
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
            logits = model({'pos': xyz, 'x': xyz.transpose(1, 2).contiguous()})
        if logits.shape != (2, 14) or not torch.isfinite(logits).all():
            raise RuntimeError(f'Invalid model output for {name}: {logits.shape}')
        print(f'Preflight OK: {name}, parameters={sum(p.numel() for p in model.parameters()):,}', flush=True)
        del model, logits, xyz
        torch.cuda.empty_cache()
    print(json.dumps(environment(), indent=2), flush=True)


class Clouds(Dataset):
    def __init__(self, manifest_path, split, seed, augmentation):
        self.root = Path(manifest_path).parent
        self.rows = [r for r in read_json(manifest_path)['records'] if r['split'] == split]
        self.split, self.seed, self.epoch, self.augmentation = split, seed, 0, augmentation

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        xyz = np.load(self.root / row['cloud'], allow_pickle=False).copy()
        if self.split == 'train':
            rng = np.random.default_rng(int(digest([self.seed, self.epoch, row['source']])[:16], 16))
            angles = rng.uniform(-np.pi, np.pi, 3)
            cx, cy, cz = np.cos(angles)
            sx, sy, sz = np.sin(angles)
            rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            xyz = xyz @ (rz @ ry @ rx).T
            xyz *= rng.uniform(*self.augmentation['scale'])
            xyz += rng.uniform(-self.augmentation['shift'], self.augmentation['shift'], 3)
            rng.shuffle(xyz)
        return torch.from_numpy(np.ascontiguousarray(xyz, dtype=np.float32)), row['label'], i


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def checkpoint(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    temp.replace(path)


def data_loader(dataset, batch, workers, shuffle=False, seed=0):
    # Workers recreated each epoch so the epoch-dependent augmentation is visible
    # under Windows spawn as well as Linux fork. No cross-model RNG coupling.
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      pin_memory=True, drop_last=shuffle, persistent_workers=False,
                      generator=torch.Generator().manual_seed(seed))


def evaluate(model, loader, k, amp, save_predictions=False):
    cm = np.zeros((k, k), dtype=np.int64)
    prediction_rows, total_loss, count = [], 0.0, 0
    model.eval()
    with torch.no_grad():
        for xyz, target, indices in loader:
            xyz, target = xyz.cuda(non_blocking=True), target.cuda(non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model({'pos': xyz, 'x': xyz.transpose(1, 2).contiguous()})
                loss = torch.nn.functional.cross_entropy(logits, target)
            if not torch.isfinite(logits).all():
                raise RuntimeError('Non-finite evaluation logits')
            prediction = logits.argmax(1)
            y, pred = target.cpu().numpy(), prediction.cpu().numpy()
            np.add.at(cm, (y, pred), 1)
            total_loss += loss.item() * len(y)
            count += len(y)
            if save_predictions:
                probabilities = logits.float().softmax(1).cpu().numpy()
                for index, actual, guessed, probs in zip(indices.tolist(), y.tolist(), pred.tolist(), probabilities):
                    r = {'source': loader.dataset.rows[index]['source'], 'true': actual, 'predicted': guessed}
                    r.update({f'p_{i}': float(p) for i, p in enumerate(probs)})
                    prediction_rows.append(r)
    result = metrics_from_confusion(cm.tolist())
    result.update(loss=total_loss / max(count, 1), confusion=cm.tolist(), samples=count)
    return result, prediction_rows


def inference_benchmark(model, cloud, batch_size, amp, iterations):
    model.eval()
    xyz = cloud.unsqueeze(0).repeat(batch_size, 1, 1).cuda()
    data = {'pos': xyz, 'x': xyz.transpose(1, 2).contiguous()}
    timings = []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
        for _ in range(20):
            model(data)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(iterations):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            model(data)
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end))
    return {'latency_batch_size': batch_size, 'inference_ms_per_batch': float(np.mean(timings)),
            'inference_ms_per_sample': float(np.mean(timings)) / batch_size,
            'inference_p95_ms_per_batch': float(np.percentile(timings, 95)),
            'inference_samples_per_second': 1000 * batch_size / float(np.mean(timings)),
            'inference_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'timing_scope': 'CUDA model forward only; data loading/H2D excluded; training-point input'}


def train(job_path):
    job = read_json(job_path)
    p, seed, name = job['protocol'], job['seed'], job['method']
    run_dir = Path(job_path).parent
    seed_all(seed)
    env_path = run_dir / 'environment.json'
    current_env = environment()
    if env_path.exists():
        previous = read_json(env_path)
        for key in ('torch', 'numpy', 'cuda_runtime', 'cudnn', 'gpu'):
            if previous[key] != current_env[key]:
                raise RuntimeError(f'Resume environment changed: {key}. Use a separate output directory.')
    write_json(env_path, current_env)
    model = build_model(name).cuda()
    parameters = sum(x.numel() for x in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=p['lr'], weight_decay=p['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=p['epochs'], eta_min=p['min_lr'])
    scaler = torch.cuda.amp.GradScaler(enabled=p['amp'])
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=p['label_smoothing'])
    # Standard PointNet feature-transform regularizer; explicitly model-specific.
    transforms = []
    if name == 'pointnet':
        model.encoder.fstn.register_forward_hook(lambda module, inputs, output: transforms.append(output) if model.training else None)
    train_set = Clouds(job['manifest'], 'train', seed, p['augmentation'])
    val_set = Clouds(job['manifest'], 'val', seed, p['augmentation'])
    if len(train_set) < p['micro_batch']:
        raise ValueError('Training set is smaller than one micro-batch')
    val_loader = data_loader(val_set, p['eval_batch'], p['workers'])
    history, first_epoch, best_key, best_epoch, total_seconds = [], 1, (-1., -1.), 0, 0.0
    latest = run_dir / 'latest.pth'
    if latest.exists():
        state = torch.load(latest, map_location='cpu', weights_only=False)
        if state['identity'] != job['identity']:
            raise RuntimeError('Checkpoint protocol/data/code identity mismatch')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        history, first_epoch = state['history'], state['epoch'] + 1
        best_key, best_epoch, total_seconds = tuple(state['best_key']), state['best_epoch'], state['wall_seconds']
        random.setstate(state['rng_python'])
        np.random.set_state(state['rng_numpy'])
        torch.set_rng_state(state['rng_torch'])
        torch.cuda.set_rng_state_all(state['rng_cuda'])
        del state
        print(f'Resuming {name} seed={seed} at epoch {first_epoch}', flush=True)
    accumulation = p['effective_batch'] // p['micro_batch']
    loader = logits = loss = xyz = target = transform = None
    for epoch in range(first_epoch, p['epochs'] + 1):
        train_set.epoch = epoch
        loader = data_loader(train_set, p['micro_batch'], p['workers'], True, seed + epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, seen = 0., 0
        cm = np.zeros((p['num_classes'], p['num_classes']), dtype=np.int64)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for step, (xyz, target, _) in enumerate(loader):
            xyz, target = xyz.cuda(non_blocking=True), target.cuda(non_blocking=True)
            transforms.clear()
            with torch.cuda.amp.autocast(enabled=p['amp']):
                logits = model({'pos': xyz, 'x': xyz.transpose(1, 2).contiguous()})
                loss = criterion(logits, target)
            if transforms:
                transform = transforms[-1].float()
                eye = torch.eye(transform.shape[-1], device=transform.device)
                loss = loss + p['pointnet_transform_regularization'] * torch.linalg.matrix_norm(transform @ transform.transpose(1, 2) - eye, ord='fro').mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite training loss at epoch {epoch} batch {step}')
            window_start = (step // accumulation) * accumulation
            window_size = min(accumulation, len(loader) - window_start)
            scaler.scale(loss / window_size).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), p['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach()) * len(target)
            seen += len(target)
            np.add.at(cm, (target.cpu().numpy(), logits.detach().argmax(1).cpu().numpy()), 1)
            if (step + 1) % 100 == 0:
                print(f'{name} seed={seed} epoch={epoch} batch={step+1}/{len(loader)} loss={total_loss/seen:.5f}', flush=True)
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - started
        peak_memory = torch.cuda.max_memory_allocated()
        val, _ = evaluate(model, val_loader, p['num_classes'], p['amp'])
        train_metrics = metrics_from_confusion(cm.tolist())
        record = {'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'], 'train_loss': total_loss / seen,
                  'train_oa': train_metrics['oa'], 'val_loss': val['loss'], 'val_oa': val['oa'],
                  'val_macc': val['macc'], 'val_macro_f1': val['macro_f1'], 'train_seconds': train_seconds,
                  'train_peak_allocated_bytes': peak_memory, 'train_samples_seen': seen}
        history.append(record)
        # Fixed selection rule: macro F1 first, OA tie-break. Never consult test.
        key = (val['macro_f1'], val['oa'])
        if key > best_key:
            best_key, best_epoch = key, epoch
            checkpoint(run_dir / 'best.pth', {'model': model.state_dict(), 'epoch': epoch,
                       'validation': val, 'identity': job['identity']})
        scheduler.step()
        total_seconds += time.perf_counter() - started
        checkpoint(latest, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                   'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'epoch': epoch,
                   'history': history, 'best_key': best_key, 'best_epoch': best_epoch,
                   'wall_seconds': total_seconds, 'identity': job['identity'],
                   'rng_python': random.getstate(), 'rng_numpy': np.random.get_state(),
                   'rng_torch': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all()})
        write_csv(run_dir / 'history.csv', history)
        print(json.dumps(record), flush=True)
    best = torch.load(run_dir / 'best.pth', map_location='cpu', weights_only=False)
    if best['identity'] != job['identity']:
        raise RuntimeError('Best checkpoint identity mismatch')
    model.load_state_dict(best['model'])
    selected_validation = best['validation']
    del best
    # Construct and evaluate test only after the complete training/selection.
    test_set = Clouds(job['manifest'], 'test', seed, p['augmentation'])
    test, predictions = evaluate(model, data_loader(test_set, p['eval_batch'], p['workers']), p['num_classes'], p['amp'], True)
    write_csv(run_dir / 'test_predictions.csv', predictions)
    classes = read_json(job['manifest'])['classes']
    write_csv(run_dir / 'per_class.csv', [{'class': c, **{key: test[key][i] for key in ('support', 'precision', 'recall', 'f1')}} for i, c in enumerate(classes)])
    write_json(run_dir / 'confusion.json', {'classes': classes, 'matrix': test['confusion']})
    del optimizer, scheduler, scaler, loader, val_loader, logits, loss, xyz, target, transform
    transforms.clear()
    torch.cuda.empty_cache()
    benchmark = inference_benchmark(model, test_set.__class__(job['manifest'], 'train', seed, p['augmentation'])[0][0],
                                    p['latency_batch'], p['amp'], p['latency_iterations'])
    result = {'method': name, 'seed': seed, 'identity': job['identity'], 'best_epoch': best_epoch,
              'manifest_fingerprint': read_json(job['manifest'])['fingerprint'],
              'selection': 'validation macro_f1, tie-break validation OA, earliest tied epoch',
              'parameters': parameters, 'training_validation_wall_seconds': total_seconds,
              'train_peak_allocated_bytes': max(r['train_peak_allocated_bytes'] for r in history),
              'selected_validation': selected_validation, 'test': test, 'benchmark': benchmark,
              'environment': environment()}
    write_json(run_dir / 'result.json', result)
    print('Completed:', run_dir, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', nargs='+')
    parser.add_argument('--amp', type=int, choices=(0, 1), default=1)
    parser.add_argument('--job', type=Path)
    args = parser.parse_args()
    if args.check:
        preflight(args.check, bool(args.amp))
    elif args.job:
        train(args.job)
    else:
        parser.error('Use --check or --job')
