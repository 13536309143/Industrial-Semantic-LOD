"""Copy an independently verified inference snapshot without touching its source."""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/pointnext_lod'))
from artifact_cache import file_hash, files_identity, record, stage_identity


def reuse(source, destination, glb):
    source, destination, glb = source.resolve(strict=True), destination.resolve(), glb.resolve(strict=True)
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite existing snapshot: {destination}')
    state = json.loads((source / 'inference_status.json').read_text(encoding='utf-8'))
    if state['glb_sha256'] != file_hash(glb):
        raise ValueError('Scene content changed; fresh inference required')
    if state['checkpoint_sha256'] != file_hash(ROOT / 'models/pointnext/industrial-part-14class-best.pth'):
        raise ValueError('Checkpoint changed; fresh inference required')
    stem = glb.stem.replace(' ', '_')
    stamps = [source / f'{stem}_pointnext_provenance.json', source / f'{stem}_full.provenance.json',
              source / f'{stem}_candidates.provenance.json']
    verified = []
    for path in stamps:
        payload = json.loads(path.read_text(encoding='utf-8'))
        if files_identity(payload['inputs']) != payload['inputs']:
            raise ValueError(f'Inference inputs/code changed: {path}')
        if files_identity(payload['outputs']) != payload['outputs']:
            raise ValueError(f'Inference outputs changed: {path}')
        if any(not Path(p).is_relative_to(source) for p in payload['outputs']):
            raise ValueError('Unexpected output outside source snapshot')
        verified.append((path, payload))
    shutil.copytree(source, destination)

    def relocated(path):
        path = Path(path)
        return destination / path.relative_to(source) if path.is_relative_to(source) else path

    for path, payload in verified:
        outputs = [relocated(p) for p in payload['outputs']]
        for old, expected in payload['outputs'].items():
            if file_hash(relocated(old)) != expected:
                raise ValueError('Snapshot copy integrity failure')
        identity = stage_identity([relocated(p) for p in payload['inputs']], payload['parameters'])
        record(relocated(path), identity, outputs)
    state['snapshot_reuse'] = {'source': str(source), 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
                              'glb_sha256': state['glb_sha256'], 'source_stamps_sha256': {str(p): file_hash(p) for p in stamps},
                              'timing_scope': 'Stage timings belong to original inference; no fresh GPU inference executed.'}
    for stage in state['stages']:
        stage['reused_snapshot'] = True
        stage['original_wall_seconds'] = stage['wall_seconds']
    state['status'] = 'inference_complete_analysis_pending'
    state['snapshot_directory'] = str(destination)
    (destination / 'inference_status.json').write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    (destination / 'snapshot_reuse.json').write_text(json.dumps(state['snapshot_reuse'], ensure_ascii=False, indent=2), encoding='utf-8')
    return state['snapshot_reuse']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--glb', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reuse(args.source_dir, args.output_dir, args.glb), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
