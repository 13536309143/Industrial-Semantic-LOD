"""Reject private research artifacts in the Git index or pushed history.

This checks committed paths and deployment metadata, not arbitrary prose for
all possible confidential information. Review public comparison text manually.
"""
import argparse
import json
from pathlib import PurePosixPath
import subprocess
import sys

PRIVATE_ROOTS = (
    'experiments/', 'analysis_reports/', 'outputs/', 'lod_analysis_outputs/',
    'datasets/', '_downloaded_resources/', '.artifacts/', 'private/', 'data/',
    'docs/assets/',
)
PRIVATE_FILES = {'tools/maintenance/export_thesis_results.py'}
DATA_SUFFIXES = {
    '.csv', '.tsv', '.jsonl', '.ndjson', '.npy', '.npz', '.xlsx', '.xls',
    '.docx', '.pptx', '.pdf', '.pth', '.pt', '.ckpt', '.onnx', '.safetensors',
    '.glb', '.gltf', '.zippp', '.zip', '.7z', '.tar', '.gz', '.bundle',
    '.log', '.dll', '.exe', '.pdb',
}
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.svg'}
UPSTREAM_ROOTS = (
    'tools/pointclipv2/PointCLIP_V2/',
    'tools/pointnext_lod/PointNeXt/', 'thirdparty/',
)
DEPLOYMENT = 'models/pointnext/deployment.json'
PUBLIC_DEPLOYMENT_KEYS = {
    'schema', 'method', 'class_count', 'input_points',
    'checkpoint_relative_path', 'class_order_file', 'weights_distributed',
}
PUBLIC_UPSTREAM_RESOURCES = {
    'tools/pointclipv2/PointCLIP_V2/zeroshot_cls/clip/bpe_simple_vocab_16e6.txt.gz',
    'tools/pointclipv2/PointCLIP_V2/zeroshot_seg/clip/bpe_simple_vocab_16e6.txt.gz',
    'tools/pointnext_lod/PointNeXt/openpoints/dataset/matterport3d/category_mapping.tsv',
}


def git(*args, input=None):
    result = subprocess.run(['git', *args], input=input, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=True)
    return result.stdout


def check(tree=None):
    raw = git('ls-files', '-z') if tree is None else git('ls-tree', '-r', '--name-only', '-z', tree)
    paths = [p.decode('utf-8') for p in raw.split(b'\0') if p]
    failures = []
    for path in paths:
        if path in PUBLIC_UPSTREAM_RESOURCES:
            continue
        suffix = PurePosixPath(path).suffix.lower()
        if (path.startswith(PRIVATE_ROOTS) or path in PRIVATE_FILES
                or suffix in DATA_SUFFIXES or path.endswith('.provenance.json')
                or (suffix in IMAGE_SUFFIXES and not path.startswith(UPSTREAM_ROOTS))):
            failures.append(path)
        if path == DEPLOYMENT:
            blob = git('show', (tree + ':' if tree else ':') + path)
            payload = json.loads(blob)
            if set(payload) - PUBLIC_DEPLOYMENT_KEYS or payload.get('weights_distributed') is not False:
                failures.append(path + ' (non-public metadata fields)')
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--staged', action='store_true')
    mode.add_argument('--history', nargs='+')
    mode.add_argument('--pre-push', action='store_true')
    args = parser.parse_args()
    if args.staged:
        failures = check()
    else:
        refs = args.history or [line.split()[1] for line in sys.stdin
                                if line.strip() and set(line.split()[1]) != {'0'}]
        if not refs:
            return 0
        commits = git('rev-list', '--stdin', input=('\n'.join(refs) + '\n').encode()).decode().splitlines()
        failures = []
        checked = set()
        for commit in commits:
            tree = git('show', '-s', '--format=%T', commit).decode().strip()
            if tree in checked:
                continue
            checked.add(tree)
            failures = check(tree)
            if failures:
                break
    if failures:
        print('Publication check rejected private artifact paths:', file=sys.stderr)
        for path in sorted(set(failures))[:25]:
            print('  ' + path, file=sys.stderr)
        print('Keep full research material outside the public Git tree.', file=sys.stderr)
        return 1
    print('Publication artifact check passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
