"""Content-based provenance for completed offline inference stages (stdlib only)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_identity(paths) -> dict:
    result = {}
    for item in paths:
        path = Path(item).resolve(strict=True)
        if path.is_dir():
            files = sorted(p for p in path.rglob("*") if p.is_file()
                           and p.suffix.lower() in {".py", ".pyd", ".so"})
            if not files:
                raise ValueError(f"No code files in {path}")
        else:
            files = [path]
        for file in files:
            result[str(file)] = file_hash(file)
    return result


def stage_identity(inputs, parameters) -> dict:
    payload = {"schema": 1, "inputs": files_identity(inputs), "parameters": parameters}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "fingerprint": hashlib.sha256(encoded).hexdigest()}


def matches(stamp: Path, identity: dict, outputs) -> bool:
    try:
        previous = json.loads(Path(stamp).read_text(encoding="utf-8"))
        return (previous.get("schema") == 1
                and previous.get("fingerprint") == identity["fingerprint"]
                and previous.get("outputs") == files_identity(outputs))
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False


def record(stamp: Path, identity: dict, outputs) -> None:
    # Call only after successful inference; output hashes detect truncation/edits.
    payload = {**identity, "outputs": files_identity(outputs)}
    path = Path(stamp)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "record"])
    parser.add_argument("--stamp", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, action="append", required=True)
    parser.add_argument("--param", action="append", default=[])
    args = parser.parse_args()
    try:
        identity = stage_identity(args.input, sorted(args.param))
        if args.action == "record":
            record(args.stamp, identity, args.output)
            return 0
        valid = matches(args.stamp, identity, args.output)
        print(f"{'Reuse verified' if valid else 'Rebuild missing/stale'} stage: {args.stamp}")
        return 0 if valid else 1
    except (OSError, ValueError) as exc:
        print(f"Provenance error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
