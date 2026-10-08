#!/usr/bin/env python3
"""Replay the pinned production manifest without replacing existing containers."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
from string import Template
from assets import ensure_assets

MANIFEST = Path(__file__).resolve().parents[1] / 'profiles/ultrafast-production-create.json'


def resolve_manifest():
    defaults = {
        'MODEL_DIR': Path.home() / 'models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-mtpdense-g32',
        'PLE_DIR': Path.home() / 'models/ple-table-fp8',
        'DRAFT_VOCAB_FILE': Path.home() / '.cache/qwen38-v16b/draft-vocab-ids-K65536.txt',
    }
    paths = {}
    for key, default in defaults.items():
        value = os.environ.get(key, str(default))
        if not value.strip() or ':' in value or '\n' in value or '\r' in value:
            raise SystemExit(f'{key} must be a nonempty path without colons or newlines')
        paths[key] = str(Path(value).expanduser().resolve())
    argv = json.loads(MANIFEST.read_text())
    if argv[:2] != ['docker', 'create'] or not all(isinstance(x, str) for x in argv):
        raise SystemExit('Invalid create manifest')
    return [Template(value).substitute(paths) for value in argv]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--create', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--no-download', action='store_true', help='Fail if any required asset is missing or incomplete')
    args = parser.parse_args()
    argv = resolve_manifest()
    name = argv[argv.index('--name') + 1]
    if args.dry_run:
        print(shlex.join(argv))
        return
    # Daemon failures must not be mistaken for an absent container.
    subprocess.run(['docker', 'info', '--format', '{{.ServerVersion}}'], check=True, stdout=subprocess.DEVNULL)
    existing = subprocess.run(['docker', 'container', 'ls', '-a', '--format', '{{.Names}}'], check=True, capture_output=True, text=True)
    if name in existing.stdout.splitlines():
        raise SystemExit(f'Container {name} already exists; refusing replacement or restart')
    image = next(x for x in argv if x.startswith('sha256:'))
    # Fail before large downloads when the required helper/serving image is absent.
    subprocess.run(['docker', 'image', 'inspect', image], check=True, stdout=subprocess.DEVNULL)
    mounts = {}
    for i, value in enumerate(argv):
        if value == '-v':
            host, container, _ = argv[i + 1].split(':')
            mounts[container] = Path(host)
    ensure_assets(mounts['/model'], mounts['/ple-table'], mounts['/draft-vocab/ids.txt'],
                  image, offline=args.no_download)
    subprocess.run(argv, check=True)
    if args.run:
        subprocess.run(['docker', 'start', '--attach', name], check=True)


if __name__ == '__main__':
    main()
