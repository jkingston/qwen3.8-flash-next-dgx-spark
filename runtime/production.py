#!/usr/bin/env python3
"""Replay the pinned production manifest without replacing existing containers."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

MANIFEST = Path(__file__).resolve().parents[1] / 'profiles/ultrafast-production-create.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--create', action='store_true')
    mode.add_argument('--run', action='store_true')
    args = parser.parse_args()
    argv = json.loads(MANIFEST.read_text())
    if argv[:2] != ['docker', 'create'] or not all(isinstance(x, str) for x in argv):
        raise SystemExit('Invalid create manifest')
    name = argv[argv.index('--name') + 1]
    if args.dry_run:
        print(shlex.join(argv))
        return
    # Daemon failures must not be mistaken for an absent container.
    subprocess.run(['docker', 'info', '--format', '{{.ServerVersion}}'], check=True, stdout=subprocess.DEVNULL)
    existing = subprocess.run(['docker', 'container', 'ls', '-a', '--format', '{{.Names}}'], check=True, capture_output=True, text=True)
    if name in existing.stdout.splitlines():
        raise SystemExit(f'Container {name} already exists; refusing replacement or restart')
    for i, value in enumerate(argv):
        if value == '-v':
            host = Path(argv[i + 1].split(':', 1)[0])
            if not host.exists():
                raise SystemExit(f'Required asset missing: {host}')
    image = next(x for x in argv if x.startswith('sha256:'))
    subprocess.run(['docker', 'image', 'inspect', image], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(argv, check=True)
    if args.run:
        subprocess.run(['docker', 'start', '--attach', name], check=True)


if __name__ == '__main__':
    main()
