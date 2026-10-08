#!/usr/bin/env bash
# Current production profile; never overwrite an existing container.
set -euo pipefail
exec python3 "$(dirname "${BASH_SOURCE[0]}")/production.py" --run "$@"
