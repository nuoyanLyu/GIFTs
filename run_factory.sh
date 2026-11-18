#!/usr/bin/env bash
# Compatibility name used by the existing training, merge and smoke scripts.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$project_dir/run_ragen2.sh" "$@"
