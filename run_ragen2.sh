#!/usr/bin/env bash
# Run this checkout with the independent ragen2 conda environment.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1

env_prefix="${RAGEN2_ENV_PREFIX:-}"
if [[ -z "$env_prefix" && -n "${CONDA_PREFIX:-}" && "${CONDA_PREFIX##*/}" == ragen2 ]]; then
  env_prefix="$CONDA_PREFIX"
fi
if [[ -z "$env_prefix" ]]; then
  conda_exe="${CONDA_EXE:-}"
  if [[ -z "$conda_exe" ]]; then
    conda_exe="$(type -P conda || true)"
  fi
  if [[ -z "$conda_exe" || ! -x "$conda_exe" ]]; then
    echo 'Activate conda environment ragen2, or set RAGEN2_ENV_PREFIX to its prefix.' >&2
    exit 2
  fi
  # conda run establishes CONDA_PREFIX; the second invocation takes the branch above.
  exec "$conda_exe" run --no-capture-output -n ragen2 bash "$project_dir/run_ragen2.sh" "$@"
fi

env_prefix="${env_prefix%/}"
if [[ "${env_prefix##*/}" != ragen2 || ! -d "$env_prefix/conda-meta" || ! -x "$env_prefix/bin/python" ]]; then
  echo "Expected a conda prefix named ragen2: $env_prefix" >&2
  exit 2
fi

export PYTHONPATH="$project_dir/verl:$project_dir"
export PATH="$env_prefix/bin:$PATH"
export CONDA_PREFIX="$env_prefix"
export CONDA_DEFAULT_ENV=ragen2
export RAGEN_API_KEYS_PATH="$project_dir/ragen/env/api-keys.json"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
cd "$project_dir"

# A stale symlink/incorrect prefix must fail before importing training code.
"$env_prefix/bin/python" -B -c \
  'import pathlib,sys; expected=pathlib.Path(sys.argv[1]).resolve(); actual=pathlib.Path(sys.prefix).resolve(); sys.exit(0 if actual == expected else "Python prefix mismatch: " + str(actual))' \
  "$env_prefix"
exec "$env_prefix/bin/python" "$@"
