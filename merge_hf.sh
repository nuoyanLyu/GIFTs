#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo 'Usage: bash merge_hf.sh FSDP_ACTOR_DIRECTORY NEW_HF_DIRECTORY' >&2
  exit 2
fi
project_dir="$(cd "$(dirname "$0")" && pwd)"
actor_dir="$1"
target_dir="$2"
if [[ ! -d "$actor_dir/huggingface" || -e "$target_dir" ]]; then
  echo 'Actor directory must contain huggingface metadata; target must be new.' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=''
exec bash "$project_dir/run_factory.sh" -m verl.model_merger merge \
  --backend fsdp --local_dir "$actor_dir" --target_dir "$target_dir"
