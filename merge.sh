#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash merge.sh CHECKPOINT_ROOT [HF_ROOT [FIRST_STEP [LAST_STEP]]]
Exports global_step_N/actor as HF_ROOT/step_N for N = 50, 100, 150, ...
Defaults: HF_ROOT=CHECKPOINT_ROOT/hf, FIRST_STEP=50, LAST_STEP=latest saved multiple of 50.
The checkpoint root must contain the veRL latest_checkpointed_iteration.txt tracker.
EOF
}

if [[ ${1:-} == --help ]]; then usage; exit 0; fi
if (( $# < 1 || $# > 4 )); then usage >&2; exit 2; fi

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
checkpoint_root="$(realpath -m -- "$1")"
hf_root="$(realpath -m -- "${2:-$checkpoint_root/hf}")"
first="${3:-50}"
tracker="$checkpoint_root/latest_checkpointed_iteration.txt"
if [[ ! -s "$tracker" ]]; then
  echo "No saved checkpoint tracker at $tracker" >&2
  exit 3
fi
latest="$(cat "$tracker")"
if [[ ! "$latest" =~ ^[0-9]{1,9}$ ]]; then
  echo "Invalid checkpoint tracker at $tracker" >&2
  exit 2
fi
latest=$((10#$latest))
last="${4:-$((latest / 50 * 50))}"
if [[ ! "$first" =~ ^[1-9][0-9]{0,8}$ || ! "$last" =~ ^[0-9]{1,9}$ ]] ||
   (( first % 50 || 10#$last % 50 )); then
  echo 'FIRST_STEP and LAST_STEP must be multiples of 50.' >&2
  exit 2
fi
last=$((10#$last))
if (( last < first || last > latest )); then
  echo "Requested steps $first through $last are not available (latest saved step $latest)." >&2
  exit 3
fi

mkdir -p -- "$hf_root"
exec 9>"$hf_root/.merge.lock"
flock -n 9 || { echo "Another merge is writing to $hf_root" >&2; exit 4; }
stage=''
trap 'if [[ -n "$stage" ]]; then rm -rf -- "$stage"; fi' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

# Verify the weight files and record the identities of the training shards used.
validate_export() {
  python3 - "$@" <<'PY'
import json
import sys
from pathlib import Path

actor, target = map(Path, sys.argv[1:3])
world_size = json.loads((actor / "fsdp_config.json").read_text())["world_size"]
if not isinstance(world_size, int) or world_size < 1:
    raise ValueError(f"Invalid FSDP world size: {world_size}")
shards = {}
for rank in range(world_size):
    shard = actor / f"model_world_size_{world_size}_rank_{rank}.pt"
    stat = shard.stat()
    if not shard.is_file() or not stat.st_size:
        raise ValueError(f"Missing or empty model shard: {shard}")
    shards[shard.name] = [stat.st_ino, stat.st_size, stat.st_mtime_ns]
if sys.argv[3] == "source":
    json.loads((actor / "huggingface/config.json").read_text())
    sys.exit(0)

record = {"actor": str(actor), "shards": shards}
index = target / "model.safetensors.index.json"
if index.is_file():
    weight_map = json.loads(index.read_text())["weight_map"]
    if not weight_map:
        raise ValueError(f"Empty Hugging Face weight index: {index}")
    files = set(weight_map.values())
else:
    files = {"model.safetensors"}
for name in files | {"config.json", "tokenizer_config.json"}:
    path = target / name
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f"Missing or empty Hugging Face file: {path}")
json.loads((target / "config.json").read_text())
json.loads((target / "tokenizer_config.json").read_text())
manifest = target / "merge_source.json"
if sys.argv[3] == "record":
    manifest.write_text(json.dumps(record, indent=2) + "\n")
elif json.loads(manifest.read_text()) != record:
    raise ValueError(f"Existing output does not match the current checkpoint: {target}")
PY
}

for ((step=first; step<=last; step+=50)); do
  actor="$checkpoint_root/global_step_$step/actor"
  target="$hf_root/step_$step"
  if [[ -e "$target" || -L "$target" ]]; then
    validate_export "$actor" "$target" verify
    echo "Reusing verified step $step at $target"
    continue
  fi
  validate_export "$actor" "$target" source
  stage="$(mktemp -d "$hf_root/.step_$step.XXXXXXXX")"
  echo "Merging step $step from $actor"
  bash "$project_dir/merge_hf.sh" "$actor" "$stage/hf"
  validate_export "$actor" "$stage/hf" record
  mv -nT -- "$stage/hf" "$target"
  rmdir -- "$stage"
  stage=''
  echo "Exported step $step to $target"
done
