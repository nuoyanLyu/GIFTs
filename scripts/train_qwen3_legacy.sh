#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_qwen3_legacy.sh TASK --gpus IDS --num-gpus N \
    --output-dir NEW_DIRECTORY [Hydra overrides...]

TASK is math, nash, or nmt. GPU_LIST, NGPUS, and OUTPUT_DIR may supply the
corresponding resource options. The script never probes or selects free GPUs.
Use --cfg job --resolve to inspect the configuration without starting training;
--output-dir may be omitted for this inspection.

Examples:
  bash scripts/train_qwen3_legacy.sh math --gpus 4 --num-gpus 1 --cfg job --resolve
  bash scripts/train_qwen3_legacy.sh nash --gpus 4 --num-gpus 1 \
    --output-dir /data1/USER/checkpoints/qwen3-nash-run1 trainer.total_training_steps=200

The selected legacy config supplies the task, lengths, and batch sizes.
The legacy GRPO flags remain separate from the model settings.
run_qwen3_rl.sh selects model_profile=qwen3_4b_nothink for the model,
native non-thinking template, answer parser, and sampling defaults.
EOF
}

if [[ $# -eq 0 ]]; then usage >&2; exit 2; fi
if [[ "$1" == "--help" || "$1" == "-h" ]]; then usage; exit 0; fi
task="$1"
shift
case "$task" in
  math) config_name='_10_math' ;;
  nash) config_name='_16_nash_new' ;;
  nmt) config_name='_29_nmt_batch' ;;
  *) echo "Unknown task: $task (choose math, nash, or nmt)." >&2; exit 2 ;;
esac

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
gpu_list="${GPU_LIST:-}"
num_gpus="${NGPUS:-}"
output_dir="${OUTPUT_DIR:-}"
extra=()
inspect_config=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus|--num-gpus|--output-dir)
      if [[ $# -lt 2 || -z "$2" ]]; then echo "$1 requires a value." >&2; exit 2; fi
      case "$1" in
        --gpus) gpu_list="$2" ;;
        --num-gpus) num_gpus="$2" ;;
        --output-dir) output_dir="$2" ;;
      esac
      shift 2
      ;;
    --cfg|-c)
      if [[ $# -lt 2 ]]; then echo "$1 requires a value." >&2; exit 2; fi
      inspect_config=true
      extra+=("$1" "$2")
      shift 2
      ;;
    --cfg=*) inspect_config=true; extra+=("$1"); shift ;;
    --config-name|--config-name=*|--config-path|--config-path=*)
      echo 'Choose the config using TASK; do not override config-name/config-path.' >&2
      exit 2
      ;;
    --) shift; extra+=("$@"); break ;;
    --help|-h) usage; exit 0 ;;
    *) extra+=("$1"); shift ;;
  esac
done

if [[ ! "$gpu_list" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]]; then
  echo 'Set --gpus (or GPU_LIST) explicitly, for example 4 or 4,5.' >&2
  exit 2
fi
if [[ ! "$num_gpus" =~ ^[1-9][0-9]*$ ]]; then
  echo 'Set --num-gpus (or NGPUS) explicitly to a positive integer.' >&2
  exit 2
fi
IFS=',' read -r -a gpu_ids <<< "$gpu_list"
if [[ "${#gpu_ids[@]}" -ne "$num_gpus" ]]; then
  echo 'The GPU list length must equal --num-gpus.' >&2
  exit 2
fi
for ((i=0; i<${#gpu_ids[@]}; i++)); do
  for ((j=i+1; j<${#gpu_ids[@]}; j++)); do
    if [[ "${gpu_ids[i]}" == "${gpu_ids[j]}" ]]; then
      echo 'The GPU list contains a duplicate device.' >&2
      exit 2
    fi
  done
done

# The + expansion also handles an empty array under macOS Bash 3.2 with set -u.
for value in ${extra[@]+"${extra[@]}"}; do
  case "$value" in
    --cfg|-c|--cfg=*) inspect_config=true ;;
    --config-name|--config-name=*|--config-path|--config-path=*)
      echo 'Choose the config using TASK; do not override config-name/config-path.' >&2
      exit 2
      ;;
  esac
  key="${value%%=*}"
  key="${key#++}"
  key="${key#+}"
  key="${key#\~}"
  case "$key" in
    system.CUDA_VISIBLE_DEVICES|trainer.n_gpus_per_node|actor_rollout_ref.rollout.tensor_model_parallel_size|trainer.default_local_dir)
      echo "Set resource/output field $key using this script's explicit options." >&2
      exit 2
      ;;
  esac
done

if [[ -z "$output_dir" ]]; then
  if "$inspect_config"; then
    # This is a displayed destination only; --cfg does not create it or train.
    output_dir="$project_dir/checkpoints/qwen3-${task}-configuration-preview"
  else
    echo 'Set a new --output-dir (or OUTPUT_DIR); old 1.5B run directories are not reused.' >&2
    exit 2
  fi
fi
case "$output_dir" in
  /*) ;;
  *) output_dir="$PWD/$output_dir" ;;
esac
if ! "$inspect_config" && [[ -e "$output_dir" ]]; then
  echo 'The output directory already exists. Choose a new run directory.' >&2
  exit 2
fi

# Match the old scripts' actual USE_GRPO expansion. User-provided scientific
# overrides remain explicit and come after these defaults. Hardware fields use
# the resource options above, with TP matching the selected legacy convention.
exec bash "$project_dir/run_qwen3_rl.sh" "$config_name" \
  algorithm.adv_estimator=grpo \
  agent_proxy.reward_normalization.method=mean_std \
  actor_rollout_ref.actor.use_kl_loss=True \
  "system.CUDA_VISIBLE_DEVICES=\"$gpu_list\"" \
  "trainer.n_gpus_per_node=$num_gpus" \
  "actor_rollout_ref.rollout.tensor_model_parallel_size=$num_gpus" \
  "trainer.default_local_dir=$output_dir" \
  ${extra[@]+"${extra[@]}"}
