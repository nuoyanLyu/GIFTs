#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash run_qwen3_rl.sh CONFIG_NAME [Hydra GPU/game/budget overrides]' >&2
  exit 2
fi
config_name="$1"
shift
project_dir="$(cd "$(dirname "$0")" && pwd)"
profile_overrides=(model_profile=qwen3_4b_nothink)
if [[ -n "${RAGEN_MODEL_PATH:-}" ]]; then
  profile_overrides+=("model_path=$RAGEN_MODEL_PATH")
fi
hydra_options=()
overrides=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cfg|-c|--package|--shell-completion)
      [[ $# -ge 2 ]] || { echo "$1 requires a value." >&2; exit 2; }
      hydra_options+=("$1" "$2"); shift 2 ;;
    --info)
      hydra_options+=("$1"); shift
      if [[ $# -gt 0 ]]; then
        case "$1" in all|config|defaults|defaults-tree|plugins|searchpath)
          hydra_options+=("$1"); shift ;;
        esac
      fi ;;
    --resolve|--help|-h|--hydra-help|--version|--run|-r|--multirun|-m)
      hydra_options+=("$1"); shift ;;
    --cfg=*|--package=*|--info=*|--shell-completion=*)
      hydra_options+=("$1"); shift ;;
    --config-name|--config-name=*|--config-path|--config-path=*)
      echo 'Select the configuration using CONFIG_NAME.' >&2; exit 2 ;;
    --) shift; overrides+=("$@"); break ;;
    *) overrides+=("$1"); shift ;;
  esac
done
# Hydra's nargs=* overrides must stay contiguous after its command options.
exec bash "$project_dir/run_factory.sh" train.py --config-name "$config_name" \
  ${hydra_options[@]+"${hydra_options[@]}"} "${profile_overrides[@]}" \
  ${overrides[@]+"${overrides[@]}"}
