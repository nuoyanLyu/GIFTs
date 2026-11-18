#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    cat <<'EOF'
Usage: reason_test/run_qwen3.sh MODEL_DIR GPU RUN_DIR [options] [test ...]
Options:
  --max-samples N  Limit each selected test to N items
  --max-tokens N   Override the default budget (600 tokens, IFEval 2500)
  --seed N         Set the evaluation seed (default 42)
Tests: nash tictactoe undercover mmlu mmlu-pro math social common ifeval logic
With no test names, run all ten in the listed order.
EOF
    exit 0
fi

if [[ $# -lt 3 ]]; then
    echo 'Expected MODEL_DIR GPU RUN_DIR. See --help.' >&2
    exit 2
fi
model_dir="$(cd "$1" && pwd)"
device="$2"
run_dir="$3"
shift 3
if [[ ! -f "$model_dir/config.json" || ! "$device" =~ ^[0-9]+$ ]]; then
    echo 'Specify a local HF model directory and a single GPU index.' >&2
    exit 2
fi

options=()
tests=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-samples|--max-tokens|--seed)
            if [[ $# -lt 2 || ! "$2" =~ ^[0-9]+$ ]]; then
                echo "Expected an integer after $1." >&2
                exit 2
            fi
            flag="${1:2}"
            options+=("--${flag/-/_}" "$2")
            shift 2
            ;;
        nash|tictactoe|undercover|mmlu|mmlu-pro|math|social|common|ifeval|logic)
            for selected in "${tests[@]}"; do
                if [[ "$selected" == "$1" ]]; then
                    echo "Duplicate test: $1" >&2
                    exit 2
                fi
            done
            tests+=("$1")
            shift
            ;;
        *) echo "Unknown option or test: $1" >&2; exit 2 ;;
    esac
done
if [[ ${#tests[@]} == 0 ]]; then
    tests=(nash tictactoe undercover mmlu mmlu-pro math social common ifeval logic)
fi

if [[ -e "$run_dir" ]]; then
    echo "Choose a new run directory: $run_dir" >&2
    exit 2
fi
mkdir -p "$run_dir"
run_dir="$(cd "$run_dir" && pwd)"
test_dir="$(cd "$(dirname "$0")" && pwd)"
project_dir="$(dirname "$test_dir")"
python="${RAGEN2_PYTHON:-$HOME/anaconda3/envs/ragen2/bin/python}"
scorer="${IFEVAL_SOURCE:-$HOME/RAGEN/third_party/ifeval}"
scorer_deps="${IFEVAL_DEPS:-$test_dir/.ifeval_deps}"
cd "$project_dir"

printf 'model_dir=%s\ngpu=%s\npython=%s\nenable_thinking=False\n' "$model_dir" "$device" "$python" > "$run_dir/manifest.txt"
printf 'tests=%s\n' "${tests[*]}" >> "$run_dir/manifest.txt"
sha256sum "$test_dir"/*.py "$test_dir/run_qwen3.sh" > "$run_dir/source.sha256"
for test_name in "${tests[@]}"; do
    device_arg=--device
    case "$test_name" in
        nash) script=nash-new ;;
        tictactoe|undercover|social|common) script="$test_name" ;;
        mmlu) script=mmlu_dl ;;
        mmlu-pro) script=mmlu_pro_dl ;;
        math) script=lv3to5_dl ;;
        ifeval) script=ifEval; device_arg=--cuda_id ;;
        logic) script=logic ;;
    esac
    extra_path=''
    if [[ "$test_name" == ifeval ]]; then
        extra_path="$scorer:$scorer_deps:"
    fi
    command=("$python" -u "$test_dir/$script.py" --model_dir "$model_dir" --model_name "$(basename "$model_dir")"
        "$device_arg" "$device" --qwen3 --output_dir "$run_dir/$test_name" "${options[@]}")
    printf '%q ' "${command[@]}" >> "$run_dir/manifest.txt"
    printf '\n' >> "$run_dir/manifest.txt"
    echo "Running $test_name"
    env PYTHONPATH="$extra_path$project_dir${PYTHONPATH:+:$PYTHONPATH}" "${command[@]}" 2>&1 | tee "$run_dir/$test_name.log"
done
