#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export VLLM_USE_V1=0
export HF_DATASETS_TRUST_REMOTE_CODE=1

MODEL_NAME="phi-tiny"
LANGUAGE=""
TASK=""
ADAPTER_PATH=""
REQUESTED_GPUS="0"

count_gpus() {
  local devices="${1// /}"
  if [[ -z "$devices" ]]; then
    echo 1
    return
  fi

  local IFS=','
  read -ra gpu_ids <<< "$devices"
  local count=0
  local gpu_id
  for gpu_id in "${gpu_ids[@]}"; do
    if [[ -n "$gpu_id" ]]; then
      ((count++))
    fi
  done
  echo "$count"
}

usage() {
  cat <<'EOF'
Usage: run_eval_only.sh [options]

Options:
  -m, --model-name NAME       Model name or local checkpoint path
  -l, --language LANG         Target language code
  -t, --task TASK             Target task name
  -a, --adapter-path PATH     Optional LoRA adapter path to merge before eval
  -g, --gpus LIST             CUDA_VISIBLE_DEVICES value
  -h, --help                  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    -l|--language)
      LANGUAGE="$2"
      shift 2
      ;;
    -t|--task)
      TASK="$2"
      shift 2
      ;;
    -a|--adapter-path)
      ADAPTER_PATH="$2"
      shift 2
      ;;
    -g|--gpus)
      REQUESTED_GPUS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

export CUDA_VISIBLE_DEVICES="$REQUESTED_GPUS"
GPU_COUNT="$(count_gpus "$REQUESTED_GPUS")"

# Construct run name based on provided arguments
RUN_SUFFIX=""
if [[ -n "$LANGUAGE" ]]; then
  RUN_SUFFIX="${RUN_SUFFIX}-${LANGUAGE}"
fi
if [[ -n "$TASK" ]]; then
  RUN_SUFFIX="${RUN_SUFFIX}-${TASK}"
fi

cmd=(
  uv run python "$SCRIPT_DIR/run_eval.py"
  --model_name "$MODEL_NAME"
  --run_name "eval-${MODEL_NAME##*/}${RUN_SUFFIX}"
  --tensor_parallel_size "$GPU_COUNT"
)

if [[ -n "$LANGUAGE" ]]; then
  cmd+=(--language "$LANGUAGE")
fi
if [[ -n "$TASK" ]]; then
  cmd+=(--task "$TASK")
fi

if [[ -n "$ADAPTER_PATH" ]]; then
  cmd+=(--adapter_path "$ADAPTER_PATH")
fi

exec "${cmd[@]}"
