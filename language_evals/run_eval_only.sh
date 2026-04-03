#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_NAME="phi-tiny"
LANGUAGE="pes"
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

cmd=(
  python "$SCRIPT_DIR/evaluate.py"
  --model_name "$MODEL_NAME"
  --language "$LANGUAGE"
  --run_name "eval-${MODEL_NAME##*/}-${LANGUAGE}"
)

if [[ -n "$ADAPTER_PATH" ]]; then
  cmd+=(--adapter_path "$ADAPTER_PATH")
fi

exec "${cmd[@]}"
