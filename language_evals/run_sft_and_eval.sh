#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_NAME="phi-tiny"
LANGUAGE="pes"
DATASET_NAME="my_custom_dataset"
WANDB_PROJECT="moe-sft-eval"
RAM_DISK_OUTPUT="/dev/shm/moe_adapter_output"
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

accelerate_config_for_gpus() {
  case "$1" in
    1) echo "$SCRIPT_DIR/accelerate_config_1gpu.yaml" ;;
    2|4|8) echo "$SCRIPT_DIR/accelerate_config_multi_gpu.yaml" ;;
    *)
      echo "Unsupported GPU count: $1. Expected 1, 2, 4, or 8 GPUs." >&2
      exit 1
      ;;
  esac
}

usage() {
  cat <<'EOF'
Usage: run_sft_and_eval.sh [options]

Options:
  -m, --model-name NAME       Model name or local checkpoint path
  -l, --language LANG         Target language code
  -d, --dataset-name NAME     LlamaFactory dataset name
  -o, --output-dir PATH       RAM disk / adapter output path
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
    -d|--dataset-name)
      DATASET_NAME="$2"
      shift 2
      ;;
    -o|--output-dir)
      RAM_DISK_OUTPUT="$2"
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
ACCELERATE_CONFIG="$(accelerate_config_for_gpus "$GPU_COUNT")"

ACCELERATE_ARGS=(accelerate launch --config_file "$ACCELERATE_CONFIG")
if [[ "$GPU_COUNT" -gt 1 ]]; then
  ACCELERATE_ARGS+=(--num_processes "$GPU_COUNT")
fi

RUN_NAME="train_eval-${MODEL_NAME##*/}-${LANGUAGE}"

"${ACCELERATE_ARGS[@]}" "$SCRIPT_DIR/sft.py" \
  --model_name "$MODEL_NAME" \
  --dataset_name "$DATASET_NAME" \
  --ram_disk_output "$RAM_DISK_OUTPUT" \
  --wandb_project "$WANDB_PROJECT" \
  --run_name "$RUN_NAME"

python "$SCRIPT_DIR/evaluate.py" \
  --model_name "${RAM_DISK_OUTPUT}_merged" \
  --language "$LANGUAGE" \
  --run_name "$RUN_NAME" \
  --cleanup_model_path "${RAM_DISK_OUTPUT}_merged"
