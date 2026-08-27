#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$REPO_ROOT/routing_analysis/get_routing_weights.py"

GPU_IDS="${GPU_IDS:-0,1}"

# Fill this list with the checkpoint directories you want to process.
# Each entry can be either a relative name or an absolute path.
CHECKPOINTS_ROOT="/data2/lucasbandarkar/checkpoints"
CHECKPOINTS=(
    # "granite_hun_baseline-target_lm_200k"
    # "granite_hun__L10-35_nofreeze_199k/"
    # "granite_hun__L10-35_nofreeze_201k/"
    # "granite_hun__L10-35_nofreeze_198k/"
    # "granite_hun__L10-35_routers_200k/"
    # "granite_hun__L10-35_routers_199k/"
    # "granite_hun__L10-35_routers_201k/"
    # "granite_sin_baseline-target_lm_200k/"
    # "granite_sin__L10-35_200k/"
    # "granite_sin__L10-35_nofreeze_200k"
    # "granite_sin__L10-20_routers_packed_199k"
    # "qwen3_kan_baseline-target_lm_200k/"
    # "qwen3_kan__L7-34_nofreeze_200k/"
    # "qwen3_kan__L7-34_routers_200k/"
    # "qwen3_kir_baseline-target_lm_200k/"
    # "qwen3_kir__L7-34_nofreeze_200k"
    # "qwen3_kir__L7-34_routers_200k/"
    # "qwen3_tel_baseline-target_lm_200k/"
    # "qwen3_tel__L7-34_nofreeze_200k/"
    # "qwen3_tel__L7-34_routers_200k/"
    # "gpt_kan_baseline-target_lm_200k"
    # "gpt_kan__L4-17_nofreeze_199k"
    # "gpt_kan__L4-17_routers_200k/"
    # "gpt_kir_baseline-target_lm_200k/"
    # gpt_kir__L4-17_nofreeze_200k
    # "gpt_kir__L4-17_routers_200k/"
    # "marco_hun_baseline-target_lm_199k/"
    # "marco_hun__L7-19_nofreeze_199k/"
    # "marco_hun__L7-19_nofreeze_200k/"
    # "marco_hun__L7-19_routers_200k/" will need new one
    # "marco_sin_baseline-target_lm_200k/"
    # "marco_sin__L7-19_nofreeze_200k/"
    # "marco_sin__L7-19_routers_201k"
)

infer_nickname() {
  local base_name="$1"
  local nickname
  for nickname in \
    qwen3_30b qwen35 qwen3 \
    olmoe mixtral llama4 phimoe moonlight \
    gpt nemotron kimi llada ling phi-tiny \
    ernie granite marco marco_nano; do
    if [[ "$base_name" == "$nickname"* ]]; then
      echo "$nickname"
      return 0
    fi
  done
  return 1
}

infer_target_language() {
  local base_name="$1"
  local candidate
  local short_code
  for candidate in \
    eng_Latn kan_Knda kir_Cyrl ben_Beng tel_Telu sin_Sinh hun_Latn vie_Latn pes_Arab; do
    short_code="${candidate%%_*}"
    if [[ "$base_name" == *"$short_code"* ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
  echo "No checkpoints configured. Edit CHECKPOINTS in this script first."
  exit 0
fi

cd "$REPO_ROOT"

for checkpoint_dir in "${CHECKPOINTS[@]}"; do
  checkpoint_path="${checkpoint_dir%/}"
  if [[ "$checkpoint_path" != /* ]]; then
    checkpoint_path="$CHECKPOINTS_ROOT/$checkpoint_path"
  fi
  if [[ ! -d "$checkpoint_path" ]]; then
    echo "Skipping missing checkpoint: $checkpoint_path" >&2
    continue
  fi

  base_name="$(basename "$checkpoint_path")"
  nickname="$(infer_nickname "$base_name" || true)"
  target_language="$(infer_target_language "$base_name" || true)"

  if [[ -z "$nickname" || -z "$target_language" ]]; then
    echo "Could not infer the model nickname or target language from $checkpoint_path" >&2
    continue
  fi

  if [[ "$nickname" == "qwen3" ]]; then
    nickname="qwen3_30b"
  fi
  if [[ "$nickname" == "marco" ]]; then
    nickname="marco_nano"
  fi

  language_codes="eng_Latn,$target_language"
  echo "Processing checkpoint: $checkpoint_path"
  echo "  nickname: $nickname"
  echo "  languages: $language_codes"

  if [[ "$nickname" == "granite" || "$nickname" == "gpt" ]]; then
    conda run -n xlcl python "$SCRIPT_PATH" \
      -m "$nickname" \
      -g "$GPU_IDS" \
      -t 3 \
      --language-codes "$language_codes" \
      --trained_ckpt "$checkpoint_path"
  else
    python3 "$SCRIPT_PATH" \
      -m "$nickname" \
      -g "$GPU_IDS" \
      -t 3 \
      --language-codes "$language_codes" \
      --trained_ckpt "$checkpoint_path"
  fi

  echo
 done
