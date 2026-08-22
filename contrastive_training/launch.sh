#!/usr/bin/env bash

# Sequential training launcher.
#
# Edit the run_training calls at the bottom, then launch from tmux with:
#   bash launch.sh

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

CONDA_ENV="${CONDA_ENV:-xlcl}"
GPU_IDS="${GPU_IDS:-2,3,4,5}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-accelerate_config_4gpu.yaml}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/launch_$(date +%Y%m%d_%H%M%S).log}"

# By default, failed runs are recorded and the next run starts anyway.
# Set EXIT_NONZERO_ON_FAILURE=1 if you want launch.sh itself to return failure
# after all queued runs finish.
EXIT_NONZERO_ON_FAILURE="${EXIT_NONZERO_ON_FAILURE:-0}"

mkdir -p "$(dirname "$LOG_FILE")" || {
  echo "Failed to create log directory for: ${LOG_FILE}" >&2
  exit 1
}

touch "$LOG_FILE" || {
  echo "Failed to write log file: ${LOG_FILE}" >&2
  exit 1
}

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)" || {
    echo "Failed to locate conda base" >&2
    exit 1
  }
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh" || {
    echo "Failed to source conda.sh from: ${CONDA_BASE}" >&2
    exit 1
  }
  if ! conda activate "$CONDA_ENV"; then
    echo "Failed to activate conda env: ${CONDA_ENV}" >&2
    exit 1
  fi
else
  echo "conda is not available on PATH; cannot activate ${CONDA_ENV}" >&2
  exit 1
fi

if ! command -v accelerate >/dev/null 2>&1; then
  echo "accelerate is not available after activating ${CONDA_ENV}" >&2
  exit 1
fi

if [[ ! -f "$ACCELERATE_CONFIG" ]]; then
  echo "Accelerate config not found: ${ACCELERATE_CONFIG}" >&2
  exit 1
fi

RUN_TOTAL=0
RUN_FAILED=0

timestamp() {
  date "+%Y-%m-%d %H:%M:%S %Z"
}

log_banner() {
  local title="$1"
  local command="$2"

  {
    printf '\n'
    printf '################################################################################\n'
    printf '# %s\n' "$title"
    printf '# %s\n' "$(timestamp)"
    printf '# Command:\n'
    printf '#   %s\n' "$command"
    printf '################################################################################\n'
    printf '\n'
  } >> "$LOG_FILE"
}

show_only_progress() {
  # tqdm and HF progress bars generally write carriage-returning percentage/rate
  # lines to stderr. Everything on stderr is still logged; this filter only
  # decides what remains visible in the tmux pane.
  tr '\r' '\n' | awk '
    NF && (/[0-9]+%[[:space:]]*\|/ || /[0-9.]+[[:space:]]*it\/s/ || /[0-9.]+[[:space:]]*s\/it/ || /<.*,[[:space:]]*[0-9.]+[[:space:]]*(it\/s|s\/it)/) {
      printf "\r\033[K%s", $0
      printed = 1
      fflush()
    }
    END {
      if (printed) {
        printf "\n"
      }
    }
  '
}

run_training() {
  local label="$1"
  shift

  local -a cmd=(
    accelerate launch
    --config_file "$ACCELERATE_CONFIG"
    train.py
    "$@"
  )

  local printable_cmd="CUDA_VISIBLE_DEVICES=\"${GPU_IDS}\" ${cmd[*]}"
  RUN_TOTAL=$((RUN_TOTAL + 1))

  echo
  echo "[$(timestamp)] Starting run ${RUN_TOTAL}: ${label}"
  echo "Logging to: ${LOG_FILE}"
  log_banner "START RUN ${RUN_TOTAL}: ${label}" "$printable_cmd"

  CUDA_VISIBLE_DEVICES="$GPU_IDS" "${cmd[@]}" \
    > >(tee -a "$LOG_FILE" >/dev/null) \
    2> >(tee -a "$LOG_FILE" | show_only_progress >&2)

  local status=$?

  {
    printf '\n'
    printf '################################################################################\n'
    printf '# END RUN %s: %s\n' "$RUN_TOTAL" "$label"
    printf '# %s\n' "$(timestamp)"
    printf '# Exit status: %s\n' "$status"
    printf '################################################################################\n'
    printf '\n'
  } >> "$LOG_FILE"

  if [[ "$status" -ne 0 ]]; then
    RUN_FAILED=$((RUN_FAILED + 1))
    echo
    echo "[$(timestamp)] Run ${RUN_TOTAL} failed with status ${status}; continuing."
  else
    echo
    echo "[$(timestamp)] Run ${RUN_TOTAL} finished successfully."
  fi

  # Never abort the queue because one run failed.
  return 0
}

main() {
  echo "Activated conda env: ${CONDA_ENV}"
  echo "Using GPUs: ${GPU_IDS}"
  echo "Writing full logs to: ${LOG_FILE}"

  run_training "tha marco baseline 201k lr1e-6" \
    -l tha -m marco --baseline --no_optimizations -r 1e-6 -s 201

  run_training "tha marco contrastive 200k lr1.2e-6" \
    -l tha -m marco -a 0.1 -y 7 -x 19 --no_optimizations -r 1e-6 -s 201 --freezing_mode 2

  run_training "tha granite baseline 201k lr8e-7" \
    -l tha -m granite --baseline --no_optimizations -r 8e-7 -s 201 -b 8


  # run_training "kir gpt baseline 199k lr6e-7" \
  #   -l kir -m gpt --baseline --no_optimizations -r 6e-7 -s 199 -b 16
  
  # run_training "kir gpt contrastive 199k lr6e-7" \
  #   -l kir -m gpt -a 0.1 -y 4 -x 17 --no_optimizations -r 6e-7 -s 199 --freezing_mode 2 -b 16
  
  # run_training "sin gpt contrastive 200k lr7e-7" \
  #   -l sin -m gpt -a 0.2 -y 4 -x 17 --no_optimizations -r 7e-7 -s 200 --freezing_mode 2 -b 8

  # Duplicate/edit these for your sweep:
  #
  # run_training "tel gpt baseline 199k lr6e-7" \
  #   -l tel -m gpt --baseline --no_optimizations -r 6e-7 -s 199
  #
  # run_training "kan gpt contrastive L4-17 200k lr1e-6" \
  #   -l kan -m gpt --no_optimizations -y 4 -x 17 -r 1e-6 -s 200
  #
  # run_training "sin qwen3 nofreeze 200k lr6e-7" \
  #   -l sin -m qwen3 -f 2 -r 6e-7 -s 200

  echo
  echo "[$(timestamp)] Queue complete: ${RUN_TOTAL} run(s), ${RUN_FAILED} failure(s)."
  echo "Full log: ${LOG_FILE}"

  if [[ "$RUN_FAILED" -ne 0 && "$EXIT_NONZERO_ON_FAILURE" == "1" ]]; then
    exit 1
  fi
}

main "$@"
