#!/usr/bin/env bash

# Sequential eval launcher.
#
# Edit the run_eval calls at the bottom, then launch from tmux with:
#   bash launch.sh

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

CONDA_ENV="${CONDA_ENV:-moevllm}"
UV_ENV_PATH="${UV_ENV_PATH:-${HOME}/.venvs/${CONDA_ENV}}"
ACTIVE_ENV_KIND=""
GPU_IDS="${GPU_IDS:-7}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/eval_launch_$(date +%Y%m%d_%H%M%S).log}"
EVAL_SCRIPT="${EVAL_SCRIPT:-./run_eval_only.sh}"

# By default, failed evals are recorded and the next eval starts anyway.
# Set EXIT_NONZERO_ON_FAILURE=1 if you want launch.sh itself to return failure
# after all queued evals finish.
EXIT_NONZERO_ON_FAILURE="${EXIT_NONZERO_ON_FAILURE:-0}"

mkdir -p "$(dirname "$LOG_FILE")" || {
  echo "Failed to create log directory for: ${LOG_FILE}" >&2
  exit 1
}

touch "$LOG_FILE" || {
  echo "Failed to write log file: ${LOG_FILE}" >&2
  exit 1
}

activate_uv_env() {
  local env_path="$UV_ENV_PATH"

  if [[ ! -f "${env_path}/bin/activate" ]]; then
    echo "uv env activation script not found: ${env_path}/bin/activate" >&2
    return 1
  fi

  # shellcheck disable=SC1091
  source "${env_path}/bin/activate" || {
    echo "Failed to activate uv env: ${env_path}" >&2
    return 1
  }

  ACTIVE_ENV_KIND="uv"
  return 0
}

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)" || {
    echo "Failed to locate conda base" >&2
    if ! activate_uv_env; then
      exit 1
    fi
  }
  if [[ "$ACTIVE_ENV_KIND" != "uv" ]]; then
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh" || {
      echo "Failed to source conda.sh from: ${CONDA_BASE}" >&2
      if ! activate_uv_env; then
        exit 1
      fi
    }
  fi
  if [[ "$ACTIVE_ENV_KIND" != "uv" ]]; then
    if conda activate "$CONDA_ENV"; then
      ACTIVE_ENV_KIND="conda"
    else
      echo "Failed to activate conda env: ${CONDA_ENV}; trying uv env: ${UV_ENV_PATH}" >&2
      if ! activate_uv_env; then
        exit 1
      fi
    fi
  fi
else
  echo "conda is not available on PATH; trying uv env: ${UV_ENV_PATH}" >&2
  if ! activate_uv_env; then
    exit 1
  fi
fi

if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "Eval script not found: ${EVAL_SCRIPT}" >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python is not available after activating ${ACTIVE_ENV_KIND} env: ${CONDA_ENV}" >&2
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

show_terminal_output() {
  # Keep normal stderr visible, but redraw tqdm/HF progress updates in place.
  tr '\r' '\n' | awk '
    function finish_progress_line() {
      if (progress_active) {
        printf "\n"
        progress_active = 0
      }
    }

    NF && (/[0-9]+%[[:space:]]*\|/ || /[0-9.]+[[:space:]]*it\/s/ || /[0-9.]+[[:space:]]*s\/it/ || /<.*,[[:space:]]*[0-9.]+[[:space:]]*(it\/s|s\/it)/) {
      printf "\r\033[K%s", $0
      progress_active = 1
      fflush()
      next
    }

    {
      finish_progress_line()
      print
      fflush()
    }

    END {
      finish_progress_line()
    }
  '
}

args_include_option() {
  local short_option="$1"
  local long_option="$2"
  shift 2

  local arg
  for arg in "$@"; do
    if [[ "$arg" == "$short_option" || "$arg" == "$long_option" || "$arg" == "${long_option}="* ]]; then
      return 0
    fi
  done

  return 1
}

run_eval() {
  local -a eval_args=("$@")
  local -a cmd=(
    bash "$EVAL_SCRIPT"
  )

  if ! args_include_option -g --gpus "${eval_args[@]}"; then
    cmd+=(-g "$GPU_IDS")
  fi

  cmd+=("${eval_args[@]}")

  local label="${eval_args[*]}"
  local printable_cmd="${cmd[*]}"
  RUN_TOTAL=$((RUN_TOTAL + 1))

  echo
  echo "[$(timestamp)] Starting eval ${RUN_TOTAL}: ${label}"
  echo "Logging to: ${LOG_FILE}"
  log_banner "START EVAL ${RUN_TOTAL}: ${label}" "$printable_cmd"

  "${cmd[@]}" \
    > >(tee -a "$LOG_FILE") \
    2> >(tee -a "$LOG_FILE" | show_terminal_output >&2)

  local status=$?

  {
    printf '\n'
    printf '################################################################################\n'
    printf '# END EVAL %s: %s\n' "$RUN_TOTAL" "$label"
    printf '# %s\n' "$(timestamp)"
    printf '# Exit status: %s\n' "$status"
    printf '################################################################################\n'
    printf '\n'
  } >> "$LOG_FILE"

  if [[ "$status" -ne 0 ]]; then
    RUN_FAILED=$((RUN_FAILED + 1))
    echo
    echo "[$(timestamp)] Eval ${RUN_TOTAL} failed with status ${status}; continuing."
  else
    echo
    echo "[$(timestamp)] Eval ${RUN_TOTAL} finished successfully."
  fi

  # Never abort the queue because one eval failed.
  return 0
}

main() {
  echo "Activated ${ACTIVE_ENV_KIND} env: ${CONDA_ENV}"
  echo "Default GPUs: ${GPU_IDS}"
  echo "Writing full logs to: ${LOG_FILE}"

  # run_eval -m /data2/lucasbandarkar/checkpoints/gpt_tel_baseline-target_lm_201k/checkpoint-5000/ -l te
  # run_eval -m /data2/lucasbandarkar/checkpoints/gpt_tel_baseline-target_lm_201k/ -l te
  # run_eval -m /data2/lucasbandarkar/checkpoints/gpt_tel__L4-17_nofreeze_201k/ -l te
  # run_eval -m /data2/lucasbandarkar/checkpoints/marco_hun__L7-19_routers_199k/ -l hu
  # run_eval -m /data2/lucasbandarkar/checkpoints/marco_hun__L7-19_routers_200k/ -l hu
  # run_eval -m /data2/lucasbandarkar/checkpoints/granite_vie_baseline-target_lm_199k/checkpoint-1250/ -l vi
  # run_eval -m AIDC-AI/Marco-Mini-Global-Base -l th
  run_eval -m /data2/lucasbandarkar/checkpoints/granite_sin_baseline-target_lm_201k/checkpoint-2500/ -l si
  run_eval -m /data2/lucasbandarkar/checkpoints/granite_sin_baseline-target_lm_201k/ -l si
  run_eval -m /data2/lucasbandarkar/checkpoints/granite_tha_baseline-target_lm_200k/checkpoint-5000/ -l th
  run_eval -m /data2/lucasbandarkar/checkpoints/granite_tha_baseline-target_lm_200k/ -l th

  # run_eval -m AIDC-AI/Marco-Mini-Global-Base -l el
  # run_eval -m ibm-granite/granite-4.0-h-tiny -l th

  # Duplicate/edit these for your eval queue:
  #
  # run_eval -m ../contrastive_training/checkpoints/my_run/checkpoint-5000 -l hu
  # run_eval -m ../contrastive_training/checkpoints/my_run/checkpoint-5000 -l hu -t flores
  #
  # You can override shared defaults from the shell:
  #   GPU_IDS=7 CONDA_ENV=moevllm bash launch.sh
  #   GPU_IDS=7 CONDA_ENV=moevllm UV_ENV_PATH=~/.venvs/moevllm bash launch.sh

  echo
  echo "[$(timestamp)] Queue complete: ${RUN_TOTAL} eval(s), ${RUN_FAILED} failure(s)."
  echo "Full log: ${LOG_FILE}"

  if [[ "$RUN_FAILED" -ne 0 && "$EXIT_NONZERO_ON_FAILURE" == "1" ]]; then
    exit 1
  fi
}

main "$@"
