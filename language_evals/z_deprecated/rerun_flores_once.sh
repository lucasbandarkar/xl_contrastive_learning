#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
CHECKPOINT_ROOT="/data2/lucasbandarkar/checkpoints"
GPU_ID="${GPU_ID:-1}"
LANGUAGE="${LANGUAGE:-id}"
TASK="${TASK:-flores}"
STRICT_MISSING_ORIGINAL="${STRICT_MISSING_ORIGINAL:-0}"
STRICT_MISSING_CHECKPOINT="${STRICT_MISSING_CHECKPOINT:-0}"
SITE_CUSTOMIZE_DIR="$(mktemp -d)"

if [[ "${CONDA_DEFAULT_ENV:-}" != "moevllm" && "${ALLOW_NON_MOEVLLM:-0}" != "1" ]]; then
  echo "Expected to be in the moevllm conda environment." >&2
  echo "Run: conda activate moevllm" >&2
  echo "Or set ALLOW_NON_MOEVLLM=1 to bypass this check." >&2
  exit 1
fi

cleanup_on_error() {
  local exit_code=$?
  echo "Failed on line $1 with exit code $exit_code." >&2
  exit "$exit_code"
}
cleanup_on_exit() {
  rm -rf "$SITE_CUSTOMIZE_DIR"
}
trap 'cleanup_on_error $LINENO' ERR
trap cleanup_on_exit EXIT

if [[ "$TASK" == "flores" ]]; then
  cat > "$SITE_CUSTOMIZE_DIR/sitecustomize.py" <<'PY'
import os
import pathlib
import tempfile

_OriginalTemporaryDirectory = tempfile.TemporaryDirectory
_task_utils_dir = os.environ.get("FLORES_TASK_UTILS_DIR")


class _FloresAwareTemporaryDirectory(_OriginalTemporaryDirectory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        path = pathlib.Path(self.name)
        if path.name.startswith("flores_tasks_") and _task_utils_dir:
            link = path / "task_utils"
            target = pathlib.Path(_task_utils_dir)
            if not link.exists():
                link.symlink_to(target, target_is_directory=True)


tempfile.TemporaryDirectory = _FloresAwareTemporaryDirectory
PY

  export FLORES_TASK_UTILS_DIR="$SCRIPT_DIR/task_utils"
  export PYTHONPATH="$SITE_CUSTOMIZE_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

models=(
  "$CHECKPOINT_ROOT/ernie_pes__L6-21_routers_10k"
  "$CHECKPOINT_ROOT/ernie_pes_baseline-target_lm_5k"
  "$CHECKPOINT_ROOT/ernie_pes_baseline-translation_sft_10k"
  "$CHECKPOINT_ROOT/granite_ind_baseline-target_lm_10k"
  "$CHECKPOINT_ROOT/granite_pes__L10-35_30k"
  "$CHECKPOINT_ROOT/granite_pes__L10-35_routers_60k"
  "$CHECKPOINT_ROOT/granite_pes__L14-14_20k"
  "$CHECKPOINT_ROOT/granite_pes_baseline-target_lm_10k"
  "$CHECKPOINT_ROOT/granite_pes_baseline-target_lm_20k"
  "$CHECKPOINT_ROOT/granite_pes_baseline-target_lm_6k"
  "$CHECKPOINT_ROOT/granite_pes_baseline-translation_sft_10k"
  "baidu/ERNIE-4.5-21B-A3B-PT"
  "microsoft/Phi-mini-MoE-instruct"
  "inclusionAI/Ling-mini-2.0"
  "microsoft/Phi-tiny-MoE-instruct"
  "ibm-granite/granite-4.0-h-tiny"
)

merge_task_summary() {
  local model_address="$1"
  local model_name="${model_address%/}"
  local model_leaf="${model_name##*/}"
  local original_dir="$RESULTS_DIR/eval-${model_leaf}-${LANGUAGE}"
  local task_dir="$RESULTS_DIR/eval-${model_leaf}-${LANGUAGE}-${TASK}"
  local original_summary="$original_dir/summary.json"
  local task_summary="$task_dir/summary.json"

  if [[ ! -f "$task_summary" ]]; then
    echo "Missing temporary ${TASK} summary: $task_summary" >&2
    return 1
  fi

  if [[ ! -f "$original_summary" ]]; then
    echo "Missing original summary to update: $original_summary" >&2
    return 1
  fi

  python - "$original_summary" "$task_summary" "$TASK" <<'PY'
import json
import sys
from pathlib import Path

original_path = Path(sys.argv[1])
task_path = Path(sys.argv[2])
task_name = sys.argv[3]

with original_path.open() as f:
    original = json.load(f)

with task_path.open() as f:
    task_only = json.load(f)

if task_name not in task_only:
    raise SystemExit(f"{task_path} does not contain a {task_name!r} key")

old_value = original.get(task_name)
original[task_name] = task_only[task_name]

with original_path.open("w") as f:
    json.dump(original, f, indent=4, ensure_ascii=False)
    f.write("\n")

print(f"Updated {original_path}")
print(f"  old {task_name}: {old_value}")
print(f"  new {task_name}: {original[task_name]}")
PY

  rm -rf "$task_dir"
  echo "Deleted temporary folder: $task_dir"
}

cd "$REPO_ROOT"

for model in "${models[@]}"; do
  model_name="${model%/}"
  model_leaf="${model_name##*/}"
  original_summary="$RESULTS_DIR/eval-${model_leaf}-${LANGUAGE}/summary.json"

  echo
  echo "============================================================"
  echo "Rerunning ${TASK} for: $model"
  echo "Expected original summary: $original_summary"
  echo "============================================================"

  if [[ "$model" == "$CHECKPOINT_ROOT/"* && ! -d "$model" ]]; then
    echo "Checkpoint directory does not exist: $model" >&2
    if [[ "$STRICT_MISSING_CHECKPOINT" == "1" ]]; then
      exit 1
    fi
    echo "Skipping $model"
    continue
  fi

  if [[ ! -f "$original_summary" ]]; then
    echo "Original summary is missing, so there is nothing to update: $original_summary" >&2
    if [[ "$STRICT_MISSING_ORIGINAL" == "1" ]]; then
      exit 1
    fi
    echo "Skipping $model"
    continue
  fi

  "$SCRIPT_DIR/run_eval_only.sh" -m "$model" -l "$LANGUAGE" -t "$TASK" -g "$GPU_ID"
  merge_task_summary "$model"
done

echo
echo "All ${TASK} reruns merged successfully."
