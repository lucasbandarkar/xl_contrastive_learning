#!/usr/bin/env bash
set -euo pipefail

# Rerun MMLU-ProX for Vietnamese and Hungarian granite runs and replace only the
# existing "mmlu_prox" entry in each final summary.json.
#
# Run from language_evals/ as:
#   z_deprecated/rerun_granite_vi_mmlu_prox_once.sh
#
# Optional env:
#   GPU_ID=0
#   CHECKPOINT_ROOT=/data2/lucasbandarkar/checkpoints
#   DRY_RUN=1
#   STRICT_MISSING_CHECKPOINT=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANGUAGE_EVALS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$LANGUAGE_EVALS_DIR/results"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data2/lucasbandarkar/checkpoints}"
GPU_ID="${GPU_ID:-2}"
DRY_RUN="${DRY_RUN:-0}"
STRICT_MISSING_CHECKPOINT="${STRICT_MISSING_CHECKPOINT:-0}"

export VLLM_USE_V1=0
export HF_DATASETS_TRUST_REMOTE_CODE=1

if [[ -d /data2 ]]; then
  DEFAULT_EVAL_RUNNER="python"
  DEFAULT_APPLY_VLLM_PHIMOE_PATCH="0"
else
  DEFAULT_EVAL_RUNNER="uv"
  DEFAULT_APPLY_VLLM_PHIMOE_PATCH="1"
fi

EVAL_RUNNER="${EVAL_RUNNER:-$DEFAULT_EVAL_RUNNER}"
export APPLY_VLLM_PHIMOE_PATCH="${APPLY_VLLM_PHIMOE_PATCH:-$DEFAULT_APPLY_VLLM_PHIMOE_PATCH}"

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

case "$EVAL_RUNNER" in
  uv)
    python_cmd=(uv run python)
    ;;
  python)
    python_cmd=(python)
    ;;
  *)
    echo "Unknown EVAL_RUNNER: $EVAL_RUNNER. Expected 'uv' or 'python'." >&2
    exit 1
    ;;
esac

GPU_COUNT="$(count_gpus "$GPU_ID")"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

cleanup_on_error() {
  local exit_code=$?
  echo "Failed on line $1 with exit code $exit_code." >&2
  exit "$exit_code"
}
trap 'cleanup_on_error $LINENO' ERR

discover_targets() {
  python - "$RESULTS_DIR" "$CHECKPOINT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

results_dir = Path(sys.argv[1])
checkpoint_root = Path(sys.argv[2])

hf_models = {
    "granite-4.0-h-tiny": "ibm-granite/granite-4.0-h-tiny",
}

target_langs = {"vi", "hu"}

for summary_path in sorted(results_dir.glob("eval-granite*/summary.json")):
    try:
        summary = json.loads(summary_path.read_text())
    except Exception as exc:
        print(f"BAD_JSON\t{summary_path}\t{exc}", file=sys.stderr)
        continue

    if "mmlu_prox" not in summary:
        continue

    result_dir = summary_path.parent.name
    if not result_dir.startswith("eval-"):
        print(f"UNPARSEABLE\t{summary_path}\tmissing eval- prefix", file=sys.stderr)
        continue

    stem = result_dir.removeprefix("eval-")
    model_leaf = None
    lang = None
    for candidate in sorted(target_langs, key=len, reverse=True):
        suffix = f"-{candidate}"
        if stem.endswith(suffix):
            model_leaf = stem[:-len(suffix)]
            lang = candidate
            break

    if model_leaf is None or lang is None:
        continue

    if model_leaf in hf_models:
        model = hf_models[model_leaf]
        status = "available"
    else:
        checkpoint = checkpoint_root / model_leaf
        model = str(checkpoint)
        status = "available" if checkpoint.is_dir() else "missing_checkpoint"

    print("\t".join([str(summary_path), model_leaf, lang, model, status]))
PY
}

merge_mmlu_prox_summary() {
  local original_summary="$1"
  local task_summary="$2"

  python - "$original_summary" "$task_summary" <<'PY'
import json
import sys
from pathlib import Path

original_path = Path(sys.argv[1])
task_path = Path(sys.argv[2])

original = json.loads(original_path.read_text())
task_only = json.loads(task_path.read_text())

if "mmlu_prox" not in task_only:
    raise SystemExit(f"{task_path} does not contain an 'mmlu_prox' key")

old_value = original.get("mmlu_prox")
original["mmlu_prox"] = task_only["mmlu_prox"]

with original_path.open("w") as f:
    json.dump(original, f, indent=4, ensure_ascii=False)
    f.write("\n")

print(f"Updated {original_path}")
print(f"  old mmlu_prox: {old_value}")
print(f"  new mmlu_prox: {original['mmlu_prox']}")
PY
}

cd "$LANGUAGE_EVALS_DIR"

while IFS=$'\t' read -r summary_path model_leaf lang model status; do
  [[ -n "${summary_path:-}" ]] || continue

  tmp_run_name="eval-${model_leaf}-${lang}-mmlu-prox-refresh"
  tmp_dir="$RESULTS_DIR/$tmp_run_name"
  tmp_summary="$tmp_dir/summary.json"

  echo
  echo "============================================================"
  echo "Summary: $summary_path"
  echo "Model:   $model"
  echo "Lang:    $lang"
  echo "Task:    mmlu_prox"
  echo "Status:  $status"
  echo "============================================================"

  if [[ "$status" == "missing_checkpoint" ]]; then
    if [[ "$STRICT_MISSING_CHECKPOINT" == "1" ]]; then
      echo "Checkpoint directory does not exist: $model" >&2
      exit 1
    fi
    echo "Skipping missing checkpoint: $model"
    continue
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN=1 would rerun MMLU-ProX and merge into $summary_path"
    continue
  fi

  rm -rf "$tmp_dir"

  "${python_cmd[@]}" "$LANGUAGE_EVALS_DIR/run_eval.py" \
    --model_name "$model" \
    --language "$lang" \
    --task mmlu_prox \
    --run_name "$tmp_run_name" \
    --tensor_parallel_size "$GPU_COUNT" \
    --no_debug_samples

  if [[ ! -f "$tmp_summary" ]]; then
    echo "Missing temporary MMLU-ProX summary: $tmp_summary" >&2
    exit 1
  fi

  merge_mmlu_prox_summary "$summary_path" "$tmp_summary"
  rm -rf "$tmp_dir"
  echo "Deleted temporary folder: $tmp_dir"
done < <(discover_targets)

echo
echo "Granite Vietnamese/Hungarian MMLU-ProX refresh complete."
