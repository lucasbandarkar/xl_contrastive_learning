#!/usr/bin/env bash
set -euo pipefail

# Historical name, revived for the Global-MGSM prompt/stop-sequence fix.
# Run from language_evals/ as: z_deprecated/rerun_flores_once.sh
# Scans results/*/summary.json, reruns only MGSM, and
# replaces just the existing "mgsm" entry. Files with suffixes such as
# summary_at240k.json are intentionally ignored.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANGUAGE_EVALS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$LANGUAGE_EVALS_DIR/results"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data2/lucasbandarkar/checkpoints}"
GPU_ID="${GPU_ID:-0}"
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

sys.path.insert(0, str(results_dir.parent))
from language_to_task import LANGUAGE_TO_TASK  # noqa: E402

hf_models = {
    "ERNIE-4.5-21B-A3B-PT": "baidu/ERNIE-4.5-21B-A3B-PT",
    "Ling-mini-2.0": "inclusionAI/Ling-mini-2.0",
    "Phi-mini-MoE-instruct": "microsoft/Phi-mini-MoE-instruct",
    "Phi-tiny-MoE-instruct": "microsoft/Phi-tiny-MoE-instruct",
    "Qwen3.5-35B-A3B": "Qwen/Qwen3.5-35B-A3B",
    "granite-4.0-h-tiny": "ibm-granite/granite-4.0-h-tiny",
}

langcodes = sorted(LANGUAGE_TO_TASK, key=len, reverse=True)

for summary_path in sorted(results_dir.glob("*/summary.json")):
    try:
        summary = json.loads(summary_path.read_text())
    except Exception as exc:
        print(f"BAD_JSON\t{summary_path}\t{exc}", file=sys.stderr)
        continue

    if "mgsm" not in summary:
        continue

    result_dir = summary_path.parent.name
    if not result_dir.startswith("eval-"):
        print(f"UNPARSEABLE\t{summary_path}\tmissing eval- prefix", file=sys.stderr)
        continue

    stem = result_dir.removeprefix("eval-")
    model_leaf = None
    lang = None
    for candidate in langcodes:
        suffix = f"-{candidate}"
        if stem.endswith(suffix):
            model_leaf = stem[:-len(suffix)]
            lang = candidate
            break

    if model_leaf is None or lang is None:
        print(f"UNPARSEABLE\t{summary_path}\tcould not infer language", file=sys.stderr)
        continue

    if model_leaf in hf_models:
        model = hf_models[model_leaf]
        status = "available"
    else:
        checkpoint = checkpoint_root / model_leaf
        if checkpoint.is_dir():
            model = str(checkpoint)
            status = "available"
        else:
            model = str(checkpoint)
            status = "missing_checkpoint"

    print("\t".join([str(summary_path), model_leaf, lang, model, status]))
PY
}

merge_mgsm_summary() {
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

if "mgsm" not in task_only:
    raise SystemExit(f"{task_path} does not contain an 'mgsm' key")

old_value = original.get("mgsm")
original["mgsm"] = task_only["mgsm"]

with original_path.open("w") as f:
    json.dump(original, f, indent=4, ensure_ascii=False)
    f.write("\n")

print(f"Updated {original_path}")
print(f"  old mgsm: {old_value}")
print(f"  new mgsm: {original['mgsm']}")
PY
}

delete_mgsm_value() {
  local original_summary="$1"
  local missing_model="$2"

  python - "$original_summary" "$missing_model" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
missing_model = sys.argv[2]
summary = json.loads(summary_path.read_text())
old_value = summary.pop("mgsm", None)

with summary_path.open("w") as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)
    f.write("\n")

print(f"Deleted mgsm from {summary_path}")
print(f"  missing checkpoint: {missing_model}")
print(f"  old mgsm: {old_value}")
PY
}

cd "$LANGUAGE_EVALS_DIR"

while IFS=$'\t' read -r summary_path model_leaf lang model status; do
  [[ -n "${summary_path:-}" ]] || continue

  tmp_run_name="eval-${model_leaf}-${lang}-mgsm-refresh"
  tmp_dir="$RESULTS_DIR/$tmp_run_name"
  tmp_summary="$tmp_dir/summary.json"

  echo
  echo "============================================================"
  echo "Summary: $summary_path"
  echo "Model:   $model"
  echo "Lang:    $lang"
  echo "Status:  $status"
  echo "============================================================"

  if [[ "$status" == "missing_checkpoint" ]]; then
    if [[ "$STRICT_MISSING_CHECKPOINT" == "1" ]]; then
      echo "Checkpoint directory does not exist: $model" >&2
      exit 1
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "DRY_RUN=1 would delete mgsm from $summary_path"
    else
      delete_mgsm_value "$summary_path" "$model"
    fi
    continue
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN=1 would rerun MGSM and merge into $summary_path"
    continue
  fi

  rm -rf "$tmp_dir"

  "${python_cmd[@]}" "$LANGUAGE_EVALS_DIR/run_eval.py" \
    --model_name "$model" \
    --language "$lang" \
    --task mgsm \
    --run_name "$tmp_run_name" \
    --tensor_parallel_size "$GPU_COUNT" \
    --no_debug_samples

  if [[ ! -f "$tmp_summary" ]]; then
    echo "Missing temporary MGSM summary: $tmp_summary" >&2
    exit 1
  fi

  merge_mgsm_summary "$summary_path" "$tmp_summary"
  rm -rf "$tmp_dir"
  echo "Deleted temporary folder: $tmp_dir"
done < <(discover_targets)

echo
echo "MGSM refresh complete."
