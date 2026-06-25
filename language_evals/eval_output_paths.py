from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CHECKPOINT_NAME_RE = re.compile(r"^checkpoint-(\d+)(?:_vllm)?$")
RUN_SAMPLE_COUNT_RE = re.compile(r"(?:^|[_-])(\d+)k(?:$|[_-])", re.IGNORECASE)


@dataclass(frozen=True)
class EvalOutputNaming:
    run_name: str
    summary_filename: str = "summary.json"
    checkpoint_step: int | None = None
    latest_checkpoint_step: int | None = None
    estimated_samples_k: int | None = None

    @property
    def is_checkpoint_eval(self) -> bool:
        return self.checkpoint_step is not None

    def __iter__(self):
        yield self.run_name
        yield self.summary_filename


def parse_checkpoint_step(name: str) -> int | None:
    match = CHECKPOINT_NAME_RE.match(name)
    if match is None:
        return None
    return int(match.group(1))


def find_checkpoint_ancestor(model_path: str | Path | None) -> tuple[Path, int] | None:
    if model_path is None:
        return None

    path = Path(model_path)
    for candidate in [path, *path.parents]:
        step = parse_checkpoint_step(candidate.name)
        if step is not None:
            return candidate, step
    return None


def parse_run_sample_count_k(run_name: str | None) -> int | None:
    if not run_name:
        return None

    matches = RUN_SAMPLE_COUNT_RE.findall(run_name)
    if not matches:
        return None
    return int(matches[-1])


def metadata_sample_count_k(training_metadata: dict[str, Any] | None) -> int | None:
    if not training_metadata:
        return None

    data = training_metadata.get("data", {})
    for key in ("data_limit", "train_size"):
        sample_count = data.get(key)
        if isinstance(sample_count, int) and sample_count > 0:
            return max(int(sample_count / 1000 + 0.5), 1)
    return None


def metadata_output_dir_name(training_metadata: dict[str, Any] | None) -> str | None:
    if not training_metadata:
        return None

    output_dir_name = training_metadata.get("run", {}).get("output_dir_name")
    if isinstance(output_dir_name, str) and output_dir_name:
        return output_dir_name
    return None


def latest_checkpoint_step(run_dir: Path) -> int | None:
    checkpoint_steps = []
    try:
        for child in run_dir.iterdir():
            if child.is_dir():
                step = parse_checkpoint_step(child.name)
                if step is not None:
                    checkpoint_steps.append(step)
    except OSError:
        pass

    if not checkpoint_steps:
        return None
    return max(checkpoint_steps)


def replace_checkpoint_leaf_in_run_name(
    run_name: str,
    checkpoint_leaf: str,
    checkpoint_step: int,
    parent_run_leaf: str,
) -> str:
    for leaf in (checkpoint_leaf, f"checkpoint-{checkpoint_step}"):
        prefix = f"eval-{leaf}"
        if run_name == prefix or run_name.startswith(f"{prefix}-"):
            return f"eval-{parent_run_leaf}{run_name[len(prefix):]}"
    return run_name


def estimate_checkpoint_samples_k(
    checkpoint_step: int,
    final_step: int,
    run_sample_count_k: int,
) -> int | None:
    if final_step <= 0 or run_sample_count_k <= 0:
        return None
    return max(int(checkpoint_step * run_sample_count_k / final_step + 0.5), 1)


def resolve_eval_output_naming(
    model_path_or_original: str | Path,
    original_model_path_or_run_name: str | Path,
    requested_run_name: str | None = None,
    training_metadata: dict[str, Any] | None = None,
    requested_summary_filename: str = "summary.json",
    *,
    model_path: str | Path | None = None,
) -> EvalOutputNaming:
    if requested_run_name is None:
        original_model_path = model_path_or_original
        requested_run_name = str(original_model_path_or_run_name)
        resolved_model_path = model_path
    else:
        resolved_model_path = model_path_or_original
        original_model_path = original_model_path_or_run_name

    checkpoint_context = find_checkpoint_ancestor(original_model_path)
    if checkpoint_context is None:
        checkpoint_context = find_checkpoint_ancestor(resolved_model_path)
    if checkpoint_context is None:
        return EvalOutputNaming(
            run_name=requested_run_name,
            summary_filename=requested_summary_filename,
        )

    checkpoint_dir, checkpoint_step = checkpoint_context
    run_dir = checkpoint_dir.parent
    metadata_run_name = metadata_output_dir_name(training_metadata)
    parent_run_leaf = run_dir.name
    if parse_run_sample_count_k(parent_run_leaf) is None and metadata_run_name:
        parent_run_leaf = metadata_run_name

    adjusted_run_name = replace_checkpoint_leaf_in_run_name(
        run_name=requested_run_name,
        checkpoint_leaf=checkpoint_dir.name,
        checkpoint_step=checkpoint_step,
        parent_run_leaf=parent_run_leaf,
    )

    final_step = latest_checkpoint_step(run_dir)
    run_sample_count_k = (
        parse_run_sample_count_k(parent_run_leaf)
        or parse_run_sample_count_k(metadata_run_name)
        or metadata_sample_count_k(training_metadata)
    )
    estimated_samples_k = None
    summary_filename = requested_summary_filename
    if summary_filename == "summary.json":
        summary_filename = f"summary_checkpoint-{checkpoint_step}.json"

    if final_step is not None and run_sample_count_k is not None:
        estimated_samples_k = estimate_checkpoint_samples_k(
            checkpoint_step=checkpoint_step,
            final_step=final_step,
            run_sample_count_k=run_sample_count_k,
        )
        if estimated_samples_k is not None and requested_summary_filename == "summary.json":
            summary_filename = f"summary_{estimated_samples_k}k.json"

    return EvalOutputNaming(
        run_name=adjusted_run_name,
        summary_filename=summary_filename,
        checkpoint_step=checkpoint_step,
        latest_checkpoint_step=final_step,
        estimated_samples_k=estimated_samples_k,
    )


def task_output_suffix(summary_filename: str) -> str:
    if summary_filename == "summary.json":
        return ""

    summary_stem = Path(summary_filename).stem
    if summary_stem.startswith("summary"):
        return summary_stem.removeprefix("summary")
    return f"_{summary_stem}"
