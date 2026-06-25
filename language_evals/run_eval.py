from __future__ import annotations
import os

import gc
import json
import shutil
from argparse import ArgumentParser
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional
from vllm import SamplingParams
from lm_eval.api.registry import get_model

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from language_to_task import LANGUAGE_TO_TASK
from task_evaluators import TASK_EVALUATOR_REGISTRY
from export_fsdp_checkpoint import maybe_export_fsdp_checkpoint
from eval_output_paths import resolve_eval_output_naming, task_output_suffix


def prepare_model(model_name_or_path: str, adapter_path: str | None = None):
    if adapter_path:
        print(f"🧠 Loading base MoE and merging adapter {adapter_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model = model.merge_and_unload()
        
        merged_path = f"{adapter_path}_merged"
        print(f"💾 Saving merged model to {merged_path} for vLLM...")
        model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)

        del model
        del base_model
        gc.collect()
        torch.cuda.empty_cache()

        return merged_path, True
    else:
        return model_name_or_path, False


def load_training_metadata(model_path: str, original_model_path: str) -> tuple[dict | None, Path | None]:
    model_dir = Path(model_path)
    original_model_dir = Path(original_model_path)
    candidates = [
        model_dir / "training_metadata.json",
        original_model_dir / "training_metadata.json",
    ]

    if model_dir.name.endswith("_vllm"):
        candidates.append(model_dir.with_name(model_dir.name[:-5]) / "training_metadata.json")
    if model_dir.name.startswith("checkpoint-"):
        candidates.append(model_dir.parent / "training_metadata.json")

    for path in candidates:
        if path.exists():
            with path.open("r") as f:
                return json.load(f), path
    return None, None


def invert_language_to_task() -> dict[str, list[str]]:
    task_to_languages: dict[str, list[str]] = {}
    for language, task_ids in LANGUAGE_TO_TASK.items():
        for task_id in task_ids:
            task_to_languages.setdefault(task_id, []).append(language)
    return task_to_languages


def select_run_targets(language: str | None, task: str | None) -> tuple[list[str], list[str]]:
    task_to_languages = invert_language_to_task()

    if language is None and task is None:
        raise ValueError("At least one of `--language` or `--task` must be provided.")

    languages = [lang.strip() for lang in language.split(",")] if language else []

    if languages and task is not None:
        for lang in languages:
            if lang not in LANGUAGE_TO_TASK:
                raise ValueError(f"Unknown language `{lang}`.")
            if task not in LANGUAGE_TO_TASK[lang]:
                raise ValueError(f"Task `{task}` is not mapped for language `{lang}`.")
        return languages, [task]

    if languages:
        all_tasks = set()
        for lang in languages:
            if lang not in LANGUAGE_TO_TASK:
                raise ValueError(f"Unknown language `{lang}`.")
            all_tasks.update(LANGUAGE_TO_TASK[lang])
        return languages, sorted(list(all_tasks))

    if task not in task_to_languages:
        raise ValueError(f"Unknown task `{task}`.")
    return task_to_languages[task], [task]

@dataclass
class VLLMWrapper:
    lm_eval_model: Any
    lm_eval_model_args: Optional[dict]
    direct_vllm_model: Optional[Any] = None
    _sampling_params_cls: Optional[Any] = None
    model_path: str = ""
    is_qwen3_model: bool = False
    is_gpt_oss_model: bool = False

    def build_sampling_params(self, **kwargs):
        if self._sampling_params_cls:
            return self._sampling_params_cls(**kwargs)
        raise RuntimeError("SamplingParams class not available.")


def is_qwen3_model(model_path: str) -> bool:
    lowered_path = model_path.lower()
    if "qwen3" in lowered_path or "marco" in lowered_path:
        return True

    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return False

    with config_path.open() as f:
        model_config = json.load(f)
    model_type = str(model_config.get("model_type", "")).lower()
    architectures = [str(arch).lower() for arch in model_config.get("architectures", [])]
    return model_type.startswith("qwen3") or any("qwen3" in arch for arch in architectures)


def is_gpt_oss_model(model_path: str) -> bool:
    lowered_path = model_path.lower()
    if "gpt-oss" in lowered_path or "gpt_oss" in lowered_path:
        return True

    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return False

    with config_path.open() as f:
        model_config = json.load(f)
    model_type = str(model_config.get("model_type", "")).lower()
    architectures = [str(arch).lower() for arch in model_config.get("architectures", [])]
    return model_type == "gpt_oss" or any("gptoss" in arch or "gpt_oss" in arch for arch in architectures)


def build_vllm_wrapper(
    model_path: str,
    needs_direct_vllm: bool,
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
) -> VLLMWrapper:
    apply_phimoe_patch = os.environ.get("APPLY_VLLM_PHIMOE_PATCH", "1") != "0"
    if apply_phimoe_patch:
        print("Applying vLLM PhiMoE patch because APPLY_VLLM_PHIMOE_PATCH=1.")
        from vllm_phimoe_patch import apply_vllm_phimoe_patch
        apply_vllm_phimoe_patch(model_path)

    lm_eval_model_cls = get_model("vllm")

    qwen3_model = is_qwen3_model(model_path)
    gpt_oss_model = is_gpt_oss_model(model_path)
    
    vllm_kwargs = {
        "pretrained": model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": True,
        "enable_thinking": True,
        "max_model_len": max_model_len,
        # "gpu_memory_utilization": 0.6,  # Adjust as needed to prevent OOM
    }

    config_path = Path(model_path) / "config.json"
    if not apply_phimoe_patch and config_path.exists():
        with config_path.open() as f:
            model_config = json.load(f)
        if model_config.get("architectures") == ["PhimoeForCausalLM"]:
            vllm_kwargs["hf_overrides"] = {"architectures": ["PhiMoEForCausalLM"]}

    if "qwen3.5" in model_path.lower():
        vllm_kwargs["gdn_prefill_backend"] = "triton"
    if qwen3_model:
        vllm_kwargs["think_end_token"] = "</think>"
    if gpt_oss_model:
        vllm_kwargs["think_end_token"] = "<|channel|>final<|message|>"

    lm_eval_model_instance = lm_eval_model_cls(**vllm_kwargs)

    return VLLMWrapper(
        lm_eval_model=lm_eval_model_instance,
        lm_eval_model_args=None,
        direct_vllm_model=lm_eval_model_instance.model if needs_direct_vllm else None,
        _sampling_params_cls=SamplingParams,
        model_path=model_path,
        is_qwen3_model=qwen3_model,
        is_gpt_oss_model=gpt_oss_model,
    )


def evaluate_model(
    model_path: str,
    original_model_path: str,
    selected_languages: list[str],
    selected_tasks: list[str],
    run_name: str,
    cleanup_model_path: str | None,
    summary_file_name: str = "summary.json",
    tensor_parallel_size: int = 1,
    debug_single_task: bool = False,
):
    training_metadata, training_metadata_path = load_training_metadata(model_path, original_model_path)
    output_naming = resolve_eval_output_naming(
        original_model_path,
        run_name,
        model_path=model_path,
        training_metadata=training_metadata,
        requested_summary_filename=summary_file_name,
    )
    if output_naming.is_checkpoint_eval:
        if output_naming.run_name != run_name:
            print(f"Checkpoint eval results will be grouped under `{output_naming.run_name}`.")
        if output_naming.estimated_samples_k is not None:
            print(
                f"Checkpoint-{output_naming.checkpoint_step} is approximately "
                f"{output_naming.estimated_samples_k}k samples "
                f"based on latest checkpoint-{output_naming.latest_checkpoint_step}; "
                f"writing `{output_naming.summary_filename}`."
            )
        else:
            print(
                f"Could not infer sample count for checkpoint-{output_naming.checkpoint_step}; "
                f"writing `{output_naming.summary_filename}`."
            )

    run_name = output_naming.run_name
    summary_file_name = output_naming.summary_filename
    task_suffix = task_output_suffix(summary_file_name)

    max_model_len = 4096
    if "te" in selected_languages:
        max_model_len = 7000
        print(f"Using max_model_len={max_model_len} for languages: {selected_languages}")
    elif "bn" in selected_languages:
        max_model_len = 5000
        print(f"Using max_model_len={max_model_len} for languages: {selected_languages}")

    vllm_wrapper = build_vllm_wrapper(
        model_path,
        needs_direct_vllm="multiloko" in selected_tasks,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
    )

    output_dir = Path(__file__).resolve().parent / "results" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for task_id in selected_tasks:
        output_file = output_dir / f"{task_id}{task_suffix}.json"
        if output_file.exists() and not debug_single_task:
            print(f"Skipping `{task_id}`, results already exist in {output_file.name}")
            with output_file.open("r") as f:
                summary[task_id] = json.load(f)
        else:
            if output_file.exists():
                print(f"Overwriting existing `{task_id}` results in {output_file.name} for debug samples.")
            evaluator_cls = TASK_EVALUATOR_REGISTRY[task_id]
            evaluator_instance = evaluator_cls(vllm_wrapper)
            task_results = evaluator_instance.lm_eval_evaluate(
                selected_languages,
                debugging=debug_single_task,
                output_file=str(output_file),
            )
            summary[task_id] = task_results
            print(f"Completed `{task_id}` for languages: {selected_languages}")
            print(f"Temporary results written to {output_file.name}")

    if debug_single_task:
        print(f"Wrote debug samples to {output_dir} without creating {summary_file_name}.")
        if cleanup_model_path:
            print(f"Cleaning up model directory: {cleanup_model_path}")
            shutil.rmtree(cleanup_model_path, ignore_errors=True)
        return

    if training_metadata is not None:
        summary["training_metadata"] = training_metadata
        print(f"Loaded training metadata from {training_metadata_path}")
    else:
        print("No training metadata found for evaluated model.")

    summary_path = output_dir / summary_file_name
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    for task_id in selected_tasks:
        output_file = output_dir / f"{task_id}{task_suffix}.json"
        if output_file.exists():
            output_file.unlink()

    print(f"Wrote summary to {summary_path} and cleaned up per-task files.")

    if cleanup_model_path:
        print(f"Cleaning up model directory: {cleanup_model_path}")
        shutil.rmtree(cleanup_model_path, ignore_errors=True)


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Base model name or path")
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Run all mapped tasks for a language. Can be a comma-separated list of languages.",
    )
    parser.add_argument("--task", type=str, default=None, help="Run a task across all mapped languages")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--base_model", type=str, default=None, help="Original HF model ID for FSDP checkpoint export")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument(
        "--cleanup_model_path",
        type=str,
        default=None,
        help="Path to clean up after evaluation",
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--no_debug_samples", action="store_true", help="For single language+task runs, write normal summary output instead of sample-debug task JSON.",)
    args = parser.parse_args()

    selected_languages, selected_tasks = select_run_targets(args.language, args.task)
    debug_single_task = (
        args.language is not None
        and args.task is not None
        and not args.no_debug_samples
    )

    model_path, created_merge = prepare_model(args.model_name, args.adapter_path)
    model_path = maybe_export_fsdp_checkpoint(model_path, args.base_model)
    cleanup_path = model_path if created_merge else args.cleanup_model_path
    try:
        evaluate_model(
            model_path=model_path,
            original_model_path=args.model_name,
            selected_languages=selected_languages,
            selected_tasks=selected_tasks,
            run_name=args.run_name,
            cleanup_model_path=cleanup_path,
            tensor_parallel_size=args.tensor_parallel_size,
            debug_single_task=debug_single_task,
        )
    except Exception:
        if cleanup_path:
            print(f"Cleaning up model directory after failure: {cleanup_path}")
            shutil.rmtree(cleanup_path, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
