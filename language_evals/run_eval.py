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

    if language is not None and task is not None:
        if language not in LANGUAGE_TO_TASK:
            raise ValueError(f"Unknown language `{language}`.")
        if task not in LANGUAGE_TO_TASK[language]:
            raise ValueError(f"Task `{task}` is not mapped for language `{language}`.")
        return [language], [task]

    if language is not None:
        if language not in LANGUAGE_TO_TASK:
            raise ValueError(f"Unknown language `{language}`.")
        return [language], LANGUAGE_TO_TASK[language]

    if task not in task_to_languages:
        raise ValueError(f"Unknown task `{task}`.")
    return task_to_languages[task], [task]

@dataclass
class VLLMWrapper:
    lm_eval_model: Any
    lm_eval_model_args: Optional[dict]
    direct_vllm_model: Optional[Any] = None
    _sampling_params_cls: Optional[Any] = None

    def build_sampling_params(self, **kwargs):
        if self._sampling_params_cls:
            return self._sampling_params_cls(**kwargs)
        raise RuntimeError("SamplingParams class not available.")

def build_vllm_wrapper(model_path: str, needs_direct_vllm: bool, tensor_parallel_size: int = 1) -> VLLMWrapper:
    
    # Apply a vLLM patch (fixes MoE head_dim mismatch)
    from vllm_phimoe_patch import apply_vllm_phimoe_patch
    if "phi" in model_path.lower():
        apply_vllm_phimoe_patch(model_path)

    lm_eval_model_cls = get_model("vllm")
    
    vllm_kwargs = {
        "pretrained": model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": True,
        "enable_thinking": True,
    }

    if "qwen3.5" in model_path.lower():
        vllm_kwargs["gdn_prefill_backend"] = "triton"
        vllm_kwargs["think_end_token"] = "</think>"

    lm_eval_model_instance = lm_eval_model_cls(**vllm_kwargs)

    return VLLMWrapper(
        lm_eval_model=lm_eval_model_instance,
        lm_eval_model_args=None,
        direct_vllm_model=lm_eval_model_instance.model if needs_direct_vllm else None,
        _sampling_params_cls=SamplingParams,
    )


def evaluate_model(
    model_path: str,
    language: str | None,
    task: str | None,
    run_name: str,
    cleanup_model_path: str | None,
    tensor_parallel_size: int = 1,
):
    selected_languages, selected_tasks = select_run_targets(language, task)

    vllm_wrapper = build_vllm_wrapper(
        model_path,
        needs_direct_vllm="multiloko" in selected_tasks,
        tensor_parallel_size=tensor_parallel_size,
    )

    output_dir = Path(__file__).resolve().parent / "results" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for task_id in selected_tasks:
        output_file = output_dir / f"{task_id}.json"
        if output_file.exists():
            print(f"Skipping `{task_id}`, results already exist in {output_file.name}")
            with output_file.open("r") as f:
                summary[task_id] = json.load(f)
        else:
            evaluator_cls = TASK_EVALUATOR_REGISTRY[task_id]
            evaluator_instance = evaluator_cls(vllm_wrapper)
            task_results = evaluator_instance.lm_eval_evaluate(
                selected_languages,
                output_file=str(output_file),
            )
            summary[task_id] = task_results
            print(f"Completed `{task_id}` for languages: {selected_languages}")
            print(f"Temporary results written to {output_file.name}")

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    for task_id in selected_tasks:
        output_file = output_dir / f"{task_id}.json"
        if output_file.exists():
            output_file.unlink()

    print(f"Wrote summary to {output_dir} and cleaned up per-task files.")

    if cleanup_model_path:
        print(f"Cleaning up model directory: {cleanup_model_path}")
        shutil.rmtree(cleanup_model_path, ignore_errors=True)


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Base model name or path")
    parser.add_argument("--language", type=str, default=None, help="Run all mapped tasks for a language")
    parser.add_argument("--task", type=str, default=None, help="Run a task across all mapped languages")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument(
        "--cleanup_model_path",
        type=str,
        default=None,
        help="Path to clean up after evaluation",
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    selected_languages, selected_tasks = select_run_targets(args.language, args.task)

    model_path, created_merge = prepare_model(args.model_name, args.adapter_path)
    cleanup_path = model_path if created_merge else args.cleanup_model_path
    try:
        evaluate_model(
            model_path=model_path,
            language=args.language,
            task=args.task,
            run_name=args.run_name,
            cleanup_model_path=cleanup_path,
            tensor_parallel_size=args.tensor_parallel_size,
        )
    except Exception:
        if cleanup_path:
            print(f"Cleaning up model directory after failure: {cleanup_path}")
            shutil.rmtree(cleanup_path, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
