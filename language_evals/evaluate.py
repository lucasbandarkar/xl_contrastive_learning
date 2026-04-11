from __future__ import annotations

import gc
import shutil
from argparse import ArgumentParser
from pathlib import Path

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

    if language is not None:
        if language not in LANGUAGE_TO_TASK:
            raise ValueError(f"Unknown language `{language}`.")
        task_ids = LANGUAGE_TO_TASK[language]
        return [language], task_ids

    if task is None:
        raise ValueError("Exactly one of `--language` or `--task` must be provided.")
    if task not in task_to_languages:
        raise ValueError(f"Unknown task `{task}`.")
    return task_to_languages[task], [task]

def build_model_config(model_path: str, needs_direct_vllm: bool) -> EvalModelConfig:
    direct_vllm_model = None
    sampling_params_cls = None
    if needs_direct_vllm:
        from vllm import LLM, SamplingParams

        direct_vllm_model = LLM(
            model=model_path,
            trust_remote_code=True,
        )
        sampling_params_cls = SamplingParams

    return EvalModelConfig(
        lm_eval_model="vllm",
        lm_eval_model_args={
            "pretrained": model_path,
            "enable_thinking": True,
            "think_end_token": "</think>",
        },
        direct_vllm_model=direct_vllm_model,
        _sampling_params_cls=sampling_params_cls,
    )


def evaluate_model(
    model_path: str,
    language: str | None,
    task: str | None,
    run_name: str,
    cleanup_model_path: str | None,
):
    selected_languages, selected_tasks = select_run_targets(language, task)

    model_config = build_model_config(
        model_path,
        needs_direct_vllm="multiloko" in selected_tasks,
    )

    output_dir = Path(__file__).resolve().parent / "results" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for task_id in selected_tasks:
        evaluator_cls = TASK_EVALUATOR_REGISTRY[task_id]
        evaluator_instance = evaluator_cls(model_config)
        output_file = output_dir / f"{task_id}.json"
        task_results = evaluator_instance.lm_eval_evaluate(
            selected_languages,
            output_file=str(output_file),
        )
        summary[task_id] = task_results
        print(f"Completed `{task_id}` for languages: {selected_languages}")

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as f:
        import json

        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"Wrote per-task results and summary to {output_dir}")

    if cleanup_model_path:
        print(f"Cleaning up model directory: {cleanup_model_path}")
        shutil.rmtree(cleanup_model_path, ignore_errors=True)


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Base model name or path")
    selection_group = parser.add_mutually_exclusive_group(required=True)
    selection_group.add_argument("--language", type=str, help="Run all mapped tasks for a language")
    selection_group.add_argument("--task", type=str, help="Run a task across all mapped languages")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument(
        "--cleanup_model_path",
        type=str,
        default=None,
        help="Path to clean up after evaluation",
    )
    args = parser.parse_args()

    selected_languages, selected_tasks = select_run_targets(args.language, args.task)
    preflight_checks(selected_languages, selected_tasks)

    model_path, created_merge = prepare_model(args.model_name, args.adapter_path)
    cleanup_path = model_path if created_merge else args.cleanup_model_path
    try:
        evaluate_model(
            model_path=model_path,
            language=args.language,
            task=args.task,
            run_name=args.run_name,
            cleanup_model_path=cleanup_path,
        )
    except Exception:
        if cleanup_path:
            print(f"Cleaning up model directory after failure: {cleanup_path}")
            shutil.rmtree(cleanup_path, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
