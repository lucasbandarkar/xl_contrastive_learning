from argparse import ArgumentParser
import shutil
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import lm_eval

def language_to_eval_tasks(language: str):
    task_prefixes = [
        "belebele_",
        "global_mmlu_full_",
        "mmlu_prox_",
        "mgsm_direct_"
    ]
    language_to_eval_tasks_map = {
        "en": ["hellaswag", "gsm8k", "truthfulqa", "mgsm_direct_en", "belebele_eng_Latn"],
        "pes": ["belebele_pes_Arab"],
        "fr": ["mgsm_direct_fr", "belebele_fra_Latn"],
        "ar": ["global_mmlu_full_ar_medical", "mmlu_prox_ar_biology"],
        "bn": ["mgsm_direct_bn"],
    }
    return language_to_eval_tasks_map.get(language, [])

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

def evaluate_model(model_path: str, language: str, run_name: str, cleanup_model_path: str | None):
    eval_tasks = language_to_eval_tasks(language)
    
    if not eval_tasks:
        print(f"⚠️ No evaluation tasks defined for language: {language}. Skipping evaluation.")
        return

    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args={
            "pretrained": model_path,
            "dtype": "auto",
            "gpu_memory_utilization": 0.8,
            "tensor_parallel_size": max(1, torch.cuda.device_count()),
        },
        tasks=eval_tasks,
        batch_size="auto",
        log_samples=True
    )

    if results is not None:
        print(f"✅ Evaluation complete for language: {language}.")
        print(lm_eval.utils.make_table(results))
    else:
        print(f"❌ Evaluation for language '{language}' returned no results.")

    if cleanup_model_path:
        print(f"🧹 Cleaning up model directory: {cleanup_model_path}")
        shutil.rmtree(cleanup_model_path, ignore_errors=True)

def main():
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help="Base model name or path")
    parser.add_argument('--language', type=str, required=True)
    parser.add_argument('--adapter_path', type=str, default=None)
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--cleanup_model_path', type=str, default=None, help="Path to clean up after evaluation")
    args = parser.parse_args()

    model_path, created_merge = prepare_model(args.model_name, args.adapter_path)
    cleanup_path = model_path if created_merge else args.cleanup_model_path
    evaluate_model(model_path, args.language, args.run_name, cleanup_path)

if __name__ == "__main__":
    main()