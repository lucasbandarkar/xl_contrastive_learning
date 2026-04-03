from argparse import ArgumentParser
import shutil
import gc
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from llamafactory.train.tuner import run_exp

def train_model(model_name_or_path, dataset_name, ram_disk_output, wandb_project, run_name):
    train_args = {
        "stage": "sft",
        "do_train": True,
        "model_name_or_path": model_name_or_path,
        "dataset": dataset_name,
        "template": "default",
        "finetuning_type": "lora",
        "lora_target": "all",
        "lora_rank": 64,
        "output_dir": ram_disk_output,
        "overwrite_output_dir": True,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "fsdp": "full_shard auto_wrap",
        "lr_scheduler_type": "cosine",
        "logging_steps": 10,
        "save_strategy": "no",
        "learning_rate": 5e-5,
        "num_train_epochs": 1.0,
        "fp16": True,
        "report_to": "wandb",
        "wandb_project": wandb_project,
        "wandb_run_name": run_name,
    }

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if local_rank == 0:
        print(f"🚀 Starting LlamaFactory SFT on {model_name_or_path}...")
        
    run_exp(train_args)
    
    if local_rank == 0:
        print("✅ Training complete. LoRA adapter saved to RAM disk.")
        print("🧠 Loading base MoE and merging adapter in memory...")
        
        # Clear distributed environment variables so device_map="auto" can use all GPUs for merging
        for key in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_PORT", "MASTER_ADDR"]:
            os.environ.pop(key, None)

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, ram_disk_output)
        model = model.merge_and_unload() 

        merged_path = f"{ram_disk_output}_merged"
        print(f"💾 Saving merged model to {merged_path} for vLLM...")
        model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)

        print(f"🧹 Cleaning up unmerged RAM disk adapter: {ram_disk_output}")
        shutil.rmtree(ram_disk_output, ignore_errors=True)

def main():
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--ram_disk_output', type=str, required=True)
    parser.add_argument('--wandb_project', type=str, default="moe-sft-eval")
    parser.add_argument('--run_name', type=str, required=True)
    args = parser.parse_args()
    train_model(args.model_name, args.dataset_name, args.ram_disk_output, args.wandb_project, args.run_name)

if __name__ == "__main__":
    main()