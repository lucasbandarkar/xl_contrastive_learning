from argparse import ArgumentParser
from contrastive_trainer import ContrastiveTrainer, ContrastiveLMTrainer
from trl import SFTTrainer
import torch
from modeling import load_models
from parallel_dataset import load_parallel_datasets, ParallelDataCollator, ConcatenatedDataCollator
from transformers import TrainingArguments
import json
import shutil
from pathlib import Path
import huggingface_hub

def create_training_args(
        model_nickname,
        learning_rate=1e-6,
        lr_scheduler='constant_with_warmup',
        partial_training=False,
        batch_size=None,
        test_run=False,
        fsdp=False,
    ) -> TrainingArguments:
    if not batch_size:
        # ideally, auto_find_batch_size would prevent us from having to use this hacky approach
        batch_size = get_model_config(model_nickname)['batch_sizes']['partial' if partial_training else 'full']

    grad_accum_steps, warmup_steps = calc_grad_accum_steps(batch_size, 32)
    kwargs = {}
    if fsdp:
        kwargs['gradient_checkpointing'] = True

    return TrainingArguments(
        output_dir=f"./checkpoints/{create_output_directory_name()}",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # gradient_accumulation_steps= grad_accum_steps,
        auto_find_batch_size=True, # pretty complicated to have this work with fixed effective batch size, to figure out later
        use_cache=False,
        optim='adafactor',
        num_train_epochs=1,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler,
        warmup_steps=warmup_steps,
        neftune_noise_alpha=5.0,
        # max_grad_norm=0.0, # FSDP2 requires disabling to 0.0 to prevent "RuntimeError: No backend type associated with device type cpu"
        max_grad_norm=1.0, # ok if using FSDP1
        adam_beta2=0.99, # default value is 0.999
        save_strategy="no" if test_run else "steps",
        save_steps=1000 if not test_run else None,
        bf16=False, # disable to maintain Pure BF16 (avoids FP32 master weight upcast bug)
        eval_strategy='steps',
        # eval_steps=1/25, # eval 20 times through training
        save_total_limit=1 if not test_run else None,
        logging_steps=200,
        remove_unused_columns=False, # essential for our ParallelDataCollator
        log_on_each_node=False,
        # report_to=report
        **kwargs
    )

def get_model_config(model_nickname):
    """ fetches config file for model """
    with open("training_configs.json", "r") as file:
        model_config = json.load(file)[model_nickname]
    return model_config

def calc_grad_accum_steps(device_batch_size, effective_batch_size):
    """
    In order to maintain sufficiently large effective batch size, calculates number of gradient accumulation steps with current training conditions.
    """
    world_size = 1
    grad_accum_steps=1
    while grad_accum_steps * world_size * device_batch_size < effective_batch_size:
        grad_accum_steps *= 2
        
    num_warmup_steps = int(24000 / effective_batch_size) # warmup lasts 24k samples, regardless of what batch size is
    return grad_accum_steps, num_warmup_steps

def create_output_directory_name():
    return "moe_contrastive_training_test"

def main(args):
    multi_gpu = False
    if torch.cuda.device_count() > 1:
        multi_gpu = True
    
    # Layers are one-indexed in the CLI, while the trainer/model use the same convention.
    # Partial model loading only needs the upper bound of the requested range.
    max_layer = int(args.max_layer) if args.earlyexit else None
    model, tokenizer = load_models(args.nickname, max_layer=max_layer, fsdp=multi_gpu)

    data_limit = 1000 if args.test_run else None
    dataset_train, dataset_valid, key_src, key_tgt = load_parallel_datasets("opus", args.language, data_limit=data_limit)

    Collator = ParallelDataCollator ## ConcatenatedDataCollator if args.baseline else ParallelDataCollator
    custom_data_collator = Collator(tokenizer, key_src, key_tgt)


    if args.earlyexit:
        training_args = create_training_args(args.nickname, partial_training=True, test_run=args.test_run)
        trainer = ContrastiveTrainer(
            model,
            args=training_args,
            train_dataset=dataset_train,
            eval_dataset=dataset_valid,
            data_collator=custom_data_collator,
            min_layer=args.min_layer,
            max_layer=args.max_layer,
        )
    elif args.baseline:
        training_args = create_training_args(args.nickname, test_run=args.test_run, fsdp=multi_gpu)
        trainer = SFTTrainer(
            model,
            args=training_args,
            train_dataset=dataset_train,
            eval_dataset=dataset_valid,
        )
    else:
        training_args = create_training_args(args.nickname, test_run=args.test_run, fsdp=multi_gpu)
        trainer = ContrastiveLMTrainer(
            model,
            args=training_args,
            train_dataset=dataset_train,
            eval_dataset=dataset_valid,
            data_collator=custom_data_collator,
            min_layer=args.min_layer,
            max_layer=args.max_layer,
        )
    if not args.baseline:
        if args.routers_only:
            trainer.configure_router_only_training(args.max_layer, multi_gpu)
        else:
            trainer.configure_early_layer_only_training(args.max_layer, multi_gpu)
    
    trainer.train()

    output_dir = trainer.args.output_dir
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Copy custom model code files (e.g. configuration_slimmoe.py, modeling_slimmoe.py)
    # into the checkpoint so it can be loaded standalone with trust_remote_code=True
    cache_dir = Path(huggingface_hub.snapshot_download(model.config._name_or_path, local_files_only=True))
    for entry in model.config.auto_map.values():
        py_file = entry.split(".")[0] + ".py"  # e.g. "configuration_slimmoe"
        src = cache_dir / py_file
        if src.exists():
            shutil.copy2(src, Path(output_dir) / py_file)
            print(f"Copied {py_file} into {output_dir}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-m', '--nickname', type=str, default="phi-tiny", help="the nickname of the model with which to name files")
    parser.add_argument('-l', '--language', type=str, default='pes', help="the target language")
    # accelerate should take care of gpus ?
    # parser.add_argument('-g', '--gpus', type=str, required=True, help="the comma-separated list of gpus to evaluate on")
    # parser.add_argument('-y', '--layernum', type=int, default=None, help="backward-compatible shortcut for setting both min_layer and max_layer to the same layer")
    parser.add_argument('-y', '--min_layer', type=int, default=None, help="the first layer at which to calculate contrastive loss")
    parser.add_argument('-x', '--max_layer', type=int, default=14, help="the last layer at which to calculate contrastive loss")
    parser.add_argument('-e', '--earlyexit', action="store_true", help="whether to early exit and not calculate LM loss")
    parser.add_argument('-r', '--routers_only', action="store_true", help="if passed, only router/gate weights are trained")
    parser.add_argument('-t', '--test_run', action="store_true", help="passed if you want to just do a test run with small data size")
    parser.add_argument('--baseline', action="store_true", help="no applying of contrastive training, this is for control")
    args = parser.parse_args()

    if args.min_layer is None:
        args.min_layer = args.max_layer

    if args.min_layer > args.max_layer:
        raise ValueError(f"min_layer ({args.min_layer}) cannot be greater than max_layer ({args.max_layer}).")

    main(args)
