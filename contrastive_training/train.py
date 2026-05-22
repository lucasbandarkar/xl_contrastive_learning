from argparse import ArgumentParser
from contrastive_trainer import ContrastiveTrainer, ContrastiveLMTrainer, TargetLMTrainer, TranslationSFTTrainer
import torch
from modeling import load_models
from parallel_dataset import (
    load_parallel_datasets,
    ParallelDataCollator,
    TargetLanguageCausalLMCollator,
    format_translation_sft_dataset,
)
from transformers import TrainingArguments
import json
import shutil
from pathlib import Path
import huggingface_hub


def create_training_args(
        model_nickname,
        output_dir=None,
        learning_rate=1e-6,
        lr_scheduler='constant_with_warmup',
        partial_training=False,
        batch_size=None,
        test_run=False,
        num_gpus=1,
        is_aws=False,
        freezing_mode=0,
        use_model_cache=False,
        save_checkpoints=True,
    ) -> TrainingArguments:
    model_config = get_model_config(model_nickname)
    training_mode = 'partial' if partial_training else 'full'

    if batch_size is None:
        # uses auto_find_batch_size if config hasn't been set
        # auto_find_batch_size is convenient but controlling step size is complicated
        # also, FSDP was not liking auto_find_batch_size
        use_auto_batch_size = model_config.get('auto_find_batch_size', True)
        batch_size = get_configured_batch_size(model_config, training_mode, num_gpus, is_aws)
    else:
        use_auto_batch_size = False

    if use_auto_batch_size:
        grad_accum_steps = 1
        save_steps = 1000
        warmup_steps = 200
    else:
        grad_accum_steps, warmup_steps = calc_grad_accum_steps(batch_size, 32, num_gpus)
        save_steps = 1250 # about every 40000 samples
    
    kwargs = {}
    if num_gpus > 1 or model_config.get("gradient_checkpointing", False):
        kwargs['gradient_checkpointing'] = True

    location = "." if is_aws else "/data2/lucasbandarkar"

    return TrainingArguments(
        output_dir=f"{location}/checkpoints/{output_dir}",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps= grad_accum_steps,
        auto_find_batch_size=use_auto_batch_size,
        use_cache=use_model_cache,
        # optim='adafactor',
        optim='adamw_torch_fused', # more memory, but faster
        num_train_epochs=1,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler,
        warmup_steps=warmup_steps,
        neftune_noise_alpha=None if partial_training or freezing_mode == 1 else 5.0,
        # max_grad_norm=0.0, # FSDP2 requires disabling to 0.0 to prevent "RuntimeError: No backend type associated with device type cpu"
        max_grad_norm=1.0, # ok if using FSDP1
        adam_beta2=0.99, # default value is 0.999
        save_strategy="steps" if save_checkpoints and not test_run else "no",
        save_steps=save_steps if save_checkpoints and not test_run else None,
        bf16=False, # disable to maintain Pure BF16 (avoids FP32 master weight upcast bug)
        eval_strategy='steps',
        # eval_steps=1/25, # eval 20 times through training
        save_total_limit=3 if save_checkpoints and not test_run else None,
        logging_steps=warmup_steps, # about 1k samples
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

def get_configured_batch_size(model_config, training_mode, num_gpus, is_aws):
    # allows for dict for different GPU sizes
    batch_dict_name ="batch_sizes_smaller_gpu" if is_aws else "batch_sizes_by_gpus"
    gpu_overrides = model_config.get(batch_dict_name, {}).get(training_mode, {})
    return gpu_overrides.get(str(num_gpus), 32) # default batch size to 32

def calc_grad_accum_steps(device_batch_size, effective_batch_size, world_size=1):
    """
    In order to maintain sufficiently large effective batch size, calculates number of gradient accumulation steps with current training conditions.
    """
    grad_accum_steps=1
    while grad_accum_steps * world_size * device_batch_size < effective_batch_size:
        grad_accum_steps *= 2
        
    num_warmup_steps = int(1000 / effective_batch_size) # warmup lasts 1k samples, regardless of what batch size is
    return grad_accum_steps, num_warmup_steps

def configure_resume_training(training_args, resume_from_checkpoint):
    trainer_state_path = Path(resume_from_checkpoint) / "trainer_state.json"
    with open(trainer_state_path, "r") as file:
        trainer_state = json.load(file)

    checkpoint_epoch = trainer_state.get("epoch", 1.0)
    training_args.num_train_epochs = float(checkpoint_epoch + 1)
    training_args.output_dir = str(Path(resume_from_checkpoint).parent)
    print(
        f"Resuming from {resume_from_checkpoint} at epoch {checkpoint_epoch:.3f}; "
        f"training until epoch {training_args.num_train_epochs:.3f}."
    )

def create_output_directory_name(args, size_suffix):
    prefix = f"{args.nickname}_{args.language}"
    
    training_details = ""
    if args.baseline:
        training_details += f"baseline-{args.baseline_mode}"
        if args.baseline_mode == "frozen_lm":
            training_details += f"_L{args.max_layer}"
    else:
        if args.earlyexit:
            training_details += "earlyexit"

        if args.min_layer:
            training_details += f"_L{args.min_layer}-{args.max_layer}"
        else:
            training_details = f"_L{args.max_layer}"
        
        if args.freezing_mode == 1: 
            training_details += "_routers"

    return f"{prefix}_{training_details}_{size_suffix}"

def is_aws_server():
    """Detect if we're running on an AWS server by checking if the /data2/ directory is missing."""
    # /data2/ exists on the pluslab servers, but not on AWS.
    return not Path("/data2/").is_dir()

def main(args):
    num_gpus = torch.cuda.device_count()
    multi_gpu = num_gpus > 1
    
    # Layers are one-indexed in the CLI, while the trainer/model use the same convention.
    # Partial model loading only needs the upper bound of the requested range.
    max_layer = int(args.max_layer) if args.earlyexit else None
    aws = is_aws_server()
    model_config = get_model_config(args.nickname)
    freezing_mode = args.freezing_mode
    if freezing_mode is None:
        freezing_mode = model_config.get("freezing_mode", 0)

    model, tokenizer = load_models(
        args.nickname,
        max_layer=max_layer,
        fsdp=multi_gpu,
        is_aws=aws,
        use_model_cache=not args.disable_cache,
        use_optimizations=not args.no_optimizations,
    )

    data_limit = 1000 if args.test_run else args.samples * 1000
    dataset_train, dataset_valid, key_src, key_tgt = load_parallel_datasets(
        args.data,
        args.language,
        data_limit=data_limit,
        disable_cache=args.disable_cache,
    )

    ckpt_suffix = "test" if args.test_run else f"{args.samples}k"
    output_dir_name = create_output_directory_name(args, ckpt_suffix)

    if args.earlyexit:
        training_args = create_training_args(
            args.nickname,
            output_dir=output_dir_name,
            partial_training=True,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            test_run=args.test_run,
            num_gpus=num_gpus,
            is_aws=aws,
            freezing_mode=freezing_mode,
            use_model_cache=not args.disable_cache,
            save_checkpoints=not args.no_checkpoint,
        )
        trainer = ContrastiveTrainer(
            model,
            args=training_args,
            train_dataset=dataset_train,
            eval_dataset=dataset_valid,
            data_collator=ParallelDataCollator(tokenizer, key_src, key_tgt, model_config.get("max_length")),
            min_layer=args.min_layer,
            max_layer=args.max_layer,
            alpha_contrastive=args.contrastive_alpha,
        )
    elif args.baseline and args.baseline_mode in ["target_lm", "frozen_lm"]:
        training_args = create_training_args(
            args.nickname,
            output_dir=output_dir_name,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            test_run=args.test_run,
            num_gpus=num_gpus,
            is_aws=aws,
            freezing_mode=freezing_mode,
            use_model_cache=not args.disable_cache,
            save_checkpoints=not args.no_checkpoint,

        )
        trainer = TargetLMTrainer(
            model,
            args=training_args,
            train_dataset=dataset_train,
            eval_dataset=dataset_valid,
            data_collator=TargetLanguageCausalLMCollator(tokenizer, key_src, key_tgt),
        )
    elif args.baseline and args.baseline_mode == "translation_sft":
        training_args = create_training_args(
            args.nickname,
            output_dir=output_dir_name,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            test_run=args.test_run,
            num_gpus=num_gpus,
            is_aws=aws,
            freezing_mode=freezing_mode,
            use_model_cache=not args.disable_cache,
            save_checkpoints=not args.no_checkpoint,
        )
        training_args.group_by_length = True # only possible for translation sft
        trainer = TranslationSFTTrainer(
            model,
            args=training_args,
            train_dataset=format_translation_sft_dataset(
                dataset_train,
                tokenizer,
                key_src,
                key_tgt,
                disable_cache=args.disable_cache,
            ),
            eval_dataset=format_translation_sft_dataset(
                dataset_valid,
                tokenizer,
                key_src,
                key_tgt,
                disable_cache=args.disable_cache,
            ),
            processing_class=tokenizer,
        )
    else:
        training_args = create_training_args(
            args.nickname,
            output_dir=output_dir_name,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            test_run=args.test_run,
            num_gpus=num_gpus,
            is_aws=aws,
            freezing_mode=freezing_mode,
            use_model_cache=not args.disable_cache,
            save_checkpoints=not args.no_checkpoint,
        )
        trainer = ContrastiveLMTrainer(
            model,
            args=training_args,
            train_dataset=dataset_train,
            eval_dataset=dataset_valid,
            data_collator=ParallelDataCollator(tokenizer, key_src, key_tgt, model_config.get("max_length")),
            min_layer=args.min_layer,
            max_layer=args.max_layer,
            alpha_contrastive=args.contrastive_alpha,
        )

    if args.baseline and args.baseline_mode == "frozen_lm":
        trainer.configure_early_layer_only_training(args.max_layer, multi_gpu)
    elif not args.baseline:
        """
        freezing modes:
        0: only train early layers {default}
        1: only train router/gate weights
        2: no parameters frozen
        """
        if freezing_mode == 1:
            trainer.configure_router_only_training(args.max_layer, multi_gpu)
        elif freezing_mode == 0:
            trainer.configure_early_layer_only_training(args.max_layer, multi_gpu)
    
    if args.resume_training:
        configure_resume_training(
            trainer.args,
            args.resume_training,
        )

    trainer.train(resume_from_checkpoint=args.resume_training)

    if args.test_run:
        return

    output_dir = Path(trainer.args.output_dir)
    trainer.save_model(str(output_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output_dir)

        # Copy custom model code files (e.g. configuration_slimmoe.py, modeling_slimmoe.py)
        # into the checkpoint so it can be loaded standalone with trust_remote_code=True.
        auto_map = getattr(model.config, "auto_map", None)
        if auto_map:
            for entry in auto_map.values():
                py_file = entry.split(".")[0] + ".py"  # e.g. "configuration_slimmoe.py"
                try:
                    src = huggingface_hub.hf_hub_download(
                        model.config._name_or_path,
                        py_file,
                        local_files_only=True,
                    )
                except Exception:
                    src = huggingface_hub.hf_hub_download(model.config._name_or_path, py_file)
                shutil.copy2(src, output_dir / py_file)
                print(f"Copied {py_file} into {output_dir}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-m', '--nickname', type=str, default="phi-tiny", help="the nickname of the model with which to name files")
    parser.add_argument('-l', '--language', type=str, default='pes', help="the target language")
    parser.add_argument('-d', '--data', type=str, default='mixed', choices=['opus', 'mixed'], help='data source to use for contrastive/CPT data')
    # accelerate should take care of gpus ?
    # parser.add_argument('-g', '--gpus', type=str, required=True, help="the comma-separated list of gpus to evaluate on")
    # parser.add_argument('-y', '--layernum', type=int, default=None, help="backward-compatible shortcut for setting both min_layer and max_layer to the same layer")
    parser.add_argument('-y', '--min_layer', type=int, default=None, help="the first layer at which to calculate contrastive loss")
    parser.add_argument('-x', '--max_layer', type=int, default=14, help="the last layer to train for early-layer freezing, and the last layer at which to calculate contrastive loss")
    parser.add_argument('-e', '--earlyexit', action="store_true", help="whether to early exit and not calculate LM loss")
    parser.add_argument('-f', '--freezing_mode', type=int, default=0, help="0 trains early layers; 1 trains only router/gate weights")
    parser.add_argument('-t', '--test_run', action="store_true", help="passed if you want to just do a test run with small data size")
    parser.add_argument('-b', '--batch_size', type=int, default=None, help="manual per-device batch size; overrides training_configs.json batch sizes and disables auto_find_batch_size")
    parser.add_argument('-s', "--samples", type=int, default=10, help="number of samples to train on, IN THOUSANDS (e.g. 10 means 10,000)")
    parser.add_argument('-a', '--contrastive_alpha', type=float, default=1.0, help="weight for contrastive loss in contrastive training modes")
    parser.add_argument('-r', '--learning_rate', type=float, default=1e-6, help="training learning rate")
    parser.add_argument('--resume_training', type=str, default=None, help="checkpoint directory to resume from & do another epoch, e.g. .../checkpoint-313")
    parser.add_argument('--disable_cache', action="store_true", help="disable HF model use_cache and avoid dataset map cache where supported")
    parser.add_argument('--no_optimizations', '--no-optimizations', action="store_true", help="bypass optimizations.py and use the legacy modeling.py load path")
    parser.add_argument('--no_checkpoint', '--no-checkpoint', action="store_true", help="disable intermediate trainer checkpoints; final model saving still runs")
    parser.add_argument('--baseline', action="store_true", help="no applying of contrastive training, this is for control")
    parser.add_argument(
        '--baseline_mode',
        choices=["translation_sft", "target_lm", "frozen_lm"],
        default="translation_sft",
        help="which baseline to run when --baseline is passed",
    )
    args = parser.parse_args()

    if args.min_layer is None:
        args.min_layer = args.max_layer

    if args.min_layer > args.max_layer:
        raise ValueError(f"min_layer ({args.min_layer}) cannot be greater than max_layer ({args.max_layer}).")

    main(args)
