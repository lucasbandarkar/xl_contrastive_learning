# Vanilla SFT and Eval

This folder now has two bash entrypoints:

- `run_sft_and_eval.sh` for training, then evaluating the merged model (NOT YET IMPLEMENTED)
- `run_eval_only.sh` for evaluation only

Examples:

```bash
./run_sft_and_eval.sh -m phi-tiny -l pes -d my_custom_dataset -g 0
./run_eval_only.sh -m microsoft/Phi-tiny-MoE-instruct -l si -g 1

./run_eval_only.sh -m ../contrastive_training/checkpoints/moe_contrastive_training_test -l si -g 0
```

Common flags:

- `-m, --model-name`
- `-l, --language`
- `-g, --gpus`
- `-d, --dataset-name` for training
- `-a, --adapter-path` for eval-only

The `-g/--gpus` flag should be a comma-separated list like `0`, `0,1`, or `0,1,2,3`.

## Eval runtime defaults

`run_eval_only.sh` chooses different defaults based on the server:

- On PlusLab servers, detected by `/data2`, it uses the currently active `python`/conda environment and skips `apply_vllm_phimoe_patch` for Phi-tiny.
- On AWS, it preserves Clark's uv workflow by using `uv run python` and applying
  `apply_vllm_phimoe_patch`.

You can override either default explicitly:

```bash
EVAL_RUNNER=uv APPLY_VLLM_PHIMOE_PATCH=1 ./run_eval_only.sh -m microsoft/Phi-tiny-MoE-instruct -l si -g 0
EVAL_RUNNER=python APPLY_VLLM_PHIMOE_PATCH=0 ./run_eval_only.sh -m microsoft/Phi-tiny-MoE-instruct -l si -g 0
```

## Create environment

This env works for Phi-tiny & ling, but not for Qwen3.5.

```bash
uv python install 3.12.11
uv venv ~/.venvs/moevllm --python 3.12.11
source ~/.venvs/moevllm/bin/activate

uv pip install torch==2.10.0 torchvision torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install transformers==4.57.6
uv pip install vllm==0.19.0
uv pip install datasets==3.6.0 lm-eval==0.4.10 hf_transfer==0.1.9 peft==0.16.0 ray
```

For Qwen3.5, i reused the environment at the bottom of `routing_analysis/README.md`, which i named `qwen35`

## Evaluating a new language

See `language_to_task.py` for instructions on how to add another language.

