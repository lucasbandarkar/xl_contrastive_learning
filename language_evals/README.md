# Vanilla SFT and Eval

This folder now has two bash entrypoints:

- `run_sft_and_eval.sh` for training, then evaluating the merged model (NOT YET IMPLEMENTED)
- `run_eval_only.sh` for evaluation only

Examples:

```bash
./run_sft_and_eval.sh -m phi-tiny -l pes -d my_custom_dataset -g 0
./run_eval_only.sh -m microsoft/Phi-tiny-MoE-instruct -l si -g 1
```

Common flags:

- `-m, --model-name`
- `-l, --language`
- `-g, --gpus`
- `-d, --dataset-name` for training
- `-a, --adapter-path` for eval-only

The `-g/--gpus` flag should be a comma-separated list like `0`, `0,1`, or `0,1,2,3`.

## Create environment

```bash
conda create --name moevllm python=3.12.11 -y
conda activate moevllm
pip install torch==2.10.0 torchvision torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu126
pip install transformers==4.57.6
pip install vllm==0.19.0
pip install datasets==3.6.0 lm-eval==0.4.10
```

## Evaluating a new language

See `language_to_task.py` for instructions on how to add another language.


