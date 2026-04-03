# Vanilla SFT and Eval

This folder now has two bash entrypoints and two local `accelerate` configs:

- `run_sft_and_eval.sh` for training, then evaluating the merged model
- `run_eval_only.sh` for evaluation only
- `accelerate_config_1gpu.yaml`
- `accelerate_config_multi_gpu.yaml`

Examples:

```bash
./run_sft_and_eval.sh -m phi-tiny -l pes -d my_custom_dataset -g 0
./run_eval_only.sh -m phi-tiny -l pes -a /dev/shm/moe_adapter_output -g 0
```

Common flags:

- `-m, --model-name`
- `-l, --language`
- `-g, --gpus`
- `-d, --dataset-name` for training
- `-a, --adapter-path` for eval-only

The `-g/--gpus` flag should be a comma-separated list like `0`, `0,1`, or `0,1,2,3`.
The wrapper picks `accelerate_config_1gpu.yaml` for one GPU and `accelerate_config_multi_gpu.yaml` for 2 or 4 GPUs.

## Create environment

conda create --name moeeval python=3.12.11 -y
conda activate moeeval
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r eval_requirements.txt