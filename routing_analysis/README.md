This folder contains old code from https://github.com/mohsenfayyaz/moe, critical to https://openreview.net/forum?id=ZoZR0x7tTD

Instructions to recreate the `moe` env for these files:

1. `conda create --name moe python=3.13.4` - I named my env for finetuning "moe"
1. `conda activate moe`
1. `conda install -c conda-forge pip - install pip`
1. `conda install pytorch==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia` - install torch and related packages (and all its dependencies) using conda to ensure right CUDA compatibility. On H100, do pytorch==2.7.1 pytorch-cuda=12.6, on A100, do pytorch==2.5.1 pytorch-cuda=12.4. I haven't tried on A6000 servers.
1. `pip install -r lucas_requirements.txt` - use the requirements.txt I created, I'm not sure if it is complete. Please let me know if there's anything that needs to be added to make the files run appropriately

Note: For models that are newer than transformers==4.52.4, you can either use the xlcl env detailed in `contrastive_training/README.md` or try:

For Qwen3.5 i have been using this:
```bash
conda create -n qwen35 python=3.12.12
conda activate qwen35
pip install torch==2.10.0 torchvision==2.10.0 --index-url https://download.pytorch.org/whl/cu126
pip install transformers==4.57.6
pip install -r matplotlib pandas seaborn
pip install peft==0.16.0 uv==0.11.6 # for evals
uv pip install vllm==0.18.0 # for evals
pip install lm_eval==0.4.10 # for evals
```