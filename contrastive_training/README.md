#### Lucas status 3/26/2026

Got a training steps run + eval steps on Phi-moe-tiny-instruct when no FSDP involved (just one GPU):

uv option:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
CUDA_VISIBLE_DEVICES="0" uv run accelerate launch --config_file accelerate_config_1gpu.yaml train.py -l pes -m granite -f 1 --no-checkpoint
```

Packed training:
```
CUDA_VISIBLE_DEVICES="0" uv run accelerate launch --config_file accelerate_config_1gpu.yaml \
  train.py -l pes -m qwen3-moe-tiny -f 1 -b 16 --packed --disable_cache --no-checkpoint
```

`--packed` packs target/source examples into one flattened row, passes reset `position_ids`,
`seq_idx`, and FlashAttention `cu_seq_lens_*`, and computes the same target LM loss and mean-token
router KL loss without padding tokens. Granite and Qwen3 MoE use split-forward packed paths that
drop source tokens after the selected router-loss layer; other MoE models use the generic
router-logit path when their Transformers forward accepts `cu_seq_lens_*`. The current packed
collator still batches by example count (`-b`); `max_length` only truncates each sequence before
packing. If packed training becomes the default path, the next batching change should be
token-budget packing: pack examples until total sequence length reaches `N`.

Measured speedups on one L40S:

| Freeze mode | Variant | Batch | Grad accum | Runtime | Seconds/step | Samples/s | Steps/s | Eval loss | Packed speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Router-only (`-f 1`) | padded | 16 | 2 | 216.88s | 0.693 | 46.11 | 1.443 | 12.89 | 1.000x |
| Router-only (`-f 1`) | packed | 16 | 2 | 102.26s | 0.327 | 97.79 | 3.061 | 12.69 | 2.121x |
| No-freeze (`-f 2`) | padded | 16 | 2 | 246.49s | 0.788 | 40.57 | 1.270 | 12.77 | 1.000x |
| No-freeze (`-f 2`) | packed | 16 | 2 | 118.89s | 0.380 | 84.11 | 2.633 | 12.51 | 2.073x |

Granite-4.0-h-tiny did not benefit
from this packed path in a small router-only microbenchmark, because Granite/Mamba sequence
index overhead dominates the padding savings.


```
conda create -n xlcl python=3.14
conda activate xlcl
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r xlcl_requirements.txt
pip install causal-conv1d==1.6.1 mamba-ssm==2.3.1 --no-build-isolation
pip install liger-kernel==0.8.0
MAX_JOBS=4 python -m pip install flash-attn==2.8.3 --no-build-isolation
```

For H100s, the 12.6 cuda and nvcc toolkit required me to start over:
```
conda create -n xlcl python=3.14 uv cuda-toolkit=12.6.0 -c nvidia -c conda-forge -y
conda activate xlcl
uv pip install torch==2.10.0+cu126 torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install -r xlcl_requirements.txt
uv pip install causal-conv1d==1.6.1 mamba-ssm==2.3.1 --no-build-isolation
uv pip install liger-kernel==0.8.0
MAX_JOBS=4 uv pip install flash-attn==2.8.3 --no-build-isolation
```

And then: `CUDA_VISIBLE_DEVICES="7" accelerate launch --config_file accelerate_config_1gpu.yaml train.py -l pes -t`

Additionally, can launch `accelerate_config_4gpu.yaml` for multi-gpu (2,3,or 4) with additional `--num_processes X` flag

#### Notes for Clark & Cara 4/12/2026

Background:
- LlamaFactory is an optimized version of the `trl` package, which I previously used for trainings. But it lacks flexibility, so for now the code is written using `trl`. Later, we'll probably have to adapt to LlamaFactory or figure out other ways to speed it up.

Code:
- LlamaFactory wraps the `trl` package (which has the `SFTTrainer` class), which itself wraps the `transformers` package which has the most generic `Trainer` class. Essentially I write custom Trainer classes in `contrastive_trainer.py` and override a bunch of the functions.
- I create an implementation for the contrastive loss; the central piece of all of this. The goal of everything else is to set up the data, weights, etc to be able to use this contrastive loss to increase routing alignment.
- a key element of the contrastive loss is how to aggregate across tokens in a sequence. I just listed out a number of `token_aggregation` methods.
- `key_layer` is the layer at which we do contrastive learning. That is, the layer where we use the routing weights to calculate the loss before backpropagating.
- ContrastiveTrainer is to train with *just* the contrastive loss. I'm not sure if this will work on its own, or if it'll break the model to fine-tune with a single, very specific objective. As a result, I also develop ContrastiveLMTrainer where the contrastive loss is accompanied with the typical LM loss (when doing LM-loss training on a post-trained model, we call this Continual Pretraining (CPT)). `alpha_contrastive` is how much to weigh one vs the other. 
- `modeling.py` class creates cutsom ...ForCausalLM classes (using Mixin) that ensures the router logits are returned in order to use them for the loss. The `NICKNAME_TO_MODEL_MAP` and `training_configs.json` is my very personal style of code that allows me to pass in shorthand names into the CLI and store the optimized batch size & other hyperparams that I've found for those models. `training_configs.json` has dummy values for now. 
- The "Partial..." classes is because if we are training with just the contrastive loss, we technically don't need to do inference beyond the `key_layer` (since it's not important for the training). To save huge amounts of time, these classes attempt to completely halt the forward pass after the key layer. I think I tried to replicate some implementations of early-exit decoding I found. For now, this is incomplete and we'll come back to this.
- a notable customization that's required is that instead of applying a loss to one sample, contrastive loss requires forward pass on TWO samples (in 2 different languages). The implementation of a custom DataCollator is in `parallel_dataset.py`. 

Future Complications:
- the only way we're going to be able to do this training on large models is by using FSDP or other types of model parallelism (because MoE models are large). We're going to have to make sure that none of our custom behavior breaks LlamaFactory's ability to optimize the computation graph. For now, I think we can ignore this and just have a basic implementation for Phi-Moe-tiny instruct, which should be trainable on smaller GPUs.
- Even if LlamaFactory doesn't break, it may just be unable to truly do its thing to speed up training, which would be too bad. But also like, I have no idea how it works, it's magic to me.
