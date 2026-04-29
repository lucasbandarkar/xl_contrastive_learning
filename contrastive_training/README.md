`CUDA_VISIBLE_DEVICES="6,7" accelerate launch --num_processes 2 train.py -l pes`

`CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --num_processes 4 train.py -l pes`

See accelerate_config.txt

#### Lucas status 3/26/2026

Got a training steps run + eval steps on Phi-moe-tiny-instruct when no FSDP involved (just one GPU):

uv option:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
CUDA_VISIBLE_DEVICES="0" uv run accelerate launch --config_file accelerate_config_1gpu.yaml train.py -l pes -t
```


```
conda create -n xlcl python=3.14
conda activate xlcl
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r xlcl_requirements.txt
```

And then: `CUDA_VISIBLE_DEVICES="7" accelerate launch --config_file accelerate_config_1gpu.yaml train.py -l pes -t`

So far `accelerate_config_1gpu.yaml` is the only config that has worked.

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
