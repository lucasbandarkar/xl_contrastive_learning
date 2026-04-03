`CUDA_VISIBLE_DEVICES="6,7" accelerate launch --num_processes 2 train.py -l pes`

`CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --num_processes 4 train.py -l pes`

See accelerate_config.txt

#### Lucas status 3/26/2026

Got a training steps run + eval steps on Phi-moe-tiny-instruct when no FSDP involved (just one GPU):

```
conda create -n xlcl python=3.14
conda activate xlcl
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r xlcl_requirements.txt
```

And then: `CUDA_VISIBLE_DEVICES="7" accelerate launch --config_file accelerate_config_1gpu.yaml train.py -l pes -t`

#### Notes for Lionel 10/7/2025

Yeah sorry that it's so incomplete and half-baked. Hopefully the explanations below + AI tools will be able to help make sense of the stuff I've worked on so far. TBH I'm pretty proud of what I have come up with so far, even though I don't know if it'll actually work. Reminder: I didn't even run this once as I hadn't set up enough.

Background:
- LlamaFactory is an optimized version of the `trl` package, which I previously used for trainings. I decided to invest in the optimized way since speed and memory costs are going to end up being super important for later experiments/implementations

Setup:
- I never actually ran this code so even though i technically built this code with llamafactory==0.9.3 in mind, idk if it'll matter what version you use and I'd rather recommending the latest version you can use with the CUDA drivers on the a6000s.
- I dumped the `accelerate` package config I had set up in `accelerate_config.txt`

Code:
- LlamaFactory wraps the `trl` package (which has the `SFTTrainer` class), which itself wraps the `transformers` package which has the most generic `Trainer` class. Can you see this conversation I had with ChatGPT ? https://chatgpt.com/share/68e4cf0c-54c0-800e-9dff-e19289d5938e. If so, it might give some context for what I was trying to do. Essentially I write custom Trainer classes in `contrastive_trainer.py` and override a bunch of the functions.
- I had copied some code from LlamaFactory into `llamafactory_overrides` because in order to use LlamaFactory's optimizations with our code, we're going to have to override certain parts of llamafactory so that it calls our classes & methods. It doesn't look like I got very far with this, but that I had determined that there was a way to be able to use LlamaFactory's convenient command-line interface (CLI).
- I create an implementation for the contrastive loss; the central piece of all of this. The goal of everything else is to set up the data, weights, etc to be able to use this contrastive loss to increase routing alignment.
- a key element of the contrastive loss is how to aggregate across tokens in a sequence. I just listed out a number of `token_aggregation` methods.
- `key_layer` is the layer at which we do contrastive learning. That is, the layer where we use the routing weights to calculate the loss before backpropagating.
- ContrastiveTrainer is to train with *just* the contrastive loss. I'm not sure if this will work on its own, or if it'll break the model to fine-tune with a single, very specific objective. As a result, I also develop ContrastiveLMTrainer where the contrastive loss is accompanied with the typical LM loss (when doing LM-loss training on a post-trained model, we call this Continual Pretraining (CPT)). `alpha_contrastive` is how much to weigh one vs the other. 
- `modeling.py` class creates cutsom ...ForCausalLM classes (using Mixin) that ensures the router logits are returned in order to use them for the loss. The `NICKNAME_TO_MODEL_MAP` and `training_configs.json` is my very personal style of code that allows me to pass in shorthand names into the CLI and store the optimized batch size & other hyperparams that I've found for those models. `training_configs.json` has dummy values for now. 
- The "Partial..." classes is because if we are training with just the contrastive loss, we technically don't need to do inference beyond the `key_layer` (since it's not important for the training). To save huge amounts of time, these classes attempt to completely halt the forward pass after the key layer. I think I tried to replicate some implementations of early-exit decoding I found.
- a notable customization that's required is that instead of applying a loss to one sample, contrastive loss requires forward pass on TWO samples (in 2 different languages). iirc, this didn't look like it'd be that complicated. The implementation of a custom DataCollator is in `parallel_dataset.py`. It doesn't look like I ever got to actually adding this to the Trainer classes.

Future Complications:
- the only way we're going to be able to do this training on large models is by using FSDP or other types of model parallelism (because MoE models are large). We're going to have to make sure that none of our custom behavior breaks LlamaFactory's ability to optimize the computation graph. For now, I think we can ignore this and just have a basic implementation for OLMoE 7B (can even test on OLMoE 1B), which should be trainable on A6000.
- Even if LlamaFactory doesn't break, it may just be unable to truly do its thing to speed up training, which would be too bad. But also like, I have no idea how it works, it's magic to me.

High-Level Plan for Implementing this method
1. Understand what I've written so far
2. Get a dummy training using `trl` (can ignore LlamaFactory) to work with a small model. That'll mean finish bringing together the dataset, the loss function, the model training classes altogether and get it to work.
3. Get it to actually do increase routing alignment with very small runs (it'll likely be super slow). That is measure the routing alignment before and after training and see that it learned to route more similarly, even just slightly.
4. Get it to work with LlamaFactory so that it'll be way faster and easier to iterate.
5. Then begins the experiments; trying to figure out what will actually lead to improvement on these models.

I think each step in itself constitutes a large amount of work :) Any progress, big or small, through these steps will be truly awesome.
