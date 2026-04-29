from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers import AutoModelForCausalLM, Qwen3MoeForCausalLM, MixtralForCausalLM, OlmoeForCausalLM, AutoConfig, AutoTokenizer
import torch
from accelerate import dispatch_model
from accelerate.utils import infer_auto_device_map
from typing import Optional, Tuple, Union, Dict, Any
import gc



# copied from routing_analysis/get_routing_weights.py
NICKNAME_TO_MODEL_MAP = {
    "qwen3_30b": "Qwen/Qwen3-30B-A3B",
    "olmoe": "allenai/OLMoE-1B-7B-0125-Instruct",
    # "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    # "llama4": "meta-llama/Llama-4-Scout-17B-16E-Instruct", # WASNT WORKING (conda issue ?)
    # "phimoe": "microsoft/Phi-3.5-MoE-instruct",
    # "moonlight": "moonshotai/Moonlight-16B-A3B-Instruct", # WASNT WORKING (too complicated to get Deepseek remote code to work)
    "gpt": "openai/gpt-oss-20b",
    "qwen35": "Qwen/Qwen3.5-35B-A3B",
    "nemotron": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8", # transformers doesn't have output_router_logits implemented
    # "kimi": "moonshotai/Kimi-Linear-48B-A3B-Instruct", # doesn't have output_router_logits enabled
    # "llada": "inclusionAI/LLaDA2.1-mini",
    "ling": "inclusionAI/Ling-mini-2.0",
    "phi-tiny": "microsoft/Phi-tiny-MoE-instruct",
    "ernie": "baidu/ERNIE-4.5-21B-A3B-PT",
}

def load_models(model_name, max_layer=None):
    # load custom MoE model objects, can I use Mohsen's src/modeling/ modifications ?
    # most importantly need to modify forward() function
    # for memory, is there a way to load just the layers that matter ?
    if model_name in NICKNAME_TO_MODEL_MAP.keys():
        model_name = NICKNAME_TO_MODEL_MAP[model_name] # now model_name is a HF model name
    else: ## maybe model is a checkpoint address
        return True
    
    if max_layer:
        model = PartialMoEModelForCausalLM.from_pretrained(model_name, max_layer)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Fix FSDP auto-wrap mismatch: Force _no_split_modules to match the EXACT class name 
    if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
        actual_layer_cls = model.model.layers[0].__class__.__name__
        model._no_split_modules = [actual_layer_cls]
        
    model.train()
    return model, tokenizer

class PartialForwardMoEModelMixin:
    """Mixin class to add router logits functionality to MoE models."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._early_exit_layer = None
    
    def set_early_exit_layer(self, layer_idx: Optional[int] = None):
        """Set the layer after which to perform early exit (saves compute)."""
        self._early_exit_layer = layer_idx
    
    def _extract_router_logits_from_layer(self, layer_output, layer_idx: int):
        """Extract router logits from a transformer layer output."""
        # This depends on the specific MoE implementation
        # For Mixtral-style models, router logits are typically in the output
        if hasattr(layer_output, 'router_logits'):
            return layer_output.router_logits
        elif isinstance(layer_output, tuple) and len(layer_output) > 1:
            # Sometimes router logits are returned as additional outputs
            for item in layer_output[1:]:
                if hasattr(item, 'router_logits') or (
                    isinstance(item, torch.Tensor) and item.dim() >= 2
                ):
                    return item
        return None
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        """Forward pass with early exit after specified layer."""
        # manually iterate through layers, collect router_logits,  and exit early
        
        hidden_states = self.model.embed_tokens(input_ids)

        # Collect router logits
        all_router_logits = () if output_router_logits else None
        
        # Process layers up to early exit point
        max_layer = min(self._early_exit_layer, len(self.model.layers))
        for decoder_layer in self.model.layers[:max_layer]:            
            # Forward through layer
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                output_router_logits=output_router_logits,
                **kwargs
            )
            
            hidden_states = layer_outputs[0]

            if output_router_logits and hasattr(layer_outputs, 'router_logits'):
                all_router_logits += (layer_outputs.router_logits,)
                    
        if return_dict:
            return MoeCausalLMOutputWithPast(
                logits=None,  # No language modeling head
                hidden_states=hidden_states,
                router_logits=all_router_logits if output_router_logits else None,
            )
        else:
            return (None, all_router_logits) if output_router_logits else (None,)
    
class PartialMoEModelForCausalLM(PartialForwardMoEModelMixin, torch.nn.Module):
    """
    Memory-efficient partial MoE model that only loads layers up to max_layer.
    This should save memory by not loading unnecessary layers.
    """
    def __init__(self, config, layer_number: int):
        super().__init__()
        self.config = config
        self.set_early_exit_layer(layer_number)
        
        # Load only the components we need
        self._load_partial_model()
    
    def _load_partial_model(self):
        """Load only the layers we need for partial forward pass."""
        # Load full model temporarily
        full_model = AutoModelForCausalLM.from_pretrained(
            self.config._name_or_path, 
            torch_dtype=torch.bfloat16
        )
        
        # Extract only what we need, for LlamaForCausalLM structure
        self.model = torch.nn.Module()
        self.model.embed_tokens = full_model.model.embed_tokens
        self.model.layers = torch.nn.ModuleList(full_model.model.layers[:self._early_exit_layer])
        self.model.norm = full_model.model.norm
        
        # # Infer and dispatch across GPUs
        # device_map = infer_auto_device_map(
        #     self.model, 
        #     max_memory={i: torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())},
        #     # no_split_module_classes=["Qwen3MoeDecoderLayer"], 
        # )
        # self.model = dispatch_model(self.model, device_map=device_map)

        # Clean up full model from memory
        del full_model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, max_layer: int, **kwargs):
        """overriding from_pretrained ensures we create an instance of this class"""
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        config._name_or_path = pretrained_model_name_or_path
        
        return cls(config, max_layer)

# class PartialQwen3MoeForCausalLM(PartialForwardMoEModelMixin, Qwen3MoeForCausalLM):

# class PartialMixtralForCausalLM(PartialForwardMoEModelMixin, MixtralForCausalLM):

# class PartialOlmoeForCausalLM(PartialForwardMoEModelMixin, OlmoeForCausalLM):


"""
Alternative to the above: add a forward hook to layer X that raises an exception

Zero3 param gathering is handled by the normal model.forward() preamble, Flash Attention/Liger are unaffected, and you only pay for k layers of compute.
The torch.compile graph-break issue remains but is acceptable for the partial-pass mode since you're getting a genuine compute saving anyway.
"""
class EarlyExitException(Exception):
    def __init__(self, router_logits_so_far):
        self.router_logits = router_logits_so_far

def partial_forward_with_routing(model, inputs, max_layer, target_layers):
    collected = {}

    def make_hook(idx):
        def hook(module, inp, out):
            # Collect router logits if this is a MoE layer
            if hasattr(out, '__len__') and len(out) > 1:
                collected[idx] = out[-1]  # router logits position
            if idx == max_layer - 1:
                raise EarlyExitException(collected)
        return hook

    hooks = []
    for idx, layer in enumerate(model.model.layers):
        if idx <= max_layer:
            hooks.append(layer.register_forward_hook(make_hook(idx)))
    
    try:
        model(**inputs, output_router_logits=True)
    except EarlyExitException as e:
        router_logits = e.router_logits
    finally:
        for h in hooks:
            h.remove()
    
    return {k: v for k, v in router_logits.items() if k in target_layers}