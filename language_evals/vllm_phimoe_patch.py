import os
import sys
from pathlib import Path
from vllm.model_executor.models import ModelRegistry
import vllm.model_executor.models.phimoe as vllm_phimoe
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.attention import Attention
from unittest.mock import MagicMock

def apply_vllm_phimoe_patch(model_path: str):
    """
    Injects high-performance native vLLM support for Phi-tiny-MoE models
    by patching head_dim calculations and registering a custom architecture.
    """

    # 2. Define a high-performance Patched Attention class that respects the tiny head_dim (128)
    class PatchedPhiMoEAttention(vllm_phimoe.PhiMoEAttention):
        def __init__(self, hidden_size, num_heads, num_kv_heads, **kwargs):
            from vllm.distributed import get_tensor_model_parallel_world_size
            from torch import nn
            
            nn.Module.__init__(self)
            self.hidden_size = hidden_size
            tp_size = get_tensor_model_parallel_world_size()
            self.total_num_heads = num_heads
            self.num_heads = self.total_num_heads // tp_size
            self.total_num_kv_heads = num_kv_heads
            self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
            
            # --- THE FIX: Force head_dim to 128 instead of hidden_size // num_heads ---
            self.head_dim = 128 
            
            self.q_size = self.num_heads * self.head_dim
            self.kv_size = self.num_kv_heads * self.head_dim
            self.scaling = self.head_dim**-0.5
            self.rope_theta = kwargs.get("rope_theta", 10000)
            self.rope_scaling = kwargs.get("rope_scaling")
            prefix = kwargs.get("prefix", "")

            self.qkv_proj = QKVParallelLinear(hidden_size, self.head_dim, self.total_num_heads,
                                              self.total_num_kv_heads, bias=True, quant_config=kwargs.get("quant_config"),
                                              prefix=f"{prefix}.qkv_proj")
            self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, hidden_size,
                                            bias=True, quant_config=kwargs.get("quant_config"),
                                            prefix=f"{prefix}.o_proj")
            self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim, 
                                       max_position=kwargs.get("max_position", 4096),
                                       base=int(self.rope_theta), is_neox_style=True, rope_scaling=self.rope_scaling)
            self.attn = Attention(self.num_heads, self.head_dim, self.scaling, num_kv_heads=self.num_kv_heads,
                                  cache_config=kwargs.get("cache_config"), quant_config=kwargs.get("quant_config"),
                                  prefix=f"{prefix}.attn")

    # 3. Create a patched Decoder Layer that uses our optimized Attention
    class PatchedPhiMoEDecoderLayer(vllm_phimoe.PhiMoEDecoderLayer):
        def __init__(self, config, cache_config=None, quant_config=None, prefix=""):
            from torch import nn
            nn.Module.__init__(self)
            self.hidden_size = config.hidden_size
            rope_theta = getattr(config, "rope_theta", 10000)
            self.self_attn = PatchedPhiMoEAttention(
                hidden_size=self.hidden_size, num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads, rope_theta=rope_theta,
                cache_config=cache_config, quant_config=quant_config,
                max_position=config.max_position_embeddings, rope_scaling=config.rope_scaling,
                prefix=f"{prefix}.self_attn")
            self.block_sparse_moe = vllm_phimoe.PhiMoE(
                num_experts=config.num_local_experts, top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
                quant_config=quant_config, prefix=f"{prefix}.block_sparse_moe")
            self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    # 4. Create the final CausalLM class that uses these patched layers
    class PatchedPhiMoEForCausalLM(vllm_phimoe.PhiMoEForCausalLM):
        def __init__(self, *, vllm_config, prefix=""):
            from vllm.model_executor.models.utils import make_layers
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            config = vllm_config.model_config.hf_config
            self.model.start_layer, self.model.end_layer, self.model.layers = make_layers(
                config.num_hidden_layers,
                lambda prefix: PatchedPhiMoEDecoderLayer(config, vllm_config.cache_config, vllm_config.quant_config, prefix=prefix),
                prefix=f"{prefix}.model.layers")

    # Register this patched model for the checkpoint's architecture name
    ModelRegistry.register_model("PhimoeForCausalLM", PatchedPhiMoEForCausalLM)

    # 5. Fallback/Mock logic for remote code import
    sys.modules["flash_attn"] = MagicMock()
    sys.modules["flash_attn.layers"] = MagicMock()
    sys.modules["flash_attn.layers.rotary"] = MagicMock()
    sys.path.append(str(Path(model_path).resolve()))
