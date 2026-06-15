"""Training optimizations adapted from Lacuna."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Any

import torch
from transformers import AutoConfig
from transformers.integrations.hub_kernels import allow_all_hub_kernels
from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN, _apply_liger_kernel
from torchtitan.models.moe import MoE, MoEArgs

logger = logging.getLogger(__name__)
FLASH_ATTN2_IMPLEMENTATION = "flash_attention_2"
TORCHTITAN_MOE_MODEL_TYPES = {"qwen3_moe", "glm4_moe"}
TORCHTITAN_INCOMPATIBLE_MODEL_IDS = {
    "PrimeIntellect/qwen3-moe-tiny": "checkpoint uses fused gate_up_proj/down_proj expert tensors, not TorchTitan's split w1/w2/w3 Qwen3 layout",
}
FLASH_ATTN3_MODEL_TYPES = {
    "glm4_moe",
    "gpt_oss",
    "granite",
    "granitemoehybrid",
    "nemotron",
    "olmoe",
    "phi3",
    "qwen3_moe",
    "qwen3_5_moe",
    "qwen3_next",
}

MODEL_ALIASES = {
    "qwen3-next-thinking": "Qwen/Qwen3-Next-80B-A3B-Thinking",
    "Qwen3-Next-80B-A3B-Thinking": "Qwen/Qwen3-Next-80B-A3B-Thinking",
    "kimi-linear": "moonshotai/Kimi-Linear-48B-A3B-Instruct",
    "Kimi-Linear-48B-A3B-Instruct": "moonshotai/Kimi-Linear-48B-A3B-Instruct",
    "moonlight": "moonshotai/Moonlight-16B-A3B-Instruct",
    "Moonlight-16B-A3B-Instruct": "moonshotai/Moonlight-16B-A3B-Instruct",
    "trinity-mini": "arcee-ai/Trinity-Mini",
    "arcee-ai/Trinity-Mini": "arcee-ai/Trinity-Mini",
    "glm-4.7-flash": "zai-org/GLM-4.7-Flash",
    "GLM-4.7-Flash": "zai-org/GLM-4.7-Flash",
    "nemotron": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "ling": "inclusionAI/Ling-mini-2.0",
    "Ling-mini-2.0": "inclusionAI/Ling-mini-2.0",
    "llada": "inclusionAI/LLaDA2.1-mini",
    "Llada2.1-mini": "inclusionAI/LLaDA2.1-mini",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "gpt": "openai/gpt-oss-20b",
    "olmoe": "allenai/OLMoE-1B-7B-0125-Instruct",
    "OLMoE-1B-7B-0125-Instruct": "allenai/OLMoE-1B-7B-0125-Instruct",
    "phimoe": "microsoft/Phi-3.5-MoE-instruct",
    "Phi-3.5-MoE-instruct": "microsoft/Phi-3.5-MoE-instruct",
    "phi-tiny": "microsoft/Phi-tiny-MoE-instruct",
    "Phi-tiny-MoE-instruct": "microsoft/Phi-tiny-MoE-instruct",
    "phi-mini": "microsoft/Phi-mini-MoE-instruct",
    "Phi-mini-MoE-instruct": "microsoft/Phi-mini-MoE-instruct",
    "granite": "ibm-granite/granite-4.0-h-tiny",
    "Granite-4.0-h-tiny": "ibm-granite/granite-4.0-h-tiny",
    "qwen35": "Qwen/Qwen3.5-35B-A3B",
    "Qwen3.5-35B-A3B": "Qwen/Qwen3.5-35B-A3B",
    "qwen3": "Qwen/Qwen3-30B-A3B",
    "Qwen3-30B-A3B": "Qwen/Qwen3-30B-A3B",
    "qwen3_30b": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "qwen3-30b-instruct-2507": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen3-30B-A3B-Instruct-2507": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "qwen3-moe-tiny": "PrimeIntellect/qwen3-moe-tiny",
    "sarvam-30b": "sarvamai/sarvam-30b",
    "ernie": "baidu/ERNIE-4.5-21B-A3B-PT",
    "baidu/ERNIE-4.5-21B-A3B-PT": "baidu/ERNIE-4.5-21B-A3B-PT",
    "laguna-xs2": "poolside/Laguna-XS.2",
    "poolside/Laguna XS.2": "poolside/Laguna-XS.2",
    "poolside/Laguna-XS.2": "poolside/Laguna-XS.2",
}


@dataclass(frozen=True)
class OptimizationConfig:
    attn_implementation: str | None = None
    torch_compile_mode: str | None = None
    torch_compile_fullgraph: bool = False
    torch_compile_dynamic: bool | None = None


def resolve_model_name(model_name: str) -> str:
    return MODEL_ALIASES.get(model_name, model_name)


def prepare_model_load(
    model_name: str,
    use_model_cache: bool = False,
    optimization_config: OptimizationConfig | None = None,
):
    model_name = resolve_model_name(model_name)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.use_cache = use_model_cache

    optimization_config = optimization_config or build_optimization_config(config.model_type)
    logger.info(
        "Using attention implementation `%s` for model type `%s`.",
        optimization_config.attn_implementation,
        config.model_type,
    )
    apply_model_patches(config.model_type, optimization_config, model_name=model_name)

    model_kwargs = {
        "dtype": torch.bfloat16,
        "attn_implementation": optimization_config.attn_implementation,
    }
    if optimization_config.attn_implementation and "/" in optimization_config.attn_implementation:
        model_kwargs["allow_all_kernels"] = True
    model_kwargs = {key: value for key, value in model_kwargs.items() if value is not None}

    kernel_context = (
        allow_all_hub_kernels()
        if optimization_config.attn_implementation and "/" in optimization_config.attn_implementation
        else nullcontext()
    )

    return model_name, config, optimization_config, model_kwargs, kernel_context


def finalize_loaded_model(
    model: torch.nn.Module,
    optimization_config: OptimizationConfig,
    use_model_cache: bool = False,
    fsdp: bool = False,
) -> torch.nn.Module:
    model.config.use_cache = use_model_cache
    applied_attn = getattr(model.config, "_attn_implementation", None)
    if optimization_config.attn_implementation and applied_attn != optimization_config.attn_implementation:
        logger.info(
            "Requested attention implementation `%s`, but loaded model reports `%s`.",
            optimization_config.attn_implementation,
            applied_attn,
        )

    if fsdp and hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
        actual_layer_cls = model.model.layers[0].__class__.__name__
        model._no_split_modules = [actual_layer_cls]

    model = model.to(torch.bfloat16)
    model = apply_torch_compile(model, optimization_config)
    model.train()
    return model


def build_optimization_config(model_type: str) -> OptimizationConfig:
    if model_type in FLASH_ATTN3_MODEL_TYPES:
        attn_implementation = FLASH_ATTN2_IMPLEMENTATION
    else:
        logger.info(
            "FlashAttention is not enabled for model type `%s`; no known optimized attention module is registered.",
            model_type,
        )
        attn_implementation = None

    return OptimizationConfig(attn_implementation=attn_implementation)


def apply_liger_kernel(model_type: str) -> bool:
    """Apply Liger's Transformers monkey patch for a supported model type."""
    if model_type not in MODEL_TYPE_TO_APPLY_LIGER_FN:
        logger.info("Liger kernels are not available for model type `%s`; continuing without Liger patches.", model_type)
        return False

    _apply_liger_kernel(model_type)
    logger.info("Applied Liger kernels for model type `%s`.", model_type)
    return True


def apply_torchtitan_moe(model_type: str, model_name: str | None = None) -> bool:
    """Patch supported MoE decoder layers to use TorchTitan grouped GEMM MoE."""
    if model_name in TORCHTITAN_INCOMPATIBLE_MODEL_IDS:
        logger.info(
            "TorchTitan MoE is not applied to `%s`: %s.",
            model_name,
            TORCHTITAN_INCOMPATIBLE_MODEL_IDS[model_name],
        )
        return False

    if model_type == "qwen3_moe":
        _apply_qwen3_torchtitan_moe()
    elif model_type == "glm4_moe":
        _apply_glm4_torchtitan_moe()
    else:
        logger.info(
            "TorchTitan MoE is not available for model type `%s`; supported model types are %s.",
            model_type,
            sorted(TORCHTITAN_MOE_MODEL_TYPES),
        )
        return False

    logger.info("Applied TorchTitan grouped-GEMM MoE for model type `%s`.", model_type)
    return True


def _apply_qwen3_torchtitan_moe() -> None:
    from transformers.models.qwen3_moe import modeling_qwen3_moe
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig, Qwen3MoeDecoderLayer

    def get_moe_args(config: Qwen3MoeConfig) -> MoEArgs:
        return MoEArgs(
            num_experts=config.num_experts,
            num_shared_experts=0,
            top_k=config.num_experts_per_tok,
            score_func="softmax",
            route_norm=config.norm_topk_prob,
            route_scale=1.0,
            score_before_experts=False,
            use_grouped_mm=True,
        )

    class Qwen3MoeTorchTitanDecoderLayer(Qwen3MoeDecoderLayer):
        def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
            super().__init__(config, layer_idx)
            if (layer_idx not in config.mlp_only_layers) and (
                config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
            ):
                self.mlp = MoE(
                    get_moe_args(config),
                    dim=config.hidden_size,
                    hidden_dim=config.moe_intermediate_size,
                )

    modeling_qwen3_moe.Qwen3MoeDecoderLayer = Qwen3MoeTorchTitanDecoderLayer


def _apply_glm4_torchtitan_moe() -> None:
    from transformers.models.glm4_moe import modeling_glm4_moe
    from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeConfig, Glm4MoeDecoderLayer

    def get_moe_args(config: Glm4MoeConfig) -> MoEArgs:
        return MoEArgs(
            num_experts=config.n_routed_experts,
            num_shared_experts=config.n_shared_experts,
            top_k=config.num_experts_per_tok,
            score_func="sigmoid",
            route_norm=config.norm_topk_prob,
            route_scale=config.routed_scaling_factor,
            score_before_experts=False,
            use_grouped_mm=True,
        )

    class Glm4MoeTorchTitanDecoderLayer(Glm4MoeDecoderLayer):
        def __init__(self, config: Glm4MoeConfig, layer_idx: int):
            super().__init__(config, layer_idx)
            if layer_idx >= config.first_k_dense_replace:
                self.mlp = MoE(
                    get_moe_args(config),
                    dim=config.hidden_size,
                    hidden_dim=config.moe_intermediate_size,
                )

    modeling_glm4_moe.Glm4MoeDecoderLayer = Glm4MoeTorchTitanDecoderLayer


def apply_model_patches(model_type: str, config: OptimizationConfig, model_name: str | None = None) -> None:
    apply_liger_kernel(model_type)
    apply_torchtitan_moe(model_type, model_name=model_name)


def apply_torch_compile(model: torch.nn.Module, config: OptimizationConfig) -> torch.nn.Module:
    """Compile decoder layers in-place, matching Lacuna's layer-level strategy."""
    if config.torch_compile_mode is None:
        logger.info("Torch compile is disabled.")
        return model

    torch._dynamo.config.cache_size_limit = 256
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.capture_scalar_outputs = True

    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ValueError("Torch compile was requested, but model.model.layers was not found.")

    for idx, layer in enumerate(layers):
        if hasattr(layer, "compile"):
            layer.compile(
                mode=config.torch_compile_mode,
                fullgraph=config.torch_compile_fullgraph,
                dynamic=config.torch_compile_dynamic,
            )
        else:
            layers[idx] = torch.compile(
                layer,
                mode=config.torch_compile_mode,
                fullgraph=config.torch_compile_fullgraph,
                dynamic=config.torch_compile_dynamic,
            )

    logger.info("Applied torch.compile to %d decoder layers with mode `%s`.", len(layers), config.torch_compile_mode)
    return model
