from dataclasses import dataclass
from types import MethodType
from typing import Any, Union

import torch

from parallel_dataset import PACKED_SEQUENCE_TARGET


@dataclass
class PackedForwardOutput:
    router_logits: torch.Tensor
    logits: torch.Tensor


def tensor_to_int(value: Union[torch.Tensor, int]) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def target_token_count(inputs: dict[str, Union[torch.Tensor, Any]]) -> int:
    if "target_token_count" in inputs:
        return tensor_to_int(inputs["target_token_count"])
    return int(inputs["sequence_type"].eq(PACKED_SEQUENCE_TARGET).sum().item())


def select_packed_target_tokens(
    tensor: torch.Tensor,
    inputs: dict[str, Union[torch.Tensor, Any]],
) -> torch.Tensor:
    if "target_token_count" in inputs:
        return tensor[:, : target_token_count(inputs)]
    return tensor[inputs["sequence_type"].eq(PACKED_SEQUENCE_TARGET)].view(1, -1, tensor.size(-1))


def select_packed_target_positions(
    tensor: torch.Tensor,
    inputs: dict[str, Union[torch.Tensor, Any]],
) -> torch.Tensor:
    if "target_token_count" in inputs:
        return tensor[:, : target_token_count(inputs)]
    return tensor[inputs["sequence_type"].eq(PACKED_SEQUENCE_TARGET)].view(1, -1)


def normalize_packed_router_logits(raw_router_logits, num_tokens: int, min_layer: int, max_layer: int) -> torch.Tensor:
    if raw_router_logits is None:
        raise ValueError(
            "Model did not return router_logits. Ensure the model supports "
            "output_router_logits=True or add a model-specific packed adapter."
        )

    selected_router_logits = raw_router_logits[min_layer - 1:max_layer]
    router_layers = []

    if isinstance(selected_router_logits, torch.Tensor):
        router_logits = selected_router_logits
        if router_logits.dim() == 4 and router_logits.size(1) == 1:
            router_logits = router_logits.squeeze(1)
        elif router_logits.dim() == 2:
            router_logits = router_logits.unsqueeze(0)
        if router_logits.dim() != 3:
            raise ValueError(f"Expected packed router logits with shape [layers, tokens, experts], got {router_logits.shape}.")
        if router_logits.size(1) != num_tokens:
            router_logits = router_logits.reshape(router_logits.size(0), num_tokens, -1)
        return router_logits

    for layer_router_logits in selected_router_logits:
        if layer_router_logits.dim() == 3 and layer_router_logits.size(0) == 1:
            layer_router_logits = layer_router_logits.squeeze(0)
        elif layer_router_logits.dim() == 3:
            layer_router_logits = layer_router_logits.reshape(num_tokens, -1)
        elif layer_router_logits.dim() != 2:
            raise ValueError(f"Unexpected packed router-logit shape: {layer_router_logits.shape}.")

        if layer_router_logits.size(0) != num_tokens:
            layer_router_logits = layer_router_logits.reshape(num_tokens, -1)
        router_layers.append(layer_router_logits)

    if not router_layers:
        raise ValueError(
            f"No router logits found for layers [{min_layer}, {max_layer}]. "
            "Check that the model exposes enough MoE layers."
        )
    return torch.stack(router_layers, dim=0)


PACKED_FORWARD_KEYS = (
    "input_ids",
    "position_ids",
    "seq_idx",
    "cu_seq_lens_q",
    "cu_seq_lens_k",
    "max_length_q",
    "max_length_k",
    "target_cu_seq_lens_q",
    "target_cu_seq_lens_k",
    "target_max_length_q",
    "target_max_length_k",
    "target_token_count",
)


def get_model_config(model):
    for candidate in (model, getattr(model, "module", None), getattr(model, "_fsdp_wrapped_module", None)):
        config = getattr(candidate, "config", None)
        if config is not None:
            return config
    return None


def get_packed_forward(model, inputs: dict[str, Union[torch.Tensor, Any]], min_layer: int, max_layer: int) -> PackedForwardOutput:
    config = get_model_config(model)
    attn_implementation = getattr(config, "_attn_implementation", None)
    if attn_implementation != "flash_attention_2":
        raise ValueError(
            "Packed training requires flash_attention_2 so cu_seq_lens isolate sequences. "
            f"Current attention implementation: {attn_implementation!r}."
        )

    packed_kwargs = {key: inputs[key] for key in PACKED_FORWARD_KEYS if key in inputs}
    outputs = model(
        **packed_kwargs,
        attention_mask=None,
        use_cache=False,
        output_router_logits=True,
        contrastive_packed_forward=True,
        contrastive_min_layer=min_layer,
        contrastive_max_layer=max_layer,
    )
    if not isinstance(outputs, PackedForwardOutput):
        raise TypeError("Packed model forward must return PackedForwardOutput.")
    return outputs


def apply_packed_forward_patch(model: torch.nn.Module) -> torch.nn.Module:
    if getattr(model, "_contrastive_packed_forward_patch", False):
        return model

    original_forward = model.forward

    def forward_with_packed(self, *args, contrastive_packed_forward=False, contrastive_min_layer=None, contrastive_max_layer=None, **kwargs):
        if not contrastive_packed_forward:
            return original_forward(*args, **kwargs)
        if args:
            raise ValueError("Packed split-forward expects keyword inputs.")
        return packed_split_forward(self, original_forward, kwargs, int(contrastive_min_layer), int(contrastive_max_layer))

    model._contrastive_original_forward = original_forward
    model.forward = MethodType(forward_with_packed, model)
    model._contrastive_packed_forward_patch = True
    return model


def packed_split_forward(
    model: torch.nn.Module,
    original_forward,
    inputs: dict[str, Union[torch.Tensor, Any]],
    min_layer: int,
    max_layer: int,
) -> PackedForwardOutput:
    if model.config.model_type == "granitemoehybrid":
        target_hidden_states = granite_packed_split_forward(model, inputs, max_layer)
        raw_router_logits = tuple(layer.block_sparse_moe.router_logits for layer in model.model.layers[:max_layer])
        logits = target_logits(model, target_hidden_states)
    elif model.config.model_type == "qwen3_moe":
        target_hidden_states, raw_router_logits = qwen3_moe_packed_split_forward(model, inputs, max_layer)
        logits = target_logits(model, target_hidden_states)
    else:
        outputs = original_forward(
            input_ids=inputs["input_ids"],
            position_ids=inputs["position_ids"],
            attention_mask=None,
            use_cache=False,
            output_router_logits=True,
            cu_seq_lens_q=inputs["cu_seq_lens_q"],
            cu_seq_lens_k=inputs["cu_seq_lens_k"],
            max_length_q=tensor_to_int(inputs["max_length_q"]),
            max_length_k=tensor_to_int(inputs["max_length_k"]),
        )
        logits = select_packed_target_tokens(outputs.logits, inputs)
        raw_router_logits = getattr(outputs, "router_logits", None)

    router_logits = normalize_packed_router_logits(
        raw_router_logits,
        num_tokens=inputs["input_ids"].size(1),
        min_layer=min_layer,
        max_layer=max_layer,
    )
    return PackedForwardOutput(router_logits=router_logits, logits=logits)


def target_logits(model: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    logits = model.lm_head(hidden_states)
    return logits / getattr(model.config, "logits_scaling", 1.0)


def qwen3_moe_sparse_layer_indices(layers) -> list[int]:
    return [
        layer_idx
        for layer_idx, layer in enumerate(layers)
        if hasattr(getattr(layer, "mlp", None), "gate") and hasattr(getattr(layer, "mlp", None), "experts")
    ]


def qwen3_moe_packed_split_forward(model, inputs: dict[str, Union[torch.Tensor, Any]], max_layer: int):
    backbone = model.model
    layers = backbone.layers
    sparse_layer_indices = qwen3_moe_sparse_layer_indices(layers)
    if max_layer > len(sparse_layer_indices):
        raise ValueError(f"max_layer={max_layer} exceeds Qwen3 MoE sparse layer count {len(sparse_layer_indices)}.")
    split_layer_exclusive = sparse_layer_indices[max_layer - 1] + 1

    hidden_states = backbone.embed_tokens(inputs["input_ids"])
    position_ids = inputs["position_ids"]
    cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
    position_embeddings = backbone.rotary_emb(hidden_states, position_ids=position_ids)

    all_token_kwargs = {
        "cu_seq_lens_q": inputs["cu_seq_lens_q"],
        "cu_seq_lens_k": inputs["cu_seq_lens_k"],
        "max_length_q": tensor_to_int(inputs["max_length_q"]),
        "max_length_k": tensor_to_int(inputs["max_length_k"]),
    }
    for decoder_layer in layers[:split_layer_exclusive]:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **all_token_kwargs,
        )

    raw_router_logits = tuple(
        layer.mlp.router_logits
        for layer in layers[:split_layer_exclusive]
        if hasattr(getattr(layer, "mlp", None), "router_logits")
    )

    target_hidden_states = select_packed_target_tokens(hidden_states, inputs)
    target_position_ids = select_packed_target_positions(inputs["position_ids"], inputs)
    target_cache_position = torch.arange(target_hidden_states.shape[1], device=target_hidden_states.device)
    target_position_embeddings = backbone.rotary_emb(target_hidden_states, position_ids=target_position_ids)

    target_kwargs = {
        "cu_seq_lens_q": inputs["target_cu_seq_lens_q"],
        "cu_seq_lens_k": inputs["target_cu_seq_lens_k"],
        "max_length_q": tensor_to_int(inputs["target_max_length_q"]),
        "max_length_k": tensor_to_int(inputs["target_max_length_k"]),
    }
    for decoder_layer in layers[split_layer_exclusive:]:
        target_hidden_states = decoder_layer(
            target_hidden_states,
            attention_mask=None,
            position_ids=target_position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=target_cache_position,
            position_embeddings=target_position_embeddings,
            **target_kwargs,
        )

    return backbone.norm(target_hidden_states), raw_router_logits


def granite_packed_split_forward(model, inputs: dict[str, Union[torch.Tensor, Any]], max_layer: int):
    backbone = model.model
    layers = backbone.layers
    if max_layer > len(layers):
        raise ValueError(f"max_layer={max_layer} exceeds model layer count {len(layers)}.")

    hidden_states = backbone.embed_tokens(inputs["input_ids"])
    hidden_states = hidden_states * backbone.embedding_multiplier
    position_ids = inputs["position_ids"]
    cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
    position_embeddings = backbone.rotary_emb(hidden_states, position_ids) if backbone.rotary_emb is not None else None

    all_token_kwargs = {
        "seq_idx": inputs["seq_idx"],
        "cu_seq_lens_q": inputs["cu_seq_lens_q"],
        "cu_seq_lens_k": inputs["cu_seq_lens_k"],
        "max_length_q": tensor_to_int(inputs["max_length_q"]),
        "max_length_k": tensor_to_int(inputs["max_length_k"]),
    }
    for decoder_layer in layers[:max_layer]:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **all_token_kwargs,
        )

    target_hidden_states = select_packed_target_tokens(hidden_states, inputs)
    target_position_ids = select_packed_target_positions(inputs["position_ids"], inputs)
    target_seq_idx = select_packed_target_positions(inputs["seq_idx"], inputs)
    target_cache_position = torch.arange(target_hidden_states.shape[1], device=target_hidden_states.device)
    target_position_embeddings = (
        backbone.rotary_emb(target_hidden_states, target_position_ids)
        if backbone.rotary_emb is not None
        else None
    )

    target_kwargs = {
        "seq_idx": target_seq_idx,
        "cu_seq_lens_q": inputs["target_cu_seq_lens_q"],
        "cu_seq_lens_k": inputs["target_cu_seq_lens_k"],
        "max_length_q": tensor_to_int(inputs["target_max_length_q"]),
        "max_length_k": tensor_to_int(inputs["target_max_length_k"]),
    }
    for decoder_layer in layers[max_layer:]:
        target_hidden_states = decoder_layer(
            target_hidden_states,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            cache_position=target_cache_position,
            position_embeddings=target_position_embeddings,
            **target_kwargs,
        )

    return backbone.norm(target_hidden_states)
