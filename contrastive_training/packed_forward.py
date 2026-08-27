from dataclasses import dataclass
from types import MethodType
from typing import Any, Optional, Union

import torch

from parallel_dataset import PACKED_SEQUENCE_TARGET
from parallel_dataset import PACKED_SEQUENCE_SOURCE


@dataclass
class PackedForwardOutput:
    router_logits: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    contrastive_hidden_states: Optional[torch.Tensor] = None


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


def packed_sequence_hidden_means(
    hidden_states: torch.Tensor,
    inputs: dict[str, Union[torch.Tensor, Any]],
) -> torch.Tensor:
    """Mean-pool packed hidden states by source/target type and sample id.

    Returns [2, samples, hidden], where index 0 is source and index 1 is target.
    This keeps the contrastive path from retaining token-level hidden activations
    for every selected layer.
    """
    if hidden_states.dim() != 3:
        raise ValueError(f"Expected packed hidden states with shape [1, tokens, hidden], got {hidden_states.shape}.")

    token_hidden = hidden_states.reshape(-1, hidden_states.size(-1))
    sample_ids = inputs["sample_ids"].reshape(-1).long()
    sequence_type = inputs["sequence_type"].reshape(-1).long()
    if token_hidden.size(0) != sample_ids.numel():
        raise ValueError(
            f"Packed hidden/token metadata mismatch: {token_hidden.size(0)} hidden rows "
            f"but {sample_ids.numel()} sample ids."
        )

    valid_tokens = sample_ids.ge(0)
    if not valid_tokens.any():
        return token_hidden.new_zeros((2, 0, token_hidden.size(-1)))

    num_samples = int(sample_ids[valid_tokens].max().item()) + 1
    sums = token_hidden.new_zeros((2, num_samples, token_hidden.size(-1)))
    counts = token_hidden.new_zeros((2, num_samples, 1))

    for seq_type in (PACKED_SEQUENCE_SOURCE, PACKED_SEQUENCE_TARGET):
        token_mask = valid_tokens & sequence_type.eq(seq_type)
        if not token_mask.any():
            continue

        indices = sample_ids[token_mask]
        sums[seq_type].index_add_(0, indices, token_hidden[token_mask])
        counts[seq_type].index_add_(
            0,
            indices,
            torch.ones((indices.numel(), 1), device=token_hidden.device, dtype=token_hidden.dtype),
        )

    return sums / counts.clamp(min=1)


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


def get_packed_forward(
    model,
    inputs: dict[str, Union[torch.Tensor, Any]],
    min_layer: int,
    max_layer: int,
    use_split_forward: bool = True,
    contrastive_space: str = "router",
) -> PackedForwardOutput:
    attn_implementation = getattr(model.config, "_attn_implementation", None)
    if attn_implementation != "flash_attention_2":
        raise ValueError(
            "Packed training requires flash_attention_2 so cu_seq_lens isolate sequences. "
            f"Current attention implementation: {attn_implementation!r}."
        )

    if contrastive_space not in {"router", "hidden"}:
        raise ValueError(f"Unsupported contrastive space: {contrastive_space!r}.")

    if contrastive_space == "hidden" and model.config.model_type != "qwen3_moe":
        raise ValueError("Packed hidden-state contrastive loss is currently implemented only for Qwen3 MoE.")

    if not use_split_forward:
        return packed_full_model_forward(model, inputs, min_layer, max_layer, contrastive_space=contrastive_space)

    if model.config.model_type == "granitemoehybrid":
        if contrastive_space == "hidden":
            raise ValueError("Packed hidden-state contrastive loss is currently implemented only for Qwen3 MoE.")
        hidden_states = granite_packed_split_forward(model, inputs, max_layer)
        raw_router_logits = tuple(layer.block_sparse_moe.router_logits for layer in model.model.layers[:max_layer])
    elif model.config.model_type == "qwen3_moe":
        hidden_states, raw_router_logits, contrastive_hidden_states = qwen3_moe_packed_split_forward(
            model,
            inputs,
            min_layer,
            max_layer,
            collect_hidden_states=contrastive_space == "hidden",
        )
    else:
        if contrastive_space == "hidden":
            raise ValueError("Packed hidden-state contrastive loss is currently implemented only for Qwen3 MoE.")
        outputs = model.model(
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
        hidden_states_all = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        hidden_states = select_packed_target_tokens(hidden_states_all, inputs)
        raw_router_logits = getattr(outputs, "router_logits", None)

    if contrastive_space == "hidden":
        return PackedForwardOutput(
            hidden_states=hidden_states,
            contrastive_hidden_states=contrastive_hidden_states,
        )

    router_logits = normalize_packed_router_logits(
        raw_router_logits,
        num_tokens=inputs["input_ids"].size(1),
        min_layer=min_layer,
        max_layer=max_layer,
    )
    return PackedForwardOutput(router_logits=router_logits, hidden_states=hidden_states)


def packed_full_model_forward(
    model,
    inputs: dict[str, Union[torch.Tensor, Any]],
    min_layer: int,
    max_layer: int,
    contrastive_space: str = "router",
) -> PackedForwardOutput:
    """Packed forward that enters through the top-level module, which FSDP can unshard."""
    common_kwargs = {
        "input_ids": inputs["input_ids"],
        "position_ids": inputs["position_ids"],
        "attention_mask": None,
        "use_cache": False,
        "cu_seq_lens_q": inputs["cu_seq_lens_q"],
        "cu_seq_lens_k": inputs["cu_seq_lens_k"],
        "max_length_q": tensor_to_int(inputs["max_length_q"]),
        "max_length_k": tensor_to_int(inputs["max_length_k"]),
    }

    if contrastive_space == "hidden":
        if model.config.model_type != "qwen3_moe":
            raise ValueError("Packed hidden-state contrastive loss is currently implemented only for Qwen3 MoE.")
        target_positions = torch.arange(target_token_count(inputs), device=inputs["input_ids"].device)
        backbone = prepare_qwen3_hidden_pooling_context(model, inputs, min_layer, max_layer)
        try:
            outputs = model(output_router_logits=False, logits_to_keep=target_positions, **common_kwargs)
            contrastive_hidden_states = getattr(backbone, "_contrastive_hidden_states", None)
        finally:
            clear_qwen3_hidden_pooling_context(backbone)

        if contrastive_hidden_states is None:
            raise RuntimeError("Qwen3 hidden pooling context did not record any hidden states.")
        if hasattr(backbone, "_contrastive_hidden_states"):
            delattr(backbone, "_contrastive_hidden_states")

        return PackedForwardOutput(
            logits=select_packed_target_tokens(outputs.logits, inputs),
            contrastive_hidden_states=contrastive_hidden_states,
        )

    if model.config.model_type == "granitemoehybrid":
        outputs = model(**common_kwargs)
        raw_router_logits = tuple(layer.block_sparse_moe.router_logits for layer in model.model.layers)
    else:
        target_positions = torch.arange(target_token_count(inputs), device=inputs["input_ids"].device)
        outputs = model(output_router_logits=True, logits_to_keep=target_positions, **common_kwargs)
        raw_router_logits = getattr(outputs, "router_logits", None)

    router_logits = normalize_packed_router_logits(
        raw_router_logits,
        num_tokens=inputs["input_ids"].size(1),
        min_layer=min_layer,
        max_layer=max_layer,
    )
    return PackedForwardOutput(
        router_logits=router_logits,
        logits=select_packed_target_tokens(outputs.logits, inputs),
    )


def qwen3_moe_sparse_layer_indices(layers) -> list[int]:
    sparse_layer_indices = []
    for layer_idx, layer in enumerate(layers):
        mlp = getattr(unwrap_module(layer), "mlp", None)
        if hasattr(mlp, "gate") and hasattr(mlp, "experts"):
            sparse_layer_indices.append(layer_idx)
    return sparse_layer_indices


def qwen3_moe_selected_sparse_layer_indices(layers, min_layer: int, max_layer: int) -> list[int]:
    sparse_layer_indices = qwen3_moe_sparse_layer_indices(layers)
    if min_layer < 1:
        raise ValueError(f"min_layer={min_layer} is invalid; Qwen3 MoE layers are one-indexed.")
    if max_layer > len(sparse_layer_indices):
        raise ValueError(f"max_layer={max_layer} exceeds Qwen3 MoE sparse layer count {len(sparse_layer_indices)}.")
    if min_layer > max_layer:
        raise ValueError(f"min_layer={min_layer} cannot be greater than max_layer={max_layer}.")
    return sparse_layer_indices[min_layer - 1:max_layer]


def unwrap_module(module):
    """Peel common wrapper attributes while keeping ordinary modules unchanged."""
    seen = set()
    while module is not None and id(module) not in seen:
        seen.add(id(module))
        for attr in ("module", "_fsdp_wrapped_module", "_orig_mod"):
            inner = getattr(module, attr, None)
            if inner is not None and inner is not module:
                module = inner
                break
        else:
            return module
    return module


def qwen3_moe_backbone(model):
    candidates = [
        getattr(model, "model", None),
        getattr(getattr(model, "module", None), "model", None),
        getattr(unwrap_module(model), "model", None),
    ]
    for candidate in candidates:
        candidate = unwrap_module(candidate)
        if hasattr(candidate, "layers") and hasattr(candidate, "embed_tokens"):
            return candidate
    raise ValueError("Could not locate the Qwen3 MoE backbone on the provided model.")


def patch_qwen3_moe_hidden_pooling_forward(model):
    backbone = qwen3_moe_backbone(model)
    if getattr(backbone, "_contrastive_qwen3_hidden_pooling_patch", False):
        return backbone

    original_forward = backbone.forward

    def forward_with_hidden_pooling(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        cache_position=None,
        **kwargs,
    ):
        pooling_context = getattr(self, "_contrastive_hidden_pooling_context", None)
        if pooling_context is None:
            return original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        from transformers.cache_utils import DynamicCache
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
        from transformers.modeling_outputs import MoeModelOutputWithPast

        return_dict = kwargs.pop("return_dict", getattr(self.config, "return_dict", True))
        use_cache = use_cache if use_cache is not None else getattr(self.config, "use_cache", False)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
        causal_mask = mask_function(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        selected_layer_indices = pooling_context["selected_layer_indices"]
        pooling_context["hidden_states"] = []

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_idx in selected_layer_indices:
                pooling_context["hidden_states"].append(
                    packed_sequence_hidden_means(hidden_states, pooling_context["inputs"])
                )

        hidden_states = self.norm(hidden_states)
        if not pooling_context["hidden_states"]:
            raise ValueError("No Qwen3 hidden states were collected for the requested MoE layer range.")
        self._contrastive_hidden_states = torch.stack(pooling_context["hidden_states"], dim=0)

        outputs = MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )
        return outputs if return_dict else outputs.to_tuple()

    backbone._contrastive_original_forward = original_forward
    backbone.forward = MethodType(forward_with_hidden_pooling, backbone)
    backbone._contrastive_qwen3_hidden_pooling_patch = True
    return backbone


def prepare_qwen3_hidden_pooling_context(
    model,
    inputs: dict[str, Union[torch.Tensor, Any]],
    min_layer: int,
    max_layer: int,
):
    backbone = patch_qwen3_moe_hidden_pooling_forward(model)
    selected_layer_indices = qwen3_moe_selected_sparse_layer_indices(backbone.layers, min_layer, max_layer)
    if hasattr(backbone, "_contrastive_hidden_states"):
        delattr(backbone, "_contrastive_hidden_states")
    backbone._contrastive_hidden_pooling_context = {
        "inputs": inputs,
        "selected_layer_indices": set(selected_layer_indices),
        "hidden_states": [],
    }
    return backbone


def clear_qwen3_hidden_pooling_context(backbone) -> None:
    if hasattr(backbone, "_contrastive_hidden_pooling_context"):
        delattr(backbone, "_contrastive_hidden_pooling_context")


def qwen3_moe_packed_split_forward(
    model,
    inputs: dict[str, Union[torch.Tensor, Any]],
    min_layer: int,
    max_layer: int,
    collect_hidden_states: bool = False,
):
    backbone = model.model
    layers = backbone.layers
    sparse_layer_indices = qwen3_moe_sparse_layer_indices(layers)
    selected_layer_indices = set(qwen3_moe_selected_sparse_layer_indices(layers, min_layer, max_layer))
    split_layer_exclusive = sparse_layer_indices[max_layer - 1] + 1
    contrastive_hidden_layers = []

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
    for current_decoder_idx, decoder_layer in enumerate(layers[:split_layer_exclusive]):
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
        # This loop runs through decoder layers, not just sparse MoE layers. The
        # index check maps the one-indexed CLI MoE-layer range onto Qwen3's
        # actual decoder-layer indices.
        if collect_hidden_states and current_decoder_idx in selected_layer_indices:
            contrastive_hidden_layers.append(packed_sequence_hidden_means(hidden_states, inputs))

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

    contrastive_hidden_states = None
    if collect_hidden_states:
        if not contrastive_hidden_layers:
            raise ValueError("No Qwen3 hidden states were collected for the requested MoE layer range.")
        contrastive_hidden_states = torch.stack(contrastive_hidden_layers, dim=0)

    return backbone.norm(target_hidden_states), raw_router_logits, contrastive_hidden_states


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
