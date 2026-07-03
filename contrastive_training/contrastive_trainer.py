# from llamafactory.train.pt.trainer import CustomTrainer
# from trl import SFT_Trainer
from transformers import Trainer
from trl import SFTTrainer
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import re
from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from packed_forward import get_packed_forward, select_packed_target_positions
from parallel_dataset import PACKED_SEQUENCE_SOURCE, PACKED_SEQUENCE_TARGET

POSSIBLE_ROUTER_NAMES = [
    'mlp.router',
    'mlp.gate',
    'block_sparse_moe.gate',
    'block_sparse_moe.router',
]


class FreezableTrainerMixin:
    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[dict] = None) -> None:
        if getattr(getattr(self.model, "config", None), "model_type", None) != "gpt_oss":
            return super()._save(output_dir=output_dir, state_dict=state_dict)

        original_save_pretrained = self.model.save_pretrained

        def save_pretrained_in_transformers_format(*args, **kwargs):
            # GPT-OSS has a Transformers save bug when converting back to the original checkpoint format.
            kwargs["save_original_format"] = False
            return original_save_pretrained(*args, **kwargs)

        self.model.save_pretrained = save_pretrained_in_transformers_format
        try:
            return super()._save(output_dir=output_dir, state_dict=state_dict)
        finally:
            delattr(self.model, "save_pretrained")

    def print_trainable_params(self):
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def configure_router_only_training(self, max_layer, multi_gpu=False):
        """Freeze everything except MoE gate/router linear layers up to max_layer."""
        for name, param in self.model.named_parameters():
            # Prevent FSDP crash: Root FSDP module must have trainable parameters to unshard correctly
            if multi_gpu and any(k in name for k in ["embed_tokens", "norm", "lm_head"]):
                param.requires_grad = True
                continue

            is_router_param = any(identifier in name for identifier in POSSIBLE_ROUTER_NAMES)
            param.requires_grad = is_router_param and not self.is_beyond_max_layer(name, max_layer)

        self.print_trainable_params()

    def configure_early_layer_only_training(self, max_layer, multi_gpu=False):
        for name, param in self.model.named_parameters():
            param.requires_grad = self.is_before_max_layer(name, max_layer, multi_gpu)

        self.print_trainable_params()

    def is_before_max_layer(self, param_name, max_layer, multi_gpu=False):
        ## before *inclusive of* max_layer
        # Prevent FSDP crash: Root-level parameters must remain trainable
        # if multi_gpu and any(k in param_name for k in ["embed_tokens", "norm", "lm_head"]):
        #     return True

        match = re.search(r'\.layers\.(\d+)\.', param_name)
        if match:
            layer_idx = int(match.group(1))
            if layer_idx < max_layer:
                return True
            elif layer_idx == max_layer:
                # TODO: freeze everything that is the router and before
                if any(identifier in param_name for identifier in POSSIBLE_ROUTER_NAMES):
                    return True
                elif "att" in param_name: # TODO: THIS SURELY DOESN't cover everything
                    return True
        return False

    def is_beyond_max_layer(self, param_name, max_layer):
        match = re.search(r'\.layers\.(\d+)\.', param_name)
        return bool(match and int(match.group(1)) >= max_layer)


@dataclass
class RouterForwardOutput:
    router_logits: torch.Tensor  # [num_layers, batch, seq, experts]
    logits: Optional[torch.Tensor]


def standardize_router_logits(
    raw_router_logits,
    batch_size: int,
    seq_len: int,
    attention_mask: torch.Tensor,
    min_layer: int,
    max_layer: int,
) -> torch.Tensor:
    """Normalize model-specific router outputs to [layers, batch, seq, experts]."""
    if raw_router_logits is None:
        raise ValueError(
            "Model did not return router_logits. Ensure the model supports "
            "output_router_logits=True or add a model-specific adapter."
        )

    selected_router_logits = raw_router_logits[min_layer - 1:max_layer]
    mask = attention_mask.unsqueeze(-1)

    if isinstance(selected_router_logits, torch.Tensor):
        router_logits = selected_router_logits

        if router_logits.dim() == 3:
            router_logits = router_logits.unsqueeze(0)

        if router_logits.dim() != 4:
            raise ValueError(f"Expected router logits with 3 or 4 dims, got {router_logits.shape}.")

        if router_logits.shape[1] != batch_size or router_logits.shape[2] != seq_len:
            router_logits = router_logits.reshape(router_logits.shape[0], batch_size, seq_len, -1)

        return router_logits * mask.unsqueeze(0)
    else:
        router_layers = []
        for layer_router_logits in selected_router_logits:
            if layer_router_logits.dim() == 2:
                layer_router_logits = layer_router_logits.reshape(batch_size, seq_len, -1)
            elif layer_router_logits.dim() == 3:
                layer_router_logits = layer_router_logits.reshape(batch_size, seq_len, -1)
            else:
                raise ValueError(f"Unexpected router-logit shape: {layer_router_logits.shape}.")

            router_layers.append(layer_router_logits * mask)

        if not router_layers:
            raise ValueError(
                f"No router logits found for layers [{min_layer}, {max_layer}]. "
                "Check that the model exposes enough MoE layers."
            )

        return torch.stack(router_layers, dim=0)


def patch_granite_router_logits(model):
    if getattr(model, "_contrastive_granite_router_patch", False):
        return

    for layer in model.model.layers:
        moe = layer.block_sparse_moe
        original_forward = moe.forward

        def forward_with_router_logits(self, layer_input):
            bsz, length, emb_size = layer_input.size()
            layer_input = layer_input.reshape(-1, emb_size)
            _, batch_index, batch_gates, expert_size, router_logits = self.router(layer_input)
            self.router_logits = router_logits

            expert_inputs = layer_input[batch_index]
            hidden_states = self.input_linear(expert_inputs, expert_size)
            hidden_states = self.activation(hidden_states[..., : self.hidden_size]) * hidden_states[..., self.hidden_size :]
            expert_outputs = self.output_linear(hidden_states, expert_size)
            expert_outputs = expert_outputs * batch_gates[:, None]

            zeros = torch.zeros((bsz * length, self.input_size), dtype=expert_outputs.dtype, device=expert_outputs.device)
            layer_output = zeros.index_add(0, batch_index, expert_outputs)
            return layer_output.view(bsz, length, self.input_size)

        moe._contrastive_original_forward = original_forward
        moe.forward = MethodType(forward_with_router_logits, moe)

    model._contrastive_granite_router_patch = True


def patch_qwen3_moe_router_logits(model):
    if getattr(model, "_contrastive_qwen3_moe_router_patch", False):
        return

    for layer in model.model.layers:
        moe = getattr(layer, "mlp", None)
        if not (hasattr(moe, "gate") and hasattr(moe, "experts")):
            continue

        original_forward = moe.forward

        def forward_with_router_logits(self, hidden_states):
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
            router_logits, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
            self.router_logits = router_logits
            final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
            return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        moe._contrastive_original_forward = original_forward
        moe.forward = MethodType(forward_with_router_logits, moe)

    model._contrastive_qwen3_moe_router_patch = True


def contrastive_loss_fn(
        router_logits_tgt, mask_tgt,
        router_logits_src, mask_src,
        scoring_func,
        token_aggregation=1
    ):
    """
    assumes router_logits_* is of shape (batch_size, sequence_length, num_experts)


    ways of implementing this:
    1. average over all tokens in the sequence; compare expert importance across sequences
    2. last routing weight
    3. last N tokens
    4. weighted averaging ?
    X. for each target token, find 1?3? minimum distance tokens in source tokens (requires O(n^2) div calculations) 
    """

    def apply_scoring(x):
        if scoring_func == "softmax":
            return F.softmax(x, dim=-1)
        elif scoring_func == "sigmoid":
            return torch.sigmoid(x)
        return x

    def _single_layer_loss(single_router_logits_tgt, single_router_logits_src):
        single_router_logits_tgt = apply_scoring(single_router_logits_tgt)
        single_router_logits_src = apply_scoring(single_router_logits_src)

        def masked_mean(x, mask):
            # calculates mean for each item in a batch when each is potentially masked (bc its just padding)
            # expects x to be of shape (batch_size, sequence_length, num_of_experts,)
            # expects mask to be of shape (batch_size, sequence_length, )
            mask = mask.unsqueeze(-1)                  # [B, L, 1]
            masked_sum = (x * mask).sum(dim=1)         # [B, E]
            token_counts = mask.sum(dim=1).clamp(min=1)  # [B, 1]
            return masked_sum / token_counts           # [B, E]
        
        def get_last_non_padded(x, mask):
            # arange is 1-indexed so that index 0 is not ignored if it's the only 1
            seq_lens = mask.shape[1]
            positions = torch.arange(1, seq_lens + 1, device=mask.device)
            last_indices = (mask * positions).argmax(dim=1) # [B]
            
            B = x.shape[0]
            return x[torch.arange(B, device=x.device), last_indices, :]

        if token_aggregation == 1:
            # mean aggregation of normalized (non-sparsified weights) across non-padded tokens
            agg_routing_tgt = masked_mean(single_router_logits_tgt, mask_tgt)
            agg_routing_src = masked_mean(single_router_logits_src, mask_src)
        elif token_aggregation == 2:
            # only last non-padded token
            agg_routing_tgt = get_last_non_padded(single_router_logits_tgt, mask_tgt)
            agg_routing_src = get_last_non_padded(single_router_logits_src, mask_src)
        elif token_aggregation == 3:
            # last N tokens, TODO: make sure this takes care of mask
            N = 10
            agg_routing_tgt = single_router_logits_tgt[:, -N:, :].mean(dim=1)
            agg_routing_src = single_router_logits_src[:, -N:, :].mean(dim=1)
        elif token_aggregation == 4:
            # TODO: use sparsified mean, not all-expert all-token mean
            agg_routing_tgt = single_router_logits_tgt.mean(dim=1) # dummy
            agg_routing_src = single_router_logits_src.mean(dim=1) # dummy

        # Normalize to valid probability distributions instead of re-applying softmax
        agg_routing_tgt = agg_routing_tgt / (agg_routing_tgt.sum(dim=-1, keepdim=True) + 1e-9)
        agg_routing_src = agg_routing_src / (agg_routing_src.sum(dim=-1, keepdim=True) + 1e-9)

        # agg_routing_* should be of shape (batch_size, num_experts)
        # kl_div expects input in log-space and target as probabilities
        kl_div_loss = F.kl_div(torch.log(agg_routing_tgt + 1e-9),
                               agg_routing_src, reduction='batchmean')
        return kl_div_loss

    if isinstance(router_logits_tgt, (tuple, list)):
        layer_losses = [
            _single_layer_loss(tgt_layer, src_layer)
            for tgt_layer, src_layer in zip(router_logits_tgt, router_logits_src)
        ]
        return torch.stack(layer_losses).mean()

    if isinstance(router_logits_tgt, torch.Tensor) and router_logits_tgt.dim() == 4:
        # Shape: [num_layers, batch, seq, experts]
        layer_losses = [
            _single_layer_loss(router_logits_tgt[i], router_logits_src[i])
            for i in range(router_logits_tgt.size(0))
        ]
        return torch.stack(layer_losses).mean()

    return _single_layer_loss(router_logits_tgt, router_logits_src)
    
    
    # elif token_aggregation == 4:


def packed_contrastive_loss_fn(
        router_logits,
        sample_ids,
        sequence_type,
        scoring_func,
        token_aggregation=1,
    ):
    """Compute the same mean-token routing KL loss from packed token rows."""
    if token_aggregation != 1:
        raise ValueError("Packed contrastive loss currently supports only mean token aggregation.")

    if router_logits.dim() == 4:
        if router_logits.size(1) != 1:
            raise ValueError(f"Packed router logits should have batch size 1, got {router_logits.shape}.")
        router_logits = router_logits.squeeze(1)
    elif router_logits.dim() == 2:
        router_logits = router_logits.unsqueeze(0)
    elif router_logits.dim() != 3:
        raise ValueError(f"Expected packed router logits with 2, 3, or 4 dims, got {router_logits.shape}.")

    sample_ids = sample_ids.reshape(-1).long()
    sequence_type = sequence_type.reshape(-1).long()
    valid_tokens = sample_ids.ge(0)
    if not valid_tokens.any():
        return router_logits.sum() * 0.0

    num_samples = int(sample_ids[valid_tokens].max().item()) + 1

    def apply_scoring(x):
        if scoring_func == "softmax":
            return F.softmax(x, dim=-1)
        elif scoring_func == "sigmoid":
            return torch.sigmoid(x)
        return x

    def aggregate_by_sample(token_scores, token_mask):
        num_layers, _, experts = token_scores.shape
        sums = token_scores.new_zeros((num_layers, num_samples, experts))
        counts = token_scores.new_zeros((num_samples, 1))
        if token_mask.any():
            indices = sample_ids[token_mask]
            expanded_indices = indices.view(1, -1, 1).expand(num_layers, -1, experts)
            sums.scatter_add_(1, expanded_indices, token_scores[:, token_mask, :])
            counts.index_add_(
                0,
                indices,
                torch.ones((indices.numel(), 1), device=token_scores.device, dtype=token_scores.dtype),
            )
        return sums, counts

    target_mask = valid_tokens & sequence_type.eq(PACKED_SEQUENCE_TARGET)
    source_mask = valid_tokens & sequence_type.eq(PACKED_SEQUENCE_SOURCE)
    token_scores = apply_scoring(router_logits)
    target_sums, target_counts = aggregate_by_sample(token_scores, target_mask)
    source_sums, source_counts = aggregate_by_sample(token_scores, source_mask)
    has_pair = target_counts.squeeze(-1).gt(0) & source_counts.squeeze(-1).gt(0)
    if not has_pair.any():
        return token_scores.sum() * 0.0

    agg_routing_tgt = target_sums[:, has_pair, :] / target_counts[has_pair].view(1, -1, 1).clamp(min=1)
    agg_routing_src = source_sums[:, has_pair, :] / source_counts[has_pair].view(1, -1, 1).clamp(min=1)
    agg_routing_tgt = agg_routing_tgt / (agg_routing_tgt.sum(dim=-1, keepdim=True) + 1e-9)
    agg_routing_src = agg_routing_src / (agg_routing_src.sum(dim=-1, keepdim=True) + 1e-9)
    per_sample_kl = F.kl_div(torch.log(agg_routing_tgt + 1e-9), agg_routing_src, reduction='none').sum(dim=-1)
    return per_sample_kl.mean(dim=1).mean()


def _is_packed_batch(inputs: dict[str, Union[torch.Tensor, Any]]) -> bool:
    packed = inputs.get("packed", False)
    if isinstance(packed, torch.Tensor):
        return bool(packed.item())
    return bool(packed)


def packed_target_lm_loss(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    lm_head: torch.nn.Module,
    loss_fct: torch.nn.Module,
    logits_scaling: float = 1.0,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Project only packed positions that contribute to target LM loss."""
    shift_hidden_states = hidden_states[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    valid_lm_tokens = shift_labels.ne(-100)
    if not valid_lm_tokens.any():
        return shift_hidden_states.sum() * 0.0, None, shift_labels[valid_lm_tokens]

    selected_hidden_states = shift_hidden_states[valid_lm_tokens]
    selected_labels = shift_labels[valid_lm_tokens]
    selected_logits = lm_head(selected_hidden_states)
    selected_logits = selected_logits / logits_scaling
    lm_loss = loss_fct(selected_logits.float(), selected_labels)
    return lm_loss, selected_logits, selected_labels


def packed_target_lm_loss_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_fct: torch.nn.Module,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Compute packed target LM loss from logits already produced by model.forward."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    valid_lm_tokens = shift_labels.ne(-100)
    if not valid_lm_tokens.any():
        return shift_logits.sum() * 0.0, None, shift_labels[valid_lm_tokens]

    selected_logits = shift_logits[valid_lm_tokens]
    selected_labels = shift_labels[valid_lm_tokens]
    lm_loss = loss_fct(selected_logits.float(), selected_labels)
    return lm_loss, selected_logits, selected_labels

class TargetLMTrainer(FreezableTrainerMixin, Trainer):
    pass


class TranslationSFTTrainer(FreezableTrainerMixin, SFTTrainer):
    pass


class ContrastiveLMTrainer(FreezableTrainerMixin, Trainer):
    def __init__(
        self,
        *args,
        min_layer,
        max_layer,
        alpha_contrastive=1.0,
        scoring_func="softmax",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.min_layer = int(min_layer)
        self.max_layer = int(max_layer)
        self.key_layer = self.min_layer
        self.alpha_contrastive = alpha_contrastive
        self.scoring_func = scoring_func
        self.eval_loss_key = "lm_loss_tgt"
        self.lm_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100) # Standard LM loss

        if self.model.config.model_type == "granitemoehybrid":
            patch_granite_router_logits(self.model)
        elif self.model.config.model_type == "qwen3_moe":
            patch_qwen3_moe_router_logits(self.model)

        # This trainer computes a custom mean-reduced loss and does not use
        # num_items_in_batch. Without this override, packed batches' `labels`
        # key makes Trainer report loss multiplied by gradient accumulation.
        self.model_accepts_loss_kwargs = False

    def lm_loss(self, shift_logits, shift_labels):
        valid_labels = shift_labels.ne(-100)
        if not valid_labels.any():
            return shift_logits.sum() * 0.0
        return self.lm_loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
    
    def get_routing_logits(self, model, input_ids, attention_mask):
        ## need to implement for model's without output_router_logits option
        ## https://github.com/hiyouga/LlamaFactory/blob/main/src/llamafactory/model/model_utils/moe.py
        if model.config.model_type == "granitemoehybrid":
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ) ## NOTE: since doing LM loss, need to do full forward pass
            raw_router_logits = tuple(layer.block_sparse_moe.router_logits for layer in model.model.layers)
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_router_logits=True,
            ) ## NOTE: since doing LM loss, need to do full forward pass
            raw_router_logits = outputs.router_logits

        B, L = input_ids.shape
        router_logits = standardize_router_logits(
            raw_router_logits,
            batch_size=B,
            seq_len=L,
            attention_mask=attention_mask,
            min_layer=self.min_layer,
            max_layer=self.max_layer,
        )
        return RouterForwardOutput(router_logits=router_logits, logits=outputs.logits)

    def compute_packed_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ):
        router_outputs = get_packed_forward(
            model,
            inputs,
            self.min_layer,
            self.max_layer,
            use_split_forward=not getattr(self, "is_fsdp_enabled", False),
        )
        logits_scaling = getattr(model.config, "logits_scaling", 1.0)
        target_labels = select_packed_target_positions(inputs["labels"], inputs)
        if router_outputs.logits is not None:
            lm_loss_tgt, logits_tgt, lm_labels_tgt = packed_target_lm_loss_from_logits(
                router_outputs.logits,
                target_labels,
                self.lm_loss_fct,
            )
        else:
            lm_loss_tgt, logits_tgt, lm_labels_tgt = packed_target_lm_loss(
                router_outputs.hidden_states,
                target_labels,
                model.lm_head,
                self.lm_loss_fct,
                logits_scaling=logits_scaling,
            )

        contrastive_loss_val = packed_contrastive_loss_fn(
            router_outputs.router_logits,
            inputs["sample_ids"],
            inputs["sequence_type"],
            self.scoring_func,
        )
        total_loss = lm_loss_tgt + 200 * self.alpha_contrastive * contrastive_loss_val

        if return_outputs:
            log_outputs = {
                "logits_tgt": logits_tgt,
                "labels_tgt": lm_labels_tgt,
                "logits_src": None,
                "lm_loss_tgt": lm_loss_tgt.detach(),
                "contrastive_loss_val": contrastive_loss_val.detach(),
                "total_loss_computed": total_loss.detach(),
            }
            return total_loss, log_outputs
        return total_loss

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        if _is_packed_batch(inputs):
            return self.compute_packed_loss(model, inputs, return_outputs=return_outputs)

        # Concatenate along the batch dimension for a single efficient forward pass
        combined_input_ids = torch.cat([inputs['input_ids_tgt'], inputs['input_ids_src']], dim=0)
        combined_attention_mask = torch.cat([inputs['attention_mask_tgt'], inputs['attention_mask_src']], dim=0)
        
        router_outputs = self.get_routing_logits(model, combined_input_ids, combined_attention_mask)
        combined_layer_router_logits = router_outputs.router_logits
        combined_logits = router_outputs.logits
        
        # Split back into target and source
        batch_size = inputs['input_ids_tgt'].size(0)
        masked_router_logits_tgt, masked_router_logits_src = combined_layer_router_logits.split(batch_size, dim=1)
        logits_tgt = combined_logits[:batch_size]
        
        # Clone the views to prevent inplace backward modification errors from FSDP
        masked_router_logits_tgt = masked_router_logits_tgt.clone()
        masked_router_logits_src = masked_router_logits_src.clone()
        if getattr(self, "is_fsdp_enabled", False):
            logits_tgt = logits_tgt.clone()


        shift_logits_tgt = logits_tgt[..., :-1, :].contiguous()
        labels_tgt = inputs.get('labels_tgt', inputs['input_ids_tgt'])
        shift_labels_tgt = labels_tgt[..., 1:].contiguous()
        lm_loss_tgt = self.lm_loss(shift_logits_tgt, shift_labels_tgt)

        contrastive_loss_val = contrastive_loss_fn(
            masked_router_logits_tgt,
            inputs['attention_mask_tgt'],
            masked_router_logits_src,
            inputs['attention_mask_src'],
            self.scoring_func
        )
        # 200 because contrastive_loss_val is typically in ~0.05 and lm_loss_tgt is typically in range of ~10-15
        total_loss = lm_loss_tgt + 200 * self.alpha_contrastive * contrastive_loss_val
        
        if return_outputs:
            log_outputs = {
                "logits_tgt": logits_tgt,
                "logits_src": combined_logits[batch_size:].detach(), # detach for now since we aren't using it to save vram
                "lm_loss_tgt": lm_loss_tgt.detach(),
                "contrastive_loss_val": contrastive_loss_val.detach(),
                "total_loss_computed": total_loss.detach(),
            }
            return total_loss, log_outputs
        return total_loss
    
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            # eval_loss tracks LM-only loss for ContrastiveLMTrainer; training still optimizes the mixed objective.
            _, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = outputs[self.eval_loss_key].detach()
            
            # Extract the specific parts for evaluation
            # We focus on the target language for the 'logits' and 'labels'
            logits = outputs.get("logits_tgt")
            if _is_packed_batch(inputs):
                labels = outputs.get("labels_tgt")
            else:
                labels = inputs.get("labels_tgt", inputs.get("input_ids_tgt"))

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits, labels)

class ContrastiveTrainer(ContrastiveLMTrainer):
    def __init__(self, *args, **kwargs):
        self.training_type = 'full'
        super().__init__(*args, **kwargs)
        self.eval_loss_key = "contrastive_loss_val"
        
    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        ## NOTE: since only doing contrastive loss, don't need to do full forward pass
        ## logic taken care of by PartialMoEModelForCausalLM class in modeling.py

        # Concatenate along the batch dimension for a single efficient forward pass
        combined_input_ids = torch.cat([inputs['input_ids_tgt'], inputs['input_ids_src']], dim=0)
        combined_attention_mask = torch.cat([inputs['attention_mask_tgt'], inputs['attention_mask_src']], dim=0)
        
        router_outputs = self.get_routing_logits(model, combined_input_ids, combined_attention_mask)
        combined_router_logits = router_outputs.router_logits
        
        batch_size = inputs['input_ids_tgt'].size(0)
        masked_router_logits_tgt, masked_router_logits_src = combined_router_logits.split(batch_size, dim=1)
        
        # Clone the views to prevent inplace backward modification errors from FSDP
        masked_router_logits_tgt = masked_router_logits_tgt.clone()
        masked_router_logits_src = masked_router_logits_src.clone()

        contrastive_loss_val = contrastive_loss_fn(
            masked_router_logits_tgt,
            inputs['attention_mask_tgt'],
            masked_router_logits_src,
            inputs['attention_mask_src'],
            self.scoring_func
        )
        if return_outputs:
            # Ensure consistent output structure for logging in prediction_step.
            log_outputs = {
                "lm_loss_tgt": contrastive_loss_val.new_zeros(()),
                "contrastive_loss_val": contrastive_loss_val.detach(),
                "total_loss_computed": contrastive_loss_val.detach(),
            }
            return contrastive_loss_val, log_outputs
        return contrastive_loss_val
