# from llamafactory.train.pt.trainer import CustomTrainer
# from trl import SFT_Trainer
from transformers import Trainer
from datasets import Dataset
from parallel_dataset import ParallelDataCollator
import torch
import torch.nn.functional as F
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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

class ContrastiveLMTrainer(Trainer):
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
        self.lm_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100) # Standard LM loss
    
    def get_routing_logits(self, model, input_ids, attention_mask):
        ## need to implement for model's without output_router_logits option
        ## https://github.com/hiyouga/LlamaFactory/blob/main/src/llamafactory/model/model_utils/moe.py
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_router_logits=True,
        ) ## NOTE: since doing LM loss, need to do full forward pass

        B, L = input_ids.shape
        selected_router_logits = []
        for layer_router_logits in outputs.router_logits[self.min_layer - 1:self.max_layer]:
            selected_router_logits.append(
                layer_router_logits.view(B, L, -1) * attention_mask.unsqueeze(-1)
            )
        if not selected_router_logits:
            raise ValueError(
                f"No router logits found for layers [{self.min_layer}, {self.max_layer}]. "
                "Check that the model exposes enough MoE layers."
            )
        return torch.stack(selected_router_logits, dim=0), outputs.logits

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):

        # Concatenate along the batch dimension for a single efficient forward pass
        combined_input_ids = torch.cat([inputs['input_ids_tgt'], inputs['input_ids_src']], dim=0)
        combined_attention_mask = torch.cat([inputs['attention_mask_tgt'], inputs['attention_mask_src']], dim=0)
        
        combined_layer_router_logits, combined_logits = self.get_routing_logits(model, combined_input_ids, combined_attention_mask)
        
        # Split back into target and source
        batch_size = inputs['input_ids_tgt'].size(0)
        masked_router_logits_tgt, masked_router_logits_src = combined_layer_router_logits.split(batch_size, dim=1)
        logits_tgt, logits_src = combined_logits.split(batch_size, dim=0)
        
        # Clone the views to prevent inplace backward modification errors from FSDP
        masked_router_logits_tgt = masked_router_logits_tgt.clone()
        masked_router_logits_src = masked_router_logits_src.clone()
        logits_tgt = logits_tgt.clone()
        logits_src = logits_src.clone()


        shift_logits_tgt = logits_tgt[..., :-1, :].contiguous()
        shift_labels_tgt = inputs['input_ids_tgt'][..., 1:].contiguous()
        lm_loss_tgt = self.lm_loss_fct(shift_logits_tgt.view(-1, shift_logits_tgt.size(-1)), shift_labels_tgt.view(-1))

        contrastive_loss_val = contrastive_loss_fn(
            masked_router_logits_tgt,
            inputs['attention_mask_tgt'],
            masked_router_logits_src,
            inputs['attention_mask_src'],
            self.scoring_func
        )
        # 200 because contrastive_loss_val is typically in ~0.05 and lm_loss_tgt is typically in range of ~10-15
        total_loss = lm_loss_tgt + 200 * self.alpha_contrastive * contrastive_loss_val
        
        log_outputs = {
            "logits_tgt": logits_tgt,
            "logits_src": logits_src,
            "lm_loss_tgt": lm_loss_tgt.item(), # Convert to scalar for logging
            "contrastive_loss_val": contrastive_loss_val.item(), # Convert to scalar for logging
            "total_loss_computed": total_loss.item() # Add total loss to outputs for logging
        }
        return (total_loss, log_outputs) if return_outputs else total_loss
    
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            # We call compute_loss with return_outputs=True to get our split logits
            # This ensures the eval loss is calculated exactly like the training loss
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.detach()

            # self.log({
            #     "eval_lm_loss_tgt": outputs["lm_loss_tgt"],
            #     "eval_contrastive_loss_val": outputs["contrastive_loss_val"],
            #     "eval_total_loss_computed": outputs["total_loss_computed"]
            # })
            
            # Extract the specific parts for evaluation
            # We focus on the target language for the 'logits' and 'labels'
            logits = outputs.get("logits_tgt")
            labels = inputs.get("input_ids_tgt")

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits, labels)

    def configure_router_only_training(self, max_layer, multi_gpu=False):
        """Freeze everything except MoE gate/router linear layers up to max_layer."""
        # this happens before Trainer wraps model for distributed training, so is safe (and more convenient) to use self.model
        
        # TODO: make sure all names of the router are captured here
        possible_names = ['mlp.router', 'block_sparse_moe.gate', 'mlp.gate']
        
        for name, param in self.model.named_parameters():
            # Prevent FSDP crash: Root FSDP module must have trainable parameters to unshard correctly
            if multi_gpu and any(k in name for k in ["embed_tokens", "norm", "lm_head"]):
                param.requires_grad = True
                continue
                
            if any(identifier in name for identifier in possible_names):
                # Check if the parameter belongs to a layer beyond max_layer
                if self.is_beyond_max_layer(name, max_layer):
                    continue
            param.requires_grad = False
        
        self.print_trainable_params()
    
    def configure_early_layer_only_training(self, max_layer, multi_gpu=False):
        for name, param in self.model.named_parameters():
            # Check if the parameter belongs to a layer beyond max_layer
            param.requires_grad = self.is_before_max_layer(name, max_layer, multi_gpu)
        
        self.print_trainable_params()

    def is_before_max_layer(self, param_name, max_layer, multi_gpu=False):
        ## before *inclusive of* max_layer
        # Prevent FSDP crash: Root-level parameters must remain trainable
        if multi_gpu and any(k in param_name for k in ["embed_tokens", "norm", "lm_head"]):
            return True
            
        match = re.search(r'\.layers\.(\d+)\.', param_name)
        if match:
            layer_idx = int(match.group(1))
            if layer_idx <= max_layer:
                return True
        return False

    def print_trainable_params(self):
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
class ContrastiveTrainer(ContrastiveLMTrainer):
    def __init__(self, *args, **kwargs):
        self.training_type = 'full'
        super().__init__(*args, **kwargs)
        
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
        
        combined_router_logits, _ = self.get_routing_logits(model, combined_input_ids, combined_attention_mask)
        
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
        # Ensure consistent output structure for logging in prediction_step
        log_outputs = {
            "lm_loss_tgt": 0.0, # No LM loss in ContrastiveTrainer
            "contrastive_loss_val": contrastive_loss_val.item(),
            "total_loss_computed": contrastive_loss_val.item() # Total loss is just contrastive loss here
        }
        return (contrastive_loss_val, log_outputs) if return_outputs else contrastive_loss_val
