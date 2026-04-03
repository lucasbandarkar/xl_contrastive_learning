import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
# from .modeling.modeling_kimi import KimiLinearForCausalLM

class MoeModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if ("llada" in model_name.lower()) or ("Kimi" in model_name):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if ("deepseek" in model_name) or ("llada" in model_name.lower()):
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True)
        # elif "Kimi" in model_name:
        #     ### haven't gotten this to work
        #     self.model = KimiLinearForCausalLM.from_pretrained(
        #         model_name, torch_dtype="auto", device_map="auto", 
        #         trust_remote_code=True, attn_implementation="kernels-community/flash-attn2"
        #     )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        
        if "Kimi" in model_name:
            self.num_experts_activated = self.model.config.num_experts_per_token
        else:
            self.num_experts_activated = self.model.config.num_experts_per_tok

    def get_routings(self, messages):
        """
        Get the router choices for a given set of messages.
        Args:
            model: The Mixtral model.
            tokenizer: The tokenizer for the model.
            messages: A list of messages to be processed.
        Returns:
            router_choices: The router choices for the given messages.
        """
        model = self.model
        tokenizer = self.tokenizer
        if transformers.__version__.startswith("5"):
            # apply_chat_template returns BatchEncoding object, not tensor in transformers v5
            input_ids = tokenizer.apply_chat_template(
                messages, 
                enable_thinking=False,
                tokenize=True,
                return_tensors="pt"
            ).input_ids.to("cuda")
        else:
            input_ids = tokenizer.apply_chat_template(messages, enable_thinking=False, return_tensors="pt").to("cuda")

        with torch.no_grad():
            # generation = model.generate(input_ids, max_new_tokens=5, do_sample=False)
            outputs = model(input_ids, output_router_logits=True, return_dict=True)
            router_logits = torch.stack([rl for rl in outputs["router_logits"]], dim=0)  # torch.Size([32, 70, 8])
            return_dict = {
                "router_logits": router_logits.float().detach().cpu().numpy(),  # torch.Size([32, 70, 8]) ~ (layer, token, expert)
                # "router_choices": torch.argmax(router_logits, dim=-1).float().detach().cpu().numpy(),  # torch.Size([32, 70]) ~ (layer, token)
                "messages": messages,
                "input_ids": input_ids.detach().cpu().numpy(),
                "input_ids_decoded": [tokenizer.decode(x, skip_special_tokens=False) for i, x in enumerate(input_ids[0])],
                "input_ids_decoded_verbose": [(i, tokenizer.decode(x, skip_special_tokens=False)) for i, x in enumerate(input_ids[0])],
                # "outputs": outputs,
                # "generation": tokenizer.decode(generation[0], skip_special_tokens=True),
            }
        return return_dict

    def get_assistant_routings_only(self, chat_item):
        """
        (this method has not been tested yet)
        Assumes chat format: [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        """
        full_routing = self.get_routings(chat_item)
        
        # Find assistant token boundary
        user_part_only = [msg for msg in chat_item if msg["role"] == "user"]
        user_input_ids = self.tokenizer.apply_chat_template(user_part_only, enable_thinking=False, return_tensors="pt")
        assistant_start_idx = user_input_ids.shape[1]
        
        # Slice to get only assistant tokens
        router_logits = full_routing["router_logits"][:, assistant_start_idx:, :]
        input_ids = full_routing["input_ids"][:, assistant_start_idx:]
        
        return {
            "router_logits": router_logits,
            "messages": chat_item,
            "input_ids": input_ids,
            "input_ids_decoded": [self.tokenizer.decode(x, skip_special_tokens=False) for x in input_ids[0]],
            "input_ids_decoded_verbose": [(i, self.tokenizer.decode(x, skip_special_tokens=False)) for i, x in enumerate(input_ids[0])],
            "assistant_start_idx": assistant_start_idx,
        }