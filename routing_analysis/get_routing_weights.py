import torch
import torch.nn.functional as F
import datasets
from tqdm import tqdm
import numpy as np
import json, gc, os, re, shutil
from itertools import combinations
from pathlib import Path
import sys
import models
from utils import MGSMINSTRUCT_LANGUAGES
from argparse import ArgumentParser

""" 
INSTRUCTIONS FOR GATED REPOS !!!
if you're getting an error related to a gated repo on HF that you don't have access for, you need to:
1. visit the model page on HF and accept the waiver once logged in
2. Wait for the request to be officially approved
3. Create a HF user access token (or use a pre-existing one) and copy the value
4. Then below fill in the line HF_TOKEN="<hf_token_value>"
"""
HF_TOKEN = ""
## 15 for now, add more later
FLORES_LANGUAGES = ["eng_Latn", "bam_Latn", "fra_Latn", "tha_Thai", "pes_Arab", "arb_Arab", "arb_Latn", "ary_Arab", "hin_Deva", "zho_Hans", "srp_Cyrl", "lit_Latn", "ory_Orya", "ben_Beng", "asm_Beng"]
EXPERIMENTAL_LANGUAGES = ["eng_Latn", "kan_Knda", "kir_Cyrl", "ben_Beng", "tel_Telu", "sin_Sinh", "hun_Latn", "vie_Latn", "pes_Arab", ]
NICKNAME_TO_MODEL_MAP = {
    "qwen3_30b": "Qwen/Qwen3-30B-A3B",
    "olmoe": "allenai/OLMoE-1B-7B-0125-Instruct",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "llama4": "meta-llama/Llama-4-Scout-17B-16E-Instruct", # WASNT WORKING (conda issue ?)
    "phimoe": "microsoft/Phi-3.5-MoE-instruct",
    "moonlight": "moonshotai/Moonlight-16B-A3B-Instruct", # WASNT WORKING (too complicated to get Deepseek remote code to work)
    "gpt": "openai/gpt-oss-20b",
    "qwen35": "Qwen/Qwen3.5-35B-A3B",
    "nemotron": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8", # transformers doesn't have output_router_logits implemented
    "kimi": "moonshotai/Kimi-Linear-48B-A3B-Instruct", # doesn't have output_router_logits enabled
    "llada": "inclusionAI/LLaDA2.1-mini",
    "ling": "inclusionAI/Ling-mini-2.0",
    "phi-tiny": "microsoft/Phi-tiny-MoE-instruct",
    "ernie": "baidu/ERNIE-4.5-21B-A3B-PT",
    "granite": "ibm-granite/granite-4.0-h-tiny",
    "marco": "AIDC-AI/Marco-Mini-Global-Base",
    "marco_nano": "AIDC-AI/Marco-Nano-Instruct"
}

"""
MAJOR WARNING:
In moe env, transformers 4.52.4
Here, output_router_logits for Qwen3 returns pre-softmax logits, as indicated below
However, in transformers 5.3.0 Qwen3&3.5 returns post-softmax logits
If using such a version, you need to set scoring_func to "none" in MODEL_CONFIGS for Qwen3&3.5
"""
MODEL_CONFIGS = {
    "qwen3_30b": [48, 128, 8, {}], # [num_layers, num_experts, num_experts_active_per_tok, special_gating_function_params]
    "olmoe": [16, 64, 8, {}],
    "mixtral": [32, 8, 2],
    "llama4": [48, 16, 1],
    "phimoe": [32, 16, 2],
    "moonlight": [27, 64, 6],
    "gpt": [24, 32, 4, {}],
    "qwen35": [40, 256, 8, {"scoring_func": "none"}], # router logits are already post-softmax
    "nemotron": [52, 128, 6],
    "kimi": [26, 256, 8, {"scoring_func": "sigmoid"}], # might break as it has 1 dense layer + 26 moe
    "llada": [20, 256, 8, {"scoring_func": "sigmoid"}], # might break as it has 1 dense layer + 20 moe
    "ling": [20, 256, 8, {"scoring_func": "sigmoid"}],
    "phi-tiny": [32, 16, 2],
    "ernie": [28, 64, 6, {}], # has one dense layer
    "granite": [40, 64, 6, {}],
    "marco": [28, 256, 8, {}],
    "marco_nano": [28, 232, 8, {}]
}
DATA_FOLDER = "/data2/lucasbandarkar/moe/collected_routing_data/"

REPO_ROOT = Path(__file__).resolve().parents[1]
LANGUAGE_EVALS_DIR = REPO_ROOT / "language_evals"
if str(LANGUAGE_EVALS_DIR) not in sys.path:
    sys.path.append(str(LANGUAGE_EVALS_DIR))

class RunnerByDataset:
    def __init__(self, language_codes):
        self.langcodes = language_codes
        self.load_datasets()
    
    def load_datasets(self):
        pass
    def get_model_outputs(self, num_samples):
        pass

class FloresRunner(RunnerByDataset):
    def __init__(self, language_codes):
        self.dataset_name = "flores"
        super().__init__(language_codes)

    def load_datasets(self):
        # Load all datasets into a dictionary
        self.datasets_dict = {}
        for lang_code in self.langcodes:
            short_lang_key = lang_code.split('_')[0]
            ds = datasets.load_dataset("facebook/belebele", name=lang_code, split='test')
            df = ds.to_pandas()

            # Drop rows with the same 'link' (flores passage identifier)
            # Keep the first occurrence. This means we'll only process each unique passage once per language.
            df_unique_passages = df.drop_duplicates(subset=['link']).copy()

            # Sort the unique passages by 'link' for consistent ordering
            df_sorted = df_unique_passages.sort_values(by='link').reset_index(drop=True)
            self.datasets_dict[short_lang_key] = df_sorted
            print(f"  - {lang_code}: {len(df_sorted)} unique passages after deduplication.")
        
        self.N = len(list(self.datasets_dict.values())[0])
        # return datasets_dict

    def get_model_outputs(self, model, num_samples):
        outputs = {lang.split('_')[0]: [] for lang in self.datasets_dict.keys()}

        if num_samples < self.N:
            samples = np.random.choice(range(self.N), num_samples, replace=False)
        else:
            samples = range(self.N)
        # Runs all parallel datasets
        for i in tqdm(samples):
            if i == 181: # this sample always causes OOM, likely in Bambara ?
                continue
            try:
                current_outputs = {}
                for lang_key, dataset in self.datasets_dict.items():
                    msg = [{"role": "user", "content": dataset.iloc[i]['flores_passage']}]
                    output = model.get_routings(msg)
                    current_outputs[lang_key] = output
                
                # add all at once because of risk of OOMs; want all outputs lists to be index-aligned
                for lang_key, output in current_outputs.items():
                    outputs[lang_key].append(output)
                    
            except torch.cuda.OutOfMemoryError as e:
                print(f"CUDA OOM at index {i} for {lang_key}, skipping this sample for all languages")
                torch.cuda.empty_cache() # attempts to remove data from GPU to enable the loop to keep going
                del msg
                del output
                gc.collect()
                torch.cuda.empty_cache()
        return outputs
    
class MGSMInstructRunner(RunnerByDataset):
    def __init__(self, language_codes):
        self.dataset_name = "mgsm_ins"
        super().__init__(language_codes)
        
    def load_datasets(self):
        mgsminstruct = datasets.load_dataset("Mathoctopus/GSM8KInstruct_Parallel", split="train")

        def detect_language(entry):
            extracted_language = entry['prompt'].split()[17][:-1]
            return MGSMINSTRUCT_LANGUAGES[extracted_language]
        
        mgsminstruct = mgsminstruct.map(lambda x: {"lang": detect_language(x)})

        self.datasets_dict = {lang: [] for lang in MGSMINSTRUCT_LANGUAGES.values()}
        self.datasets_dict.pop("te", None) # Telugu only has one sample ??
        counter = 0
        new_text_buffer = {}
        for example in mgsminstruct:
            lang = example["lang"]
            if lang == 'en':
                if counter == 9: # checking that a sample in each language existed
                    for lang, stored_example in new_text_buffer.items(): # flush new text buffer
                        self.datasets_dict[lang].append(stored_example)
                new_text_buffer = {'en': example}
                counter = 0
            elif lang == 'te':
                continue
            else:
                new_text_buffer[lang] = example
                counter += 1
        self.N = len(list(self.datasets_dict.values())[0])

    def get_model_outputs(self, model, num_samples):
        outputs = {lang: [] for lang in self.datasets_dict.keys()}

        if num_samples < self.N:
            np.random.seed(0)
            samples = np.random.choice(range(self.N), num_samples, replace=False)
        else:
            samples = range(self.N)
        
        for i in tqdm(samples):
            try:
                current_outputs = {}
                for lang_key, dataset in self.datasets_dict.items():
                    prompt = dataset[i]['prompt']
                    extracted_question = prompt.split('###')[1].split('\n')[1]
                    msg = [{"role": "user", "content": extracted_question},
                            {"role": "assistant", "content": dataset[i]['chosen']}]
                    output = model.get_routings(msg)
                    current_outputs[lang_key] = output
                
                # add all at once because of risk of OOMs; want all outputs lists to be index-aligned
                for lang_key, output in current_outputs.items():
                    outputs[lang_key].append(output)
                    
            except torch.cuda.OutOfMemoryError as e:
                print(f"CUDA OOM at index {i} for {lang_key}, skipping this sample for all languages")
                torch.cuda.empty_cache() # attempts to remove data from GPU to enable the loop to keep going
                del msg
                del output
                gc.collect()
                torch.cuda.empty_cache()
        return outputs
    
class MedInstructRunner(RunnerByDataset):
    def __init__(self, language_codes):
        self.dataset_name = "medinstruct"
        super().__init__("eng")

    def load_datasets(self):
        # Load all datasets into a dictionary
        self.datasets_dict = {}
        ds = datasets.load_dataset("lavita/AlpaCare-MedInstruct-52k", split='train')
        self.datasets_dict['eng'] = ds.to_pandas()
        self.N = len(list(self.datasets_dict.values())[0])
        # return datasets_dict

    def get_model_outputs(self, model, num_samples):
        outputs = {'eng': []}

        if num_samples < self.N:
            samples = np.random.choice(range(self.N), num_samples, replace=False)
        else:
            samples = range(self.N)
        # Runs all parallel datasets
        for i in tqdm(samples):
            try:
                current_outputs = {}
                for lang_key, dataset in self.datasets_dict.items():
                    msg = [{"role": "user", "content": dataset.iloc[i]['instruction']},
                            {"role": "assistant", "content": dataset.iloc[i]['output']}]
                    output = model.get_routings(msg)
                    current_outputs[lang_key] = output
                
                # add all at once because of risk of OOMs; want all outputs lists to be index-aligned
                for lang_key, output in current_outputs.items():
                    outputs[lang_key].append(output)
                    
            except torch.cuda.OutOfMemoryError as e:
                print(f"CUDA OOM at index {i} for {lang_key}, skipping this sample for all languages")
                torch.cuda.empty_cache() # attempts to remove data from GPU to enable the loop to keep going
                del msg
                del output
                gc.collect()
                torch.cuda.empty_cache()
        return outputs

def calculate_expert_importance_per_sequence(router_logits_by_lang, modelname):
    ''' (DEPRECATED) expert importance is average of routing weights across tokens in a sequence '''
    expert_importance = {}
    for lang,lang_router_weights in router_logits_by_lang.items():
        expert_importance[lang] = []
        for array in lang_router_weights:
            # Reduce along dimension 1 (the second dimension)
            if modelname == "qwen35":
                collapsed_tensor = torch.mean(torch.from_numpy(array), dim=1)
            else:
                collapsed_tensor = torch.mean(F.softmax(torch.from_numpy(array), dim=-1), dim=1)
            expert_importance[lang].append(collapsed_tensor)
    return expert_importance

def calculate_total_actual_weight_per_sequence(router_logits_by_lang, modelname):
    # scoring_func and renormalize are parameters of vllm's FusedMoE
    # https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/layer.py#L275
    topk_experts = MODEL_CONFIGS[modelname][2]
    scoring_func = "softmax" # defaults
    renormalize = True # defaults
    gate_configs = MODEL_CONFIGS[modelname][3]
    if "scoring_func" in gate_configs:
        scoring_func = gate_configs["scoring_func"]
    if "renormalize" in gate_configs:
        renormalize = False


    total_actual_weight = {}
    for lang,lang_router_weights in router_logits_by_lang.items():
        total_actual_weight[lang] = []
        for array in lang_router_weights:
            if scoring_func == "softmax": # for now most model's default
                post_activ = F.softmax(torch.from_numpy(array), dim=-1)
            elif scoring_func == "sigmoid":
                # for GLM-4.5-Air, Kimi, Llada
                post_activ = torch.sigmoid(torch.from_numpy(array))
            elif scoring_func == "none":
                post_activ = torch.from_numpy(array)                
            
            # We find the values of the k-th largest element to create a mask
            topk_values, _ = torch.topk(post_activ, k=topk_experts, dim=-1)
            # Get the threshold (the smallest value among the top-k)
            threshold = topk_values[..., -1].unsqueeze(-1)
            # Create a mask: True for top-k elements, False otherwise
            mask = post_activ >= threshold
            topk_discrete = post_activ * mask.to(post_activ.dtype)
            if renormalize:
                # Avoid division by zero for safety
                renormalized = topk_discrete / (topk_discrete.sum(dim=-1, keepdim=True) + 1e-20)
                collapsed_tensor = torch.mean(renormalized, dim=1)
            else:
                collapsed_tensor = torch.mean(topk_discrete, dim=1)
            total_actual_weight[lang].append(collapsed_tensor)
    return total_actual_weight

def get_last_token_routing(router_logits_by_lang):
    last_token_routing = {}
    for lang,lang_router_weights in router_logits_by_lang.items():
        last_token_routing[lang] = []
        for array in lang_router_weights:
            # Reduce along dimension 1 (the second dimension)
            last_token_routing[lang].append(torch.from_numpy(array)[:, -1, :])
    return last_token_routing

def calculate_activation_counts(router_logits_by_lang, modelname):
    activation_counts = {}
    for lang,lang_router_weights in router_logits_by_lang.items():
        activation_counts[lang] = []
        for array in lang_router_weights:
            tensor = torch.from_numpy(array)
            experts_dim = -1
            topk_experts = MODEL_CONFIGS[modelname][2]
            _, top_indices = torch.topk(tensor, k=topk_experts, dim=experts_dim)

            # Create binary tensor with zeros and ones
            activations = torch.zeros_like(tensor).scatter_(experts_dim, top_indices, 1)
            collapsed_tensor = torch.sum(activations, dim=1)
            activation_counts[lang].append(collapsed_tensor)
    return activation_counts

def dump_data_to_json(data_dict, preexisting_results, numlangs, dataset_name, filename_suffix, prefix, original_filepath=None):
    if preexisting_results:
        json_compatible_data = preexisting_results
    else:
        json_compatible_data = {}
    for key, list_of_arrays in data_dict.items():
        converted_list = []
        for array in list_of_arrays:
            # .tolist() converts a NumPy array (of any dimension) into a nested Python list
            converted_list.append(array.tolist())
        json_compatible_data[key] = converted_list

    fname = f"{prefix}_{dataset_name}_{filename_suffix}_{numlangs}langs.json"
    output_filepath = os.path.join(DATA_FOLDER, fname)
    try:
        with open(output_filepath, 'w') as f:
            json.dump(json_compatible_data, f, indent=4) # indent for pretty printing
        print(f"\nSuccessfully dumped data to {output_filepath}")
    except Exception as e:
        print(f"Error dumping to JSON: {e}")

def fetch_preexisting_results(dataset_name, model_nickname, mode_prefix):
    # Look for files matching the pattern "expert_importance_flores_qwen3_30b_15langs.json"
    pattern = re.compile(rf"{mode_prefix}_{dataset_name}_{re.escape(model_nickname)}_(\d+)langs\.json")
    
    for fname in os.listdir(DATA_FOLDER):
        match = pattern.match(fname)
        if match:
            num_existing_langs = int(match.group(1))
            print(f"Found existing results file: {fname} with {num_existing_langs} languages.")
            original_filepath = os.path.join(DATA_FOLDER, fname)
            try:
                with open(original_filepath, 'r') as f:
                    existing_results = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {original_filepath}: {e}")
                existing_results = {} # Reset if file is corrupt
                original_filepath = None # Don't consider this file as valid
            except Exception as e:
                print(f"Error reading {original_filepath}: {e}")
                existing_results = {}
                original_filepath = None
            return existing_results

def get_arguments(args):
    ''' parses command line arguments '''
    langcodes = ['eng']
    if args.dataset == 'flores':
        if args.language_codes:
            langcodes = [code.strip() for code in args.language_codes.split(',') if code.strip()]
        elif args.languages:
            langcodes = EXPERIMENTAL_LANGUAGES[:args.languages]
        else:
            langcodes = EXPERIMENTAL_LANGUAGES
    elif args.dataset == 'mgsminstruct':
        if args.languages:
            langcodes = list(MGSMINSTRUCT_LANGUAGES.values())[:args.languages]
        else:
            langcodes = list(MGSMINSTRUCT_LANGUAGES.values())

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    mode_map = {
        0: "expert_importance",
        1: "last_token_routing",
        2: "activation_counts",
        3: "total_actual_weight",
    }
    mode = mode_map[args.mode]
    return langcodes, mode

def get_model_name(args):
    base_model = NICKNAME_TO_MODEL_MAP[args.nickname]
    if not args.trained_ckpt:
        return base_model, None

    from export_fsdp_checkpoint import resolve_fsdp_checkpoint

    fsdp_result = resolve_fsdp_checkpoint(args.trained_ckpt, base_model=base_model)
    checkpoint_path = fsdp_result.model_path
    cleanup_path = checkpoint_path if fsdp_result.created_export else None
    print(f"Loading trained checkpoint from {checkpoint_path}")
    return checkpoint_path, cleanup_path

def get_output_suffix(args):
    if not args.trained_ckpt:
        return args.nickname

    checkpoint_path = Path(args.trained_ckpt).resolve()
    if checkpoint_path.name.startswith("checkpoint-"):
        suffix = f"{checkpoint_path.parent.name}_{checkpoint_path.name}"
    else:
        suffix = checkpoint_path.name
    return f"{args.nickname}_{suffix}"

def main(args):
    language_codes, mode = get_arguments(args)
    model_name, cleanup_path = get_model_name(args)
    output_suffix = get_output_suffix(args)

    # check if requested data has already been collected for certain languages
    preexisting_results = fetch_preexisting_results(args.dataset, output_suffix, mode)
    if preexisting_results:
        evaluated_languages = set(preexisting_results.keys())
        print(f"Languages already evaluated: {evaluated_languages}")
        language_codes_to_evaluate = [lang for lang in language_codes if lang[:3] not in evaluated_languages]
    else:
        language_codes_to_evaluate = language_codes

    if len(language_codes_to_evaluate) == 0:
        print("All languages already evaluated. No new evaluations needed.")
        return
    
    print(f"Languages to evaluate: {language_codes_to_evaluate}")
    try:
        model = models.MoeModel(model_name=model_name)
        if args.dataset == 'flores':
            runner = FloresRunner(language_codes_to_evaluate)
        elif args.dataset == 'medinstruct':
            runner = MedInstructRunner(['eng'])
        else:
            runner = MGSMInstructRunner(language_codes_to_evaluate)

        raw_outputs = runner.get_model_outputs(model, 500)
        router_weights = {}
        for lang,lang_outputs in raw_outputs.items():
            router_weights[lang] = [lang_outputs[i]['router_logits'] for i in range(len(lang_outputs))]

        if mode == "expert_importance":
            ### initial expert_importance deprecated, but still kept for investigating old data
            data = calculate_total_actual_weight_per_sequence(router_weights, args.nickname)
        elif mode == "last_token_routing":
            data = get_last_token_routing(router_weights)
        elif mode == "activation_counts":
            data = calculate_activation_counts(router_weights, args.nickname)
        elif mode == "total_actual_weight":
            data = calculate_total_actual_weight_per_sequence(router_weights, args.nickname)

        dump_data_to_json(data, preexisting_results, len(language_codes), args.dataset, output_suffix, mode)
    finally:
        if cleanup_path:
            print(f"Cleaning up model directory: {cleanup_path}")
            shutil.rmtree(cleanup_path, ignore_errors=True)


## Example call: python get_routing_weights.py -m mixtral -g 0,1 -t 0 -d mgsminstruct
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-m', '--nickname', type=str, required=True, help="the nickname of the model with which to name files and select model configs")
    parser.add_argument('-l', '--languages', type=int, help="the number of languages to evaluate")
    parser.add_argument('--language-codes', type=str, default=None, help="comma-separated language codes to evaluate, e.g. eng_Latn,kan_Knda")
    parser.add_argument('-g', '--gpus', type=str, default="6,7", help="the comma-separated list of gpus to evaluate on")
    parser.add_argument('-t', '--mode', type=int, help="code for data to collect 0: expert importance, 1: last token, 2: activation counts, 3: total_actual_weight")
    parser.add_argument('-d', '--dataset', type=str, default='flores', help="flores (max 15 langs) or mgsminstruct (max 10 langs)")
    parser.add_argument('--trained_ckpt', type=str, default=None, help="contrastive_training output dir or checkpoint-NNN dir to load instead of the base HF model")
    args = parser.parse_args()
    main(args)
    
