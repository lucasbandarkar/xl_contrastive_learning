import sys, os
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from typing import List, Dict, Any

class ParallelDataCollator:
    # does it make sense to inherit from DPODataCollatorWithPadding ??
    def __init__(self, tokenizer: AutoTokenizer, src_language_key, tgt_language_key):
        self.tokenizer = tokenizer
        self.src_key = src_language_key
        self.tgt_key = tgt_language_key

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collates a list of features into a batch of tokenized inputs for two languages.

        Args:
            features (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
                                              represents a single data sample. Each sample
                                              is expected to have a 'translation' key,
                                              which is itself a dictionary containing
                                              'en' (English sentence) and 'fa' (Farsi sentence) keys.
                                              Example: `{'translation': {'en': 'Hello', 'fa': 'سلام'}}`
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the tokenized inputs for both source- and target-langauge sentences.
                                    The keys are structured as follows:
                                     - 'input_ids_src': Token IDs for source language (typically English) sentences.
                                     - 'attention_mask_src': Attention masks for source sentences.
                                     - 'input_ids_tgt': Token IDs for target language sentences.
                                     - 'attention_mask_tgt': Attention masks for target sentences.
        """
        # Extract both language's sentences from the batch features using the language keys passed into the constructor
        src_sentences = [feature["translation"][self.src_key] for feature in features]
        tgt_sentences = [feature["translation"][self.tgt_key] for feature in features]

        # Tokenize both together to ensure they are padded to the same maximum length.
        # This allows us to concatenate them along the batch dimension later.
        all_batch = self.tokenizer(
            src_sentences + tgt_sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        
        batch_size = len(features)

        # Combine the tokenized inputs for both languages into a single dictionary.
        # This dictionary will be the input that the training loop receives.
        batch = {
            "input_ids_src": all_batch["input_ids"][:batch_size],
            "attention_mask_src": all_batch["attention_mask"][:batch_size],
            "input_ids_tgt": all_batch["input_ids"][batch_size:],
            "attention_mask_tgt": all_batch["attention_mask"][batch_size:],
        }
        return batch
    
class ParallelInstructDataCollator:
    # same as above but with instruction data
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # TODO: build this for instruction data, that is each input has prompt and answer
        return batch

    
class ConcatenatedDataCollator:
    def __init__(self, tokenizer: AutoTokenizer, src_language_key, tgt_language_key):
        self.tokenizer = tokenizer
        self.src_key = src_language_key
        self.tgt_key = tgt_language_key

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract both language's sentences from the batch features using the language keys passed into the constructor
        src_sentences = [feature["translation"][self.src_key] for feature in features]
        tgt_sentences = [feature["translation"][self.tgt_key] for feature in features]

        # TODO: implement the data formatting for this baseline
        return batch


def load_parallel_datasets(dataset: str, language: str, data_limit=None):
    if dataset.lower() == "opus":
        OPUS_LANGUAGE_MAP = {
            "pes": "fa",
            "ben": "bn",
        }
        tgt_lang = OPUS_LANGUAGE_MAP[language]
        split_name = f"en-{tgt_lang}" if tgt_lang > "en" else f"{tgt_lang}-en"
        dataset = load_dataset("Helsinki-NLP/opus-100", split_name)

        if data_limit:
            train_limit = min(data_limit, len(dataset['train']))
            valid_limit = min(data_limit, len(dataset['validation']))
            return dataset['train'].select(range(train_limit)), dataset['validation'].select(range(valid_limit)), "en", tgt_lang
        return dataset['train'], dataset['validation'], "en", tgt_lang
    elif dataset.lower() == "math":
        ## combines mathoctopus' MGSM Instruct and MSVAMP
        # Add the project root to sys.path to enable absolute imports from project subdirectories
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..')) # Go up one level from 'contrastive_training' to 'moe'
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from expert_steering.mgsm_utils import MGSMINSTRUCT_LANGUAGES

        mgsminstruct = load_dataset("Mathoctopus/GSM8KInstruct_Parallel", split="train")

        def detect_language(entry):
            extracted_language = entry['prompt'].split()[17][:-1]
            return MGSMINSTRUCT_LANGUAGES[extracted_language]
        
        mgsminstruct = mgsminstruct.map(lambda x: {"lang": detect_language(x)})
        english_sample = None
        for example in mgsminstruct:
            lang = example["lang"]
            if lang == 'en':
                # English samples should come first
                english_sample = example
            if lang == language:
                tgt_sample = example
                if english_sample:
                    # TODO: add to dataset and format
                    pass
                # reset english sample
                english_sample = None


        # TODO: finish the processing of MGSMInstruct and also of "Mathoctopus/MSVAMP"
        msvamp = load_dataset("Mathoctopus/GSM8KInstruct_Parallel", split="train")
