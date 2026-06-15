import sys, os
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from typing import List, Dict, Any

from dataset_helpers import (
    detect_mgsm_language,
    format_feature_text,
    format_translation_sft_batch,
    load_mixed_parallel_dataset,
    opus_language_code,
    opus_split_name,
)

MAX_LENGTH = 384 # for control
TRANSLATION_SFT_MAX_LENGTH = 512 # longer because two texts needed in one sample
PACKED_SEQUENCE_SOURCE = 0
PACKED_SEQUENCE_TARGET = 1


class ParallelDataCollator:
    # does it make sense to inherit from DPODataCollatorWithPadding ??
    def __init__(self, tokenizer: AutoTokenizer, src_language_key, tgt_language_key, max_length=None):
        self.tokenizer = tokenizer
        self.src_key = src_language_key
        self.tgt_key = tgt_language_key
        self.max_length = max_length or MAX_LENGTH

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
        src_sentences = [
            format_feature_text(feature, self.src_key, "src", self.tokenizer)
            for feature in features
        ]
        tgt_sentences = [
            format_feature_text(feature, self.tgt_key, "tgt", self.tokenizer)
            for feature in features
        ]

        # Tokenize both together to ensure they are padded to the same maximum length.
        # This allows us to concatenate them along the batch dimension later.
        all_batch = self.tokenizer(
            src_sentences + tgt_sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
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
        labels_tgt = batch["input_ids_tgt"].clone()
        labels_tgt[batch["attention_mask_tgt"] == 0] = -100
        batch["labels_tgt"] = labels_tgt
        return batch


class PackedParallelDataCollator:
    """Pack target/source pairs into one row for FlashAttention varlen training."""

    def __init__(self, tokenizer: AutoTokenizer, src_language_key, tgt_language_key, max_length=None):
        self.tokenizer = tokenizer
        self.src_key = src_language_key
        self.tgt_key = tgt_language_key
        self.max_length = max_length or MAX_LENGTH

    def _tokenize(self, sentences: List[str]) -> List[List[int]]:
        encoded = self.tokenizer(
            sentences,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=False,
        )
        return encoded["input_ids"]

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        tgt_sentences = [feature["translation"][self.tgt_key] for feature in features]
        src_sentences = [feature["translation"][self.src_key] for feature in features]
        all_sequences = self._tokenize(tgt_sentences + src_sentences)
        batch_size = len(features)
        tgt_sequences = all_sequences[:batch_size]
        src_sequences = all_sequences[batch_size:]

        input_ids: list[int] = []
        labels: list[int] = []
        position_ids: list[int] = []
        seq_idx: list[int] = []
        sample_ids: list[int] = []
        sequence_type: list[int] = []
        cu_seq_lens = [0]
        target_cu_seq_lens = [0]
        max_sequence_length = 0
        target_max_sequence_length = 0
        target_total_tokens = 0
        sequence_index = 0

        def append_sequence(token_ids: list[int], sample_index: int, seq_type: int) -> None:
            nonlocal sequence_index, max_sequence_length, target_max_sequence_length, target_total_tokens
            if not token_ids:
                raise ValueError(f"Packed sequence for sample {sample_index} tokenized to zero tokens.")

            start = len(input_ids)
            input_ids.extend(token_ids)
            position_ids.extend(range(len(token_ids)))
            seq_idx.extend([sequence_index] * len(token_ids))
            sample_ids.extend([sample_index] * len(token_ids))
            sequence_type.extend([seq_type] * len(token_ids))
            cu_seq_lens.append(len(input_ids))
            max_sequence_length = max(max_sequence_length, len(token_ids))

            if seq_type == PACKED_SEQUENCE_TARGET:
                target_labels = list(token_ids)
                target_labels[0] = -100
                labels.extend(target_labels)
                target_total_tokens += len(token_ids)
                target_cu_seq_lens.append(target_total_tokens)
                target_max_sequence_length = max(target_max_sequence_length, len(token_ids))
            else:
                labels.extend([-100] * len(token_ids))

            if len(input_ids) != start + len(token_ids):
                raise RuntimeError("Packed sequence length accounting failed.")
            sequence_index += 1

        for sample_index, tgt_ids in enumerate(tgt_sequences):
            append_sequence(tgt_ids, sample_index, PACKED_SEQUENCE_TARGET)
        for sample_index, src_ids in enumerate(src_sequences):
            append_sequence(src_ids, sample_index, PACKED_SEQUENCE_SOURCE)

        if not input_ids:
            raise ValueError("PackedParallelDataCollator received only empty tokenized sequences.")

        cu_seq_lens_tensor = torch.tensor(cu_seq_lens, dtype=torch.int32)
        target_cu_seq_lens_tensor = torch.tensor(target_cu_seq_lens, dtype=torch.int32)
        return {
            "packed": torch.tensor(True),
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "labels": torch.tensor([labels], dtype=torch.long),
            "position_ids": torch.tensor([position_ids], dtype=torch.long),
            "seq_idx": torch.tensor([seq_idx], dtype=torch.int32),
            "sample_ids": torch.tensor([sample_ids], dtype=torch.long),
            "sequence_type": torch.tensor([sequence_type], dtype=torch.long),
            "cu_seq_lens_q": cu_seq_lens_tensor,
            "cu_seq_lens_k": cu_seq_lens_tensor.clone(),
            "max_length_q": max_sequence_length,
            "max_length_k": max_sequence_length,
            "target_cu_seq_lens_q": target_cu_seq_lens_tensor,
            "target_cu_seq_lens_k": target_cu_seq_lens_tensor.clone(),
            "target_max_length_q": target_max_sequence_length,
            "target_max_length_k": target_max_sequence_length,
            "target_token_count": target_total_tokens,
        }


class TargetLanguageCausalLMCollator:
    """Build target-language-only batches for an LM-only CPT baseline."""

    def __init__(self, tokenizer: AutoTokenizer, src_language_key, tgt_language_key, max_length=MAX_LENGTH):
        self.tokenizer = tokenizer
        self.src_key = src_language_key
        self.tgt_key = tgt_language_key
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        tgt_sentences = [
            format_feature_text(feature, self.tgt_key, "tgt", self.tokenizer)
            for feature in features
        ]
        batch = self.tokenizer(
            tgt_sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


def load_parallel_datasets(dataset: str, language: str, data_limit=None, disable_cache=False):
    if dataset == "mixed":
        return load_mixed_parallel_dataset(language, data_limit)
    elif dataset == "opus":
        tgt_lang = opus_language_code(language)
        split_name = opus_split_name(tgt_lang)
        dataset = load_dataset("Helsinki-NLP/opus-100", split_name, keep_in_memory=disable_cache)

        if data_limit:
            train_limit = min(data_limit, len(dataset['train']))
            valid_limit = min(data_limit, len(dataset['validation']))
            return dataset['train'].select(range(train_limit)), dataset['validation'].select(range(valid_limit)), "en", tgt_lang
        return dataset['train'], dataset['validation'], "en", tgt_lang
    elif dataset == "math":
        ## combines mathoctopus' MGSM Instruct and MSVAMP
        # Add the project root to sys.path to enable absolute imports from project subdirectories
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..')) # Go up one level from 'contrastive_training' to 'moe'
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from expert_steering.mgsm_utils import MGSMINSTRUCT_LANGUAGES

        mgsminstruct = load_dataset("Mathoctopus/GSM8KInstruct_Parallel", split="train", keep_in_memory=disable_cache)

        mgsminstruct = mgsminstruct.map(lambda x: {"lang": detect_mgsm_language(x, MGSMINSTRUCT_LANGUAGES)})
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
        msvamp = load_dataset("Mathoctopus/GSM8KInstruct_Parallel", split="train", keep_in_memory=disable_cache)



def format_translation_sft_dataset(
    dataset,
    tokenizer,
    src_language_key,
    tgt_language_key,
    max_length=TRANSLATION_SFT_MAX_LENGTH,
    disable_cache=False,
):
    direction_pairs = [
        (src_language_key, tgt_language_key),
        (tgt_language_key, src_language_key),
    ]

    return dataset.map(
        lambda batch: format_translation_sft_batch(batch, tokenizer, direction_pairs, max_length),
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=not disable_cache,
        keep_in_memory=disable_cache,
    )
