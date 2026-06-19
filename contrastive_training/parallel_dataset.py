import sys, os
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
import torch
from torch.utils.data import Sampler
from typing import List, Dict, Any, Optional

from dataset_helpers import (
    detect_mgsm_language,
    format_feature_text,
    format_translation_sft_batch,
    load_mixed_parallel_dataset,
    _load_feature_messages,
    MIXED_PARALLEL_VALIDATION_LIMIT,
    MAX_LENGTH_BY_LANGUAGE,
    filtered_parallel_dataset_path,
    opus_language_code,
    opus_split_name,
)

DEFAULT_MAX_LENGTH = 384 # for control
TRANSLATION_SFT_MAX_LENGTH = 512 # longer because two texts needed in one sample
PACKED_SEQUENCE_SOURCE = 0
PACKED_SEQUENCE_TARGET = 1

def resolve_max_length(language_key, max_length) -> int:
    return max_length or MAX_LENGTH_BY_LANGUAGE.get(language_key, DEFAULT_MAX_LENGTH)

class ParallelDataCollator:
    # does it make sense to inherit from DPODataCollatorWithPadding ??
    def __init__(self, tokenizer: AutoTokenizer, src_language_key, tgt_language_key, max_length=None):
        self.tokenizer = tokenizer
        self.src_key = src_language_key
        self.tgt_key = tgt_language_key
        self.max_length = resolve_max_length(tgt_language_key, max_length)

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
        self.max_length = resolve_max_length(tgt_language_key, max_length)

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
        tgt_sentences = [
            format_feature_text(feature, self.tgt_key, "tgt", self.tokenizer)
            for feature in features
        ]
        src_sentences = [
            format_feature_text(feature, self.src_key, "src", self.tokenizer)
            for feature in features
        ]
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


class PackedLengthBatchSampler(Sampler[list[int]]):
    """Yield variable-size batches capped by total packed target+source tokens."""

    def __init__(
        self,
        dataset,
        tokenizer: AutoTokenizer,
        src_language_key,
        tgt_language_key,
        max_length: int,
        max_tokens: int,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ):
        if max_tokens <= 0:
            raise ValueError(f"Packed token budget must be positive, got {max_tokens}.")
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.src_key = src_language_key
        self.tgt_key = tgt_language_key
        self.max_length = max_length
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.lengths = self._compute_lengths()

    def _compute_lengths(self) -> list[int]:
        lengths = []
        chunk_size = 1024
        for start in range(0, len(self.dataset), chunk_size):
            features = [self.dataset[idx] for idx in range(start, min(start + chunk_size, len(self.dataset)))]
            tgt_sentences = [
                format_feature_text(feature, self.tgt_key, "tgt", self.tokenizer)
                for feature in features
            ]
            src_sentences = [
                format_feature_text(feature, self.src_key, "src", self.tokenizer)
                for feature in features
            ]
            sequences = self.tokenizer(
                tgt_sentences + src_sentences,
                padding=False,
                truncation=True,
                max_length=self.max_length,
                return_attention_mask=False,
            )["input_ids"]
            batch_size = len(features)
            lengths.extend(
                max(1, len(sequences[idx])) + max(1, len(sequences[idx + batch_size]))
                for idx in range(batch_size)
            )
        return lengths

    def _indices(self) -> list[int]:
        if not self.shuffle:
            return list(range(len(self.lengths)))
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return torch.randperm(len(self.lengths), generator=generator).tolist()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self) -> list[list[int]]:
        batches = []
        batch = []
        token_count = 0
        for idx in self._indices():
            sample_tokens = self.lengths[idx]
            if batch and token_count + sample_tokens > self.max_tokens:
                batches.append(batch)
                batch = []
                token_count = 0
            batch.append(idx)
            token_count += sample_tokens
        if batch and not self.drop_last:
            batches.append(batch)
        return batches

    def __iter__(self):
        yield from self._batches()

    def __len__(self):
        return len(self._batches())


class TargetLanguageCausalLMCollator:
    """Build target-language-only batches for an LM-only CPT baseline."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        src_language_key,
        tgt_language_key,
        max_length=None,
    ):
        self.tokenizer = tokenizer
        self.src_key = src_language_key
        self.tgt_key = tgt_language_key
        self.max_length = resolve_max_length(tgt_language_key, max_length)

    def _format_chat_feature(self, feature: Dict[str, Any]) -> Optional[tuple[List[int], List[int]]]:
        messages = _load_feature_messages(feature.get("messages_tgt_json"))
        if not messages or not getattr(self.tokenizer, "chat_template", None):
            return None

        assistant_idx = next(
            (idx for idx in range(len(messages) - 1, -1, -1) if messages[idx].get("role") == "assistant"),
            None,
        )
        if assistant_idx is None:
            return None

        prompt_text = self.tokenizer.apply_chat_template(
            messages[:assistant_idx],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.tokenizer.apply_chat_template(
            messages[:assistant_idx + 1],
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        if full_ids[:len(prompt_ids)] != prompt_ids:
            return None

        completion_ids = full_ids[len(prompt_ids):]
        if not completion_ids:
            return None

        if len(prompt_ids) + len(completion_ids) <= self.max_length:
            input_ids = prompt_ids + completion_ids
            return input_ids, input_ids.copy()

        if len(completion_ids) >= self.max_length:
            prompt_budget = min(len(prompt_ids), max(1, self.max_length // 4))
            completion_budget = self.max_length - prompt_budget
        else:
            prompt_budget = self.max_length - len(completion_ids)
            completion_budget = len(completion_ids)

        input_ids = prompt_ids[-prompt_budget:] + completion_ids[:completion_budget]
        return input_ids, input_ids.copy()

    def _format_feature(self, feature: Dict[str, Any]) -> tuple[List[int], List[int]]:
        chat_ids_and_labels = self._format_chat_feature(feature)
        if chat_ids_and_labels is not None:
            return chat_ids_and_labels

        text = format_feature_text(feature, self.tgt_key, "tgt", self.tokenizer)
        input_ids = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )["input_ids"]
        return input_ids, input_ids.copy()

    def _pad_batch(self, encoded_features: List[tuple[List[int], List[int]]]) -> Dict[str, torch.Tensor]:
        batch_size = len(encoded_features)
        max_length = max(len(input_ids) for input_ids, _ in encoded_features)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("TargetLanguageCausalLMCollator requires a pad_token_id or eos_token_id.")

        input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        labels = torch.full((batch_size, max_length), -100, dtype=torch.long)

        for row_idx, (row_input_ids, row_labels) in enumerate(encoded_features):
            row_len = len(row_input_ids)
            input_ids[row_idx, :row_len] = torch.tensor(row_input_ids, dtype=torch.long)
            attention_mask[row_idx, :row_len] = 1
            labels[row_idx, :row_len] = torch.tensor(row_labels, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        encoded_features = [self._format_feature(feature) for feature in features]
        return self._pad_batch(encoded_features)


def _load_cached_parallel_dataset(language: str, data_limit=None, required=False):
    cached_dataset_path = filtered_parallel_dataset_path(language)
    if not cached_dataset_path.exists():
        if required:
            raise FileNotFoundError(
                f"No cached filtered dataset found for language '{language}' at {cached_dataset_path}. "
                "Build it first with contrastive_training/build_filtered_parallel_dataset.py."
            )
        return None

    print(f"Using cached filtered mixed dataset for {language}: {cached_dataset_path}")
    dataset_dict = load_from_disk(str(cached_dataset_path))
    train_dataset = dataset_dict["train"]
    valid_dataset = dataset_dict["validation"]

    metadata_path = cached_dataset_path / "filtered_dataset_metadata.json"
    if metadata_path.exists():
        import json
        with open(metadata_path, "r") as file:
            metadata = json.load(file)
        src_key = metadata.get("source_key", "en")
        tgt_key = metadata["target_key"]
    else:
        translation_keys = train_dataset[0]["translation"].keys()
        src_key = "en" if "en" in translation_keys else next(iter(translation_keys))
        tgt_key = next(key for key in translation_keys if key != src_key)

    if data_limit:
        train_dataset = train_dataset.select(range(min(data_limit, len(train_dataset))))
        valid_dataset = valid_dataset.select(range(min(MIXED_PARALLEL_VALIDATION_LIMIT, len(valid_dataset))))

    return train_dataset, valid_dataset, src_key, tgt_key


def load_parallel_datasets(dataset: str, language: str, data_limit=None, disable_cache=False):
    if dataset == "mixed":
        cached_dataset = _load_cached_parallel_dataset(language, data_limit=data_limit, required=False)
        if cached_dataset is not None:
            return cached_dataset
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
