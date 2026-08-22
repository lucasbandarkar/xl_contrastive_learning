import json
import os
from datasets import Dataset, load_dataset
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

MIXED_PARALLEL_VALIDATION_LIMIT = 1000
FILTERED_PARALLEL_DATASET_ROOT = Path(
    os.environ.get("FILTERED_PARALLEL_DATASET_ROOT", "/data2/lucasbandarkar/filtered_parallel_datasets")
)

LANGUAGE_NAMES = {
    "en": "English",
    "fa": "Persian",
    "bn": "Bengali",
    "hu": "Hungarian",
    "id": "Indonesian",
    "kn": "Kannada",
    "ky": "Kyrgyz",
    "ne": "Nepali",
    "te": "Telugu",
    "vi": "Vietnamese",
    "si": "Sinhala",
    "th": "Thai",
    "yo": "Yoruba",
    "zu": "Zulu",
}

MAX_LENGTH_BY_LANGUAGE = {
    "kan": 640,
    "sin": 512,
    "ben": 512,
    "tel": 640,
}

OPUS_LANGUAGE_CODES = {
    "pes": "fa",
    "ben": "bn",
    "ben_Beng": "bn",
    "hun": "hu",
    "ind": "id",
    "kan": "kn",
    "kan_Knda": "kn",
    "kir": "ky",
    "npi": "ne",
    "npi_Deva": "ne",
    "tel": "te",
    "tel_Telu": "te",
    "tha": "th",
    "tha_Thai": "th",
    "vie": "vi",
    "sin": "si",
    "yor": "yo",
    "zul": "zu",
}

MIXED_LANGUAGE_CONFIGS = {
    "fa": {
        "opus": "fa",
        "nllb": "pes_Arab",
        "names": ("fa", "pes", "persian", "farsi"),
    },
    "bn": {
        "opus": "bn",
        "nllb": "ben_Beng",
        "names": ("bn", "ben", "bengali", "bangla"),
    },
    "hu": {
        "opus": "hu",
        "nllb": "hun_Latn",
        "names": ("hu", "hun", "hungarian", "magyar"),
    },
    "id": {
        "opus": "id",
        "nllb": "ind_Latn",
        "names": ("id", "ind", "indonesian"),
    },
    "kn": {
        "opus": "kn",
        "nllb": "kan_Knda",
        "names": ("kn", "kan", "kannada"),
    },
    "ky": {
        "opus": "ky",
        "nllb": "kir_Cyrl",
        "names": ("ky", "kir", "kyrgyz", "kirghiz"),
    },
    "ne": {
        "opus": "ne",
        "nllb": "npi_Deva",
        "names": ("ne", "npi", "nepali"),
    },
    "te": {
        "opus": "te",
        "nllb": "tel_Telu",
        "names": ("te", "tel", "telugu"),
    },
    "vi": {
        "opus": "vi",
        "nllb": "vie_Latn",
        "names": ("vi", "vie", "vietnamese"),
    },
    "si": {
        "opus": "si",
        "nllb": "sin_Sinh",
        "names": ("si", "sin", "sinhala", "sinhalese"),
    },
    "th": {
        "opus": "th",
        "nllb": "tha_Thai",
        "names": ("th", "tha", "thai"),
    },
    "yo": {
        "opus": "yo",
        "nllb": "yor_Latn",
        "names": ("yo", "yor", "yoruba", "yorùbá"),
    },
    "zu": {
        "opus": "zu",
        "nllb": "zul_Latn",
        "names": ("zu", "zul", "zulu", "isizulu"),
    },
}

NLLB_SOURCE_FIELDS = {
    "source_field": "english",
    "target_field": "translated",
}

MIXED_SOURCE_SPECS = {
    "fa": [
        {"name": "bactrianx_sft", "weight": 0.40, "kind": "bactrianx_sft"},
        {"name": "nllb", "weight": 0.30, "kind": "nllb", **NLLB_SOURCE_FIELDS},
        {"name": "opus100", "weight": 0.15, "kind": "opus100"},
        {"name": "english_persian_parallel", "weight": 0.15, "kind": "generic", "path": "shenasa/English-Persian-Parallel-Dataset", "split": "train", "two_column_order": "en-tgt",},
    ],
    "id": [
        {"name": "bactrianx_sft", "weight": 0.40, "kind": "bactrianx_sft"},
        {"name": "nllb", "weight": 0.39, "kind": "nllb", **NLLB_SOURCE_FIELDS},
        {"name": "opus100", "weight": 0.15, "kind": "opus100"},
        {"name": "nusax_mt", "weight": 0.06, "kind": "generic", "path": "indonlp/NusaX-MT", "split": "train",},
    ],
    "bn": [
        {"name": "updesh", "weight": 0.40, "kind": "updesh_sft"},
        {"name": "bactrianx_sft", "weight": 0.15, "kind": "bactrianx_sft"},
        {"name": "samanantar", "weight": 0.15, "kind": "generic", "path": "ai4bharat/samanantar", "config": "bn", "source_field": "src", "target_field": "tgt"},
        {"name": "nllb", "weight": 0.05, "kind": "nllb", **NLLB_SOURCE_FIELDS},
        {"name": "opus100", "weight": 0.05, "kind": "opus100"},
        {"name": "mcot_math", "weight": 0.20, "kind": "mcot_math_sft"},
    ],
    "hu": [
        {"name": "opus100", "weight": 0.50, "kind": "opus100"},
        {"name": "nllb", "weight": 0.50, "kind": "nllb", **NLLB_SOURCE_FIELDS},
    ],
    "kn": [
        {"name": "updesh", "weight": 0.55, "kind": "updesh_sft"},
        {"name": "samanantar", "weight": 0.25, "kind": "generic", "path": "ai4bharat/samanantar", "config": "kn", "source_field": "src", "target_field": "tgt"},
        {"name": "nllb", "weight": 0.15, "kind": "nllb", **NLLB_SOURCE_FIELDS},
        {"name": "opus100", "weight": 0.05, "kind": "opus100"},
    ],
    "ky": [
        {"name": "nllb", "weight": 0.70, "kind": "nllb", **NLLB_SOURCE_FIELDS},
        {"name": "opus100", "weight": 0.30, "kind": "opus100"},
    ],
    "ne": [
        {"name": "updesh", "weight": 0.50, "kind": "updesh_sft"},
        {"name": "bactrianx_sft", "weight": 0.25, "kind": "bactrianx_sft"},
        {"name": "nllb", "weight": 0.10, "kind": "nllb", **NLLB_SOURCE_FIELDS},
        {"name": "opus100", "weight": 0.05, "kind": "opus100"},
        {"name": "eng2nep", "weight": 0.10, "kind": "generic", "path": "momo22/eng2nep", "source_field": "English", "target_field": "Nepali"},
    ],
    "te": [
        {"name": "updesh", "weight": 0.45, "kind": "updesh_sft"},
        {"name": "bactrianx_sft", "weight": 0.25, "kind": "bactrianx_sft"},
        {"name": "samanantar", "weight": 0.25, "kind": "generic", "path": "ai4bharat/samanantar", "config": "te", "source_field": "src", "target_field": "tgt"},
        {"name": "nllb", "weight": 0.10, "kind": "nllb", **NLLB_SOURCE_FIELDS},
        {"name": "opus100", "weight": 0.05, "kind": "opus100"},
    ],
    "vi": [
        {"name": "mtet", "weight": 0.30, "kind": "generic", "path": "hiimbach/mtet", "split": "train",},
        {"name": "bactrianx_sft", "weight": 0.40, "kind": "bactrianx_sft"},
        {"name": "nllb", "weight": 0.25, "kind": "nllb", **NLLB_SOURCE_FIELDS},
        {"name": "opus100", "weight": 0.05, "kind": "opus100"},
    ],
    "si": [
        {"name": "bactrianx_sft", "weight": 0.40, "kind": "bactrianx_sft"},
        {"name": "nllb", "weight": 0.45, "kind": "nllb", **NLLB_SOURCE_FIELDS},
        {"name": "opus100", "weight": 0.15, "kind": "opus100"},
    ],
    "th": [
        {"name": "bactrianx_sft", "weight": 0.40, "kind": "bactrianx_sft"},
        {"name": "scb_mt_en_th", "weight": 0.35, "kind": "generic", "path": "pythainlp/scb_mt_enth_2020"},
        {"name": "opus100", "weight": 0.25, "kind": "opus100"},
    ],
    "yo": [
        {"name": "nllb", "weight": 0.50, "kind": "nllb", **NLLB_SOURCE_FIELDS},
        {"name": "afrinllb", "weight": 0.30, "kind": "generic", "path": "AfriNLP/AfriNLLB-train", "source_field": "source", "target_field": "target", "source_lang_field": "src_lang", "source_lang": "eng_Latn", "target_lang_field": "tgt_lang", "target_lang": "yor_Latn"},
        {"name": "opus100", "weight": 0.20, "kind": "opus100"},
    ],
    "zu": [
        {"name": "afrinllb", "weight": 0.70, "kind": "generic", "path": "AfriNLP/AfriNLLB-train", "source_field": "source", "target_field": "target", "source_lang_field": "src_lang", "source_lang": "eng_Latn", "target_lang_field": "tgt_lang", "target_lang": "zul_Latn"},
        {"name": "opus100", "weight": 0.30, "kind": "opus100"},
    ],
}

ENGLISH_FIELD_NAMES = ("en", "eng", "eng_Latn", "english")
NLLB_ENGLISH_BITEXT_DATASET = "hotchpotch/nllb-english-bitext-hq"
BACTRIAN_X_DATASET = "MBZUAI/Bactrian-X"
BACTRIAN_X_DATA_PATH_TEMPLATE = "hf://datasets/MBZUAI/Bactrian-X/data/{language}.json.gz"
MCOT_MATH_DATASET = "laihuiyuan/mCoT-MATH"
UPDESH_DATA_DIR = Path("/data2/lucasbandarkar/sft_datasets/orca-agentinstruct")
UPDESH_FILE_PATTERN_TEMPLATE = "orca-agentinstruct-1M-v1.*.{language}.jsonl"

_BACTRIAN_X_ENGLISH_BY_ID: Optional[Dict[str, Dict[str, Any]]] = None


def canonical_mixed_language(language: str) -> str:
    opus_language = OPUS_LANGUAGE_CODES.get(language, language)
    if opus_language in MIXED_LANGUAGE_CONFIGS:
        return opus_language

    supported = ", ".join(sorted(MIXED_LANGUAGE_CONFIGS))
    raise ValueError(f"Unsupported mixed language '{language}'. Supported languages: {supported}")


def filtered_parallel_dataset_path(language: str) -> Path:
    return FILTERED_PARALLEL_DATASET_ROOT / canonical_mixed_language(language)


def opus_split_name(tgt_lang: str) -> str:
    return f"en-{tgt_lang}" if tgt_lang > "en" else f"{tgt_lang}-en"


def _load_dataset_stream(path: str, name: Optional[str] = None, split: str = "train", trust_remote_code: bool = False):
    kwargs = {"split": split, "streaming": True}
    if trust_remote_code:
        kwargs["trust_remote_code"] = True

    try:
        if name:
            return load_dataset(path, name, **kwargs)
        return load_dataset(path, **kwargs)
    except TypeError as exc:
        if "trust_remote_code" in kwargs and "trust_remote_code" in str(exc):
            kwargs.pop("trust_remote_code")
            if name:
                return load_dataset(path, name, **kwargs)
            return load_dataset(path, **kwargs)
        raise


def _language_field_candidates(language_config: Dict[str, Any]) -> Tuple[str, ...]:
    field_names = (language_config["opus"], language_config["nllb"], *language_config["names"])
    candidates = []
    for value in field_names:
        candidates.append(value)
        candidates.append(value.lower())
        candidates.append(value.upper())
        candidates.append(value.title())
    return tuple(dict.fromkeys(candidates))


def _find_candidate(mapping: Dict[str, Any], candidates: Iterable[str]) -> Optional[Any]:
    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]

    lowered = {str(key).lower(): key for key in mapping}
    for candidate in candidates:
        original_key = lowered.get(str(candidate).lower())
        if original_key is not None:
            return mapping[original_key]
    return None


def _clean_parallel_text(text: Any) -> Optional[str]:
    if not isinstance(text, str):
        return None

    text = " ".join(text.split())
    if len(text) < 2 or len(text) > 2000:
        return None
    return text


def _clean_sft_text(text: Any) -> Optional[str]:
    if not isinstance(text, str):
        return None

    text = " ".join(text.split())
    if len(text) < 1:
        return None
    return text


def _normalize_messages(messages: Any) -> Optional[List[Dict[str, str]]]:
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            return None

    if not isinstance(messages, list):
        return None

    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            continue
        role = role.strip().lower()
        content = content.strip()
        if role not in {"system", "user", "assistant"} or not content:
            continue
        normalized.append({"role": role, "content": content})

    if not any(message["role"] == "assistant" for message in normalized):
        return None
    return normalized


def _messages_to_plain_text(messages: List[Dict[str, str]]) -> str:
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def _get_updesh_field(example: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    content = example.get("content")
    if isinstance(content, dict):
        for key in keys:
            if key in content:
                return content[key]

    for key in keys:
        if key in example:
            return example[key]
    return None


def _load_feature_messages(value):
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return json.loads(value)


def format_feature_text(feature, language_key, side, tokenizer):
    messages = _load_feature_messages(feature.get(f"messages_{side}_json"))
    if messages:
        if getattr(tokenizer, "chat_template", None) and hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        return _messages_to_plain_text(messages)

    return feature["translation"][language_key]


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _build_bactrianx_messages(example: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    instruction = _clean_sft_text(example.get("instruction"))
    output = _clean_sft_text(example.get("output"))
    if not instruction or not output:
        return None

    user_content = instruction
    input_text = _clean_sft_text(example.get("input"))
    if input_text:
        user_content = f"{instruction}\n\n{input_text}"

    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]


def _build_mcot_math_messages(example: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    question = _clean_sft_text(example.get("question"))
    answer = _clean_sft_text(example.get("answer"))
    if not question or not answer:
        return None

    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def _make_dataset_example(
    *,
    src_text: str,
    tgt_text: str,
    language_config: Dict[str, Any],
    source_name: str,
    example_type: str,
    messages_src: Optional[List[Dict[str, str]]] = None,
    messages_tgt: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    return {
        "translation": {
            "en": src_text,
            language_config["opus"]: tgt_text,
        },
        "example_type": example_type,
        "source_name": source_name,
        "messages_src_json": _json_dumps(messages_src) if messages_src else None,
        "messages_tgt_json": _json_dumps(messages_tgt) if messages_tgt else None,
    }


def _lang_label_matches(label: Any, candidates: Iterable[str]) -> bool:
    if not isinstance(label, str):
        return False
    normalized = label.strip().lower()
    return any(normalized == candidate.lower() for candidate in candidates)


def _extract_nusax_style_pair(example: Dict[str, Any], target_candidates: Tuple[str, ...]) -> Tuple[Optional[str], Optional[str]]:
    text_1 = example.get("text_1")
    text_2 = example.get("text_2")
    lang_1 = example.get("text_1_lang")
    lang_2 = example.get("text_2_lang")

    if not text_1 or not text_2 or not lang_1 or not lang_2:
        return None, None

    if _lang_label_matches(lang_1, ENGLISH_FIELD_NAMES) and _lang_label_matches(lang_2, target_candidates):
        return text_1, text_2
    if _lang_label_matches(lang_2, ENGLISH_FIELD_NAMES) and _lang_label_matches(lang_1, target_candidates):
        return text_2, text_1
    return None, None


def _extract_two_column_pair(example: Dict[str, Any], spec: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    if spec.get("two_column_order") not in {"en-tgt", "tgt-en"}:
        return None, None

    text_columns = [
        value
        for key, value in example.items()
        if isinstance(value, str) and key not in {"id", "idx", "source", "url"}
    ]
    if len(text_columns) != 2:
        return None, None

    if spec["two_column_order"] == "en-tgt":
        return text_columns[0], text_columns[1]
    return text_columns[1], text_columns[0]


def _extract_parallel_pair(
    example: Dict[str, Any],
    language_config: Dict[str, Any],
    spec: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    source_lang_field = spec.get("source_lang_field")
    source_lang = spec.get("source_lang")
    if source_lang_field and source_lang and example.get(source_lang_field) != source_lang:
        return None, None

    target_lang_field = spec.get("target_lang_field")
    target_lang = spec.get("target_lang")
    if target_lang_field and target_lang and example.get(target_lang_field) != target_lang:
        return None, None

    source_field = spec.get("source_field")
    target_field = spec.get("target_field")
    if source_field and target_field:
        src = example.get(source_field)
        tgt = example.get(target_field)
        if src is not None and tgt is not None:
            return src, tgt

    target_candidates = _language_field_candidates(language_config)
    translation = example.get("translation")

    if isinstance(translation, dict):
        src = _find_candidate(translation, ENGLISH_FIELD_NAMES)
        tgt = _find_candidate(translation, target_candidates)
        if src is not None and tgt is not None:
            return src, tgt

        if spec.get("two_column_order") == "en-tgt":
            src = translation.get("source")
            tgt = translation.get("target")
            if src is not None and tgt is not None:
                return src, tgt
        elif spec.get("two_column_order") == "tgt-en":
            tgt = translation.get("source")
            src = translation.get("target")
            if src is not None and tgt is not None:
                return src, tgt

    src, tgt = _extract_nusax_style_pair(example, target_candidates)
    if src is not None and tgt is not None:
        return src, tgt

    src = _find_candidate(example, ENGLISH_FIELD_NAMES)
    tgt = _find_candidate(example, target_candidates)
    if src is not None and tgt is not None:
        return src, tgt

    if spec.get("two_column_order") == "en-tgt":
        src = example.get("source")
        tgt = example.get("target")
        if src is not None and tgt is not None:
            return src, tgt
    elif spec.get("two_column_order") == "tgt-en":
        tgt = example.get("source")
        src = example.get("target")
        if src is not None and tgt is not None:
            return src, tgt

    return _extract_two_column_pair(example, spec)


def _format_parallel_example(
    example: Dict[str, Any],
    language_config: Dict[str, Any],
    spec: Dict[str, Any],
) -> Optional[Dict[str, Dict[str, str]]]:
    src, tgt = _extract_parallel_pair(example, language_config, spec)
    src = _clean_parallel_text(src)
    tgt = _clean_parallel_text(tgt)

    if not src or not tgt or src == tgt:
        return None

    return _make_dataset_example(
        src_text=src,
        tgt_text=tgt,
        language_config=language_config,
        source_name=spec["name"],
        example_type="parallel_text",
    )


def _format_updesh_sft_example(
    example: Dict[str, Any],
    language_config: Dict[str, Any],
    spec: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    messages_src = _normalize_messages(_get_updesh_field(example, ("input", "inputs")))
    messages_tgt = _normalize_messages(_get_updesh_field(example, ("output", "outputs")))
    if not messages_src or not messages_tgt:
        return None

    src_text = _messages_to_plain_text(messages_src)
    tgt_text = _messages_to_plain_text(messages_tgt)
    if src_text == tgt_text:
        return None

    return _make_dataset_example(
        src_text=src_text,
        tgt_text=tgt_text,
        language_config=language_config,
        source_name=spec["name"],
        example_type="sft_parallel",
        messages_src=messages_src,
        messages_tgt=messages_tgt,
    )


def _updesh_data_files(spec: Dict[str, Any], language_config: Dict[str, Any]) -> Tuple[Path, List[Path]]:
    data_dir = Path(spec.get("data_dir", UPDESH_DATA_DIR))
    language = spec.get("language", language_config["nllb"])
    pattern = spec.get("file_pattern", UPDESH_FILE_PATTERN_TEMPLATE.format(language=language))
    data_files = sorted(data_dir.glob(pattern))
    return data_dir, data_files


def _iter_jsonl_file(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc


def _iter_round_robin_jsonl(paths: List[Path]):
    iterators = [iter(_iter_jsonl_file(path)) for path in paths]
    while iterators:
        active_iterators = []
        for iterator in iterators:
            try:
                yield next(iterator)
                active_iterators.append(iterator)
            except StopIteration:
                continue
        iterators = active_iterators


def _source_dataset_stream(spec: Dict[str, Any], language_config: Dict[str, Any]):
    kind = spec["kind"]
    if kind == "nllb":
        # allenai/nllb is still a legacy datasets script; this Parquet-backed
        # mirror keeps the NLLB/CCMatrix English bitext usable on datasets>=4.
        return _load_dataset_stream(
            NLLB_ENGLISH_BITEXT_DATASET,
            language_config["nllb"],
            split="train",
        )

    if kind == "opus100":
        return _load_dataset_stream(
            "Helsinki-NLP/opus-100",
            opus_split_name(language_config["opus"]),
            split="train",
        )

    if kind == "generic":
        return _load_dataset_stream(
            spec["path"],
            spec.get("config"),
            split=spec.get("split", "train"),
            trust_remote_code=spec.get("trust_remote_code", False),
        )

    if kind == "updesh_sft":
        data_dir, data_files = _updesh_data_files(spec, language_config)
        if not data_files:
            raise FileNotFoundError(
                f"No UPDESH JSONL files found for {language_config['nllb']} in {data_dir}"
            )
        return _iter_round_robin_jsonl(data_files)

    raise ValueError(f"Unknown mixed-parallel source kind '{kind}'")


def _format_bactrianx_sft_example(
    src_example: Dict[str, Any],
    tgt_example: Dict[str, Any],
    language_config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if src_example.get("id") != tgt_example.get("id"):
        return None

    messages_src = _build_bactrianx_messages(src_example)
    messages_tgt = _build_bactrianx_messages(tgt_example)
    if not messages_src or not messages_tgt:
        return None

    return _make_dataset_example(
        src_text=_messages_to_plain_text(messages_src),
        tgt_text=_messages_to_plain_text(messages_tgt),
        language_config=language_config,
        source_name="bactrianx_sft",
        example_type="sft_parallel",
        messages_src=messages_src,
        messages_tgt=messages_tgt,
    )


def _split_mcot_math_lang_id(example: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    lang_id = example.get("lang-id")
    if not isinstance(lang_id, str):
        return None, None

    language, separator, example_id = lang_id.partition("-")
    if not separator or not language or not example_id:
        return None, None
    return language, example_id


def _format_mcot_math_sft_example(
    src_example: Dict[str, Any],
    tgt_example: Dict[str, Any],
    language_config: Dict[str, Any],
    spec: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    source_language, source_id = _split_mcot_math_lang_id(src_example)
    target_language, target_id = _split_mcot_math_lang_id(tgt_example)
    if source_id is None or source_id != target_id:
        return None
    if source_language != spec.get("source_lang", "en"):
        return None
    if target_language != spec.get("target_lang", language_config["opus"]):
        return None

    messages_src = _build_mcot_math_messages(src_example)
    messages_tgt = _build_mcot_math_messages(tgt_example)
    if not messages_src or not messages_tgt:
        return None

    src_text = _messages_to_plain_text(messages_src)
    tgt_text = _messages_to_plain_text(messages_tgt)
    if src_text == tgt_text:
        return None

    return _make_dataset_example(
        src_text=src_text,
        tgt_text=tgt_text,
        language_config=language_config,
        source_name=spec["name"],
        example_type="sft_parallel",
        messages_src=messages_src,
        messages_tgt=messages_tgt,
    )


def _iter_mcot_math_sft_examples(
    spec: Dict[str, Any],
    language_config: Dict[str, Any],
):
    source_language = spec.get("source_lang", "en")
    target_language = spec.get("target_lang", language_config["opus"])
    pending_source: Dict[str, Dict[str, Any]] = {}
    pending_target: Dict[str, Dict[str, Any]] = {}

    stream = _load_dataset_stream(
        spec.get("path", MCOT_MATH_DATASET),
        spec.get("config"),
        split=spec.get("split", "train"),
        trust_remote_code=spec.get("trust_remote_code", False),
    )
    for raw_example in stream:
        language, example_id = _split_mcot_math_lang_id(raw_example)
        if language is None or example_id is None:
            continue

        if language == source_language:
            tgt_example = pending_target.pop(example_id, None)
            if tgt_example is not None:
                example = _format_mcot_math_sft_example(raw_example, tgt_example, language_config, spec)
                if example is not None:
                    yield example
            else:
                pending_source[example_id] = raw_example
        elif language == target_language:
            src_example = pending_source.pop(example_id, None)
            if src_example is not None:
                example = _format_mcot_math_sft_example(src_example, raw_example, language_config, spec)
                if example is not None:
                    yield example
            else:
                pending_target[example_id] = raw_example


def _get_bactrianx_english_by_id() -> Dict[str, Dict[str, Any]]:
    global _BACTRIAN_X_ENGLISH_BY_ID
    if _BACTRIAN_X_ENGLISH_BY_ID is None:
        english_by_id = {}
        english_stream = load_dataset(
            "json",
            data_files=BACTRIAN_X_DATA_PATH_TEMPLATE.format(language="en"),
            split="train",
            streaming=True,
        )
        for example in english_stream:
            example_id = example.get("id")
            if isinstance(example_id, str):
                english_by_id[example_id] = example
        _BACTRIAN_X_ENGLISH_BY_ID = english_by_id
    return _BACTRIAN_X_ENGLISH_BY_ID


def _collect_bactrianx_sft_examples(
    language_config: Dict[str, Any],
    requested_count: int,
    seen_pairs: set,
) -> List[Dict[str, Any]]:
    if requested_count <= 0:
        return []

    collected = []
    english_by_id = _get_bactrianx_english_by_id()
    tgt_stream = load_dataset(
        "json",
        data_files=BACTRIAN_X_DATA_PATH_TEMPLATE.format(language=language_config["opus"]),
        split="train",
        streaming=True,
    )

    for tgt_raw in tgt_stream:
        src_raw = english_by_id.get(tgt_raw.get("id"))
        if src_raw is None:
            continue
        example = _format_bactrianx_sft_example(src_raw, tgt_raw, language_config)
        if example is None:
            continue

        translation = example["translation"]
        pair_key = (example["source_name"], translation["en"], translation[language_config["opus"]])
        if pair_key in seen_pairs:
            continue

        seen_pairs.add(pair_key)
        collected.append(example)
        if len(collected) >= requested_count:
            break

    return collected


def _collect_examples_from_source(
    spec: Dict[str, Any],
    language_config: Dict[str, Any],
    requested_count: int,
    seen_pairs: set,
) -> List[Dict[str, Any]]:
    if requested_count <= 0:
        return []

    if spec["kind"] == "bactrianx_sft":
        return _collect_bactrianx_sft_examples(language_config, requested_count, seen_pairs)

    collected = []
    if spec["kind"] == "mcot_math_sft":
        stream = _iter_mcot_math_sft_examples(spec, language_config)
    else:
        stream = _source_dataset_stream(spec, language_config)
    for raw_example in stream:
        if spec["kind"] == "mcot_math_sft":
            example = raw_example
        elif spec["kind"] == "updesh_sft":
            example = _format_updesh_sft_example(raw_example, language_config, spec)
        else:
            example = _format_parallel_example(raw_example, language_config, spec)
        if example is None:
            continue

        translation = example["translation"]
        pair_key = (example["source_name"], translation["en"], translation[language_config["opus"]])
        if pair_key in seen_pairs:
            continue

        seen_pairs.add(pair_key)
        collected.append(example)
        if len(collected) >= requested_count:
            break
    return collected


def _allocate_source_counts(total_count: int, specs: List[Dict[str, Any]]) -> Dict[str, int]:
    if total_count <= 0:
        return {spec["name"]: 0 for spec in specs}

    total_weight = sum(spec["weight"] for spec in specs)
    raw_counts = [(spec["name"], total_count * spec["weight"] / total_weight) for spec in specs]
    counts = {name: int(raw_count) for name, raw_count in raw_counts}
    remainder = total_count - sum(counts.values())

    fractional_order = sorted(raw_counts, key=lambda item: item[1] - int(item[1]), reverse=True)
    for name, _ in fractional_order[:remainder]:
        counts[name] += 1
    return counts


def _fill_missing_examples(
    examples: List[Dict[str, Dict[str, str]]],
    target_count: int,
    candidate_specs: List[Dict[str, Any]],
    language_config: Dict[str, Any],
    seen_pairs: set,
    loaded_counts: Dict[str, int],
    requested_counts: Dict[str, int],
) -> None:
    for spec in candidate_specs:
        missing = target_count - len(examples)
        if missing <= 0:
            return
        if loaded_counts.get(spec["name"], 0) < requested_counts.get(spec["name"], 0):
            continue

        try:
            extra_examples = _collect_examples_from_source(spec, language_config, missing, seen_pairs)
        except Exception as exc:
            print(f"Could not fill mixed-parallel remainder from {spec['name']}: {exc}")
            continue

        examples.extend(extra_examples)
        loaded_counts[spec["name"]] = loaded_counts.get(spec["name"], 0) + len(extra_examples)


def _load_streamed_parallel_sources(
    language: str,
    specs: List[Dict[str, Any]],
    data_limit: Optional[int],
) -> Tuple[Dataset, Dataset, str, str]:
    canonical_language = canonical_mixed_language(language)
    language_config = MIXED_LANGUAGE_CONFIGS[canonical_language]
    train_limit = data_limit or 100000
    valid_limit = min(train_limit, MIXED_PARALLEL_VALIDATION_LIMIT)
    train_counts = _allocate_source_counts(train_limit, specs)
    valid_counts = _allocate_source_counts(valid_limit, specs)

    train_examples = []
    valid_examples = []
    seen_pairs = set()
    loaded_counts = {}

    for spec in specs:
        source_train_count = train_counts[spec["name"]]
        source_valid_count = valid_counts[spec["name"]]
        try:
            examples = _collect_examples_from_source(
                spec,
                language_config,
                source_train_count + source_valid_count,
                seen_pairs,
            )
        except Exception as exc:
            print(f"Skipping mixed-parallel source {spec['name']} for {canonical_language}: {exc}")
            loaded_counts[spec["name"]] = 0
            continue

        train_examples.extend(examples[:source_train_count])
        valid_examples.extend(examples[source_train_count:source_train_count + source_valid_count])
        loaded_counts[spec["name"]] = len(examples)

    refill_specs = [spec for spec in specs if loaded_counts.get(spec["name"], 0) > 0]
    requested_counts = {
        spec["name"]: train_counts[spec["name"]] + valid_counts[spec["name"]]
        for spec in specs
    }
    for examples, target_count in ((train_examples, train_limit), (valid_examples, valid_limit)):
        _fill_missing_examples(
            examples,
            target_count,
            refill_specs,
            language_config,
            seen_pairs,
            loaded_counts,
            requested_counts,
        )

    if not train_examples:
        raise RuntimeError(f"No training examples could be loaded for mixed-parallel language '{language}'.")
    if not valid_examples:
        raise RuntimeError(f"No validation examples could be loaded for mixed-parallel language '{language}'.")

    if len(train_examples) < train_limit or len(valid_examples) < valid_limit:
        print(
            f"Loaded fewer mixed-parallel examples than requested for {canonical_language}: "
            f"train={len(train_examples)}/{train_limit}, valid={len(valid_examples)}/{valid_limit}"
        )

    print(
        f"Loaded streamed parallel data for {canonical_language}: "
        f"train={len(train_examples)}, valid={len(valid_examples)}, sources={loaded_counts}"
    )
    return Dataset.from_list(train_examples), Dataset.from_list(valid_examples), "en", language_config["opus"]


def load_mixed_parallel_dataset(language: str, data_limit: Optional[int]) -> Tuple[Dataset, Dataset, str, str]:
    canonical_language = canonical_mixed_language(language)
    return _load_streamed_parallel_sources(
        canonical_language,
        MIXED_SOURCE_SPECS[canonical_language],
        data_limit,
    )



def detect_mgsm_language(entry, language_map):
    extracted_language = entry["prompt"].split()[17][:-1]
    return language_map[extracted_language]

def _format_translation_sft_pair(src_text, tgt_text, src_key, tgt_key, tokenizer, max_length):
    src_name = LANGUAGE_NAMES.get(src_key, src_key)
    tgt_name = LANGUAGE_NAMES.get(tgt_key, tgt_key)
    prompt = (
        f"Translate the following text from {src_name} to {tgt_name}. "
        "Return only the translation.\n\n"
        f"{src_text}\n\n"
        "Translation:\n"
    )
    completion = tgt_text
    if tokenizer.eos_token:
        completion += tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    return {
        "input_ids": (prompt_ids + completion_ids)[:max_length],
        "completion_mask": ([0] * len(prompt_ids) + [1] * len(completion_ids))[:max_length],
    }


def format_translation_sft_batch(batch, tokenizer, direction_pairs, max_length):
    input_ids = []
    completion_masks = []

    for i, translation in enumerate(batch["translation"]):
        src_key, tgt_key = direction_pairs[i % 2]
        formatted = _format_translation_sft_pair(
            translation[src_key],
            translation[tgt_key],
            src_key,
            tgt_key,
            tokenizer,
            max_length,
        )
        input_ids.append(formatted["input_ids"])
        completion_masks.append(formatted["completion_mask"])

    return {
        "input_ids": input_ids,
        "completion_mask": completion_masks,
    }
