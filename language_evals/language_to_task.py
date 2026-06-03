"""
When you want to evaluate a new language
1. Add it to LANGUAGE_TO_TASK using its row in https://docs.google.com/spreadsheets/d/1IIpKgq466sDXOjh1IQXtDvnTvDfi3WeGIdM2X8wR2Kw/
2. Add it to FLORES_TARGET_LANGUAGE_NAMES and FLORES_LANGCODE_MAP using https://github.com/facebookresearch/flores/blob/main/flores200/README.md
3. If it's in INCLUDE, Add it to CODE_TO_INCLUDE_NAME using https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/include/default
4. If it's in Multiloko, add it to CODE_TO_MULTILOKO_NAME and ensure multiloko_utils.py has a prompt builder for it.
"""

# I have decided to not implement XLSum
LANGUAGE_TO_TASK = {
    "en": ["belebele", "mgsm", "mmlu_prox", "global_mmlu_medical", "multiloko", "global_piqa",], # no flores, no include
    "fa": ["belebele", "global_mmlu_medical", "flores", "multiloko", "include", "global_piqa",],
    "si": ["belebele", "mgsm", "global_mmlu_medical", "flores", "global_piqa",],
    "hi": ["belebele", "mmlu_prox", "global_mmlu_medical", "flores", "multiloko", "include", "global_piqa"],
    "id": ["belebele", "mmlu_prox", "global_mmlu_medical", "flores", "multiloko", "include", "global_piqa"],
    "th": ["belebele", "mgsm", "mmlu_prox", "flores", "multiloko", "global_piqa"],
    "ky": ["belebele", "mgsm", "flores", "global_piqa"],
    "kn": ["belebele", "mgsm", "flores", "global_piqa"],
    "zu": ["belebele", "mgsm", "mmlu_prox", "flores", "global_piqa"],
    "yo": ["belebele", "mgsm", "mmlu_prox", "global_mmlu_medical", "flores", "global_piqa"],
    "ne": ["belebele", "mgsm", "mmlu_prox", "global_mmlu_medical", "flores", "include", "global_piqa"],
    "hu": ["belebele", "mgsm", "mmlu_prox", "flores", "include", "global_piqa"],
    "te": ["belebele", "mgsm", "mmlu_prox", "global_mmlu_medical", "flores", "include", "global_piqa"],
    # all 8
    "bn": ["belebele", "mgsm", "mmlu_prox", "global_mmlu_medical", "flores", "multiloko", "include", "global_piqa"],
    "fr": ["belebele", "mgsm", "mmlu_prox", "global_mmlu_medical", "flores", "multiloko", "include", "global_piqa"],
    "ar": ["belebele", "mgsm", "mmlu_prox", "global_mmlu_medical", "flores", "multiloko", "include", "global_piqa"],
    "vi": ["belebele", "mgsm", "mmlu_prox", "global_mmlu_medical", "flores", "multiloko", "include", "global_piqa"],
}

FLORES_LANGCODE_MAP = {
    "en": "eng_Latn",
    "fa": "pes_Arab",
    "si": "sin_Sinh",
    "bn": "ben_Beng",
    "fr": "fra_Latn",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "id": "ind_Latn",
    "th": "tha_Thai",
    "vi": "vie_Latn",
    "ky": "kir_Cyrl",
    "kn": "kan_Knda",
    "zu": "zul_Latn",
    "yo": "yor_Latn",
    "ne": "npi_Deva",
    "hu": "hun_Latn",
    "te": "tel_Telu",
}

FLORES_TARGET_LANGUAGE_NAMES = {
    "fa": "Persian, written in Arabic script",
    "si": "Sinhala, written in Sinhala script",
    "bn": "Bengali, written in Bengali script",
    "fr": "French",
    "ar": "Arabic, written in Arabic script",
    "hi": "Hindi, written in Devanagari script",
    "id": "Indonesian",
    "th": "Thai, written in Thai script",
    "vi": "Vietnamese",
    "ky": "Kyrgyz, written in Cyrillic script",
    "kn": "Kannada, written in Kannada script",
    "zu": "Zulu",
    "yo": "Yoruba",
    "ne": "Nepali, written in Devanagari script",
    "hu": "hun_Latn",
    "te": "Telugu, written in Telugu script",
}

CODE_TO_INCLUDE_NAME = {
    "fa": "Persian",
    "bn": "Bengali",
    "fr": "French",
    "ar": "Arabic",
    "hi": "Hindi",
    "id": "Indonesian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ne": "Nepali",
    "hu": "Hungarian",
    "te": "Telugu",
}

CODE_TO_MULTILOKO_NAME = {
    "en": "english",
    "fa": "farsi",
    "bn": "bengali",
    "fr": "french",
    "ar": "arabic",
    "hi": "hindi",
    "id": "indonesian",
    "th": "thai",
    "vi": "vietnamese",
}
