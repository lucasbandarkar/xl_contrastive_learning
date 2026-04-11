"""
When you want to evaluate a new language
1. Add it to LANGUAGE_TO_TASK using its row in https://docs.google.com/spreadsheets/d/1IIpKgq466sDXOjh1IQXtDvnTvDfi3WeGIdM2X8wR2Kw/
2. Add it to FLORES_LANGCODE_MAP using https://github.com/facebookresearch/flores/blob/main/flores200/README.md
3. If it's in INCLUDE, Add it to CODE_TO_INCLUDE_NAME using https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/include/default

"""

# I have decided to not implement XLSum
LANGUAGE_TO_TASK = {
    "en": ["belebele", "mgsm", "mmlu_prox", "global_mmlu_medical", "flores", "multiloko", "global_piqa",],
    "fa": ["belebele", "global_mmlu_medical", "flores", "multiloko", "include", "global_piqa",],
    "si": ["belebele", "mgsm", "global_mmlu_medical", "flores", "global_piqa",],
    # all 8
    "bn": ["belebele", "mgsm", "mmlu_prox", "global_mmlu_medical", "flores", "multiloko", "include", "global_piqa"],
    "fr": ["belebele", "mgsm", "mmlu_prox", "global_mmlu_medical", "flores", "multiloko", "include", "global_piqa"],
    "ar": ["belebele", "mgsm", "mmlu_prox", "global_mmlu_medical", "flores", "multiloko", "include", "global_piqa"],
}

FLORES_LANGCODE_MAP = {
    "en": "eng_Latn",
    "fa": "pes_Arab",
    "si": "sin_Sinh",
    "bn": "ben_Beng",
    "fr": "fra_Latn",
    "ar": "arb_Arab",
}

CODE_TO_INCLUDE_NAME = {
    "fa": "Persian",
    "bn": "Bengali",
    "fr": "French",
    "ar": "Arabic",
}

CODE_TO_MULTILOKO_NAME = {
    "en": "english",
    "fa": "farsi",
    "bn": "bengali",
    "fr": "french",
    "ar": "arabic",
}
