# this file is adapted from https://github.com/mohsenfayyaz/moe/blob/main/expert_steering/evaluator.py
# that file contains custom code for gpt-oss and customized vLLM objects
from __future__ import annotations
import re, json
from lm_eval import tasks, evaluator
from typing import Any
import string
import tempfile
from difflib import SequenceMatcher
from pathlib import Path
import numpy as np
from datasets import load_dataset
import ast

from language_to_task import CODE_TO_INCLUDE_NAME, CODE_TO_MULTILOKO_NAME, FLORES_LANGCODE_MAP
from utils.multiloko_utils import MULTILOKO_PROMPT_BUILDERS

MMLU_PROX_LANGUAGES = ["af", "ar", "bn", "cs", "de", "en", "es", "fr", "hi", "hu", "id", "it", "ja", "ko", "mr", 
                       "ne", "pt", "ru", "sr", "sw", "te", "th", "uk", "ur", "vi", "wo", "yo", "zh", "zu"]
## there's more, but this doesnt serve any purpose
GLOBAL_MMLU_LANGUAGES = ["ar", "bn", "de", "en", "es", "fr", "hi", "id", "it", "ja", "ko", "pt", "sw", "yo", "zh"]

MGSM_LANGUAGES = ["bn", "de", "en", "es", "fr", "ru", "sw", "te", "th", "ja", "zh"] # doesn't include Global-MGSM languages

GLOBAL_MGSM_BASE_YAML = "utils/global_mgsm.yaml"
FLORES_BASE_YAML = "utils/flores.yaml"
GMMLU_MEDICAL_SAMPLES_PATH = "utils/gmmlu_medical_samples_dict.json"


class Evaluator:
    def __init__(self, vllm_object, chat_format=False, shots=0):
        self.chat_format = chat_format
        self.shots = shots
        self.answers = dict()
        self.vllm_wrapper = vllm_object

    def simple_evaluate(self, *args, **kwargs):
        kwargs.setdefault("model", self.vllm_wrapper.lm_eval_model)
        if self.vllm_wrapper.lm_eval_model_args is not None:
            kwargs.setdefault("model_args", self.vllm_wrapper.lm_eval_model_args)
        return evaluator.simple_evaluate(*args, **kwargs)

    def write_results(self, output_file: str | Path, results: dict[str, Any]) -> None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    def lm_eval_evaluate(self, langcodes, debugging=False, output_file="output.json"):
        pass


class MGSMEvaluator(Evaluator):
    def lm_eval_evaluate(self, langcodes, debugging=False, output_file="output.json"):
        base_mgsm_langcodes = [l for l in langcodes if l in MGSM_LANGUAGES]
        global_mgsm_langcodes = [l for l in langcodes if l not in MGSM_LANGUAGES]

        results_clean: dict[str, float] = {}
        if base_mgsm_langcodes:
            task_names = [f"mgsm_direct_{lang}" for lang in base_mgsm_langcodes]
            eval_kwargs = {
                "tasks": task_names,
                "num_fewshot": self.shots,
                "batch_size": "auto",
            }
            if debugging:
                eval_kwargs.update({"limit": 25, "log_samples": True})
            else:
                eval_kwargs.update({"verbosity": "WARNING"})

            results = self.simple_evaluate(**eval_kwargs)
            for task_name, metrics in results["results"].items():
                lang = task_name.removeprefix("mgsm_direct_")
                results_clean[lang] = np.round(metrics["exact_match,flexible-extract"], 3)

        results_clean.update(
            self.lm_eval_evaluate_global_mgsm(
                global_mgsm_langcodes,
                debugging=debugging,
            )
        )
        self.write_results(output_file, results_clean)
        return results_clean

    def lm_eval_evaluate_global_mgsm(self, global_mgsm_langcodes, debugging=False, output_file="output.json"):
        if not global_mgsm_langcodes:
            return {}

        with tempfile.TemporaryDirectory(prefix="global_mgsm_tasks_") as temp_dir:
            temp_dir_path = Path(temp_dir)
            task_names = []
            for lang in global_mgsm_langcodes:
                task_name = f"global_mgsm_{lang}"
                task_names.append(task_name)
                task_config = (
                    f'include: "{GLOBAL_MGSM_BASE_YAML}"\n'
                    f"task: {task_name}\n"
                    f"dataset_name: {lang}\n"
                )
                (temp_dir_path / f"{task_name}.yaml").write_text(task_config)

            task_manager = tasks.TaskManager(include_path=str(temp_dir_path))
            eval_kwargs = {
                "tasks": task_names,
                "task_manager": task_manager,
                "num_fewshot": self.shots,
                "batch_size": "auto",
            }
            if debugging:
                eval_kwargs.update({"limit": 25, "log_samples": True})
            else:
                eval_kwargs.update({"verbosity": "WARNING"})

            results = self.simple_evaluate(**eval_kwargs)

        results_clean = {}
        for task_name, metrics in results["results"].items():
            lang = task_name.removeprefix("global_mgsm_")
            results_clean[lang] = np.round(metrics["exact_match,flexible-extract"], 3)
        return results_clean


class GlobalMMLUMedicalEvaluator(Evaluator):
    def lm_eval_evaluate(self, langcodes, debugging=False, output_file="output.json"):
        task_names = [f"global_mmlu_full_{lang}" for lang in langcodes]
        medical_subjects = [
            "clinical_knowledge",
            "college_medicine",
            "human_aging",
            "medical_genetics",
            "nutrition",
            "professional_medicine",
            "virology"
        ]
        samples_dict = self.get_samples_dict(task_names)
        results = self.simple_evaluate(
            tasks=list(samples_dict.keys()),
            samples=samples_dict,
            batch_size='auto',
            verbosity="WARNING"
        )
        results_clean = {}
        for task, metrics in results['results'].items():
            results_clean[task] = metrics['acc,none']
        by_lang = {}
        for lang in langcodes:
            lang_scores = [
                results_clean[f"global_mmlu_full_{lang}_{subject}"]
                for subject in medical_subjects
            ]
            by_lang[lang] = np.round(np.mean(lang_scores), 3)

        self.write_results(output_file, by_lang)
        return by_lang

    def get_samples_dict(self, lang_tasks):
        ''' Downsampling MMLU-ProX dataset because the size is far too large per language'''

        with open(GMMLU_MEDICAL_SAMPLES_PATH, "r") as f:
            eng_dict = json.load(f)

        samples_dict = {}
        for s, index_list in eng_dict.items():
            # make it the same indices for every language
            for t in lang_tasks:
                samples_dict[f'{t}_{s}'] = index_list
        return samples_dict


class MMLUProXEvaluator(Evaluator):
    def lm_eval_evaluate(self, langcodes, debugging=False, output_file="output.json"):
        task_names = [f"mmlu_prox_{lang}" for lang in langcodes]
        sample_dict = self.get_samples_dict(task_names)
        results = self.simple_evaluate(
            tasks=task_names,
            samples=sample_dict,
            num_fewshot=self.shots,
            batch_size="auto",
            verbosity="WARNING",
            log_samples=False,
            gen_kwargs={"max_gen_toks": 1024},
        )
        results_clean = {}
        for task, metrics in results['groups'].items():
            results_clean[task] = metrics['exact_match,custom-extract']
        
        self.write_results(output_file, results_clean)
        return results_clean

    def get_samples_dict(self, lang_tasks):
        ''' Downsampling MMLU-ProX dataset because the size is far too large per language'''
        full_datasize = 11759
        min_size_for_any_subject = 381
        total_desired_size = 300
        rng = np.random.default_rng(seed=36)

        subjects = [
            "_biology",
            "_business",
            "_chemistry",
            "_computer_science",
            "_economics",
            "_engineering",
            "_health",
            "_history",
            "_law",
            "_math",
            "_other",
            "_philosophy",
            "_physics",
            "_psychology",
        ]
        samples_per_subject = total_desired_size // len(subjects)
        samples_dict = {}
        for s in subjects:
            samples = list(rng.integers(0, min_size_for_any_subject, samples_per_subject))
            # ensure same samples across languages
            for t in lang_tasks:
                samples_dict[t+s] = samples

        return samples_dict


class BelebeleEvaluator(Evaluator):
    def lm_eval_evaluate(self, langcodes, debugging=False, output_file="output.json"):
        task_names = [f"belebele_{FLORES_LANGCODE_MAP[lang]}" for lang in langcodes]
        results = self.simple_evaluate(
            tasks=task_names,
            batch_size="auto",
            verbosity="WARNING",
            log_samples=False,
        )
        results_clean = {}
        for task, metrics in results['results'].items():
            results_clean[task] = np.round(metrics['acc,none'], 3)

        self.write_results(output_file, results_clean)
        return results_clean

class GlobalPIQAEvaluator(Evaluator):
    def lm_eval_evaluate(self, langcodes, debugging=False, output_file="output.json"):
        global_piqa_langcodes = [FLORES_LANGCODE_MAP[twoletter].lower() for twoletter in langcodes]
        task_names = [f"global_piqa_completions_{lang}" for lang in global_piqa_langcodes]
        
        results = self.simple_evaluate(
            tasks=task_names,
            batch_size="auto",
            verbosity="WARNING",
            log_samples=False,
        )
        results_clean = {}
        for task, metrics in results['results'].items():
            results_clean[task] = np.round(metrics['acc_norm,none'], 3)

        self.write_results(output_file, results_clean)
        return results_clean


class INCLUDEEvaluator(Evaluator):
    def lm_eval_evaluate(self, langcodes, debugging=False, output_file="output.json"):
        include_names = [CODE_TO_INCLUDE_NAME[twoletter].lower() for twoletter in langcodes]
        task_names = [f"include_base_44_{lang}" for lang in include_names]
        
        results = self.simple_evaluate(
            tasks=task_names,
            batch_size="auto",
            verbosity="WARNING",
            log_samples=False,
        )
        results_clean = {}
        for task, metrics in results['results'].items():
            results_clean[task] = np.round(metrics['acc,none'], 3)
            
        self.write_results(output_file, results_clean)
        return results_clean


class FLoResEngXEvaluator(Evaluator):
    def lm_eval_evaluate(self, langcodes, debugging=False, output_file="output.json"):
        # We skip English ("en") since we are evaluating English -> Target translations
        eval_langcodes = [lang for lang in langcodes if lang != "en"]
        results_clean = {}
        
        if not eval_langcodes:
            self.write_results(output_file, results_clean)
            return results_clean
            
        with tempfile.TemporaryDirectory(prefix="flores_tasks_") as temp_dir:
            temp_dir_path = Path(temp_dir)
            task_names = []
            
            for lang in eval_langcodes:
                tgt_lang = FLORES_LANGCODE_MAP[lang]
                task_name = f"flores_eng_Latn_{tgt_lang}"
                task_names.append(task_name)
                
                # Jinja templating mapped to the Hugging Face dataset columns
                task_config = (
                    f'include: "{FLORES_BASE_YAML}"\n'
                    f"task: {task_name}\n"
                    f"dataset_name: eng_Latn-{tgt_lang}\n"
                    f"doc_to_text: '{{{{sentence_eng_Latn}}}} = '\n"
                    f"doc_to_target: '{{{{sentence_{tgt_lang}}}}}'\n"
                    f"dataset_kwargs:\n  trust_remote_code: True\n"
                )
                (temp_dir_path / f"{task_name}.yaml").write_text(task_config)

            task_manager = tasks.TaskManager(include_path=str(temp_dir_path))
            eval_kwargs = {
                "tasks": task_names,
                "task_manager": task_manager,
                "num_fewshot": self.shots,
                "batch_size": "auto",
            }
            if debugging:
                eval_kwargs.update({"limit": 25, "log_samples": True})
            else:
                eval_kwargs.update({"verbosity": "WARNING"})

            results = self.simple_evaluate(**eval_kwargs)

        for task_name, metrics in results["results"].items():
            tgt_lang = task_name.removeprefix("flores_eng_Latn_")
            # Map the flores langcode back to the original langcode
            original_lang = next((k for k, v in FLORES_LANGCODE_MAP.items() if v == tgt_lang), tgt_lang)
            results_clean[original_lang] = np.round(metrics.get("chrf,none", metrics.get("chrf", 0.0)), 3)
            
        self.write_results(output_file, results_clean)
        return results_clean

class MultiLoKoEvaluator(Evaluator):
    def __init__(self, vllm_object, chat_format: bool = False, shots: int = 0):
        super().__init__(vllm_object, chat_format, shots)
        self.normalization_regex = re.compile(r"\b(a|an|the)\b")


    def lm_eval_evaluate(
        self, langcodes: list[str], debugging: bool = False, output_file: str = "output.json"
    ):
        if self.vllm_wrapper.direct_vllm_model is None:
            raise RuntimeError(
                "MultiLoKo evaluation requires a direct vLLM model instance, but none was provided."
            )

        primary_scores = {}
        debug_payload = {}
        for lang in langcodes:
            language_name = CODE_TO_MULTILOKO_NAME[lang]
            dataset = load_dataset("facebook/multiloko", language_name)
            all_dev_examples = dataset["dev"]

            prepared_examples = [self.prepare_example(example) for example in all_dev_examples]
            fewshot_examples = prepared_examples[:2]
            eval_examples = prepared_examples[2:]
            
            prompts = [
                self.build_prompt(language_name, fewshot_examples, example)
                for example in eval_examples
            ]
            generations = self.generate_answers(prompts)
            metrics = self.score_generations(generations, eval_examples)
            primary_scores[lang] = metrics["em"]
            if debugging:
                debug_payload[lang] = {
                    "metrics": metrics,
                    "samples": [
                        {
                            "question": example["question"],
                            "gold": example["targets"],
                            "prediction": generation,
                        }
                        for example, generation in zip(eval_examples[:25], generations[:25])
                    ],
                }

        self.write_results(output_file, debug_payload if debugging else primary_scores)
        return debug_payload if debugging else primary_scores

    def build_prompt(
        self, language_name: str, fewshot_examples: list[dict[str, Any]], example: dict[str, Any]
    ) -> str:
        try:
            prompt_builder = MULTILOKO_PROMPT_BUILDERS[language_name]
        except KeyError as exc:
            raise KeyError(
                f"No prompt template was configured for MultiLoKo language `{language_name}`."
            ) from exc
        return prompt_builder(fewshot_examples, example["question"], example["output_type"])

    def prepare_example(self, example: dict[str, Any]) -> dict[str, Any]:
        example = dict(example)
        targets = example.get("targets")

        if isinstance(targets, str):
            try:
                targets = ast.literal_eval(targets)
            except Exception:
                targets = [targets]

        if targets:
            example["prompt_answer"] = targets[0]
            example["targets"] = targets
        elif "answer" in example:
            example["prompt_answer"] = example["answer"]
            example["targets"] = [example["answer"]]
        else:
            raise KeyError(
                "MultiLoKo examples must contain either `targets` or `answer`."
            )
        return example

    def generate_answers(self, prompts: list[str]) -> list[str]:
        sampling_params = self.vllm_wrapper.build_sampling_params(
            temperature=0.0,
            max_tokens=48,
            stop=["\n", "</think>"],
        )
        outputs = self.vllm_wrapper.direct_vllm_model.generate(
            prompts,
            sampling_params,
            use_tqdm=False,
        )
        generations = []
        for output in outputs:
            text = output.outputs[0].text.strip()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            generations.append(text)
        return generations

    def score_generations(
        self, generations: list[str], eval_examples: list[dict[str, Any]]
    ) -> dict[str, float]:
        em_scores = []
        f1_scores = []
        contains_scores = []
        edit_similarity_scores = []
        for prediction, example in zip(generations, eval_examples):
            targets = example["targets"]
            em_scores.append(float(self.exact_match(targets, prediction)))
            f1_scores.append(self.f1_score(targets, prediction))
            contains_scores.append(float(self.contains_score(targets, prediction)))
            edit_similarity_scores.append(self.edit_similarity(targets, prediction))

        return {
            "em": np.round(float(np.mean(em_scores)), 3),
            "f1": np.round(float(np.mean(f1_scores)), 3),
            "contains": np.round(float(np.mean(contains_scores)), 3),
            "edit_similarity": np.round(float(np.mean(edit_similarity_scores)), 3),
        }

    def exact_match(self, gold: str | list[str], pred: str) -> bool:
        if isinstance(gold, list):
            return any(self.exact_match(single_gold, pred) for single_gold in gold)
        return self.normalize_answer(gold) == self.normalize_answer(pred)

    def contains_score(self, gold: str | list[str], pred: str) -> bool:
        if isinstance(gold, list):
            return any(self.contains_score(single_gold, pred) for single_gold in gold)
        return self.normalize_answer(gold) in self.normalize_answer(pred)

    def f1_score(self, gold: str | list[str], pred: str) -> float:
        if isinstance(gold, list):
            return max(self.f1_score(single_gold, pred) for single_gold in gold)
        gold_tokens = self.normalize_answer(gold).split()
        pred_tokens = self.normalize_answer(pred).split()
        common = set(gold_tokens) & set(pred_tokens)
        num_same = sum(min(gold_tokens.count(tok), pred_tokens.count(tok)) for tok in common)
        if not gold_tokens or not pred_tokens:
            return float(gold_tokens == pred_tokens)
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)

    def edit_similarity(self, gold: str | list[str], pred: str) -> float:
        if isinstance(gold, list):
            return max(self.edit_similarity(single_gold, pred) for single_gold in gold)
        return SequenceMatcher(
            None,
            self.normalize_answer(gold),
            self.normalize_answer(pred),
        ).ratio()

    def normalize_answer(self, answer: str) -> str:
        answer = answer.replace("\n", " ").replace("\t", " ")
        answer = answer.lower()
        answer = self.normalization_regex.sub(" ", answer)
        answer = answer.translate(str.maketrans("", "", string.punctuation))
        return " ".join(answer.split())


TASK_EVALUATOR_REGISTRY = {
    "belebele": BelebeleEvaluator,
    "mgsm": MGSMEvaluator,
    "mmlu_prox": MMLUProXEvaluator,
    "global_mmlu_medical": GlobalMMLUMedicalEvaluator,
    "flores": FLoResEngXEvaluator,
    "multiloko": MultiLoKoEvaluator,
    "global_piqa": GlobalPIQAEvaluator,
    "include": INCLUDEEvaluator,
}
