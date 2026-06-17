from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

from dataset_helpers import (
    BACTRIAN_X_DATA_PATH_TEMPLATE,
    MIXED_PARALLEL_VALIDATION_LIMIT,
    MIXED_LANGUAGE_CONFIGS,
    MIXED_SOURCE_SPECS,
    _allocate_source_counts,
    _format_bactrianx_sft_example,
    _format_parallel_example,
    _format_updesh_sft_example,
    _get_bactrianx_english_by_id,
    _source_dataset_stream,
    canonical_mixed_language,
    filtered_parallel_dataset_path,
    format_feature_text,
)


def iter_source_examples(spec: Dict[str, Any], language_config: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    if spec["kind"] == "bactrianx_sft":
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
            if example is not None:
                yield example
        return

    stream = _source_dataset_stream(spec, language_config)
    for raw_example in stream:
        if spec["kind"] == "updesh_sft":
            example = _format_updesh_sft_example(raw_example, language_config, spec)
        else:
            example = _format_parallel_example(raw_example, language_config, spec)
        if example is not None:
            yield example


def pair_key(example: Dict[str, Any], target_key: str) -> tuple[str, str, str]:
    translation = example["translation"]
    return example["source_name"], translation["en"], translation[target_key]


def side_lengths(
    examples: List[Dict[str, Any]],
    tokenizer,
    source_key: str,
    target_key: str,
) -> tuple[List[int], List[int]]:
    src_texts = [format_feature_text(example, source_key, "src", tokenizer) for example in examples]
    tgt_texts = [format_feature_text(example, target_key, "tgt", tokenizer) for example in examples]
    src_batch = tokenizer(src_texts, add_special_tokens=True, padding=False, truncation=False)
    tgt_batch = tokenizer(tgt_texts, add_special_tokens=True, padding=False, truncation=False)
    return (
        [len(input_ids) for input_ids in src_batch["input_ids"]],
        [len(input_ids) for input_ids in tgt_batch["input_ids"]],
    )


def is_kept(src_len: int, tgt_len: int, max_length: int, filter_sides: str) -> bool:
    if filter_sides == "source":
        return src_len <= max_length
    if filter_sides == "target":
        return tgt_len <= max_length
    return src_len <= max_length and tgt_len <= max_length


def collect_filtered_examples(
    spec: Dict[str, Any],
    language_config: Dict[str, Any],
    tokenizer,
    requested_count: int,
    seen_pairs: set,
    *,
    max_length: int,
    filter_sides: str,
    batch_size: int,
    max_scanned: Optional[int],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if requested_count <= 0:
        return [], {
            "requested": requested_count,
            "kept": 0,
            "scanned": 0,
            "skipped_duplicate": 0,
            "skipped_length": 0,
        }

    collected: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    scanned = 0
    skipped_duplicate = 0
    skipped_length = 0

    def flush_pending() -> None:
        nonlocal skipped_duplicate, skipped_length
        if not pending:
            return
        src_lengths, tgt_lengths = side_lengths(pending, tokenizer, "en", language_config["opus"])
        for example, src_len, tgt_len in zip(pending, src_lengths, tgt_lengths):
            key = pair_key(example, language_config["opus"])
            if key in seen_pairs:
                skipped_duplicate += 1
                continue
            if not is_kept(src_len, tgt_len, max_length, filter_sides):
                skipped_length += 1
                continue
            seen_pairs.add(key)
            example["source_token_length"] = src_len
            example["target_token_length"] = tgt_len
            collected.append(example)
            if len(collected) >= requested_count:
                break
        pending.clear()

    for example in iter_source_examples(spec, language_config):
        pending.append(example)
        scanned += 1
        if len(pending) >= batch_size:
            flush_pending()
            if len(collected) >= requested_count:
                break
        if max_scanned is not None and scanned >= max_scanned:
            break

    flush_pending()
    stats = {
        "requested": requested_count,
        "kept": len(collected),
        "scanned": scanned,
        "skipped_duplicate": skipped_duplicate,
        "skipped_length": skipped_length,
    }
    return collected[:requested_count], stats


def allocate_refill_counts(missing_count: int, specs: List[Dict[str, Any]]) -> Dict[str, int]:
    if missing_count <= 0 or not specs:
        return {spec["name"]: 0 for spec in specs}
    return _allocate_source_counts(missing_count, specs)


def refill_examples(
    examples: List[Dict[str, Any]],
    target_count: int,
    candidate_specs: List[Dict[str, Any]],
    language_config: Dict[str, Any],
    tokenizer,
    seen_pairs: set,
    *,
    max_length: int,
    filter_sides: str,
    batch_size: int,
    max_scanned: Optional[int],
    source_stats: Dict[str, Any],
    split_name: str,
) -> None:
    active_specs = list(candidate_specs)
    refill_round = 0

    while len(examples) < target_count and active_specs:
        missing_count = target_count - len(examples)
        refill_round += 1
        refill_counts = allocate_refill_counts(missing_count, active_specs)
        round_added = 0
        next_active_specs = []

        print(
            f"Refill round {refill_round} for {split_name}: "
            f"missing={missing_count}, candidates={[spec['name'] for spec in active_specs]}"
        )

        for spec in active_specs:
            requested = refill_counts.get(spec["name"], 0)
            if requested <= 0:
                next_active_specs.append(spec)
                continue

            extra_examples, stats = collect_filtered_examples(
                spec,
                language_config,
                tokenizer,
                requested,
                seen_pairs,
                max_length=max_length,
                filter_sides=filter_sides,
                batch_size=batch_size,
                max_scanned=max_scanned,
            )
            examples.extend(extra_examples)
            round_added += len(extra_examples)

            refill_stats = source_stats[spec["name"]].setdefault("refill", {})
            split_stats = refill_stats.setdefault(
                split_name,
                {
                    "requested": 0,
                    "kept": 0,
                    "scanned": 0,
                    "skipped_duplicate": 0,
                    "skipped_length": 0,
                },
            )
            for key in ("requested", "kept", "scanned", "skipped_duplicate", "skipped_length"):
                split_stats[key] += stats[key]

            source_stats[spec["name"]][f"{split_name}_kept"] += len(extra_examples)
            print(
                f"  refill {spec['name']}: requested={requested} kept={len(extra_examples)} "
                f"scanned={stats['scanned']}"
            )

            if len(extra_examples) >= requested:
                next_active_specs.append(spec)

        if round_added == 0:
            break
        active_specs = next_active_specs

    if len(examples) < target_count:
        print(
            f"Warning: only found {len(examples)}/{target_count} filtered {split_name} examples "
            "after exhausting refill candidates."
        )


def save_metadata(output_dir: Path, metadata: Dict[str, Any]) -> None:
    with (output_dir / "filtered_dataset_metadata.json").open("w") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a token-length-filtered mixed parallel dataset once, then load it cheaply in training.",
    )
    parser.add_argument("--language", required=True, help="mixed dataset language, e.g. kan/kn or sin/si")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="optional override; by default saves to the shared per-language filtered dataset cache",
    )
    parser.add_argument("--tokenizer", default="AIDC-AI/Marco-Nano-Instruct", help="tokenizer used for length filtering")
    parser.add_argument("--max_length", type=int, default=768, help="maximum allowed token length per selected side")
    parser.add_argument("--train_samples", type=int, default=200000)
    parser.add_argument("--valid_samples", type=int, default=MIXED_PARALLEL_VALIDATION_LIMIT)
    parser.add_argument(
        "--filter_sides",
        choices=("both", "source", "target"),
        default="both",
        help="which side must fit within max_length",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="tokenizer batch size for filtering")
    parser.add_argument(
        "--max_scanned_per_source",
        type=int,
        default=None,
        help="optional cap on raw examples scanned per source for a quick pilot run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    language = canonical_mixed_language(args.language)
    language_config = MIXED_LANGUAGE_CONFIGS[language]
    specs = MIXED_SOURCE_SPECS[language]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    output_dir = Path(args.output_dir) if args.output_dir else filtered_parallel_dataset_path(language)

    train_counts = _allocate_source_counts(args.train_samples, specs)
    valid_counts = _allocate_source_counts(args.valid_samples, specs)

    train_examples: List[Dict[str, Any]] = []
    valid_examples: List[Dict[str, Any]] = []
    seen_pairs: set = set()
    source_stats: Dict[str, Any] = {}

    for spec in specs:
        requested = train_counts[spec["name"]] + valid_counts[spec["name"]]
        print(f"Collecting {requested} filtered examples from {spec['name']}...")
        examples, stats = collect_filtered_examples(
            spec,
            language_config,
            tokenizer,
            requested,
            seen_pairs,
            max_length=args.max_length,
            filter_sides=args.filter_sides,
            batch_size=args.batch_size,
            max_scanned=args.max_scanned_per_source,
        )
        source_train_count = min(train_counts[spec["name"]], len(examples))
        source_valid_count = min(valid_counts[spec["name"]], max(0, len(examples) - source_train_count))
        train_examples.extend(examples[:source_train_count])
        valid_examples.extend(examples[source_train_count:source_train_count + source_valid_count])
        source_stats[spec["name"]] = {
            **stats,
            "train_kept": source_train_count,
            "valid_kept": source_valid_count,
        }
        print(
            f"  kept={stats['kept']} scanned={stats['scanned']} "
            f"train={source_train_count} valid={source_valid_count}"
        )

    refill_specs = [
        spec for spec in specs
        if source_stats[spec["name"]]["kept"] >= source_stats[spec["name"]]["requested"]
        and source_stats[spec["name"]]["requested"] > 0
    ]
    refill_examples(
        train_examples,
        args.train_samples,
        refill_specs,
        language_config,
        tokenizer,
        seen_pairs,
        max_length=args.max_length,
        filter_sides=args.filter_sides,
        batch_size=args.batch_size,
        max_scanned=args.max_scanned_per_source,
        source_stats=source_stats,
        split_name="train",
    )
    refill_examples(
        valid_examples,
        args.valid_samples,
        refill_specs,
        language_config,
        tokenizer,
        seen_pairs,
        max_length=args.max_length,
        filter_sides=args.filter_sides,
        batch_size=args.batch_size,
        max_scanned=args.max_scanned_per_source,
        source_stats=source_stats,
        split_name="valid",
    )

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list(train_examples),
            "validation": Dataset.from_list(valid_examples),
        }
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(output_dir)

    metadata = {
        "language": language,
        "source_key": "en",
        "target_key": language_config["opus"],
        "tokenizer": args.tokenizer,
        "max_length": args.max_length,
        "filter_sides": args.filter_sides,
        "requested_train_samples": args.train_samples,
        "requested_valid_samples": args.valid_samples,
        "actual_train_samples": len(train_examples),
        "actual_valid_samples": len(valid_examples),
        "source_stats": source_stats,
    }
    save_metadata(output_dir, metadata)
    print(f"Saved filtered dataset to {output_dir}")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
