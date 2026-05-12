from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from accelerate.utils import merge_fsdp_weights
from transformers import AutoConfig, AutoTokenizer, GenerationConfig


def find_fsdp_weights_dir(checkpoint_dir: Path) -> Path:
    if checkpoint_dir.name.startswith("pytorch_model_fsdp"):
        return checkpoint_dir

    candidates = sorted(checkpoint_dir.glob("pytorch_model_fsdp*"))
    if candidates:
        return candidates[-1]

    nested_candidates = sorted(checkpoint_dir.glob("checkpoint-*/pytorch_model_fsdp*"))
    if nested_candidates:
        return nested_candidates[-1]

    raise FileNotFoundError(f"No pytorch_model_fsdp* directory found under {checkpoint_dir}")


def copy_if_exists(src_dir: Path, output_dir: Path, filename: str):
    src = src_dir / filename
    if src.exists():
        shutil.copy2(src, output_dir / filename)


def export_checkpoint(checkpoint_dir: Path, base_model: str, output_dir: Path):
    fsdp_weights_dir = find_fsdp_weights_dir(checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Merging FSDP weights from {fsdp_weights_dir} into {output_dir}")
    merge_fsdp_weights(
        checkpoint_dir=str(fsdp_weights_dir),
        output_path=str(output_dir),
        safe_serialization=True,
    )

    print(f"Writing config from {base_model}")
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    config.save_pretrained(output_dir)

    try:
        generation_config = GenerationConfig.from_pretrained(base_model, trust_remote_code=True)
        generation_config.save_pretrained(output_dir)
    except Exception:
        pass

    tokenizer_source = checkpoint_dir
    if not (tokenizer_source / "tokenizer_config.json").exists():
        tokenizer_source = checkpoint_dir.parent

    if (tokenizer_source / "tokenizer_config.json").exists():
        print(f"Copying tokenizer files from {tokenizer_source}")
        for filename in [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "chat_template.jinja",
        ]:
            copy_if_exists(tokenizer_source, output_dir, filename)
    else:
        print(f"Writing tokenizer from {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)

    print(f"Exported vLLM-loadable checkpoint to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        type=Path,
        help="Training checkpoint directory or a checkpoint-NNN directory containing pytorch_model_fsdp_*.",
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Original Hugging Face model ID used for training, e.g. baidu/ERNIE-4.5-21B-A3B-PT.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the merged Hugging Face checkpoint.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.checkpoint_dir.with_name(f"{args.checkpoint_dir.name}_merged")

    export_checkpoint(args.checkpoint_dir, args.base_model, output_dir)


if __name__ == "__main__":
    main()
