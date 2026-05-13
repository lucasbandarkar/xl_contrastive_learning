from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from export_fsdp_checkpoint import normalize_ernie_experts_for_vllm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint_dir",
        type=Path,
        help="Exported ERNIE _vllm checkpoint directory to repair.",
    )
    parser.add_argument(
        "--base-model",
        default="baidu/ERNIE-4.5-21B-A3B-PT",
        help="Original ERNIE HF model used to refresh tokenizer files.",
    )
    args = parser.parse_args()

    print(f"Refreshing tokenizer files from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.checkpoint_dir)

    normalize_ernie_experts_for_vllm(args.checkpoint_dir)
    print(f"Fixed ERNIE vLLM checkpoint at {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
