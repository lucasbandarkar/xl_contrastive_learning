from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from accelerate.utils import merge_fsdp_weights
from safetensors import safe_open
from safetensors.torch import save_file
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


def normalize_ernie_experts_for_vllm(output_dir: Path):
    model_file = output_dir / "model.safetensors"
    index_file = output_dir / "model.safetensors.index.json"
    if not model_file.exists():
        return

    with safe_open(model_file, framework="pt") as f:
        has_packed_experts = any(key.endswith(".mlp.experts.down_proj") for key in f.keys())

    if not has_packed_experts:
        return

    print("Unpacking ERNIE expert weights into vLLM-compatible key names")
    tmp_dir = output_dir / "_tmp_ernie_vllm_weights"
    tmp_dir.mkdir(exist_ok=True)

    weight_map = {}
    total_size = 0
    shard_id = 0
    shard_tensors = {}
    shard_bytes = 0
    max_shard_bytes = 4 * 1024**3

    def tensor_bytes(tensor):
        return tensor.numel() * tensor.element_size()

    def flush_shard():
        nonlocal shard_id, shard_tensors, shard_bytes
        if not shard_tensors:
            return
        shard_id += 1
        shard_name = f"model-{shard_id:05d}-of-XXXXX.safetensors"
        save_file(shard_tensors, tmp_dir / shard_name, metadata={"format": "pt"})
        for tensor_name in shard_tensors:
            weight_map[tensor_name] = shard_name
        shard_tensors = {}
        shard_bytes = 0

    def add_tensor(name, tensor):
        nonlocal shard_bytes, total_size
        size = tensor_bytes(tensor)
        if shard_tensors and shard_bytes + size > max_shard_bytes:
            flush_shard()
        shard_tensors[name] = tensor.contiguous()
        shard_bytes += size
        total_size += size

    with safe_open(model_file, framework="pt") as f:
        for name in f.keys():
            tensor = f.get_tensor(name)

            if name.endswith(".mlp.experts.down_proj"):
                prefix = name.removesuffix(".experts.down_proj")
                for expert_id in range(tensor.shape[0]):
                    add_tensor(
                        f"{prefix}.experts.{expert_id}.down_proj.weight",
                        tensor[expert_id],
                    )
            elif name.endswith(".mlp.experts.gate_up_proj"):
                prefix = name.removesuffix(".experts.gate_up_proj")
                split_at = tensor.shape[1] // 2
                for expert_id in range(tensor.shape[0]):
                    add_tensor(
                        f"{prefix}.experts.{expert_id}.gate_proj.weight",
                        tensor[expert_id, :split_at, :],
                    )
                    add_tensor(
                        f"{prefix}.experts.{expert_id}.up_proj.weight",
                        tensor[expert_id, split_at:, :],
                    )
            elif ".mlp.gate.moe_statics." in name:
                add_tensor(name.replace(".mlp.gate.moe_statics.", ".mlp.moe_statics."), tensor)
            else:
                add_tensor(name, tensor)

    flush_shard()

    final_shard_names = []
    for old_shard in sorted(tmp_dir.glob("model-*-of-XXXXX.safetensors")):
        final_name = old_shard.name.replace("XXXXX", f"{shard_id:05d}")
        old_shard.rename(tmp_dir / final_name)
        final_shard_names.append(final_name)
        for tensor_name, shard_name in list(weight_map.items()):
            if shard_name == old_shard.name:
                weight_map[tensor_name] = final_name

    with (tmp_dir / index_file.name).open("w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f)

    model_file.rename(output_dir / "model.safetensors.packed")
    if index_file.exists():
        index_file.rename(output_dir / "model.safetensors.index.json.packed")

    for shard_name in final_shard_names:
        shutil.move(str(tmp_dir / shard_name), output_dir / shard_name)
    shutil.move(str(tmp_dir / index_file.name), output_dir / index_file.name)
    tmp_dir.rmdir()


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

    if config.model_type == "ernie4_5_moe":
        normalize_ernie_experts_for_vllm(output_dir)

    print(f"Writing tokenizer from {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    tokenizer_source = checkpoint_dir
    if not (tokenizer_source / "chat_template.jinja").exists():
        tokenizer_source = checkpoint_dir.parent
    copy_if_exists(tokenizer_source, output_dir, "chat_template.jinja")

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
