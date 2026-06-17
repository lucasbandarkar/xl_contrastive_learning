from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from accelerate.utils import merge_fsdp_weights
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer, GenerationConfig


FSDP_BASE_MODEL_HINTS = {
    "ernie": "baidu/ERNIE-4.5-21B-A3B-PT",
    "granite": "ibm-granite/granite-4.0-h-tiny",
    "gpt": "openai/gpt-oss-20b",
    "qwen35": "Qwen/Qwen3.5-35B-A3B",
    "qwen3.5": "Qwen/Qwen3.5-35B-A3B",
    "qwen": "Qwen/Qwen3-30B-A3B",
    "qwen3": "Qwen/Qwen3-30B-A3B",
    "ling": "inclusionAI/Ling-mini-2.0",
}


def checkpoint_step_sort_key(path: Path) -> tuple[int, str]:
    for parent in [path, *path.parents]:
        if parent.name.startswith("checkpoint-"):
            step_text = parent.name.removeprefix("checkpoint-")
            if step_text.isdigit():
                return int(step_text), str(path)
    return -1, str(path)


def find_fsdp_weights_dir(checkpoint_dir: Path) -> Path:
    if checkpoint_dir.name.startswith("pytorch_model_fsdp"):
        return checkpoint_dir

    candidates = sorted(checkpoint_dir.glob("pytorch_model_fsdp*"))
    if candidates:
        return candidates[-1]

    nested_candidates = sorted(
        checkpoint_dir.glob("checkpoint-*/pytorch_model_fsdp*"),
        key=checkpoint_step_sort_key,
    )
    if nested_candidates:
        return nested_candidates[-1]

    raise FileNotFoundError(f"No pytorch_model_fsdp* directory found under {checkpoint_dir}")


def infer_fsdp_base_model(model_dir: Path) -> str | None:
    path_hints = []
    current_path = model_dir
    for _ in range(4):
        path_hints.append(current_path.name)
        current_path = current_path.parent

    hint_text = " ".join(path_hints).lower()
    for hint, repo_id in FSDP_BASE_MODEL_HINTS.items():
        if hint in hint_text:
            return repo_id
    return None


def find_existing_checkpoint_exports(model_dir: Path) -> list[Path]:
    return sorted(
        path for path in model_dir.glob("checkpoint-*_vllm")
        if (path / "config.json").exists()
    )


def normalize_exported_config_for_vllm(output_dir: Path):
    normalize_gpt_oss_dequantized_config_for_vllm(output_dir)
    normalize_qwen3_moe_config_for_vllm(output_dir)
    normalize_qwen3_moe_experts_for_vllm(output_dir)
    normalize_tokenizer_config_for_transformers(output_dir)


def normalize_qwen3_moe_config_for_vllm(output_dir: Path):
    config_path = output_dir / "config.json"
    if not config_path.exists():
        return

    with config_path.open() as f:
        config = json.load(f)

    if (
        config.get("model_type") in {"qwen3_moe", "qwen3_5_moe"}
        and "num_experts" not in config
        and "num_local_experts" in config
    ):
        config["num_experts"] = config["num_local_experts"]
        with config_path.open("w") as f:
            json.dump(config, f, indent=2)
            f.write("\n")


def normalize_tokenizer_config_for_transformers(output_dir: Path):
    tokenizer_config_path = output_dir / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        return

    with tokenizer_config_path.open() as f:
        tokenizer_config = json.load(f)

    extra_special_tokens = tokenizer_config.get("extra_special_tokens")
    if not isinstance(extra_special_tokens, list):
        return

    tokenizer_config["extra_special_tokens"] = {
        f"{token.strip('<|>')}_token": token
        for token in extra_special_tokens
    }
    with tokenizer_config_path.open("w") as f:
        json.dump(tokenizer_config, f, indent=2)
        f.write("\n")


def maybe_export_fsdp_checkpoint(model_path: str, base_model: str | None = None) -> str:
    model_dir = Path(model_path)
    if not model_dir.is_dir():
        return model_path
    if (model_dir / "config.json").exists():
        normalize_exported_config_for_vllm(model_dir)
        return model_path

    output_dir = model_dir.with_name(f"{model_dir.name}_vllm")
    try:
        fsdp_weights_dir = find_fsdp_weights_dir(model_dir)
    except FileNotFoundError:
        if (output_dir / "config.json").exists():
            normalize_exported_config_for_vllm(output_dir)
            print(f"Using existing exported FSDP checkpoint at {output_dir}")
            return str(output_dir)

        if (model_dir / "training_metadata.json").exists() or list(model_dir.glob("checkpoint-*")):
            checkpoint_exports = find_existing_checkpoint_exports(model_dir)
            hint = ""
            if checkpoint_exports:
                export_list = ", ".join(str(path) for path in checkpoint_exports)
                hint = f" Existing checkpoint-level exports: {export_list}."
            raise FileNotFoundError(
                f"{model_dir} looks like a training output directory, but it has no config.json, "
                f"no FSDP weights, and no run-level export at {output_dir}."
                f"{hint}"
            )
        return model_path

    if (output_dir / "config.json").exists():
        normalize_exported_config_for_vllm(output_dir)
        print(f"Using existing exported FSDP checkpoint at {output_dir}")
        return str(output_dir)

    if base_model is None:
        base_model = infer_fsdp_base_model(model_dir)

    if base_model is None:
        raise ValueError(
            f"Found FSDP weights at {fsdp_weights_dir}, but could not infer the base model. "
            "Pass --base_model to export this checkpoint for vLLM."
        )

    export_script = Path(__file__).resolve()
    # Keep the heavy FSDP merge in a separate process so vLLM starts
    # from a clean Python process after the large tensor rewrite.
    subprocess.run(
        [
            sys.executable, str(export_script),
            "--checkpoint-dir", str(model_dir),
            "--base-model", base_model,
            "--output-dir", str(output_dir),
        ],
        check=True,
    )
    return str(output_dir)


def copy_if_exists(src_dir: Path, output_dir: Path, filename: str):
    src = src_dir / filename
    if src.exists():
        shutil.copy2(src, output_dir / filename)


def iter_exported_safetensor_paths(output_dir: Path) -> list[Path]:
    index_file = output_dir / "model.safetensors.index.json"
    if index_file.exists():
        with index_file.open("r") as f:
            index = json.load(f)
        shard_names = sorted(set(index.get("weight_map", {}).values()))
        return [output_dir / shard_name for shard_name in shard_names]

    model_file = output_dir / "model.safetensors"
    if model_file.exists():
        return [model_file]

    return sorted(output_dir.glob("*.safetensors"))


def exported_gpt_oss_weights_are_dequantized(output_dir: Path) -> bool:
    has_dense_expert_weights = False

    for path in iter_exported_safetensor_paths(output_dir):
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                if key.endswith(
                    (
                        ".mlp.experts.gate_up_proj_blocks",
                        ".mlp.experts.down_proj_blocks",
                        ".mlp.experts.gate_up_proj_scales",
                        ".mlp.experts.down_proj_scales",
                    )
                ):
                    return False

                if key.endswith(
                    (
                        ".mlp.experts.gate_up_proj",
                        ".mlp.experts.down_proj",
                    )
                ):
                    tensor_slice = f.get_slice(key)
                    if tensor_slice.get_dtype() in {"BF16", "F16", "F32"}:
                        has_dense_expert_weights = True

    return has_dense_expert_weights


def normalize_gpt_oss_dequantized_config_for_vllm(output_dir: Path):
    config_path = output_dir / "config.json"
    if not config_path.exists():
        return

    with config_path.open("r") as f:
        config = json.load(f)

    if config.get("model_type") != "gpt_oss":
        return

    quant_config = config.get("quantization_config")
    if not isinstance(quant_config, dict) or quant_config.get("quant_method") != "mxfp4":
        return

    if not exported_gpt_oss_weights_are_dequantized(output_dir):
        return

    config.pop("quantization_config", None)
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    print("Removed stale GPT-OSS MXFP4 quantization_config from dequantized BF16 export")


def normalize_qwen3_moe_experts_for_vllm(output_dir: Path):
    config_path = output_dir / "config.json"
    if not config_path.exists():
        return

    with config_path.open("r") as f:
        config = json.load(f)

    if config.get("model_type") not in {"qwen3_moe", "qwen3_5_moe"}:
        return

    safetensor_paths = iter_exported_safetensor_paths(output_dir)
    if not safetensor_paths:
        return

    has_packed_experts = False
    has_native_experts = False
    for path in safetensor_paths:
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                if key.endswith(
                    (
                        ".mlp.experts.gate_up_proj",
                        ".mlp.experts.down_proj",
                        ".mlp.experts.w13_weight",
                        ".mlp.experts.w2_weight",
                    )
                ):
                    has_packed_experts = True
                if (
                    ".mlp.experts." in key
                    and key.endswith(
                        (
                            ".gate_proj.weight",
                            ".up_proj.weight",
                            ".down_proj.weight",
                        )
                    )
                ):
                    has_native_experts = True

    if not has_packed_experts or has_native_experts:
        return

    print("Expanding Qwen3 MoE packed expert weights for vLLM")
    tmp_dir = output_dir / "_tmp_qwen3_moe_vllm_weights"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()

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

    for path in safetensor_paths:
        with safe_open(path, framework="pt") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                if name.endswith((".mlp.experts.gate_up_proj", ".mlp.experts.w13_weight")):
                    if name.endswith(".gate_up_proj"):
                        prefix = name.removesuffix(".gate_up_proj")
                    else:
                        prefix = name.removesuffix(".w13_weight")
                    split_at = tensor.shape[1] // 2
                    for expert_id in range(tensor.shape[0]):
                        add_tensor(
                            f"{prefix}.{expert_id}.gate_proj.weight",
                            tensor[expert_id, :split_at, :],
                        )
                        add_tensor(
                            f"{prefix}.{expert_id}.up_proj.weight",
                            tensor[expert_id, split_at:, :],
                        )
                elif name.endswith((".mlp.experts.down_proj", ".mlp.experts.w2_weight")):
                    if name.endswith(".down_proj"):
                        prefix = name.removesuffix(".down_proj")
                    else:
                        prefix = name.removesuffix(".w2_weight")
                    for expert_id in range(tensor.shape[0]):
                        add_tensor(
                            f"{prefix}.{expert_id}.down_proj.weight",
                            tensor[expert_id],
                        )
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

    with (tmp_dir / "model.safetensors.index.json").open("w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f)

    for old_path in safetensor_paths:
        old_path.rename(old_path.with_suffix(old_path.suffix + ".torchtitan"))
    old_index_file = output_dir / "model.safetensors.index.json"
    if old_index_file.exists():
        old_index_file.rename(output_dir / "model.safetensors.index.json.torchtitan")

    for shard_name in final_shard_names:
        shutil.move(str(tmp_dir / shard_name), output_dir / shard_name)
    shutil.move(str(tmp_dir / "model.safetensors.index.json"), output_dir / "model.safetensors.index.json")
    tmp_dir.rmdir()


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
    normalize_exported_config_for_vllm(output_dir)

    print(f"Writing tokenizer from {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    normalize_tokenizer_config_for_transformers(output_dir)

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
