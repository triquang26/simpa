import os
import json
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from safetensors.torch import save_file as safe_save_file


def extract_mixlora_weights_clean(model: nn.Module) -> Dict[str, torch.Tensor]:
    adapter_weights = {}
    target_keywords = ["lora_", "moe_gate", "gate_", "scalar_gate"]
    for name, param in model.named_parameters():
        if param.requires_grad and any(k in name for k in target_keywords):
            adapter_weights[name] = param.detach().cpu().clone()
    print(f"Extracted {len(adapter_weights)} trainable tensors.")
    return adapter_weights


def save_mixlora_adapter_safetensors(model: nn.Module, config: Any, save_directory: str):
    os.makedirs(save_directory, exist_ok=True)
    weights = extract_mixlora_weights_clean(model)
    config_dict = config.export()
    config_dict.update({
        'target_sparsity': float(getattr(config, 'target_sparsity', 0.3)),
        'models_path': getattr(config, 'models_path', ''),
        'branch_mode': getattr(config, 'branch_mode', 'sparse_only'),
        'gate_rank': getattr(config, 'gate_rank', 16),
    })
    with open(os.path.join(save_directory, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    safe_save_file(weights, os.path.join(save_directory, "adapter_model.safetensors"))
    print(f"Adapter saved to {save_directory}")


def save_full_adapter_for_resume(model, config, tokenizer, save_dir: str):
    save_mixlora_adapter_safetensors(model, config, save_dir)
    tokenizer.save_pretrained(save_dir)


def push_to_hub_with_card(
    local_dir: str,
    repo_id: str,
    branch_mode: str,
    dataset_name: str,
    num_steps: int,
    gsm8k_accuracy: Optional[float],
    token: str,
    private: bool = False,
):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True, private=private, token=token)

    acc_line = f"- GSM8K accuracy: {gsm8k_accuracy:.4f}\n" if gsm8k_accuracy is not None else ""
    card = (
        f"# {repo_id}\n\n"
        f"MixLoRA adapter (ASLoRA) trained on top of `meta-llama/Llama-2-7b-hf`.\n\n"
        f"## Training details\n\n"
        f"- Branch mode: `{branch_mode}`\n"
        f"- Dataset: `{dataset_name}`\n"
        f"- Training steps: {num_steps}\n"
        f"{acc_line}"
        f"- Base model: `meta-llama/Llama-2-7b-hf`\n"
    )
    with open(os.path.join(local_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)

    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        token=token,
        commit_message=f"Upload ASLoRA adapter ({branch_mode})",
    )
