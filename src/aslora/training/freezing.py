import torch
import torch.nn as nn
from transformers import PreTrainedModel
from mixlora.lora_linear import LoraLinear
from aslora.modules.moe import MixLoraSparseMoe


def freeze_base_model(model: PreTrainedModel) -> PreTrainedModel:
    for param in model.parameters():
        param.requires_grad = False
    mixlora_trainable = 0

    def unfreeze_mixlora_module(module):
        nonlocal mixlora_trainable
        if isinstance(module, LoraLinear):
            for name, param in module.named_parameters():
                if 'lora_' in name or 'magnitude_vector' in name:
                    param.requires_grad = True
                    mixlora_trainable += param.numel()
            return
        if isinstance(module, MixLoraSparseMoe):
            if hasattr(module, 'gate_') and module.gate_ is not None:
                module.gate_.requires_grad = True
                mixlora_trainable += module.gate_.numel()
            if hasattr(module, 'experts_'):
                for expert in module.experts_:
                    unfreeze_mixlora_module(expert)
            if hasattr(module, 'output_gate_network'):
                for param in module.output_gate_network.parameters():
                    param.requires_grad = True
                    mixlora_trainable += param.numel()
            if hasattr(module, 'scalar_gate'):
                for param in module.scalar_gate.parameters():
                    param.requires_grad = True
                    mixlora_trainable += param.numel()
            return
        for child in module.children():
            unfreeze_mixlora_module(child)

    unfreeze_mixlora_module(model)
    print(f"Trainable MixLoRA params: {mixlora_trainable:,}")
    return model


def verify_freezing(model: PreTrainedModel) -> bool:
    mixlora_keywords = ['lora_', 'moe_gate', 'gate_', 'scalar_gate']
    base_trainable = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if not any(k in name.lower() for k in mixlora_keywords):
                base_trainable.append(name)
    if base_trainable:
        print(f"FAILED: {len(base_trainable)} base parameters are trainable!")
        return False
    return True


def count_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} ({100 * trainable / total:.4f}%) | Total: {total:,}")
    return trainable, total
