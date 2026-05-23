"""Train MixLoRA (dual_scalar_gate) on ARC-Easy with Meta-Llama-3-8B.

target_sparsity=0.7  → keep top 30% of activations by magnitude.

Run from repo root:
    uv run python scripts/train_llama3_arc_easy.py
"""
import os
import sys
import time
import tempfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments
from huggingface_hub import HfApi

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "third_party", "mixlora"))

from aslora.config import create_mixlora_config_for_llama
from aslora.data.arc_easy import create_arc_easy_dataset
from aslora.eval.arc_easy import ArcEvalCallback
from aslora.modules.injection import inject_mixlora_into_llama, initialize_mixlora_weights_for_llama
from aslora.secrets import load_secrets, login_hf_and_wandb, preflight_check
from aslora.training.freezing import count_trainable_parameters, freeze_base_model, verify_freezing
from aslora.training.saving import save_mixlora_adapter_safetensors
from aslora.training.trainer import MixLoraTrainer

torch.set_float32_matmul_precision("high")

MODEL_ID    = "meta-llama/Meta-Llama-3-8B"
HF_USERNAME = "twanghcmut"
HF_REPO_NAME = "Meta-Llama-3-8B-simpa-dual_scalar_gate-arc"


def main():
    secrets = load_secrets()
    preflight_check(secrets)
    login_hf_and_wandb(secrets)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mixlora_config = create_mixlora_config_for_llama(
        base_model=MODEL_ID,
        target_sparsity=0.7,
        branch_mode="dual_scalar_gate",
        gate_rank=16, r=16, lora_alpha=32, lora_dropout=0.05,
        num_experts=8, top_k=2, router_aux_loss_coef=0.001,
    )
    mixlora_weights = initialize_mixlora_weights_for_llama(model, mixlora_config)
    model = inject_mixlora_into_llama(model, mixlora_config, mixlora_weights)

    model = freeze_base_model(model)
    verify_freezing(model)
    count_trainable_parameters(model)

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    train_dataset = create_arc_easy_dataset(tokenizer, max_length=512, split="train")

    training_args = TrainingArguments(
        output_dir=f"./{HF_REPO_NAME}_output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=10,
        learning_rate=0.00015638947792650994,
        weight_decay=0.06637618188528296,
        warmup_ratio=0.08497465885287803,
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        logging_steps=5,
        save_strategy="no",
        report_to="wandb",
        run_name=HF_REPO_NAME,
        remove_unused_columns=False,
    )

    trainer = MixLoraTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[ArcEvalCallback(tokenizer, eval_batch_size=4)],
    )

    start_time = time.time()
    trainer.train()
    total_duration = time.time() - start_time
    total_steps = trainer.state.global_step

    print(f"\n{'='*50}\nTổng thời gian: {total_duration/60:.2f} phút  |  "
          f"Avg/step: {total_duration/total_steps:.2f}s  |  Steps: {total_steps}\n{'='*50}\n")

    HF_REPO_ARC = f"{HF_USERNAME}/{HF_REPO_NAME}"
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter_dir = os.path.join(tmpdir, "adapter")
        save_mixlora_adapter_safetensors(model, mixlora_config, adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        api = HfApi(token=secrets.get("HF_TOKEN") or None)
        api.create_repo(repo_id=HF_REPO_ARC, exist_ok=True, private=secrets.get("HF_PRIVATE", False))
        api.upload_folder(
            folder_path=adapter_dir,
            repo_id=HF_REPO_ARC,
            commit_message="Upload SIMPA dual_scalar_gate adapter — ARC-Easy",
        )
        print(f"Adapter pushed to https://huggingface.co/{HF_REPO_ARC}")


if __name__ == "__main__":
    main()
