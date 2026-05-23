"""Training callbacks for MixLoRA."""
import os

from transformers import TrainerCallback

from aslora.training.saving import push_to_hub_with_card, save_mixlora_adapter_safetensors


class EpochSaveCallback(TrainerCallback):
    """Save adapter after each epoch and optionally push to HF Hub."""

    def __init__(self, model, config, run_dir: str, hub_cfg=None, secrets=None,
                 model_short: str = "", branch_mode: str = "", dataset_name: str = ""):
        self.model = model
        self.config = config
        self.run_dir = run_dir
        self.hub_cfg = hub_cfg
        self.secrets = secrets or {}
        self.model_short = model_short
        self.branch_mode = branch_mode
        self.dataset_name = dataset_name

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if state.epoch is not None else 0
        save_dir = os.path.join(self.run_dir, f"epoch_{epoch}")
        save_mixlora_adapter_safetensors(self.model, self.config, save_dir)

        if self.hub_cfg and self.hub_cfg.push_after_train:
            repo_name = self.hub_cfg.repo_template.format(
                model_short=self.model_short, branch_mode=self.branch_mode
            )
            repo_id = f"{self.hub_cfg.username}/{repo_name}"
            try:
                push_to_hub_with_card(
                    local_dir=save_dir,
                    repo_id=repo_id,
                    branch_mode=self.branch_mode,
                    dataset_name=self.dataset_name,
                    num_steps=int(state.global_step),
                    gsm8k_accuracy=None,
                    token=self.secrets.get("HF_TOKEN") or None,
                    private=self.secrets.get("HF_PRIVATE", False),
                )
            except Exception as e:
                print(f"[HF push epoch {epoch}] failed: {e!r}")
