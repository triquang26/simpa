import torch
from transformers import Trainer


class MixLoraTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Cache the MoE modules exactly once!
        self.moe_modules = [m for m in self.model.modules() if "MixLoraSparseMoe" in str(type(m))]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        outputs = model(**inputs)
        main_loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]

        if self.moe_modules:
            total_bal = sum(m.last_bal_loss for m in self.moe_modules)
            avg_bal = total_bal / len(self.moe_modules)
        else:
            avg_bal = 0.0

        final_loss = main_loss + 0.02 * avg_bal

        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "loss/task": main_loss.item(),
                "loss/balance": avg_bal.item() if torch.is_tensor(avg_bal) else avg_bal,
                "loss/total": final_loss.item()
            })

        return (final_loss, outputs) if return_outputs else final_loss
