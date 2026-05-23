"""ARC-Easy multiple-choice benchmark via log-likelihood scoring."""
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import TrainerCallback


class ArcEvalCallback(TrainerCallback):
    """Evaluate ARC-Easy test accuracy at the end of each training epoch."""

    def __init__(self, tokenizer, eval_batch_size: int = 8):
        self.tokenizer = tokenizer
        self.eval_batch_size = eval_batch_size

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        acc = run_arc_easy_eval(model, self.tokenizer, batch_size=self.eval_batch_size)
        print(f"\n{'='*50}")
        print(f"Epoch {int(state.epoch)}: ARC-Easy Accuracy = {acc * 100:.2f}%")
        print(f"{'='*50}\n")
        model.train()


def run_arc_easy_eval(model, tokenizer, batch_size: int = 4) -> float:
    """Score ARC-Easy test set using log-prob of the answer label token.

    One forward pass per batch. Prompts are left-padded so every sequence
    ends at 'Answer:' — we read logits[:, -1, :] and compare the log-prob
    of each candidate label (A / B / C / D).
    """
    device = next(model.parameters()).device
    dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")

    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    def build_prompt(question, choices):
        prompt = "Please choose the correct answer to the question: " + question
        for label, text in zip(choices["label"], choices["text"]):
            prompt += f" ({label}) {text}"
        prompt += "\nAnswer:"
        return prompt

    correct = 0
    total = 0

    examples = list(dataset)
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(examples), batch_size), desc="Evaluating ARC-Easy"):
            batch = examples[start : start + batch_size]
            prompts = [build_prompt(ex["question"], ex["choices"]) for ex in batch]
            answer_keys = [ex["answerKey"] for ex in batch]
            choice_label_sets = [ex["choices"]["label"] for ex in batch]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            outputs = model(**inputs)
            # logits at the last token position (= first token after "Answer:")
            last_logits = outputs.logits[:, -1, :]  # [B, V]
            log_probs = F.log_softmax(last_logits, dim=-1)  # [B, V]

            for i, (answer_key, labels) in enumerate(zip(answer_keys, choice_label_sets)):
                best_label = None
                best_score = float("-inf")
                for label in labels:
                    # Encode with a leading space to match how LLaMA tokenises "A"
                    token_ids = tokenizer.encode(" " + label, add_special_tokens=False)
                    if not token_ids:
                        continue
                    score = log_probs[i, token_ids[0]].item()
                    if score > best_score:
                        best_score = score
                        best_label = label

                if best_label == answer_key:
                    correct += 1
                total += 1

    tokenizer.padding_side = orig_padding_side
    model.train()

    accuracy = correct / total if total > 0 else 0.0
    print(f"ARC-Easy Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
    return accuracy
