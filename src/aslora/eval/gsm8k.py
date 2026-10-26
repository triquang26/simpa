"""GSM8K evaluation: training callback + offline evaluator."""
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TrainerCallback, default_data_collator

from aslora.data.metamath import template_wo_input
from .utils import extract_answer, extract_gsm_num


class GSM8KEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_batch_size: int = 4, max_new_tokens: int = 512):
        self.tokenizer = tokenizer
        self.eval_batch_size = eval_batch_size
        self.max_new_tokens = max_new_tokens
        self.dataset = load_dataset("gsm8k", "main", split="test")

    def evaluate(self, model):
        # Decoder-only generation requires left-padding (see run_offline_eval).
        self.tokenizer.padding_side = "left"

        def preprocess(examples):
            examples["x"] = []
            examples["y"] = []
            for question, answer in zip(examples['question'], examples['answer']):
                examples["x"].append(template_wo_input.format(instruction=question) + " ")
                examples["y"].append(extract_gsm_num(answer))

            inputs = self.tokenizer(
                examples['x'],
                return_tensors="pt",
                max_length=768,
                padding="max_length",
                truncation=True,
                return_token_type_ids=False,
            )
            inputs["labels"] = examples["y"]
            return inputs

        eval_dataset = self.dataset.map(
            preprocess,
            batched=True,
            batch_size=1000,
            num_proc=1,
            remove_columns=self.dataset.column_names,
        )

        dataloader = DataLoader(eval_dataset, batch_size=self.eval_batch_size, collate_fn=default_data_collator)

        all_predictions = []
        all_references = []

        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating GSM8K"):
                outputs = model.generate(
                    batch["input_ids"].to(model.device),
                    attention_mask=batch["attention_mask"].to(model.device),
                    return_dict_in_generate=True,
                    output_scores=False,
                    max_new_tokens=self.max_new_tokens,
                    eos_token_id=self.tokenizer.eos_token_id,
                    top_p=0.95,
                    temperature=0.8,
                )

                predictions = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
                pred = []
                for prediction in predictions:
                    pred.append(extract_gsm_num(prediction))

                all_predictions.extend(pred)
                all_references.extend(batch["labels"].tolist())

        accuracy = sum([p == r for p, r in zip(all_predictions, all_references)]) / len(all_predictions)
        model.train()
        return accuracy

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model:
            accuracy = self.evaluate(model)
            print(f"\n{'='*50}")
            print(f"Epoch {state.epoch}: GSM8K Accuracy = {100 * accuracy:.2f}%")
            print(f"{'='*50}\n")


def run_offline_eval(model, tokenizer, batch_size: int = 4, max_new_tokens: int = 512) -> float:
    """Standalone GSM8K test-set accuracy using greedy generation."""
    device = next(model.parameters()).device
    # Decoder-only generation REQUIRES left-padding: with right-padding the pad
    # tokens sit between the prompt and the generated tokens and corrupt the
    # continuation for every non-longest sequence in a batch (systematically
    # lowers accuracy). Must be set before tokenizing the eval batches.
    tokenizer.padding_side = "left"
    dataset = load_dataset("gsm8k", "main", split="test")

    def collate_fn(batch):
        questions = [template_wo_input.format(instruction=item['question']) for item in batch]
        answers = [extract_answer(item['answer']) for item in batch]
        return questions, answers

    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for questions, gold_answers in tqdm(loader, desc="Evaluating"):
            inputs = tokenizer(
                questions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            generated_texts = tokenizer.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
            )

            for text, gold in zip(generated_texts, gold_answers):
                pred = extract_answer(text)
                if pred == gold:
                    correct += 1

            total += len(questions)

    accuracy = correct / total if total > 0 else 0.0
    print(f"Final Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    return accuracy
