"""ARC-Easy dataset loader (training-style, full Q+A sequences)."""
from datasets import load_dataset


def create_arc_easy_dataset(tokenizer, max_length: int = 512, split: str = "train"):
    """Return a tokenized HF Dataset with input_ids / attention_mask / labels.

    Each sample is formatted as:
        "Please choose the correct answer to the question: {question}
         (A) text (B) text ...
         Answer: {answerKey}"
    Labels equal input_ids (causal LM loss over the full sequence).
    """
    dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)

    def format_and_tokenize(examples):
        texts = []
        for question, choices, answer_key in zip(
            examples["question"], examples["choices"], examples["answerKey"]
        ):
            prompt = "Please choose the correct answer to the question: " + question
            for label, text in zip(choices["label"], choices["text"]):
                prompt += f" ({label}) {text}"
            prompt += "\nAnswer: " + answer_key
            texts.append(prompt)

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return dataset.map(
        format_and_tokenize,
        batched=True,
        batch_size=1000,
        remove_columns=dataset.column_names,
    )
