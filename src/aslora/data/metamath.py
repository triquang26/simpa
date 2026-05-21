"""MetaMathQA data loading and preprocessing."""
import os
import functools
import pickle
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from tqdm import tqdm


template_wo_input = '''Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
'''


def cache_to_disk(root_datadir):
    def decorator_cache(func):
        @functools.wraps(func)
        def wrapper_cache(*args, **kwargs):
            if not os.path.exists(root_datadir):
                os.makedirs(root_datadir)
            func_name = func.__name__.replace("/", "")
            cache_file = os.path.join(root_datadir, f"{func_name}.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            result = func(*args, **kwargs)
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
            return result
        return wrapper_cache
    return decorator_cache


class DatasetPreprocessor:
    def __init__(self, tokenizer, tokenizer_kwargs=None):
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs or {}


class MetaMathQA100k_Preprocessor(DatasetPreprocessor):
    def __call__(self, example):
        # Tạo chuỗi input + output
        combined_text = [(x + " " + y + self.tokenizer.eos_token) for (x, y) in zip(example["x"], example["y"])]

        # Tokenize KHÔNG padding, trả về list (không phải tensor) để xử lý độ dài động
        encodings = self.tokenizer(combined_text, **self.tokenizer_kwargs)

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        labels = []

        # Tính toán độ dài phần Instruction để mask label
        for i, text in enumerate(combined_text):
            # Lấy độ dài của phần Instruction (x)
            instruction_len = len(self.tokenizer(example["x"][i], add_special_tokens=False)["input_ids"])
            # (Lưu ý: add_special_tokens=False để tính chính xác độ dài text thô)

            # Tạo label ban đầu giống input_ids
            label = input_ids[i].copy()

            # Mask phần instruction: gán -100
            # Cộng thêm 1 vài token (như bos_token) nếu tokenizer tự thêm vào đầu
            # Nhưng để an toàn và đơn giản với Llama, ta mask đoạn đầu:
            label[:instruction_len] = [-100] * instruction_len
            labels.append(label)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_meta_math(max_tokens: int = 512, model_id: str = "meta-llama/Llama-2-7b-hf",
                   num_samples: int = 100000, cache_dir: str = "data_cache"):
    """Load and filter MetaMathQA samples, cached to disk."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(
        cache_dir,
        f"load_meta_math__n{num_samples}_t{max_tokens}_{model_id.replace('/', '_')}.pkl",
    )
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    dataset = load_dataset("meta-math/MetaMathQA", split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def preprocess(data):
        return {
            "x": template_wo_input.format(instruction=data['query']),
            "y": data["response"]
        }

    train_samples = []
    count = 0
    dataset.shuffle(seed=42)
    bar = tqdm(dataset, total=num_samples + 10000)
    total = 0
    ok = 0
    for sample in dataset:
        total += 1
        temp = preprocess(sample)
        if len(tokenizer(temp['x'] + ' ' + temp['y'])['input_ids']) >= max_tokens:
            continue
        bar.update(1)
        bar.set_description(f"ok: {ok}/{total}")
        ok += 1
        processed_sample = preprocess(sample)
        if count < num_samples:
            train_samples.append(processed_sample)
        elif count >= num_samples + 10000:
            break
        count += 1

    train_samples = Dataset.from_list(train_samples)
    datasets = DatasetDict({"train": train_samples})

    with open(cache_file, "wb") as f:
        pickle.dump(datasets, f)
    return datasets


def create_meta_math_dataset(tokenizer, max_length: int = 1024,
                             dataset_name: str = "meta-math/MetaMathQA",
                             num_samples: int = 100000,
                             cache_dir: str = "data_cache"):
    """Build the tokenized MetaMathQA train dataset."""
    model_id = getattr(tokenizer, "name_or_path", "meta-llama/Llama-2-7b-hf")
    datasets = load_meta_math(
        max_tokens=max_length,
        model_id=model_id,
        num_samples=num_samples,
        cache_dir=cache_dir,
    )

    preprocessor = MetaMathQA100k_Preprocessor(
        tokenizer=tokenizer,
        tokenizer_kwargs={
            "padding": False,      # <--- QUAN TRỌNG: Tắt padding tĩnh
            "truncation": True,
            "max_length": max_length,
            # Bỏ return_tensors="pt" để trả về list, DataCollator sẽ convert sau
        },
    )

    datasets = datasets.map(
        preprocessor,
        batched=True,
        batch_size=1000,
        num_proc=4,  # Tăng num_proc lên để xử lý nhanh hơn
        remove_columns=datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    return datasets["train"]
