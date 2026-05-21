import numpy as np
import torch
from collections import defaultdict
from typing import Any


def jaccard_topk(a: torch.Tensor, b: torch.Tensor, p: float = 0.7) -> float:
    k = max(1, int((1 - p) * a.shape[-1]))
    s_a = set(torch.topk(a.abs(), k).indices.tolist())
    s_b = set(torch.topk(b.abs(), k).indices.tolist())
    inter = len(s_a & s_b)
    return inter / (2 * k - inter)  # |union| = 2k - |inter|


def measure_jaccard_e1(
    model: Any,
    dataloader: Any,
    p: float = 0.7,
    n_batches: int = 20,
    n_pairs: int = 200,
    rng_seed: int = 42,
) -> dict:
    from aslora.modules.moe import MixLoraSparseMoe

    rng = torch.Generator()
    rng.manual_seed(rng_seed)

    moe_modules = [(name, m) for name, m in model.named_modules() if isinstance(m, MixLoraSparseMoe)]
    all_layer_scores: dict[str, list[float]] = defaultdict(list)
    processed = 0

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if processed >= n_batches:
                break
            batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            model(**batch)
            processed += 1

            for name, moe in moe_modules:
                if moe.last_hidden is None or moe.last_routing is None:
                    continue
                hidden = moe.last_hidden    # [T, d]
                routing = moe.last_routing  # [T, K]

                expert_tokens: dict[int, list[int]] = defaultdict(list)
                for t in range(routing.shape[0]):
                    for k_idx in range(routing.shape[1]):
                        expert_tokens[routing[t, k_idx].item()].append(t)

                experts = sorted(expert_tokens.keys())
                pair_scores: list[float] = []

                for i in range(len(experts)):
                    for j in range(i + 1, len(experts)):
                        toks_i = expert_tokens[experts[i]]
                        toks_j = expert_tokens[experts[j]]
                        n_sample = min(n_pairs, len(toks_i) * len(toks_j))
                        for _ in range(n_sample):
                            idx_i = toks_i[torch.randint(len(toks_i), (1,), generator=rng).item()]
                            idx_j = toks_j[torch.randint(len(toks_j), (1,), generator=rng).item()]
                            pair_scores.append(jaccard_topk(hidden[idx_i], hidden[idx_j], p))

                if pair_scores:
                    all_layer_scores[name].append(float(np.mean(pair_scores)))

    per_layer = {name: float(np.mean(v)) for name, v in all_layer_scores.items() if v}
    mean_j = float(np.mean(list(per_layer.values()))) if per_layer else float("nan")
    d = moe_modules[0][1].hidden_size if moe_modules else 4096
    k = max(1, int((1 - p) * d))
    theoretical_bound = (k * k) / (2 * k * d - k * k)  # = k / (2d - k)

    return {
        "mean_jaccard": mean_j,
        "per_layer": per_layer,
        "p": p,
        "n_batches_processed": processed,
        "theoretical_bound": theoretical_bound,
    }
