"""Measure Jaccard E1 on a trained SIMPA checkpoint (column-wise output_gate).

Reconstructs the exact architecture used to train the checkpoint at
twanghcmut/Llama-2-7b-hf-gsm8k, then loads weights and runs the per-expert
Jaccard E1 measurement on a held-out MetaMathQA batch.

Usage:
    cd /mnt/data/sftp/data/quangpt3/emnlp
    uv run python scripts/measure_jaccard_aslora.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "mixlora"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from huggingface_hub import snapshot_download

from mixlora.lora_linear import LoraLinear

from aslora.data.metamath import create_meta_math_dataset
from aslora.secrets import load_secrets, login_hf_and_wandb


# === Checkpoint-specific constants (from training) ===
REPO_ID = "twanghcmut/Llama-2-7b-hf-gsm8k"
HIDDEN_SIZE = 4096
NUM_EXPERTS = 8
TOP_K = 2
GATE_RANK = 16
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
P_SPARSIFY = 0.7  # SIMPA standard: drop 70%, keep top 30%


# === Architecture reconstruction (matches the original SIMPA) ===


class SparsifyModule(nn.Module):
    def __init__(self, target_sparsity: float = P_SPARSIFY):
        super().__init__()
        self.target_sparsity = target_sparsity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.target_sparsity <= 0:
            return x
        with torch.no_grad():
            k = max(1, int(x.shape[-1] * (1 - self.target_sparsity)))
            threshold = torch.topk(x.abs(), k, dim=-1)[0][..., -1].unsqueeze(-1)
            mask = x.abs().ge(threshold).to(x.dtype)
        return x * mask


def _make_lora_config():
    from mixlora.config import LoraConfig
    return LoraConfig(
        adapter_name_="default",
        lora_init_="original",
        lora_r_=LORA_R,
        lora_alpha_=LORA_ALPHA,
        lora_dropout_=LORA_DROPOUT,
        dtype_=torch.bfloat16,
        use_dora_=False,
        use_rslora_=False,
        target_modules_={"q_proj": True, "k_proj": True, "v_proj": True, "o_proj": True,
                         "gate_proj": True, "up_proj": True, "down_proj": True},
    )


class MixLoraSparseMoe(nn.Module):
    def __init__(self, base_mlp: nn.Module):
        super().__init__()
        # base_layer_ used internally (not in state_dict). Each LoraLinear below
        # also has its own base_layer_ submodule that points to the same nn.Linear.
        object.__setattr__(self, "base_layer_", base_mlp)

        self.hidden_size = HIDDEN_SIZE
        self.total_experts = NUM_EXPERTS
        # Match checkpoint shape: gate_ has NUM_EXPERTS rows (router routes to all 8 experts,
        # including expert_0 which is also always-active as the "shared" dense path).
        self.routed_experts_count = NUM_EXPERTS
        self.moe_topk = TOP_K
        self.temperature = 0.1

        lora_cfg = _make_lora_config()
        self.experts_ = nn.ModuleList([nn.ModuleDict() for _ in range(NUM_EXPERTS)])
        for i in range(NUM_EXPERTS):
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                base = getattr(base_mlp, proj_name)
                self.experts_[i][proj_name] = LoraLinear(base, lora_cfg)

        self.gate_ = nn.Parameter(torch.randn(self.routed_experts_count, HIDDEN_SIZE) * 0.01)
        self.gate_norm = nn.LayerNorm(HIDDEN_SIZE)
        self.act_fn_ = nn.SiLU()
        self.sparsify_module = SparsifyModule(P_SPARSIFY)

        # Column-wise output_gate: Linear(d→r)→SiLU→Linear(r→d)→Sigmoid
        self.output_gate = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, GATE_RANK, bias=False),
            nn.SiLU(),
            nn.Linear(GATE_RANK, HIDDEN_SIZE, bias=True),
            nn.Sigmoid(),
        )

        self.register_buffer("last_bal_loss", torch.tensor(0.0))
        # measurement buffers
        self.last_hidden: torch.Tensor | None = None
        self.last_routing: torch.Tensor | None = None

    def _apply_expert(self, x_dense, x_sparse, expert_dict):
        g_base = self.base_layer_.gate_proj(x_dense)
        g = expert_dict["gate_proj"].lora_forward(g_base, x_sparse) if "gate_proj" in expert_dict else g_base
        u_base = self.base_layer_.up_proj(x_dense)
        u = expert_dict["up_proj"].lora_forward(u_base, x_sparse) if "up_proj" in expert_dict else u_base
        act = self.act_fn_(g) * u
        d_base = self.base_layer_.down_proj(act)
        d = expert_dict["down_proj"].lora_forward(d_base, act) if "down_proj" in expert_dict else d_base
        return d

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_shape = hidden_states.shape
        x_flat = hidden_states.view(-1, self.hidden_size)
        input_dtype = x_flat.dtype

        # Shared expert: LoRA on x_flat (dense)
        y1 = self._apply_expert(x_flat, x_flat, self.experts_[0])

        # Routed experts: LoRA on x_sparse (sparsified)
        x_sparse = self.sparsify_module(x_flat)

        x_norm = self.gate_norm(x_flat.to(self.gate_.dtype))
        logits = F.linear(x_norm, self.gate_)
        probs = F.softmax(logits / self.temperature, dim=-1)
        weights, selected = torch.topk(probs, self.moe_topk, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        # Capture for Jaccard E1 measurement (keep on GPU for vectorized topk later)
        self.last_hidden = x_flat.detach()
        self.last_routing = selected.detach()

        y2 = torch.zeros_like(x_flat)
        # Router maps directly to expert index (no shift): gate_ has 8 rows for 8 experts.
        for router_idx in range(self.gate_.shape[0]):
            token_indices, topk_indices = torch.where(selected == router_idx)
            curr_x_dense = x_flat[token_indices]
            curr_x_sparse = x_sparse[token_indices]
            out_expert = self._apply_expert(curr_x_dense, curr_x_sparse, self.experts_[router_idx])
            w = weights[token_indices, topk_indices].unsqueeze(1)
            y2 = y2.index_add(0, token_indices, out_expert * w.to(out_expert.dtype))

        gate_val = self.output_gate(x_flat).to(input_dtype)
        final = gate_val * y1 + (1.0 - gate_val) * y2
        return final.view(batch_shape)


def inject_full_simpa(model):
    """Inject the exact SIMPA architecture (attn LoRA + mlp.mixlora_moes.default)."""
    lora_cfg = _make_lora_config()
    for layer in model.model.layers:
        # Attention LoRA (q/k/v/o)
        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            base = getattr(layer.self_attn, proj_name)
            setattr(layer.self_attn, proj_name, LoraLinear(base, lora_cfg))

        # MLP: keep base Linear layers, add mixlora_moes.default with the MoE
        moe = MixLoraSparseMoe(layer.mlp)
        layer.mlp.add_module("mixlora_moes", nn.ModuleDict({"default": moe}))

        # Override mlp.forward to call the MoE
        def make_forward(mlp_module):
            def fwd(x, **kwargs):
                return mlp_module.mixlora_moes["default"](x)
            return fwd
        layer.mlp.forward = make_forward(layer.mlp)


inject_full_aslora = inject_full_simpa


# === Jaccard E1 measurement ===


def verify_assumptions(model, dataloader, p: float = 0.7, n_batches: int = 20, n_pairs_var: int = 10000) -> dict:
    """Verify (A1) pairwise lazy-neuron and (A2) variance control.

    Definitions on a held-out batch:
        Z_i(t1, t2) = 1{i ∈ S_p(x^t1) ∩ S_p(x^t2)}
        Y(t1, t2)  = Σ_i Z_i = |S_p(x^t1) ∩ S_p(x^t2)|
    Quantities (per layer, mean across layers):
        (A1) max_i Pr[i ∈ both] = max_i (c_i/T)^2     bound: ≤ q²
             mean over i:                              bound: ≈ q² (sanity)
        (A2) Var(Y) / (d · q²)                        bound: ≤ c moderate
    """
    moe_modules = [(name, m) for name, m in model.named_modules() if isinstance(m, MixLoraSparseMoe)]
    device = next(model.parameters()).device
    q = 1.0 - p
    q2 = q * q
    d = HIDDEN_SIZE
    k_keep = max(1, int(q * d))

    per_layer_data: dict[str, dict] = defaultdict(lambda: {
        "c_i_sum": torch.zeros(d, device=device),
        "T_total": 0,
        "intersections": [],
    })

    model.eval()
    processed = 0
    with torch.no_grad():
        for batch in dataloader:
            if processed >= n_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            attn = batch["attention_mask"].view(-1).bool()
            print(f"      [batch {processed+1}/{n_batches}] forward + assumption verification...", flush=True)
            model(**batch)
            processed += 1

            for name, moe in moe_modules:
                hidden = moe.last_hidden
                if hidden is None:
                    continue
                T = hidden.shape[0]
                _, topk_idx = torch.topk(hidden.abs().float(), k_keep, dim=-1)
                M = torch.zeros(T, d, dtype=torch.bool, device=device)
                M.scatter_(1, topk_idx, True)
                M_v = M[attn]
                Tv = M_v.shape[0]
                if Tv < 2:
                    continue
                # (A1) accumulate per-coord counts
                per_layer_data[name]["c_i_sum"] += M_v.float().sum(0)
                per_layer_data[name]["T_total"] += Tv

                # (A2) sample pairs (with replacement, avoid t==t)
                n_s = min(n_pairs_var, Tv * (Tv - 1))
                ii = torch.randint(Tv, (n_s,), device=device)
                jj = torch.randint(Tv, (n_s,), device=device)
                bad = ii == jj
                if bad.any():
                    jj[bad] = (jj[bad] + 1) % Tv
                inter = (M_v[ii] & M_v[jj]).sum(-1).float()  # [n_s]
                per_layer_data[name]["intersections"].append(inter)

                moe.last_hidden = None
                moe.last_routing = None
                del M, topk_idx, M_v
            torch.cuda.empty_cache()

    layer_rows = {}
    for name, data in per_layer_data.items():
        Tt = data["T_total"]
        c_i = data["c_i_sum"]
        rate = (c_i / Tt) ** 2                   # per-coord Pr[i in both]
        all_inter = torch.cat(data["intersections"])
        var_Y = all_inter.var(unbiased=False).item()
        layer_rows[name] = {
            "max_cooc": rate.max().item(),
            "mean_cooc": rate.mean().item(),
            "mean_inter": all_inter.mean().item(),
            "var_inter": var_Y,
            "norm_var": var_Y / (d * q2),
        }

    # Aggregate across layers (means as paper reports)
    keys = ["max_cooc", "mean_cooc", "mean_inter", "var_inter", "norm_var"]
    agg = {k: float(np.mean([r[k] for r in layer_rows.values()])) for k in keys}
    agg["max_cooc_overall"] = float(max(r["max_cooc"] for r in layer_rows.values()))
    agg["p"] = p
    agg["q2"] = q2
    agg["d"] = d
    return agg, layer_rows


def measure_jaccard(model, dataloader, p: float = 0.7, n_batches: int = 20, n_pairs: int = 200) -> dict:
    """Vectorized Jaccard E1 with cross-prompt sampling.

    Assumption (A2) requires independence between paired tokens. Tokens within the same
    prompt share context (math vocab, format) and violate independence, inflating J̄.
    We restrict pair sampling to tokens from DIFFERENT sequences (cross-prompt) and
    filter padding tokens via attention_mask. This matches the paper's intent.
    """
    moe_modules = [(name, m) for name, m in model.named_modules() if isinstance(m, MixLoraSparseMoe)]
    layer_scores: dict[str, list[float]] = defaultdict(list)
    device = next(model.parameters()).device

    model.eval()
    processed = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if processed >= n_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            attn_mask_2d = batch["attention_mask"].bool()  # [B, seq_len]
            B, seq_len = attn_mask_2d.shape
            valid_mask = attn_mask_2d.view(-1)  # [B*seq_len]

            print(f"      [batch {processed+1}/{n_batches}] forward pass (B={B}, seq={seq_len}, valid={valid_mask.sum().item()})...", flush=True)
            model(**batch)
            processed += 1

            print(f"      [batch {processed}/{n_batches}] computing Jaccard (cross-prompt)...", flush=True)
            for name, moe in moe_modules:
                hidden = moe.last_hidden     # [T, d] on GPU, bf16
                routing = moe.last_routing   # [T, K] on GPU, int64
                if hidden is None or routing is None:
                    continue

                T, d = hidden.shape
                K = routing.shape[1]
                k_keep = max(1, int((1 - p) * d))
                seq_idx_per_token = torch.arange(T, device=device) // seq_len  # [T] which sequence

                # 1) Top-k bool mask per token
                _, topk_idx = torch.topk(hidden.abs().float(), k_keep, dim=-1)  # [T, k]
                masks = torch.zeros(T, d, dtype=torch.bool, device=device)
                masks.scatter_(1, topk_idx, True)

                # 2) Group VALID tokens by expert (filter padding)
                routing_flat = routing.view(-1)  # [T*K]
                token_idx_repeated = torch.arange(T, device=device).repeat_interleave(K)
                valid_rep = valid_mask[token_idx_repeated]  # [T*K]

                expert_token_lists: dict[int, torch.Tensor] = {}
                for e in routing_flat.unique().tolist():
                    sel = (routing_flat == e) & valid_rep
                    expert_token_lists[e] = token_idx_repeated[sel]

                # 3) Cross-expert × cross-prompt sampling
                experts_list = sorted(expert_token_lists.keys())
                pair_scores: list[float] = []
                for i in range(len(experts_list)):
                    for j in range(i + 1, len(experts_list)):
                        ti = expert_token_lists[experts_list[i]]
                        tj = expert_token_lists[experts_list[j]]
                        if len(ti) == 0 or len(tj) == 0:
                            continue
                        # Oversample candidates (4×) and filter cross-prompt only
                        n_cand = min(n_pairs * 4, len(ti) * len(tj))
                        idx_i_c = ti[torch.randint(len(ti), (n_cand,), device=device)]
                        idx_j_c = tj[torch.randint(len(tj), (n_cand,), device=device)]
                        cross_prompt = seq_idx_per_token[idx_i_c] != seq_idx_per_token[idx_j_c]
                        idx_i = idx_i_c[cross_prompt][:n_pairs]
                        idx_j = idx_j_c[cross_prompt][:n_pairs]
                        if len(idx_i) == 0:
                            continue
                        inter = (masks[idx_i] & masks[idx_j]).sum(dim=-1).float()
                        union = 2 * k_keep - inter
                        pair_scores.append((inter / union).mean().item())

                del masks, topk_idx
                moe.last_hidden = None
                moe.last_routing = None

                if pair_scores:
                    layer_scores[name].append(float(np.mean(pair_scores)))

            torch.cuda.empty_cache()

    per_layer = {n: float(np.mean(v)) for n, v in layer_scores.items() if v}
    mean_j = float(np.mean(list(per_layer.values()))) if per_layer else float("nan")
    d = HIDDEN_SIZE
    k = max(1, int((1 - p) * d))
    bound = (k * k) / (2 * k * d - k * k)
    return {
        "mean_jaccard": mean_j,
        "per_layer": per_layer,
        "p": p,
        "n_batches": processed,
        "theoretical_bound": bound,
    }


# === Main ===


def main():
    secrets = load_secrets()
    login_hf_and_wandb(secrets)

    print(f"[1/4] Loading base Llama-2-7b-hf...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[2/4] Injecting SIMPA architecture (attn LoRA + mlp.mixlora_moes.default + output_gate)...")
    inject_full_simpa(model)
    # Move all newly-injected modules onto the same device/dtype as base model
    device = next(model.parameters()).device
    model.to(device=device, dtype=torch.bfloat16)

    print(f"[3/4] Downloading checkpoint {REPO_ID}...")
    ckpt_dir = snapshot_download(repo_id=REPO_ID, token=secrets.get("HF_TOKEN") or None)
    print(f"      Loading sharded .bin files from {ckpt_dir}...")
    import json as _json
    import os as _os
    idx_path = _os.path.join(ckpt_dir, "pytorch_model.bin.index.json")
    with open(idx_path) as f:
        idx = _json.load(f)
    shard_files = sorted(set(idx["weight_map"].values()))
    state_dict: dict[str, torch.Tensor] = {}
    for sf in shard_files:
        shard_path = _os.path.join(ckpt_dir, sf)
        print(f"      loading {sf}...")
        shard = torch.load(shard_path, map_location="cpu", weights_only=True)
        state_dict.update(shard)
        del shard
    print(f"      Total state dict keys: {len(state_dict)}")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"      missing keys:    {len(missing)}")
    print(f"      unexpected keys: {len(unexpected)}")
    if missing[:5]:
        print(f"      first missing:   {missing[:5]}")
    if unexpected[:5]:
        print(f"      first unexpected: {unexpected[:5]}")
    del state_dict
    import gc as _gc; _gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    model.eval()

    print(f"[4/4] Building MetaMathQA dataloader (2048 samples, batch_size=8)...")
    dataset = create_meta_math_dataset(
        tokenizer,
        max_length=1024,
        dataset_name="meta-math/MetaMathQA",
        num_samples=2048,
        cache_dir="data_cache",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        collate_fn=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, pad_to_multiple_of=8),
        shuffle=False,
    )

    print(f"      Running assumption verification (A1) (A2) (p={P_SPARSIFY})...")
    agg_av, per_layer_av = verify_assumptions(model, dataloader, p=P_SPARSIFY, n_batches=20, n_pairs_var=10000)

    print()
    print("=" * 66)
    print(f"  Appendix B Table 9 — (A1) and (A2) verification, p={P_SPARSIFY}")
    print("=" * 66)
    q2 = (1 - P_SPARSIFY) ** 2
    print(f"  Reported as MEANS across all transformer layers:")
    print(f"  Max-coord. co-occurrence rate    {agg_av['max_cooc']:.4f}    bound ≤ {q2:.2f}    {'✓' if agg_av['max_cooc'] <= q2 else '×'}")
    print(f"    (overall MAX across layers     {agg_av['max_cooc_overall']:.4f})")
    print(f"  Mean co-occurrence rate          {agg_av['mean_cooc']:.4f}    bound ≤ {q2:.2f}    {'✓' if agg_av['mean_cooc'] <= q2 else '×'}")
    print(f"  Normalized var. Var(Y)/(d·q²)    {agg_av['norm_var']:.4f}    bound ≤ c (typ. 1-5)")
    print(f"  Aux: E[|S∩S'|] / (d·q²)          {agg_av['mean_inter'] / (HIDDEN_SIZE * q2):.4f}    (should be ≈ 1)")
    print("=" * 66)
    print()
    print(f"  Per-layer breakdown:")
    print(f"    {'layer':6s} {'max_cooc':>10s} {'mean_cooc':>10s} {'norm_var':>10s}")
    for name, row in sorted(per_layer_av.items()):
        short = name.replace("model.layers.", "L").replace(".mlp.mixlora_moes.default", "")
        print(f"    {short:6s} {row['max_cooc']:>10.4f} {row['mean_cooc']:>10.4f} {row['norm_var']:>10.4f}")
    print("=" * 66)
    print()

    print(f"      Running Jaccard E1 measurement (p={P_SPARSIFY}, 20 batches × 8 = 160 sequences)...")
    # Reset dataloader (consumed by verify_assumptions)
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        collate_fn=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, pad_to_multiple_of=8),
        shuffle=False,
    )
    res = measure_jaccard(model, dataloader, p=P_SPARSIFY, n_batches=20, n_pairs=200)

    print()
    print("=" * 66)
    print(f"  Jaccard E1 — SIMPA (column-wise output_gate), p={P_SPARSIFY}")
    print("=" * 66)
    print(f"  Mean Jaccard:                 {res['mean_jaccard']:.4f}")
    print(f"  Theoretical bound (Prop. 1):  {res['theoretical_bound']:.4f}")
    print(f"  Assumption holds (J̄ ≤ bound): {res['mean_jaccard'] <= res['theoretical_bound']}")
    print(f"  Batches processed:            {res['n_batches']}")
    print()
    print(f"  Per-layer breakdown:")
    for name, j in sorted(res["per_layer"].items()):
        short = name.replace("model.layers.", "L").replace(".mlp.mixlora_moes.default", "")
        print(f"    {short:30s}  J̄ = {j:.4f}")
    print("=" * 66)
    print()
    print(f"  Paper Table 6 entries:")
    print(f"    Method     p    J̄        Bound (Prop. 1)")
    print(f"    MixLoRA    0.0  1.000    1.000   (trivial: no mask → S_p = all d dims)")
    print(f"    SIMPA     0.7  {res['mean_jaccard']:.3f}    {res['theoretical_bound']:.3f}")


if __name__ == "__main__":
    main()
