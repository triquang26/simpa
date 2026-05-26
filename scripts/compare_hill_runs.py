"""Cross-run comparison of Hill estimator + p* predictions for SIMPA paper.

Reads multiple hill_results.json files (one per backbone/mode) and renders:

  fig_headline_down_proj.pdf       -- 3-panel Hill plot at down_proj across
                                      runs. Headline figure: shared y-axis,
                                      direct visual comparison of plateau.
  fig_all_locations_grid.pdf       -- 4x3 grid: 4 locations x 3 runs of Hill
                                      curves overlaying 5 layers each.
  fig_beta_per_location.pdf        -- mean beta_hat per (location, run) bar
                                      chart, with grid-line for beta=1.34.
  fig_pstar_per_loc_across_runs.pdf-- predicted p* per location, overlaid
                                      across runs, with grid p*=0.7 line.
  table_summary.tex                -- LaTeX Table 1: per-run summary.
  table_per_location.tex           -- LaTeX Table 2: per-location mean beta
                                      and p* across runs.

Default reads:
  outputs/hill_pstar/hill_results.json           (Llama-2-7B SIMPA-trained)
  outputs/hill_pstar/llama2_7b_pretrained.json   (Llama-2-7B pretrained)
  outputs/hill_pstar/llama3_8b_pretrained.json   (Llama-3-8B pretrained)

Missing files are skipped with a warning. Outputs to outputs/hill_pstar/comparison/.

Run:
    uv run python scripts/compare_hill_runs.py
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.pop("MPLBACKEND", None)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


LOCATION_ORDER = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DEDUP_LOCATIONS = ["q_proj", "o_proj", "gate_proj", "down_proj"]  # one per distinct hook tensor


DEFAULT_RUNS: list[tuple[str, str]] = [
    ("Llama-2-7B SIMPA-trained", "outputs/hill_pstar/hill_results.json"),
    ("Llama-2-7B pretrained",    "outputs/hill_pstar/llama2_7b_pretrained.json"),
    ("Llama-3-8B pretrained",    "outputs/hill_pstar/llama3_8b_pretrained.json"),
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help='Override runs as "label:path" pairs, e.g. --runs "Llama-2 trained:a.json" "Llama-3:b.json"',
    )
    ap.add_argument("--out-dir", default="outputs/hill_pstar/comparison")
    ap.add_argument(
        "--curve-layers",
        type=int,
        nargs="+",
        default=[0, 8, 15, 23, 31],
        help="Layer indices overlaid in the Hill-curve panels.",
    )
    return ap.parse_args()


def resolve_runs(args) -> list[tuple[str, Path]]:
    if args.runs:
        pairs: list[tuple[str, Path]] = []
        for entry in args.runs:
            if ":" not in entry:
                raise ValueError(f"--runs entry must be 'label:path', got {entry!r}")
            label, path = entry.split(":", 1)
            pairs.append((label.strip(), Path(path.strip())))
        return pairs
    return [(label, Path(p)) for label, p in DEFAULT_RUNS]


def load_runs(runs: list[tuple[str, Path]]) -> list[dict]:
    loaded = []
    for label, path in runs:
        if not path.exists():
            print(f"[warn] skipping {label}: {path} not found")
            continue
        with path.open() as f:
            data = json.load(f)
        data["_label"] = label
        data["_path"] = str(path)
        loaded.append(data)
        print(f"[load] {label}  <-  {path}")
    if not loaded:
        raise RuntimeError("No runs loaded; supply --runs or place JSON files at default paths.")
    return loaded


def group_by_location(per_loc: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for row in per_loc:
        out[row["location"]].append(row)
    for loc in out:
        out[loc].sort(key=lambda r: r["layer"])
    return out


# ============================================================
# Plot 1: HEADLINE figure -- down_proj Hill plot across runs
# ============================================================
def plot_headline_down_proj(loaded: list[dict], layers: list[int], out_path: Path):
    n = len(loaded)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.0), sharey=True)
    if n == 1:
        axes = [axes]
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(layers) - 1)) for i in range(len(layers))]

    for ax, run in zip(axes, loaded):
        grouped = group_by_location(run["per_location"])
        rows = grouped.get("down_proj", [])
        rows_by_layer = {r["layer"]: r for r in rows}
        for color, layer in zip(colors, layers):
            r = rows_by_layer.get(layer)
            if r is None:
                continue
            hp = r.get("hill_plot", {})
            ks, betas = hp.get("k", []), hp.get("beta_hat", [])
            if not ks:
                continue
            ax.plot(ks, betas, color=color, lw=1.8, label=f"L{layer:02d}")

        # reference plateau band 1.3-1.7
        ax.axhspan(1.3, 1.7, color="lightcoral", alpha=0.12, lw=0)
        ax.axhline(1.5, color="red", lw=0.7, ls="--", alpha=0.6)

        ax.set_xscale("log")
        ax.set_xlabel("k (upper order statistics)")
        ax.set_title(run["_label"])
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best", fontsize=8, ncol=1)

    axes[0].set_ylabel(r"$\hat{\beta}$ (Hill estimate)")
    fig.suptitle(
        r"Hill plot at down_proj: $\hat{\beta}(k)$ across backbones. "
        r"Shaded band: $\hat{\beta} \in [1.3, 1.7]$. Dashed line: $\hat{\beta}=1.5$ "
        r"$\Rightarrow p^*=0.66$.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out_path}")


# ============================================================
# Plot 2: 4 locations x N runs grid
# ============================================================
def plot_all_locations_grid(loaded: list[dict], layers: list[int], out_path: Path):
    locs = DEDUP_LOCATIONS  # 4 locations
    n_runs = len(loaded)
    fig, axes = plt.subplots(len(locs), n_runs, figsize=(4.0 * n_runs, 3.2 * len(locs)),
                             sharex=False, sharey="row")
    if len(locs) == 1:
        axes = np.array([axes])
    if n_runs == 1:
        axes = axes.reshape(-1, 1)

    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(layers) - 1)) for i in range(len(layers))]

    for col, run in enumerate(loaded):
        grouped = group_by_location(run["per_location"])
        for row, loc in enumerate(locs):
            ax = axes[row, col]
            rows = grouped.get(loc, [])
            rows_by_layer = {r["layer"]: r for r in rows}
            for color, layer in zip(colors, layers):
                r = rows_by_layer.get(layer)
                if r is None:
                    continue
                hp = r.get("hill_plot", {})
                ks, betas = hp.get("k", []), hp.get("beta_hat", [])
                if not ks:
                    continue
                ax.plot(ks, betas, color=color, lw=1.2, label=f"L{layer:02d}", alpha=0.85)
            ax.set_xscale("log")
            ax.grid(True, which="both", alpha=0.2)
            if row == 0:
                ax.set_title(run["_label"], fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{loc}\n" + r"$\hat{\beta}$", fontsize=9)
            if row == len(locs) - 1:
                ax.set_xlabel("k", fontsize=9)
            if row == 0 and col == n_runs - 1:
                ax.legend(loc="best", fontsize=7, ncol=1)

    fig.suptitle(r"Hill plots across 4 locations (rows) $\times$ runs (cols). "
                 r"Flat plateau = Pareto tail; drift = non-Pareto.", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out_path}")


# ============================================================
# Plot 3: per-location mean beta bar chart across runs
# ============================================================
def plot_beta_per_location_bar(loaded: list[dict], out_path: Path):
    n_runs = len(loaded)
    locs = ["q_proj", "o_proj", "gate_proj", "down_proj"]  # dedup
    means = np.full((len(locs), n_runs), np.nan)
    stds = np.full((len(locs), n_runs), np.nan)
    for ci, run in enumerate(loaded):
        grouped = group_by_location(run["per_location"])
        for li, loc in enumerate(locs):
            betas = [r["beta_hat"] for r in grouped.get(loc, []) if np.isfinite(r["beta_hat"])]
            if betas:
                means[li, ci] = np.mean(betas)
                stds[li, ci] = np.std(betas)

    x = np.arange(len(locs))
    w = 0.8 / max(1, n_runs)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    cmap = plt.get_cmap("tab10")
    for ci, run in enumerate(loaded):
        offsets = (ci - (n_runs - 1) / 2) * w
        ax.bar(x + offsets, means[:, ci], width=w, yerr=stds[:, ci],
               capsize=3, color=cmap(ci), label=run["_label"], alpha=0.85)

    ax.axhline(1.34, color="red", ls="--", lw=1, label=r"$\beta=1.34$ (SIMPA paper anchor)")
    ax.axhline(1.50, color="orange", ls=":", lw=1, label=r"$\beta=1.50$ (observed at down_proj)")
    ax.set_xticks(x)
    ax.set_xticklabels(locs)
    ax.set_ylabel(r"$\hat{\beta}$ (mean over 32 layers $\pm$ 1 std)")
    ax.set_title(r"Mean $\hat{\beta}$ per location across runs")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out_path}")


# ============================================================
# Plot 4: predicted p* per location across runs
# ============================================================
def plot_pstar_per_loc_across_runs(loaded: list[dict], out_path: Path):
    fig, axes = plt.subplots(1, len(loaded), figsize=(4.5 * len(loaded), 4.0), sharey=True)
    if len(loaded) == 1:
        axes = [axes]
    cmap = plt.get_cmap("tab10")
    p_grid = float(loaded[0].get("config", {}).get("p_grid_optimum", 0.7))

    for ax, run in zip(axes, loaded):
        grouped = group_by_location(run["per_location"])
        for i, loc in enumerate(DEDUP_LOCATIONS):
            rows = grouped.get(loc, [])
            xs = [r["layer"] for r in rows]
            ys = [r["pstar_pred"] for r in rows]
            ax.plot(xs, ys, marker="o", ms=3.0, lw=1.1, color=cmap(i),
                    label=loc, alpha=0.85)
        ax.axhline(p_grid, color="red", ls="--", lw=1, label=f"grid p*={p_grid}")
        ax.set_xlabel("Transformer layer")
        ax.set_title(run["_label"], fontsize=10)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        if ax is axes[0]:
            ax.set_ylabel(r"Predicted $p^*$")
        if ax is axes[-1]:
            ax.legend(loc="best", fontsize=8, ncol=1)

    fig.suptitle(r"Closed-form $p^*$ prediction across runs", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out_path}")


# ============================================================
# LaTeX tables
# ============================================================
def write_table_summary(loaded: list[dict], out_path: Path):
    lines = [
        r"% Table: per-run summary (auto-generated by scripts/compare_hill_runs.py)",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Hill estimator validation across backbones. "
        r"$\hat\beta_{\text{down}}$ is the mean Hill estimate over the 32 down\_proj "
        r"locations; $p^*_{\text{pred}}$ is the closed-form prediction from $\hat\beta_{\text{down}}$.}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Backbone & Mode & $\hat\beta_{\text{down}}$ (mean) & $p^*_{\text{pred}}$ "
        r"& $p^*_{\text{grid}}$ & Gap \\",
        r"\midrule",
    ]
    for run in loaded:
        cfg = run.get("config", {})
        backbone = cfg.get("base_model", "?")
        mode = cfg.get("mode", "?")
        grouped = group_by_location(run["per_location"])
        down_betas = [r["beta_hat"] for r in grouped.get("down_proj", [])
                      if np.isfinite(r["beta_hat"])]
        beta_down = float(np.mean(down_betas)) if down_betas else float("nan")
        from math import sqrt
        if beta_down > 1.0:
            theta = (beta_down - 1) / beta_down
            disc = (1 + theta) ** 2 + 4 * theta
            p_pred = (-(1 + theta) + sqrt(disc)) / (2 * theta)
        else:
            p_pred = float("nan")
        p_grid = float(cfg.get("p_grid_optimum", 0.7))
        gap = p_pred - p_grid if np.isfinite(p_pred) else float("nan")
        lines.append(
            f"{backbone} & {mode} & "
            f"{beta_down:.3f} & {p_pred:.3f} & {p_grid:.2f} & {gap:+.3f} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\label{tab:hill_validation_summary}",
        r"\end{table}",
    ])
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[tex] wrote {out_path}")


def write_table_per_location(loaded: list[dict], out_path: Path):
    lines = [
        r"% Table: per-location mean beta_hat across runs",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Mean Hill estimate $\hat\beta$ per adapter input location, "
        r"averaged over all 32 transformer layers. The Q/K/V columns are identical "
        r"by construction (shared input tensor). Locations with $\hat\beta > 3$ do not "
        r"satisfy the Pareto-tail assumption of Theorem 1.}",
        r"\begin{tabular}{l" + "c" * len(DEDUP_LOCATIONS) + r"}",
        r"\toprule",
        r"Backbone & " + " & ".join(loc.replace("_", "\\_") for loc in DEDUP_LOCATIONS) + r" \\",
        r"\midrule",
    ]
    for run in loaded:
        label = run["_label"]
        grouped = group_by_location(run["per_location"])
        cells = []
        for loc in DEDUP_LOCATIONS:
            bs = [r["beta_hat"] for r in grouped.get(loc, []) if np.isfinite(r["beta_hat"])]
            cells.append(f"{np.mean(bs):.2f}" if bs else "--")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\label{tab:per_location_beta}",
        r"\end{table}",
    ])
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[tex] wrote {out_path}")


# ============================================================
# Console summary
# ============================================================
def print_console_summary(loaded: list[dict]):
    print()
    print("=" * 78)
    print(f"  Cross-run summary  ({len(loaded)} runs)")
    print("=" * 78)
    header = f"  {'Run':36s} {'beta_down':>10s} {'p*_pred':>9s} {'gap':>7s}"
    print(header)
    print("-" * 78)
    for run in loaded:
        grouped = group_by_location(run["per_location"])
        bs = [r["beta_hat"] for r in grouped.get("down_proj", [])
              if np.isfinite(r["beta_hat"])]
        b_down = float(np.mean(bs)) if bs else float("nan")
        from math import sqrt
        if b_down > 1.0:
            theta = (b_down - 1) / b_down
            disc = (1 + theta) ** 2 + 4 * theta
            p_pred = (-(1 + theta) + sqrt(disc)) / (2 * theta)
        else:
            p_pred = float("nan")
        p_grid = float(run.get("config", {}).get("p_grid_optimum", 0.7))
        gap = p_pred - p_grid if np.isfinite(p_pred) else float("nan")
        print(f"  {run['_label']:36s} {b_down:10.3f} {p_pred:9.3f} {gap:+7.3f}")
    print("=" * 78)


def main():
    args = parse_args()
    runs = resolve_runs(args)
    loaded = load_runs(runs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_headline_down_proj(loaded, args.curve_layers,
                            out_dir / "fig_headline_down_proj.pdf")
    plot_all_locations_grid(loaded, args.curve_layers,
                            out_dir / "fig_all_locations_grid.pdf")
    plot_beta_per_location_bar(loaded, out_dir / "fig_beta_per_location.pdf")
    plot_pstar_per_loc_across_runs(loaded, out_dir / "fig_pstar_per_loc_across_runs.pdf")

    write_table_summary(loaded, out_dir / "table_summary.tex")
    write_table_per_location(loaded, out_dir / "table_per_location.tex")

    print_console_summary(loaded)


if __name__ == "__main__":
    main()
