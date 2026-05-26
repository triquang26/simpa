"""Hill tail-index estimator and closed-form p* solver for SIMPA analysis.

Pure-numpy module. Exports:
    - hill_estimator: β̂ from upper order statistics
    - hill_plot_data: β̂(k) over a log-spaced k grid (plateau diagnostic)
    - solve_pstar: closed-form positive root of θp² + (1+θ)p - 1 = 0
    - predicted_retention: ρ(p) = (1-p)^((β-1)/β) under Pareto tail
    - sanity_check_synthetic: round-trip check on Pareto draws
"""

from __future__ import annotations

import numpy as np


def hill_estimator(
    samples: np.ndarray, k: int | None = None, k_frac: float = 0.05
) -> float:
    """Hill tail-index estimator.

    Computes β̂ = k / Σ_{i=1..k} (log X_(n-i+1) - log X_(n-k)) using the top-k
    order statistics of the strictly-positive entries of `samples`. Returns
    np.nan if fewer than 200 strictly-positive samples remain after filtering
    or if the resulting k < 10.
    """
    arr = np.asarray(samples, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]
    n = arr.size
    if n < 200:
        return float("nan")

    if k is None:
        k = max(100, int(k_frac * n))
    k = int(k)
    # Need at least k+1 samples (k upper plus one threshold below them).
    if k < 10 or k >= n:
        return float("nan")

    arr_sorted = np.sort(arr)  # ascending
    threshold = arr_sorted[n - k - 1]
    if threshold <= 0:
        return float("nan")
    top = arr_sorted[n - k :]  # length k
    log_ratios = np.log(top) - np.log(threshold)
    s = log_ratios.sum()
    if s <= 0:
        return float("nan")
    return float(k / s)


def hill_plot_data(
    samples: np.ndarray, n_points: int = 40
) -> dict[str, list[float]]:
    """Hill plot data: β̂(k) on a log-spaced k grid for plateau diagnosis.

    Returns {"k": [...], "beta_hat": [...]} of equal length, with k ranging
    log-spaced from 100 up to n//4 (or n//2 if n is small). Returns empty
    arrays if fewer than 400 strictly-positive samples are available.
    """
    arr = np.asarray(samples, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]
    n = arr.size
    if n < 400:
        return {"k": [], "beta_hat": []}

    # Pick upper bound for k: prefer n//4, but allow n//2 if n is small.
    upper = n // 4 if n >= 1000 else n // 2
    upper = max(101, upper)
    grid = np.unique(np.geomspace(100, upper, n_points).astype(int))
    # Ensure all k < n
    grid = grid[grid < n]
    if grid.size == 0:
        return {"k": [], "beta_hat": []}

    ks: list[int] = []
    betas: list[float] = []
    for k in grid:
        b = hill_estimator(arr, k=int(k))
        if np.isfinite(b):
            ks.append(int(k))
            betas.append(float(b))
    return {"k": ks, "beta_hat": betas}


def solve_pstar(beta: float) -> float:
    """Closed-form positive root of θp² + (1+θ)p - 1 = 0 with θ = (β-1)/β.

    Clamps β to (1+1e-6, ∞) since Hill requires a finite-mean tail. As β → 1
    (θ → 0), the optimum degenerates; we return 1 - 1e-6. The returned p* is
    clamped to (0, 1).
    """
    beta = float(beta)
    if beta <= 1.0 + 1e-6:
        beta = 1.0 + 1e-6
    theta = (beta - 1.0) / beta
    if theta < 1e-12:
        return 1.0 - 1e-6
    # Quadratic θ p² + (1+θ) p - 1 = 0 → p = (-(1+θ) + sqrt((1+θ)² + 4θ)) / (2θ).
    b = 1.0 + theta
    disc = b * b + 4.0 * theta
    p = (-b + np.sqrt(disc)) / (2.0 * theta)
    # Clamp to (0, 1) (open interval)
    eps = 1e-12
    if p <= 0.0:
        return eps
    if p >= 1.0:
        return 1.0 - eps
    return float(p)


def predicted_retention(beta: float, p: float) -> float:
    """ρ(p) = (1-p)^((β-1)/β) — fraction of total tail energy retained
    when discarding a fraction p of mass under a Pareto tail with index β.

    Clamps β to (1+1e-6, ∞) and p to (0, 1).
    """
    beta = float(beta)
    if beta <= 1.0 + 1e-6:
        beta = 1.0 + 1e-6
    p = float(p)
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    exponent = (beta - 1.0) / beta
    return float((1.0 - p) ** exponent)


def sanity_check_synthetic(
    beta_true: float = 1.5, n: int = 100_000, seed: int = 0
) -> dict[str, float]:
    """Round-trip Hill on Y = U^(-1/β_true), U ~ Uniform(0,1).

    Returns dict with beta_true, beta_hat, relative error, and predicted p*.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, size=int(n))
    # Avoid u == 0 producing inf
    u = np.clip(u, 1e-300, 1.0)
    samples = u ** (-1.0 / float(beta_true))
    beta_hat = hill_estimator(samples, k_frac=0.05)
    rel_err = abs(beta_hat - beta_true) / beta_true if np.isfinite(beta_hat) else float("nan")
    pstar = solve_pstar(beta_hat) if np.isfinite(beta_hat) else float("nan")
    return {
        "beta_true": float(beta_true),
        "beta_hat": float(beta_hat),
        "rel_err": float(rel_err),
        "pstar_pred": float(pstar),
    }
