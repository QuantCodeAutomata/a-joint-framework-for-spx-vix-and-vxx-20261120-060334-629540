"""Experiment 1: one-dimensional Brownian primary process (Corollary 3.20,
Example 3.21, Figure 1).

Simulates the VIX-based signature model with X_t = W_t (standard Brownian
motion, b = alpha = 0, a-bar = 1) and the level-2 VIX functional
ell_empty = 0.3, ell_1 = 0, ell_11 = 1. Reports closed-form VIX, VIX futures,
instantaneous SPX variance and VXX along sample trajectories, verifies the
nonnegativity condition (3.16), and saves a Figure-1-style plot and table.

Run: python -m exp.exp_1
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.vix_model_1d import (
    nonnegativity_condition_holds,
    simulate_brownian_paths,
    spx_instantaneous_variance,
    vix,
    vix_futures,
    vxx,
)

RESULTS_DIR = "results"


def run() -> dict:
    ell_empty, ell_1, ell_11 = config.EXP1_ELL_EMPTY, config.EXP1_ELL_1, config.EXP1_ELL_11
    delta_years = config.EXP1_DELTA_DAYS / 365.0
    tau_years = config.EXP1_TAU_DAYS / 365.0

    times_years, paths = simulate_brownian_paths(
        config.EXP1_N_PATHS, config.EXP1_HORIZON_DAYS, config.EXP1_STEPS_PER_DAY, config.EXP1_SEED
    )
    dt_years = times_years[1] - times_years[0]

    vix_t = vix(paths, ell_empty, ell_1, ell_11)
    sigma2_t = spx_instantaneous_variance(paths, vix_t, delta_years, ell_empty, ell_1, ell_11)
    vxx_t = vxx(paths, vix_t, times_years, dt_years, config.EXP1_VXX0, ell_empty, ell_11, tau_years)
    futures_30 = vix_futures(vix_t, times_years, 30.0 / 365.0, ell_11)
    futures_60 = vix_futures(vix_t, times_years, 60.0 / 365.0, ell_11)

    assert vix_t.shape == paths.shape
    assert np.all(np.isfinite(vix_t)) and np.all(np.isfinite(vxx_t))

    # --- Nonnegativity verification (equation 3.16) -------------------------
    condition_holds = nonnegativity_condition_holds(ell_empty, ell_1, ell_11, delta_years)
    lhs = ell_empty / ell_11
    rhs = ell_1 ** 2 / (2 * ell_11 ** 2) + 7 * delta_years / 16.0
    vix_nonneg_fraction = float(np.mean(vix_t >= 0.0))
    var_nonneg_fraction = float(np.mean(sigma2_t >= 0.0))

    # --- Sample-trajectory table (Figure 1 style) ---------------------------
    sample_path_idx = 0
    report_steps = [int(round(d / 365.0 / dt_years)) for d in config.EXP1_REPORT_TIMES_DAYS]
    table = pd.DataFrame({
        "day": config.EXP1_REPORT_TIMES_DAYS,
        "VIX": vix_t[report_steps, sample_path_idx],
        "VXX": vxx_t[report_steps, sample_path_idx],
        "sigma": np.sqrt(np.maximum(sigma2_t[report_steps, sample_path_idx], 0.0)),
        "F_t(30)": futures_30[report_steps, sample_path_idx],
        "F_t(60)": futures_60[report_steps, sample_path_idx],
    })
    table["VIX/VIX_0"] = table["VIX"] / table["VIX"].iloc[0]
    table["VXX/VXX_0"] = table["VXX"] / config.EXP1_VXX0
    table["sigma2/sigma2_0"] = (table["sigma"] ** 2) / (table["sigma"].iloc[0] ** 2)
    table.to_csv(f"{RESULTS_DIR}/exp1_sample_trajectory.csv", index=False)

    # --- Cross-sectional summary across all paths at each reporting day ----
    summary_rows = []
    for day, step in zip(config.EXP1_REPORT_TIMES_DAYS, report_steps):
        summary_rows.append({
            "day": day,
            "VIX_mean": float(vix_t[step].mean()),
            "VIX_std": float(vix_t[step].std()),
            "VXX_mean": float(vxx_t[step].mean()),
            "VXX_std": float(vxx_t[step].std()),
            "sigma_mean": float(np.sqrt(np.maximum(sigma2_t[step], 0.0)).mean()),
            "F30_mean": float(futures_30[step].mean()),
            "F60_mean": float(futures_60[step].mean()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(f"{RESULTS_DIR}/exp1_cross_sectional_summary.csv", index=False)

    # --- Figure-1-style plot -------------------------------------------------
    n_plot_paths = min(5, config.EXP1_N_PATHS)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    days = times_years * 365.0
    for p in range(n_plot_paths):
        axes[0, 0].plot(days, vix_t[:, p], alpha=0.8)
        axes[0, 1].plot(days, vxx_t[:, p], alpha=0.8)
        axes[1, 0].plot(days, np.sqrt(np.maximum(sigma2_t[:, p], 0.0)), alpha=0.8)
        axes[1, 1].plot(days, futures_30[:, p], alpha=0.8, label="F_t(30)" if p == 0 else None)
        axes[1, 1].plot(days, futures_60[:, p], alpha=0.8, linestyle="--", label="F_t(60)" if p == 0 else None)
    axes[0, 0].set_title("VIX_t"); axes[0, 0].axhline(ell_empty, color="k", lw=0.5, ls=":")
    axes[0, 1].set_title("VXX_t"); axes[0, 1].axhline(config.EXP1_VXX0, color="k", lw=0.5, ls=":")
    axes[1, 0].set_title("sigma_t^S(ell)")
    axes[1, 1].set_title("VIX futures F_t(30), F_t(60)"); axes[1, 1].legend()
    for ax in axes.flat:
        ax.set_xlabel("day"); ax.grid(alpha=0.3)
    fig.suptitle("Experiment 1: 1D Brownian primary process -- joint VIX/VXX/variance/futures dynamics (Corollary 3.20)")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp1_trajectories.png", dpi=150)
    plt.close(fig)

    metrics = {
        "n_paths": config.EXP1_N_PATHS,
        "horizon_days": config.EXP1_HORIZON_DAYS,
        "steps_per_day": config.EXP1_STEPS_PER_DAY,
        "ell_empty": ell_empty, "ell_1": ell_1, "ell_11": ell_11,
        "vxx0": config.EXP1_VXX0,
        "tau_years": tau_years,
        "nonnegativity_condition_3_16": {
            "lhs_ell_empty_over_ell_11": lhs,
            "rhs_bound": rhs,
            "holds": bool(condition_holds),
        },
        "vix_nonneg_fraction_simulated": vix_nonneg_fraction,
        "variance_nonneg_fraction_simulated": var_nonneg_fraction,
        "vix_mean_at_t0": float(vix_t[0].mean()),
        "vix_mean_at_horizon": float(vix_t[-1].mean()),
        "vxx_mean_at_t0": float(vxx_t[0].mean()),
        "vxx_mean_at_horizon": float(vxx_t[-1].mean()),
        "sigma_mean_at_t0": float(np.sqrt(np.maximum(sigma2_t[0], 0.0)).mean()),
        "futures30_mean_at_t0": float(futures_30[0].mean()),
    }
    with open(f"{RESULTS_DIR}/exp1_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    m = run()
    print(json.dumps(m, indent=2))
