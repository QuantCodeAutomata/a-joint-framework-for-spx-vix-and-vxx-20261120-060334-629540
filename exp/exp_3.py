"""Experiment 3: joint SPX-VIX-VXX calibration with a mixed primary process
of two Ornstein-Uhlenbeck factors and one CIR factor (Section 4.2).

The VIX functional used is the parsimonious level-three enrichment of the
complete level-two basis with the four words (0,3,3), (1,1,1), (3,0,3) and
(3,3,0) (25 coefficients, Section 4.2). This script (1) evaluates the model
at the paper's reported calibrated parameters and reports the VIX futures
proxy and the rho^T R^{-1} rho constraint value, (2) runs a short joint
recalibration starting from that point against the market surfaces used
here (see data/generate_synthetic.py for why a synthetic surface is used in
place of the licensed Cboe DataShop quotes), and (3) estimates the Monte
Carlo approximation-error bounds of Corollary 3.29 (Table 2).

Run: python -m exp.exp_3 [--full]
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loader import DataloaderError, load_underlying_spots
from data.generate_synthetic import generate_joint_surfaces
from src import config
from src.calibrate import level2_complete_words, model_implied_vols, run_calibration
from src.calibration import rho_constraint_value
from src.model import PricingGrid, approximation_error_bounds, price_candidate, simulate_model
from src.primary_process import ou_cir_algebra

RESULTS_DIR = "results"

# Letters: 0=time, 1=slow OU, 2=fast OU, 3=CIR (Section 4.2).
CIR_LETTER = 3
ENRICHED_WORDS = [(0, 3, 3), (1, 1, 1), (3, 0, 3), (3, 3, 0)]


def _anchor_spots() -> dict:
    """Best-effort real-market anchor for the synthetic surface, reusing the
    same quote-date spot lookup as Experiment 2 (Section 4 shares its market
    data across Sections 4.1 and 4.2)."""
    try:
        spots = load_underlying_spots("2026-02-24")
        note = {"source": "quantx_dataloader", "spots": {k: v for k, v in spots.items() if k != "_meta"}}
        vxx_spot = spots.get("VXX", config.EXP3_VXX0)
    except DataloaderError as exc:
        note = {"source": "config_fallback", "reason": str(exc)}
        vxx_spot = config.EXP3_VXX0
    return {"vxx": vxx_spot, "note": note}


def run(full: bool = False) -> dict:
    n_paths = config.EXP3_N_PATHS_PAPER if full else config.EXP3_N_PATHS_RUN
    n_validation = config.EXP3_N_VALIDATION_PAPER if full else config.EXP3_N_VALIDATION_RUN
    maxiter = config.CALIBRATION_MAXITER_PAPER if full else config.CALIBRATION_MAXITER_RUN

    r = np.array(config.EXP3_R)
    r_inv = np.linalg.inv(r)
    algebra = ou_cir_algebra(
        config.EXP3_OU_KAPPA, config.EXP3_OU_THETA, config.EXP3_OU_ETA, config.EXP3_OU_X0, r,
        cir_kappa=config.EXP3_CIR_KAPPA, cir_theta=config.EXP3_CIR_THETA,
        cir_eta=config.EXP3_CIR_ETA, cir_y0=config.EXP3_CIR_Y0,
    )
    factor_x0 = (*config.EXP3_OU_X0, config.EXP3_CIR_Y0)
    delta_years = config.EXP3_DELTA_DAYS / 365.0
    t0, t1, t2, t3, _t4 = config.EXP3_FUTURES_MATURITIES_DAYS
    grid = PricingGrid(
        spx_maturity_days=config.EXP3_SPX_MATURITY_DAYS,
        vix_maturity_days=config.EXP3_VIX_MATURITY_DAYS,
        vxx_maturity_days=config.EXP3_VXX_MATURITY_DAYS,
        t0_days=t0, t1_days=t1, t2_days=t2, t3_days=t3,
    )
    horizon_days = max(config.EXP3_SPX_MATURITY_DAYS, config.EXP3_VIX_MATURITY_DAYS, config.EXP3_VXX_MATURITY_DAYS)

    anchor = _anchor_spots()
    spx_surface, vix_surface, vxx_surface = generate_joint_surfaces(
        spx_spot=config.EXP3_SPX_FORWARD,
        vix_level=config.EXP3_VIX_FUTURES_MARKET,
        vxx_spot=anchor["vxx"],
        rate=config.EXP3_INTEREST_RATE,
        spx_maturity_days=config.EXP3_SPX_MATURITY_DAYS,
        vix_maturity_days=config.EXP3_VIX_MATURITY_DAYS,
        vxx_maturity_days=config.EXP3_VXX_MATURITY_DAYS,
        spx_moneyness=config.EXP3_SPX_MONEYNESS,
        vix_moneyness=config.EXP3_VIX_MONEYNESS,
        vxx_moneyness=config.EXP3_VXX_MONEYNESS,
        n_spx=config.EXP3_N_STRIKES_SPX, n_vix=config.EXP3_N_STRIKES_VIX, n_vxx=config.EXP3_N_STRIKES_VXX,
        seed=config.RANDOM_SEED,
    )

    words = level2_complete_words(3) + ENRICHED_WORDS
    assert len(words) == 25, "level-two basis (21) plus four enriched words must give 25 coefficients (Section 4.2)"
    ell_star = {w: c for w, c in zip(words, config.EXP3_ELL_STAR)}
    rho_star = np.array(config.EXP3_RHO_STAR)
    constraint_value = rho_constraint_value(rho_star, r_inv)

    model = simulate_model(
        ell_star, algebra, delta_years, n_paths, horizon_days, config.EXP3_STEPS_PER_DAY,
        config.RANDOM_SEED, factor_x0, r, cir_letter=CIR_LETTER, cir_eta=config.EXP3_CIR_ETA,
    )
    priced = price_candidate(
        model, grid, rho_star, r_inv, config.EXP3_LEVERAGE_L, config.EXP3_FUTURES_FLOOR_M,
        config.EXP3_VARIANCE_FLOOR, config.EXP3_SPX_FORWARD, config.EXP3_INTEREST_RATE,
        anchor["vxx"], config.EXP3_VXX_FEE,
        np.array([q.strike for q in spx_surface.quotes]),
        np.array([q.strike for q in vix_surface.quotes]),
        np.array([q.strike for q in vxx_surface.quotes]),
    )

    model_val = simulate_model(
        ell_star, algebra, delta_years, n_validation, horizon_days, config.EXP3_STEPS_PER_DAY,
        config.RANDOM_SEED + 1, factor_x0, r, cir_letter=CIR_LETTER, cir_eta=config.EXP3_CIR_ETA,
    )
    bounds = approximation_error_bounds(
        model_val, grid, config.EXP3_FUTURES_FLOOR_M, config.EXP3_INTEREST_RATE, config.EXP3_VXX_FEE
    )

    # --- short illustrative joint recalibration from the reported point ---
    calib = run_calibration(
        words=words, initial_coeffs=ell_star, algebra=algebra, delta_years=delta_years,
        n_paths=n_paths, horizon_days=horizon_days, steps_per_day=config.EXP3_STEPS_PER_DAY,
        seed=config.RANDOM_SEED, factor_x0=factor_x0, correlation=r,
        rho_init=rho_star, r_inv=r_inv, grid=grid, leverage_L=config.EXP3_LEVERAGE_L,
        floor_m=config.EXP3_FUTURES_FLOOR_M, variance_floor=config.EXP3_VARIANCE_FLOOR,
        spot=config.EXP3_SPX_FORWARD, rate=config.EXP3_INTEREST_RATE, vxx0=anchor["vxx"],
        vxx_fee=config.EXP3_VXX_FEE, spx_surface=spx_surface, vix_surface=vix_surface,
        vxx_surface=vxx_surface, lambdas=config.EXP3_LOSS_LAMBDA, betas=config.EXP3_LOSS_BETA,
        maxiter=maxiter, cir_letter=CIR_LETTER, cir_eta=config.EXP3_CIR_ETA,
    )
    priced_calibrated = price_candidate(
        calib.model, grid, calib.rho_star, r_inv, config.EXP3_LEVERAGE_L, config.EXP3_FUTURES_FLOOR_M,
        config.EXP3_VARIANCE_FLOOR, config.EXP3_SPX_FORWARD, config.EXP3_INTEREST_RATE,
        anchor["vxx"], config.EXP3_VXX_FEE,
        np.array([q.strike for q in spx_surface.quotes]),
        np.array([q.strike for q in vix_surface.quotes]),
        np.array([q.strike for q in vxx_surface.quotes]),
    )

    results: dict = {
        "anchor": anchor["note"],
        "feller_ratio": config.EXP3_FELLER_RATIO,
        "n_coefficients": len(words),
        "word_basis_closure_size": len(model.ops.words),
        "vix_futures_proxy_reported_point": priced["vix_futures"],
        "vix_futures_market": config.EXP3_VIX_FUTURES_MARKET,
        "rho_T_Rinv_rho_reported_point": constraint_value,
        "rho_T_Rinv_rho_expected": 0.9992,
        "min_raw_variance_reported_point": priced["min_raw_variance"],
        "futures_floor_hit_reported_point": priced["futures_floor_hit"],
        "approximation_error_bounds": {
            "vix_bound": bounds["vix_bound"], "vxx_bound": bounds["vxx_bound"],
            "futures_floor_hit_validation": bounds["futures_floor_hit"],
        },
        "recalibration": {
            "loss_before": calib.history[0] if calib.history else None,
            "loss_after": calib.loss_star,
            "n_iterations": calib.n_iterations,
            "converged": calib.converged,
            "rho_T_Rinv_rho_after": rho_constraint_value(calib.rho_star, r_inv),
        },
    }

    smile_fig, smile_axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, surface, prices, label in (
        (smile_axes[0], spx_surface, priced_calibrated["spx_prices"], "SPX"),
        (smile_axes[1], vix_surface, priced_calibrated["vix_prices"], "VIX"),
        (smile_axes[2], vxx_surface, priced_calibrated["vxx_prices"], "VXX"),
    ):
        model_iv = model_implied_vols(prices, surface, config.EXP3_INTEREST_RATE)
        strikes = np.array([q.strike for q in surface.quotes])
        bid = np.array([q.bid_iv for q in surface.quotes])
        ask = np.array([q.ask_iv for q in surface.quotes])
        ax.fill_between(strikes, bid, ask, alpha=0.2, color="gray", label="bid-ask band")
        ax.plot(strikes, model_iv, marker="o", markersize=3, color="C1", label="model (OU+CIR, level-3)")
        ax.set_title(f"{label} smile"); ax.set_xlabel("strike"); ax.set_ylabel("implied vol")
        ax.legend(fontsize=7)
    smile_fig.suptitle("Experiment 3: SPX/VIX/VXX smiles, 2 OU + 1 CIR primary process (Section 4.2)")
    smile_fig.tight_layout()
    smile_fig.savefig(f"{RESULTS_DIR}/exp3_smiles.png", dpi=150)
    plt.close(smile_fig)

    table = pd.DataFrame([{
        "n_coefficients": results["n_coefficients"],
        "vix_futures_proxy": results["vix_futures_proxy_reported_point"],
        "vix_futures_market": results["vix_futures_market"],
        "rho_T_Rinv_rho": results["rho_T_Rinv_rho_reported_point"],
        "rho_T_Rinv_rho_paper": results["rho_T_Rinv_rho_expected"],
        "feller_ratio": results["feller_ratio"],
        "vix_error_bound": results["approximation_error_bounds"]["vix_bound"],
        "vxx_error_bound": results["approximation_error_bounds"]["vxx_bound"],
        "loss_before_recalibration": results["recalibration"]["loss_before"],
        "loss_after_recalibration": results["recalibration"]["loss_after"],
    }])
    table.to_csv(f"{RESULTS_DIR}/exp3_table2_summary.csv", index=False)

    results["config"] = {
        "n_paths": n_paths, "n_validation": n_validation, "maxiter": maxiter,
        "full_run": full, "spx_spread_range": [0.04, 0.15], "vxx_spread_range": [0.13, 1.18],
    }
    with open(f"{RESULTS_DIR}/exp3_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run the paper's full Monte Carlo / optimizer configuration")
    args = parser.parse_args()
    out = run(full=args.full)
    print(json.dumps(out, indent=2, default=float))
