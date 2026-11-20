"""Experiment 2: joint SPX-VIX-VXX calibration with a 3-factor
Ornstein-Uhlenbeck primary process (Section 4.1).

Two VIX-functional specifications are evaluated and illustratively
recalibrated against the SPX/VIX/VXX implied-volatility surfaces:

* the complete level-two specification (21 coefficients, equation 4.2);
* the parsimonious level-three enrichment adding the words (1,1,1) and
  (0,0,0) (23 coefficients, equation 4.4).

For each specification this script (1) evaluates the model at the paper's
reported calibrated parameters and reports the VIX futures proxy and the
rho^T R_OU^{-1} rho constraint value, (2) runs a short joint recalibration
starting from that point against the market surfaces used here (see
data/generate_synthetic.py for why a synthetic surface is used in place of
the licensed Cboe DataShop quotes), and (3) estimates the Monte Carlo
approximation-error bounds of Corollary 3.29 (Table 1).

Run: python -m exp.exp_2 [--full]
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


def _anchor_spots() -> dict:
    """Best-effort real-market anchor for the synthetic surface (see
    data/data_loader.py::load_underlying_spots); falls back to the
    strike-grid-implied values of config.py if the loader cannot reach the
    QuantX Dataloader in this environment (e.g. missing credentials)."""
    try:
        spots = load_underlying_spots("2026-02-24")
        note = {"source": "quantx_dataloader", "spots": {k: v for k, v in spots.items() if k != "_meta"}}
        vxx_spot = spots.get("VXX", config.EXP2_VXX0)
    except DataloaderError as exc:
        note = {"source": "config_fallback", "reason": str(exc)}
        vxx_spot = config.EXP2_VXX0
    return {"vxx": vxx_spot, "note": note}


def run(full: bool = False) -> dict:
    n_paths = config.EXP2_N_PATHS_PAPER if full else config.EXP2_N_PATHS_RUN
    n_validation = config.EXP2_N_VALIDATION_PAPER if full else config.EXP2_N_VALIDATION_RUN
    maxiter = config.CALIBRATION_MAXITER_PAPER if full else config.CALIBRATION_MAXITER_RUN

    r_ou = np.array(config.EXP2_R_OU)
    r_inv = np.linalg.inv(r_ou)
    algebra = ou_cir_algebra(config.EXP2_OU_KAPPA, config.EXP2_OU_THETA, config.EXP2_OU_ETA, config.EXP2_OU_X0, r_ou)
    delta_years = config.EXP2_DELTA_DAYS / 365.0
    t0, t1, t2, t3, _t4 = config.EXP2_FUTURES_MATURITIES_DAYS
    grid = PricingGrid(
        spx_maturity_days=config.EXP2_SPX_MATURITY_DAYS,
        vix_maturity_days=config.EXP2_VIX_MATURITY_DAYS,
        vxx_maturity_days=config.EXP2_VXX_MATURITY_DAYS,
        t0_days=t0, t1_days=t1, t2_days=t2, t3_days=t3,
    )
    horizon_days = max(config.EXP2_SPX_MATURITY_DAYS, config.EXP2_VIX_MATURITY_DAYS, config.EXP2_VXX_MATURITY_DAYS)

    anchor = _anchor_spots()
    spx_surface, vix_surface, vxx_surface = generate_joint_surfaces(
        spx_spot=config.EXP2_SPX_FORWARD,
        vix_level=config.EXP2_VIX_FUTURES_MARKET,
        vxx_spot=anchor["vxx"],
        rate=config.EXP2_INTEREST_RATE,
        spx_maturity_days=config.EXP2_SPX_MATURITY_DAYS,
        vix_maturity_days=config.EXP2_VIX_MATURITY_DAYS,
        vxx_maturity_days=config.EXP2_VXX_MATURITY_DAYS,
        spx_moneyness=config.EXP2_SPX_MONEYNESS,
        vix_moneyness=config.EXP2_VIX_MONEYNESS,
        vxx_moneyness=config.EXP2_VXX_MONEYNESS,
        n_spx=config.EXP2_N_STRIKES_SPX, n_vix=config.EXP2_N_STRIKES_VIX, n_vxx=config.EXP2_N_STRIKES_VXX,
        seed=config.RANDOM_SEED,
    )

    specs = {
        "level2": {
            "words": level2_complete_words(3),
            "ell_star": config.EXP2_ELL2_STAR,
            "rho_star": np.array(config.EXP2_RHO2_STAR),
            "expected_constraint": 0.9944,
        },
        "level3": {
            "words": level2_complete_words(3) + [(1, 1, 1), (0, 0, 0)],
            "ell_star": config.EXP2_ELL3_STAR,
            "rho_star": np.array(config.EXP2_RHO3_STAR),
            "expected_constraint": 0.9933,
        },
    }

    results: dict = {"anchor": anchor["note"], "specifications": {}}
    smile_fig, smile_axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for name, spec in specs.items():
        words = spec["words"]
        ell_star = {w: c for w, c in zip(words, spec["ell_star"])}
        rho_star = spec["rho_star"]
        constraint_value = rho_constraint_value(rho_star, r_inv)

        model = simulate_model(
            ell_star, algebra, delta_years, n_paths, horizon_days, config.EXP2_STEPS_PER_DAY,
            config.RANDOM_SEED, config.EXP2_OU_X0, r_ou,
        )
        priced = price_candidate(
            model, grid, rho_star, r_inv, config.EXP2_LEVERAGE_L, config.EXP2_FUTURES_FLOOR_M,
            config.EXP2_VARIANCE_FLOOR, config.EXP2_SPX_FORWARD, config.EXP2_INTEREST_RATE,
            anchor["vxx"], config.EXP2_VXX_FEE,
            np.array([q.strike for q in spx_surface.quotes]),
            np.array([q.strike for q in vix_surface.quotes]),
            np.array([q.strike for q in vxx_surface.quotes]),
        )

        model_val = simulate_model(
            ell_star, algebra, delta_years, n_validation, horizon_days, config.EXP2_STEPS_PER_DAY,
            config.RANDOM_SEED + 1, config.EXP2_OU_X0, r_ou,
        )
        bounds = approximation_error_bounds(model_val, grid, config.EXP2_FUTURES_FLOOR_M, config.EXP2_INTEREST_RATE, config.EXP2_VXX_FEE)

        # --- short illustrative joint recalibration from the reported point ---
        calib = run_calibration(
            words=words, initial_coeffs=ell_star, algebra=algebra, delta_years=delta_years,
            n_paths=n_paths, horizon_days=horizon_days, steps_per_day=config.EXP2_STEPS_PER_DAY,
            seed=config.RANDOM_SEED, factor_x0=config.EXP2_OU_X0, correlation=r_ou,
            rho_init=rho_star, r_inv=r_inv, grid=grid, leverage_L=config.EXP2_LEVERAGE_L,
            floor_m=config.EXP2_FUTURES_FLOOR_M, variance_floor=config.EXP2_VARIANCE_FLOOR,
            spot=config.EXP2_SPX_FORWARD, rate=config.EXP2_INTEREST_RATE, vxx0=anchor["vxx"],
            vxx_fee=config.EXP2_VXX_FEE, spx_surface=spx_surface, vix_surface=vix_surface,
            vxx_surface=vxx_surface, lambdas=config.EXP2_LOSS_LAMBDA, betas=config.EXP2_LOSS_BETA,
            maxiter=maxiter,
        )
        priced_calibrated = price_candidate(
            calib.model, grid, calib.rho_star, r_inv, config.EXP2_LEVERAGE_L, config.EXP2_FUTURES_FLOOR_M,
            config.EXP2_VARIANCE_FLOOR, config.EXP2_SPX_FORWARD, config.EXP2_INTEREST_RATE,
            anchor["vxx"], config.EXP2_VXX_FEE,
            np.array([q.strike for q in spx_surface.quotes]),
            np.array([q.strike for q in vix_surface.quotes]),
            np.array([q.strike for q in vxx_surface.quotes]),
        )

        results["specifications"][name] = {
            "n_coefficients": len(words),
            "word_basis_closure_size": len(model.ops.words),
            "vix_futures_proxy_reported_point": priced["vix_futures"],
            "vix_futures_market": config.EXP2_VIX_FUTURES_MARKET,
            "rho_T_Rinv_rho_reported_point": constraint_value,
            "rho_T_Rinv_rho_expected": spec["expected_constraint"],
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

        for ax, surface, prices, label in (
            (smile_axes[0], spx_surface, priced_calibrated["spx_prices"], "SPX"),
            (smile_axes[1], vix_surface, priced_calibrated["vix_prices"], "VIX"),
            (smile_axes[2], vxx_surface, priced_calibrated["vxx_prices"], "VXX"),
        ):
            model_iv = model_implied_vols(prices, surface, config.EXP2_INTEREST_RATE)
            strikes = np.array([q.strike for q in surface.quotes])
            bid = np.array([q.bid_iv for q in surface.quotes])
            ask = np.array([q.ask_iv for q in surface.quotes])
            if name == "level2":
                ax.fill_between(strikes, bid, ask, alpha=0.2, color="gray", label="bid-ask band")
            ax.plot(strikes, model_iv, marker="o", markersize=3, label=f"model ({name})")
            ax.set_title(f"{label} smile"); ax.set_xlabel("strike"); ax.set_ylabel("implied vol")
            if name == "level3":
                ax.legend(fontsize=7)

    smile_fig.suptitle("Experiment 2: SPX/VIX/VXX smiles, level-2 vs level-3 enrichment (Section 4.1)")
    smile_fig.tight_layout()
    smile_fig.savefig(f"{RESULTS_DIR}/exp2_smiles.png", dpi=150)
    plt.close(smile_fig)

    table_rows = []
    for name, spec in results["specifications"].items():
        table_rows.append({
            "specification": name,
            "n_coefficients": spec["n_coefficients"],
            "vix_futures_proxy": spec["vix_futures_proxy_reported_point"],
            "vix_futures_market": spec["vix_futures_market"],
            "rho_T_Rinv_rho": spec["rho_T_Rinv_rho_reported_point"],
            "rho_T_Rinv_rho_paper": spec["rho_T_Rinv_rho_expected"],
            "vix_error_bound": spec["approximation_error_bounds"]["vix_bound"],
            "vxx_error_bound": spec["approximation_error_bounds"]["vxx_bound"],
            "loss_before_recalibration": spec["recalibration"]["loss_before"],
            "loss_after_recalibration": spec["recalibration"]["loss_after"],
        })
    table = pd.DataFrame(table_rows)
    table.to_csv(f"{RESULTS_DIR}/exp2_table1_summary.csv", index=False)

    results["config"] = {
        "n_paths": n_paths, "n_validation": n_validation, "maxiter": maxiter,
        "full_run": full, "spx_spread_range": [0.04, 0.15], "vxx_spread_range": [0.13, 1.18],
    }
    with open(f"{RESULTS_DIR}/exp2_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run the paper's full Monte Carlo / optimizer configuration")
    args = parser.parse_args()
    out = run(full=args.full)
    print(json.dumps(out, indent=2, default=float))
