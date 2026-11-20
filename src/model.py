"""Shared driver wiring the signature, primary-process and pricing modules
into one joint SPX/VIX/VXX calibration pipeline (Section 4).

This module is the single place where the generator-invariant word basis is
assembled for a given VIX functional ell, Monte Carlo paths are simulated,
and SPX/VIX/VXX call prices and VIX futures proxies are produced for a given
candidate (ell, rho). Both exp_2 (3-factor OU) and exp_3 (2 OU + 1 CIR) call
these functions with their own primary-process algebra; neither experiment
re-implements the pricing recipe.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.pricing import (
    ModelOperators,
    build_operators,
    discounted_call_price,
    ell_power_vectors,
    futures_proxy_signed,
    h_of_ell,
    leverage_saturated_driver,
    raw_variance_path,
    signed_vix_functional,
    spx_price_path,
    stopped_futures,
    vxx_roll_proxy,
)
from src.primary_process import simulate_signature_paths
from src.signature import PolynomialDiffusionAlgebra, SparseVec, Word, close_words_under_L, shuffle, word_to_vector


@dataclass
class SimulatedModel:
    """Monte Carlo paths plus the pre-computed operators of one candidate
    VIX functional ell, independent of rho and of the option grids."""

    ops: ModelOperators
    ell_vec: np.ndarray
    ell_coeffs: np.ndarray  # eigenbasis coefficients of ell_vec, see pricing.ell_power_vectors
    h_vec: np.ndarray
    times_years: np.ndarray
    sig: np.ndarray            # (n_steps+1, n_paths, n_words)
    dW: np.ndarray              # (n_steps, n_paths, n_factors)
    perp_increments: np.ndarray  # (n_steps, n_paths)
    dt_years: float


def assemble_operators(ell: SparseVec, algebra: PolynomialDiffusionAlgebra, delta_years: float) -> ModelOperators:
    """Build the smallest word basis closed under L that contains both the
    support of ell (needed for the VIX/futures proxies, Appendix C) and the
    support of ell shuffle ell (needed for the raw-variance functional h(ell)
    of Theorem 3.14), then the matrices G and A on that basis.
    """
    ell_shuffle_ell = shuffle(ell, ell)
    seeds = set(ell.keys()) | set(ell_shuffle_ell.keys())
    words = close_words_under_L(seeds, algebra)
    from src.signature import build_G

    G = build_G(words, algebra)
    return build_operators(words, G, delta_years)


def simulate_model(
    ell: SparseVec,
    algebra: PolynomialDiffusionAlgebra,
    delta_years: float,
    n_paths: int,
    horizon_days: float,
    steps_per_day: int,
    seed: int,
    factor_x0: tuple[float, ...],
    correlation: np.ndarray,
    cir_letter: int | None = None,
    cir_eta: float | None = None,
) -> SimulatedModel:
    """Assemble the operators for `ell` and simulate the joint primary
    process / signature paths needed to price SPX, VIX and VXX claims up to
    `horizon_days` (the longest option maturity used)."""
    ops = assemble_operators(ell, algebra, delta_years)
    ell_vec = word_to_vector(ell, ops.words)
    ell_coeffs = ell_power_vectors(ops, ell_vec)
    h_vec = h_of_ell(ell, ops)

    n_factors = algebra.n_factors
    n_steps = int(round(horizon_days * steps_per_day))
    dt_years = (1.0 / 365.0) / steps_per_day
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(correlation)
    raw_normals = rng.normal(size=(n_steps, n_paths, n_factors))
    dW = np.einsum("ij,spj->spi", chol, raw_normals) * np.sqrt(dt_years)
    perp_increments = rng.normal(size=(n_steps, n_paths)) * np.sqrt(dt_years)

    times_years, sig, _ = simulate_signature_paths(
        algebra, ops.words, n_paths, horizon_days, steps_per_day,
        seed, factor_x0, correlation, cir_letter=cir_letter, cir_eta=cir_eta,
    )
    # simulate_signature_paths draws its own (seed-identical) Brownian
    # increments internally; recomputed here with the same seed so that the
    # price driver B^L of equation (3.11) uses the *same* realizations.
    return SimulatedModel(ops, ell_vec, ell_coeffs, h_vec, times_years, sig, dW, perp_increments, dt_years)


def update_candidate(model: SimulatedModel, ell: SparseVec) -> SimulatedModel:
    """Rebuild only the ell-dependent, cheap-to-recompute objects (ell_vec,
    its G-power contractions, h(ell)) for a new candidate functional sharing
    the same word basis, operators (G, A^{-1}) and simulated paths as
    `model`. Calibration keeps the word basis fixed to the support of the
    initial candidate and only varies the numerical coefficients, so the
    expensive Monte Carlo simulation (`simulate_model`) is run once per
    experiment specification and reused by every optimizer iteration.
    """
    ell_vec = word_to_vector(ell, model.ops.words)
    ell_coeffs = ell_power_vectors(model.ops, ell_vec)
    h_vec = h_of_ell(ell, model.ops)
    return SimulatedModel(
        model.ops, ell_vec, ell_coeffs, h_vec, model.times_years, model.sig,
        model.dW, model.perp_increments, model.dt_years,
    )


@dataclass
class PricingGrid:
    """Day offsets (in trading-day units, i.e. 1/365-year steps) of the
    maturities needed from a SimulatedModel: SPX/VXX option maturity, VIX
    option maturity, and the futures roll nodes T0, T1 used by the VXX
    before-T1 proxy (Corollary 3.20 / Appendix C)."""

    spx_maturity_days: float
    vix_maturity_days: float
    vxx_maturity_days: float
    t0_days: float
    t1_days: float
    t2_days: float
    t3_days: float


def price_candidate(
    model: SimulatedModel,
    grid: PricingGrid,
    rho: np.ndarray,
    r_inv: np.ndarray,
    leverage_L: float,
    floor_m: float,
    variance_floor: float,
    spot: float,
    rate: float,
    vxx0: float,
    vxx_fee: float,
    spx_strikes: np.ndarray,
    vix_strikes: np.ndarray,
    vxx_strikes: np.ndarray,
) -> dict:
    """Price the SPX, VIX and VXX call grids and the VIX futures proxy for
    one candidate (rho, with ell already baked into `model`).

    Returns a dict with keys 'spx_prices', 'vix_prices', 'vxx_prices'
    (arrays aligned with the strike grids), 'vix_futures' (scalar F_0(T_VIX)),
    'min_raw_variance' (float, for the variance-floor constraint) and
    'futures_floor_hit' (bool, for the i*=2 error-bound bookkeeping).
    """
    dt = model.dt_years
    times = model.times_years
    raw_var = raw_variance_path(model.sig, model.h_vec)  # (n_steps+1, n_paths)

    spx_idx = int(round(grid.spx_maturity_days * (1.0 / dt / 365.0)))
    spx_idx = min(spx_idx, raw_var.shape[0] - 1)
    sigma_path = np.sqrt(np.maximum(raw_var[: spx_idx + 1][:-1], 0.0))
    dB = leverage_saturated_driver(
        sigma_path, model.dW[:spx_idx], model.perp_increments[:spx_idx], rho, r_inv, leverage_L
    )
    spx_terminal = spx_price_path(dB, raw_var[: spx_idx + 1], dt, spot)
    spx_maturity_years = grid.spx_maturity_days / 365.0
    spx_prices = np.array([
        discounted_call_price(spx_terminal, float(k), rate, spx_maturity_years) for k in spx_strikes
    ])

    vix_idx = int(round(grid.vix_maturity_days * (1.0 / dt / 365.0)))
    vix_idx = min(vix_idx, model.sig.shape[0] - 1)
    vix_signed = signed_vix_functional(model.sig[vix_idx], model.ell_vec)
    vix_terminal_pct = 100.0 * np.abs(vix_signed)
    vix_maturity_years = grid.vix_maturity_days / 365.0
    vix_prices = np.array([
        discounted_call_price(vix_terminal_pct, float(k), rate, vix_maturity_years) for k in vix_strikes
    ])

    vxx_idx = int(round(grid.vxx_maturity_days * (1.0 / dt / 365.0)))
    vxx_idx = min(vxx_idx, model.sig.shape[0] - 1)
    t0_years, t1_years, t2_years = grid.t0_days / 365.0, grid.t1_days / 365.0, grid.t2_days / 365.0
    t3_years = grid.t3_days / 365.0
    futures_t1 = np.stack([
        futures_proxy_signed(model.sig[i], model.ops, model.ell_coeffs, t1_years - times[i])
        for i in range(vxx_idx + 1)
    ])
    futures_t2 = np.stack([
        futures_proxy_signed(model.sig[i], model.ops, model.ell_coeffs, t2_years - times[i])
        for i in range(vxx_idx + 1)
    ])
    futures_t3 = np.stack([
        futures_proxy_signed(model.sig[i], model.ops, model.ell_coeffs, t3_years - times[i])
        for i in range(vxx_idx + 1)
    ])
    futures_t1_floored = stopped_futures(futures_t1, floor_m)
    futures_t2_floored = stopped_futures(futures_t2, floor_m)
    futures_t3_floored = stopped_futures(futures_t3, floor_m)
    vxx_path = vxx_roll_proxy(
        times[: vxx_idx + 1], futures_t1_floored, futures_t2_floored, futures_t3_floored,
        t0_years, t1_years, t2_years, vxx0, dt,
    )
    vxx_terminal = vxx_path[-1] * np.exp((rate - vxx_fee) * (grid.vxx_maturity_days / 365.0))
    vxx_maturity_years = grid.vxx_maturity_days / 365.0
    vxx_prices = np.array([
        discounted_call_price(vxx_terminal, float(k), rate, vxx_maturity_years) for k in vxx_strikes
    ])

    vix_futures_0 = float(futures_proxy_signed(
        model.sig[0:1], model.ops, model.ell_coeffs, vix_maturity_years
    ).mean())

    return {
        "spx_prices": spx_prices,
        "vix_prices": vix_prices,
        "vxx_prices": vxx_prices,
        "vix_futures": vix_futures_0,
        "min_raw_variance": float(raw_var[: spx_idx + 1].min()),
        "futures_floor_hit": bool((futures_t1 <= floor_m).any() or (futures_t2 <= floor_m).any()),
        "vix_terminal_pct": vix_terminal_pct,
        "vxx_terminal": vxx_terminal,
    }


def approximation_error_bounds(
    model: SimulatedModel,
    grid: PricingGrid,
    floor_m: float,
    rate: float,
    vxx_fee: float,
) -> dict:
    """Monte Carlo estimates of the error bounds of Corollary 3.29:

    (i) VIX bound: 100 * e^{-rT} * E[eps_T(ell)], eps_T(ell) = |VIX_T -
        |VIX_T|| = 2 * (signed VIX_T)^-, the only source of the absolute
        value proxy's error identified in the proof of Corollary 3.29(i)
        (the Lipschitz bound on the call payoff reduces exactly to this
        quantity).
    (ii) VXX bound: e^{-cT} * C_T^m(ell), approximated by the Monte Carlo
        average gap between the VXX roll proxy computed with and without
        the futures stopping floor m, which is the only source of error the
        paper attributes to the tractable proxy once the futures level
        never approaches m (as reported for both Table 1 and Table 2); the
        exact constant C_T^m(ell) of Corollary 3.28 was not retrieved in
        closed form and is approximated this way, as recorded in
        results/RESULTS.md.
    """
    vix_idx = int(round(grid.vix_maturity_days * (1.0 / model.dt_years / 365.0)))
    vix_signed = signed_vix_functional(model.sig[vix_idx], model.ell_vec)
    epsilon_t = 2.0 * np.maximum(0.0, -vix_signed)
    vix_bound = 100.0 * np.exp(-rate * grid.vix_maturity_days / 365.0) * float(np.mean(epsilon_t))

    vxx_idx = int(round(grid.vxx_maturity_days * (1.0 / model.dt_years / 365.0)))
    t0_years, t1_years, t2_years = grid.t0_days / 365.0, grid.t1_days / 365.0, grid.t2_days / 365.0
    t3_years = grid.t3_days / 365.0
    times = model.times_years[: vxx_idx + 1]
    futures_t1 = np.stack([
        futures_proxy_signed(model.sig[i], model.ops, model.ell_coeffs, t1_years - times[i])
        for i in range(vxx_idx + 1)
    ])
    futures_t2 = np.stack([
        futures_proxy_signed(model.sig[i], model.ops, model.ell_coeffs, t2_years - times[i])
        for i in range(vxx_idx + 1)
    ])
    futures_t3 = np.stack([
        futures_proxy_signed(model.sig[i], model.ops, model.ell_coeffs, t3_years - times[i])
        for i in range(vxx_idx + 1)
    ])
    unfloored = vxx_roll_proxy(
        times, futures_t1, futures_t2, futures_t3, t0_years, t1_years, t2_years, 1.0, model.dt_years
    )
    floored = vxx_roll_proxy(
        times, stopped_futures(futures_t1, floor_m), stopped_futures(futures_t2, floor_m),
        stopped_futures(futures_t3, floor_m), t0_years, t1_years, t2_years, 1.0, model.dt_years,
    )
    vxx_gap = np.abs(unfloored[-1] - floored[-1])
    vxx_bound = np.exp(-vxx_fee * grid.vxx_maturity_days / 365.0) * float(np.mean(vxx_gap))
    return {"vix_bound": vix_bound, "vxx_bound": vxx_bound, "futures_floor_hit": bool((futures_t1 <= floor_m).any() or (futures_t2 <= floor_m).any())}
