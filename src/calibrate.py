"""Shared joint SPX-VIX-VXX calibration driver (Section 4, Appendix C).

Both exp_2 (3-factor OU primary process) and exp_3 (2 OU + 1 CIR primary
process) call `run_calibration` with their own primary-process algebra, word
basis and market surfaces; the Monte Carlo simulation, pricing recipe and
constrained optimizer are implemented once here and are not duplicated per
experiment. scipy.optimize.minimize (SLSQP) supplies the constrained local
optimizer, as no quant library in this environment implements the paper's
specific multi-market signature calibration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

from src.calibration import MarketSurface, joint_loss, rho_constraint_value
from src.model import PricingGrid, SimulatedModel, price_candidate, simulate_model, update_candidate
from src.pricing import black76_implied_vol
from src.signature import PolynomialDiffusionAlgebra, SparseVec, Word


def level2_complete_words(n_factors: int) -> list[Word]:
    """All words of length <= 2 over the alphabet {0, 1, ..., n_factors}
    (0 = time letter): 1 + (n_factors+1) + (n_factors+1)^2 words, i.e. 21
    for n_factors=3 (Section 4.1/4.2, "complete level-two specification")."""
    letters = list(range(n_factors + 1))
    words: list[Word] = [()]
    words += [(a,) for a in letters]
    words += [(a, b) for a in letters for b in letters]
    return words


@dataclass
class CalibrationResult:
    ell_star: SparseVec
    rho_star: np.ndarray
    loss_star: float
    n_iterations: int
    converged: bool
    model: SimulatedModel
    history: list[float]


def _build_quotes_arrays(surface: MarketSurface) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    strikes = np.array([q.strike for q in surface.quotes])
    forward = surface.quotes[0].forward
    maturity = surface.quotes[0].maturity
    return strikes, forward, maturity, surface.mid_ivs()


def run_calibration(
    words: list[Word],
    initial_coeffs: dict[Word, float],
    algebra: PolynomialDiffusionAlgebra,
    delta_years: float,
    n_paths: int,
    horizon_days: float,
    steps_per_day: int,
    seed: int,
    factor_x0: tuple[float, ...],
    correlation: np.ndarray,
    rho_init: np.ndarray,
    r_inv: np.ndarray,
    grid: PricingGrid,
    leverage_L: float,
    floor_m: float,
    variance_floor: float,
    spot: float,
    rate: float,
    vxx0: float,
    vxx_fee: float,
    spx_surface: MarketSurface,
    vix_surface: MarketSurface,
    vxx_surface: MarketSurface,
    lambdas: tuple[float, float, float],
    betas: tuple[float, float, float],
    maxiter: int,
    cir_letter: int | None = None,
    cir_eta: float | None = None,
) -> CalibrationResult:
    """Jointly calibrate the VIX functional ell (on the fixed word basis
    `words`, with starting values `initial_coeffs`) and the leverage
    correlation vector rho to the SPX/VIX/VXX implied-volatility surfaces.

    The parameter vector is [ell coefficients in `words` order, rho (3
    components)]. The expensive Monte Carlo simulation of the primary
    process and its signature is run once (on the word basis fixed by the
    support of `initial_coeffs`); every optimizer iteration only recomputes
    the cheap ell-dependent quantities h(ell) and the pricing functionals
    (Section 4.1, "Calibration settings").
    """
    seed_ell: SparseVec = {w: initial_coeffs.get(w, 0.0) for w in words}
    model0 = simulate_model(
        seed_ell, algebra, delta_years, n_paths, horizon_days, steps_per_day, seed,
        factor_x0, correlation, cir_letter=cir_letter, cir_eta=cir_eta,
    )

    spx_strikes, spx_fwd, spx_T, _ = _build_quotes_arrays(spx_surface)
    vix_strikes, vix_fwd, vix_T, _ = _build_quotes_arrays(vix_surface)
    vxx_strikes, vxx_fwd, vxx_T, _ = _build_quotes_arrays(vxx_surface)

    n_ell = len(words)

    def unpack(theta: np.ndarray) -> tuple[SparseVec, np.ndarray]:
        ell = {w: float(theta[i]) for i, w in enumerate(words)}
        rho = theta[n_ell:n_ell + 3]
        return ell, rho

    def price_at(theta: np.ndarray) -> dict:
        ell, rho = unpack(theta)
        model = update_candidate(model0, ell)
        return price_candidate(
            model, grid, rho, r_inv, leverage_L, floor_m, variance_floor,
            spot, rate, vxx0, vxx_fee, spx_strikes, vix_strikes, vxx_strikes,
        )

    def loss_at(theta: np.ndarray) -> float:
        out = price_at(theta)
        return joint_loss(
            out["spx_prices"], out["vix_prices"], out["vxx_prices"],
            spx_surface.quotes, vix_surface.quotes, vxx_surface.quotes,
            rate, lambdas, betas, block_scales,
        )

    theta0 = np.concatenate([np.array([initial_coeffs.get(w, 0.0) for w in words]), rho_init])

    # Block scales s_A from the initial feasible point (Section 4.1,
    # "Calibration settings": "block scales computed from an initial
    # feasible point"), so each block contributes O(1) at theta0.
    out0 = price_at(theta0)
    block_scales = (
        max(_raw_block(out0["spx_prices"], spx_surface.quotes, rate, betas[0]), 1e-8),
        max(_raw_block(out0["vix_prices"], vix_surface.quotes, rate, betas[1]), 1e-8),
        max(_raw_block(out0["vxx_prices"], vxx_surface.quotes, rate, betas[2]), 1e-8),
    )

    history: list[float] = []

    def objective(theta: np.ndarray) -> float:
        val = loss_at(theta)
        history.append(val)
        return val

    def variance_constraint(theta: np.ndarray) -> float:
        return price_at(theta)["min_raw_variance"] - variance_floor

    def rho_constraint(theta: np.ndarray) -> float:
        _, rho = unpack(theta)
        return 1.0 - rho_constraint_value(rho, r_inv)

    constraints = [
        NonlinearConstraint(variance_constraint, 0.0, np.inf),
        NonlinearConstraint(rho_constraint, 1e-6, np.inf),
    ]
    bounds = [(-10.0, 10.0)] * n_ell + [(-0.999, 0.999)] * 3

    result = minimize(
        objective, theta0, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": maxiter, "ftol": 1e-7},
    )

    ell_star, rho_star = unpack(result.x)
    model_star = update_candidate(model0, ell_star)
    return CalibrationResult(
        ell_star=ell_star, rho_star=rho_star, loss_star=float(result.fun),
        n_iterations=int(result.nit), converged=bool(result.success),
        model=model_star, history=history,
    )


def _raw_block(prices: np.ndarray, quotes, rate: float, beta: float) -> float:
    from src.calibration import loss_block
    return loss_block(prices, quotes, rate, beta)


def model_implied_vols(
    model_prices: np.ndarray, surface: MarketSurface, rate: float
) -> np.ndarray:
    """Model Black-76 implied volatilities on a surface's strike/forward grid."""
    return np.array([
        black76_implied_vol(p, q.forward, q.strike, rate, q.maturity)
        for p, q in zip(model_prices, surface.quotes)
    ])
