"""One-dimensional Brownian primary process: closed-form VIX, VXX, variance.

Implements Corollary 3.20 of "A joint framework for SPX, VIX and VXX": the
VIX-based signature model driven by a one-dimensional Brownian motion
X_t = W_t (b = alpha = 0, a-bar = 1, X_0 = 0), with VIX functional truncated
at level 2,

    VIX_t = ell_empty + ell_1 * X_t + ell_11 * X_t^2 / 2.

All formulas below are the closed-form expressions of Corollary 3.20; no
matrix exponential or shuffle-product machinery is needed in this
one-dimensional case because the dual operator G is explicitly nilpotent
(page 28-29 of the paper) and the paper gives the resulting closed forms
directly.
"""
from __future__ import annotations

import numpy as np


def simulate_brownian_paths(
    n_paths: int,
    horizon_days: float,
    steps_per_day: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate standard Brownian motion paths on a uniform daily grid.

    Returns (times_in_years, paths) where paths has shape
    (n_steps + 1, n_paths) and paths[0] == 0 (X_0 = 0).
    """
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    if steps_per_day <= 0:
        raise ValueError("steps_per_day must be positive")

    n_steps = int(round(horizon_days * steps_per_day))
    dt_years = (1.0 / 365.0) / steps_per_day
    rng = np.random.default_rng(seed)
    increments = rng.normal(loc=0.0, scale=np.sqrt(dt_years), size=(n_steps, n_paths))
    paths = np.concatenate([np.zeros((1, n_paths)), np.cumsum(increments, axis=0)], axis=0)
    times_years = np.arange(n_steps + 1) * dt_years
    return times_years, paths


def vix(x_t: np.ndarray, ell_empty: float, ell_1: float, ell_11: float) -> np.ndarray:
    """VIX_t = ell_empty + ell_1 * X_t + ell_11 * X_t^2 / 2 (Corollary 3.20)."""
    return ell_empty + ell_1 * x_t + ell_11 * (x_t ** 2) / 2.0


def vix_futures(
    vix_t: np.ndarray, t_years: np.ndarray, maturity_years: float, ell_11: float
) -> np.ndarray:
    """F_t(T, ell) = VIX_t + (1/2) * ell_11 * (T - t)  (Corollary 3.20)."""
    tau_to_maturity = maturity_years - t_years
    return vix_t + 0.5 * ell_11 * tau_to_maturity[:, None]


def spx_instantaneous_variance(
    x_t: np.ndarray,
    vix_t: np.ndarray,
    delta_years: float,
    ell_empty: float,
    ell_1: float,
    ell_11: float,
) -> np.ndarray:
    """sigma_t^S(ell)^2 as given explicitly in Corollary 3.20:

    sigma_t^S(ell)^2 = VIX_t^2 + Delta^2*ell_11^2/8
                        - Delta*(2*ell_1^2 + 2*ell_empty*ell_11)/4
                        - (3*Delta*ell_1*ell_11/2) * X_t
                        - (3*Delta*ell_11^2/2) * X_t^2/2
    """
    term_const = (delta_years ** 2) * (ell_11 ** 2) / 8.0
    term_const -= delta_years * (2 * ell_1 ** 2 + 2 * ell_empty * ell_11) / 4.0
    term_linear = -(3 * delta_years * ell_1 * ell_11 / 2.0) * x_t
    term_quadratic = -(3 * delta_years * ell_11 ** 2 / 2.0) * (x_t ** 2) / 2.0
    return vix_t ** 2 + term_const + term_linear + term_quadratic


def vxx(
    x_t: np.ndarray,
    vix_t: np.ndarray,
    t_years: np.ndarray,
    dt_years: float,
    vxx0: float,
    ell_empty: float,
    ell_11: float,
    tau_years: float,
) -> np.ndarray:
    """VXX_t closed form of Corollary 3.20 (equal futures spacing tau):

    VXX_t = VXX_0 * (2*VIX_t + ell_11*tau) / (2*ell_empty + ell_11*tau)
            * exp( -integral_0^t ell_11 / (2*VIX_s + ell_11*tau) ds ).

    The integral is approximated by a cumulative (left) Riemann sum on the
    simulation grid, consistent with the Euler discretization used to
    simulate the driving Brownian motion.
    """
    numerator = 2 * vix_t + ell_11 * tau_years
    denominator0 = 2 * ell_empty + ell_11 * tau_years
    integrand = ell_11 / (2 * vix_t + ell_11 * tau_years)
    # Cumulative left Riemann sum: integral_0^{t_k} f(s) ds ~= dt * sum_{j<k} f(t_j)
    cumulative_integral = np.concatenate(
        [np.zeros((1, x_t.shape[1])), np.cumsum(integrand[:-1], axis=0) * dt_years], axis=0
    )
    return vxx0 * (numerator / denominator0) * np.exp(-cumulative_integral)


def nonnegativity_condition_holds(ell_empty: float, ell_1: float, ell_11: float, delta_years: float) -> bool:
    """Sufficient condition (3.16): ell_empty/ell_11 >= ell_1^2/(2*ell_11^2) + 7*Delta/16.

    Only meaningful for ell_11 > 0; raises otherwise since the paper's
    sufficient condition is stated for that case only.
    """
    if ell_11 <= 0:
        raise ValueError("Condition (3.16) is stated for ell_11 > 0")
    lhs = ell_empty / ell_11
    rhs = ell_1 ** 2 / (2 * ell_11 ** 2) + 7 * delta_years / 16.0
    return lhs >= rhs
