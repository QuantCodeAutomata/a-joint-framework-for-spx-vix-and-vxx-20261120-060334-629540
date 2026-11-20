"""Regularized VIX-based signature model: pricing of SPX, VIX and VXX claims.

Implements the regularized model of Definition 3.22, the leverage-saturation
construction of Proposition 3.11 (equation 3.11), the VIX/VXX/futures proxies
of Appendix C, and Black-76 implied-volatility inversion. The regularized
pricing recipe (raw variance -> positive part -> leverage saturation -> Monte
Carlo discounted payoff) is a custom implementation of the paper's equations;
Black-76 inversion uses scipy.optimize.brentq, a standard scalar root finder,
since no quant library in this environment implements the paper's specific
VXX-roll proxy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from src.signature import (
    SparseVec,
    SpectralPropagator,
    Word,
    integral_matrix_exp,
    shuffle,
    spectral_propagator,
    word_to_vector,
)


@dataclass
class ModelOperators:
    """Pre-computed, ell-independent objects shared across calibration
    candidates: the dual-operator matrix G, its cached eigendecomposition
    used to evaluate exp(t G) for any t (Theorem 2.4's semigroup property),
    the matrix A and its inverse on the level-<=2n word basis, and the word
    index."""

    words: list[Word]
    index: dict[Word, int]
    G: np.ndarray
    propagator: SpectralPropagator
    A_inv: np.ndarray
    delta_years: float


def build_operators(words: list[Word], G: np.ndarray, delta_years: float) -> ModelOperators:
    A = integral_matrix_exp(G, delta_years)
    A_inv = np.linalg.inv(A)
    index = {w: i for i, w in enumerate(words)}
    # exp(t G) is evaluated via the cached eigendecomposition of G
    # (`spectral_propagator`), not a truncated power series: G is only
    # nilpotent for the driftless one-dimensional case of Corollary 3.20,
    # not for the mean-reverting OU/CIR primary processes of Section 4
    # (see `src/signature.py::nilpotent_powers` docstring).
    propagator = spectral_propagator(G)
    return ModelOperators(words=words, index=index, G=G, propagator=propagator, A_inv=A_inv, delta_years=delta_years)


def ell_power_vectors(ops: ModelOperators, ell_vec: np.ndarray) -> np.ndarray:
    """Eigenbasis coefficients of `ell_vec`, computed once per candidate ell
    and reused across every maturity/time node of a pricing pass (see
    `SpectralPropagator.coeffs`)."""
    return ops.propagator.coeffs(ell_vec)


def propagate_ell(ops: ModelOperators, ell_coeffs: np.ndarray, t: float) -> np.ndarray:
    """exp(t G) @ ell_vec via the cached spectral decomposition of G
    (Theorem 2.4 / Theorem 3.14), using the pre-contracted eigenbasis
    coefficients `ell_coeffs` so that only O(dim) work per time node remains
    (see `ell_power_vectors`)."""
    return ops.propagator.propagate(ell_coeffs, t)


def h_of_ell(ell: SparseVec, ops: ModelOperators) -> np.ndarray:
    """vec(h(ell))^T = Delta * vec(ell shuffle ell)^T @ A^{-1}  (Theorem 3.14)."""
    ell_shuffle_ell = shuffle(ell, ell)
    b_ell = word_to_vector(ell_shuffle_ell, ops.words)
    return ops.delta_years * (ops.A_inv.T @ b_ell)


def raw_variance_path(sig_level_2n: np.ndarray, h_vec: np.ndarray) -> np.ndarray:
    """v_t^raw = <h(ell), X_hat_t^{<=2n}> at every (time, path) node."""
    return sig_level_2n @ h_vec


def vix_proxy(sig_level_n: np.ndarray, ell_vec: np.ndarray) -> np.ndarray:
    """VIX proxy: |<ell, X_hat_t^{<=n}>| (Appendix C, "tractable proxy")."""
    return np.abs(sig_level_n @ ell_vec)


def signed_vix_functional(sig_level_n: np.ndarray, ell_vec: np.ndarray) -> np.ndarray:
    """Signed functional <ell, X_hat_t^{<=n}> (used for futures/VXX, before
    the absolute value and stopping are applied)."""
    return sig_level_n @ ell_vec


def futures_proxy_signed(
    sig: np.ndarray, ops: ModelOperators, ell_coeffs: np.ndarray, maturity_minus_t: float
) -> np.ndarray:
    """F~_t(T, ell) = <ell, X_hat_t^{<=n}> propagated to maturity via the
    semigroup duality <ell, P_{T-t} X_hat_t> = <e^{(T-t) G} ell, X_hat_t>
    (Theorem 2.4 / Theorem 3.14): ell is propagated forward once through the
    cached spectral decomposition of G, e^{(T-t) G} ell (`propagate_ell`),
    and then contracted against the full signature path `sig` (shape
    (..., len(words))).
    """
    propagated_ell = propagate_ell(ops, ell_coeffs, maturity_minus_t)
    return sig @ propagated_ell


def stopped_futures(futures_path: np.ndarray, floor_m: float) -> np.ndarray:
    """Approximate the stopping time tau_m by freezing each path at m after
    its first observed value at or below m (Appendix C, "On the simulation
    grid, continuous-time stopping is approximated by...")."""
    below = futures_path <= floor_m
    first_hit = np.where(below.any(axis=0), below.argmax(axis=0), futures_path.shape[0])
    out = futures_path.copy()
    n_steps, n_paths = futures_path.shape
    step_idx = np.arange(n_steps)[:, None]
    hit_mask = step_idx >= first_hit[None, :]
    frozen_value = np.take_along_axis(futures_path, np.clip(first_hit, 0, n_steps - 1)[None, :], axis=0)
    out = np.where(hit_mask, frozen_value, out)
    return out


def vxx_roll_proxy_before_t1(
    futures_t1: np.ndarray,
    futures_t2: np.ndarray,
    t_years: np.ndarray,
    t0_years: float,
    t1_years: float,
    vxx0: float,
    dt_years: float,
) -> np.ndarray:
    """VXX proxy for T <= T1 (Corollary 3.5 / Appendix C):

    VXX_T = VXX0 * [(T-T0) F_T(T2) + (T1-T) F_T(T1)]
                 / [-T0 F_0(T2) + T1 F_0(T1)]
            * exp( integral_0^T [F_s(T1) - F_s(T2)] /
                   [(s-T0) F_s(T2) + (T1-s) F_s(T1)] ds ).
    """
    numerator = (t_years[:, None] - t0_years) * futures_t2 + (t1_years - t_years[:, None]) * futures_t1
    denom0 = -t0_years * futures_t2[0] + t1_years * futures_t1[0]
    integrand = (futures_t1 - futures_t2) / (
        (t_years[:, None] - t0_years) * futures_t2 + (t1_years - t_years[:, None]) * futures_t1
    )
    cumulative = np.concatenate(
        [np.zeros((1, integrand.shape[1])), np.cumsum(integrand[:-1], axis=0) * dt_years], axis=0
    )
    return vxx0 * (numerator / denom0) * np.exp(cumulative)


def vxx_roll_proxy(
    times_years: np.ndarray,
    futures_t1: np.ndarray,
    futures_t2: np.ndarray,
    futures_t3: np.ndarray,
    t0_years: float,
    t1_years: float,
    t2_years: float,
    vxx0: float,
    dt_years: float,
) -> np.ndarray:
    """VXX proxy covering both roll regimes of Corollary 3.5 / Appendix C
    (equation C.3): ``T <= T1`` and ``T1 < T <= T2``, under equal futures
    spacing ``tau = T_i - T_{i-1}``.

    ``futures_t1``, ``futures_t2`` and ``futures_t3`` are the (already
    stopping-floored) signed futures proxies for maturities T1, T2 and T3,
    evaluated at every grid time in ``times_years`` (shape (n_steps+1,
    n_paths)); ``futures_t3`` is only used for ``s >= T1``. The option
    maturity used in the calibration (24 days) falls two days after T1 = 22
    days, so both regimes are needed on the simulation grid.
    """
    tau1 = t1_years - t0_years
    tau2 = t2_years - t1_years
    denom0 = -t0_years * futures_t2[0] + t1_years * futures_t1[0]

    integrand_before = (futures_t1 - futures_t2) / (
        (times_years[:, None] - t0_years) * futures_t2 + (t1_years - times_years[:, None]) * futures_t1
    )
    integrand_after = (futures_t2 - futures_t3) / (
        (times_years[:, None] - t1_years) * futures_t3 + (t2_years - times_years[:, None]) * futures_t2
    )

    is_before_t1 = times_years <= t1_years
    idx_t1 = int(np.searchsorted(times_years, t1_years, side="right"))

    cum_before = np.concatenate(
        [np.zeros((1, integrand_before.shape[1])), np.cumsum(integrand_before[:-1], axis=0) * dt_years], axis=0
    )
    integral_at_t1 = cum_before[min(idx_t1, cum_before.shape[0] - 1)]
    cum_after = np.concatenate(
        [np.zeros((1, integrand_after.shape[1])), np.cumsum(integrand_after[:-1], axis=0) * dt_years], axis=0
    )
    cum_after_shifted = cum_after - cum_after[min(idx_t1, cum_after.shape[0] - 1)] + integral_at_t1

    value_before = vxx0 * (
        (times_years[:, None] - t0_years) * futures_t2 + (t1_years - times_years[:, None]) * futures_t1
    ) / denom0 * np.exp(cum_before)
    value_after = vxx0 * (tau1 / tau2) * (
        (times_years[:, None] - t1_years) * futures_t3 + (t2_years - times_years[:, None]) * futures_t2
    ) / denom0 * np.exp(cum_after_shifted)

    return np.where(is_before_t1[:, None], value_before, value_after)


def leverage_saturated_driver(
    sigma_path: np.ndarray,
    corr_increments: np.ndarray,
    perp_increments: np.ndarray,
    rho: np.ndarray,
    R_inv: np.ndarray,
    leverage_L: float,
) -> np.ndarray:
    """Increments of B^L as in equation (3.11):

        dB^L = chi_L(|sigma|) rho^T R^{-1} dW + sqrt(1 - chi_L(|sigma|)^2 rho^T R^{-1} rho) dW^perp.

    sigma_path has shape (n_steps, n_paths) (sigma_t evaluated on [0, T-dt]),
    corr_increments has shape (n_steps, n_paths, d) and perp_increments has
    shape (n_steps, n_paths).
    """
    abs_sigma = np.abs(sigma_path)
    chi = np.where(abs_sigma <= leverage_L, 1.0, leverage_L / np.maximum(abs_sigma, 1e-12))
    rho_Rinv = R_inv @ rho
    quad_form = float(rho @ R_inv @ rho)
    weight_perp = np.sqrt(np.maximum(1.0 - (chi ** 2) * quad_form, 0.0))
    dB = chi * np.einsum("spj,j->sp", corr_increments, rho_Rinv) + weight_perp * perp_increments
    return dB


def spx_price_path(dB: np.ndarray, raw_variance: np.ndarray, dt_years: float, s0: float) -> np.ndarray:
    """S_T = S0 * exp( sum sqrt(v^+) dB - 1/2 sum v^+ dt )  (Definition 3.22)."""
    variance_pos = np.maximum(raw_variance[:-1], 0.0)
    log_increments = np.sqrt(variance_pos) * dB - 0.5 * variance_pos * dt_years
    log_terminal = np.sum(log_increments, axis=0)
    return s0 * np.exp(log_terminal)


def discounted_call_price(terminal: np.ndarray, strike: float, rate: float, maturity: float) -> float:
    payoff = np.maximum(terminal - strike, 0.0)
    return float(np.exp(-rate * maturity) * np.mean(payoff))


def black76_call_price(forward: float, strike: float, vol: float, rate: float, maturity: float) -> float:
    if vol <= 0 or maturity <= 0:
        return float(np.exp(-rate * maturity) * max(forward - strike, 0.0))
    d1 = (np.log(forward / strike) + 0.5 * vol ** 2 * maturity) / (vol * np.sqrt(maturity))
    d2 = d1 - vol * np.sqrt(maturity)
    return float(np.exp(-rate * maturity) * (forward * norm.cdf(d1) - strike * norm.cdf(d2)))


def black76_implied_vol(
    price: float, forward: float, strike: float, rate: float, maturity: float
) -> float:
    """Invert the Black-76 call price for implied volatility via Brent's method."""
    intrinsic = np.exp(-rate * maturity) * max(forward - strike, 0.0)
    if price <= intrinsic:
        return 0.0

    def objective(vol: float) -> float:
        return black76_call_price(forward, strike, vol, rate, maturity) - price

    try:
        return float(brentq(objective, 1e-6, 5.0, xtol=1e-8))
    except ValueError:
        return float("nan")


def black76_vega(forward: float, strike: float, vol: float, rate: float, maturity: float) -> float:
    if vol <= 0 or maturity <= 0:
        return 0.0
    d1 = (np.log(forward / strike) + 0.5 * vol ** 2 * maturity) / (vol * np.sqrt(maturity))
    return float(np.exp(-rate * maturity) * forward * norm.pdf(d1) * np.sqrt(maturity))
