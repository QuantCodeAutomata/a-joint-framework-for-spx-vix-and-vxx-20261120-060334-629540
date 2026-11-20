"""Simulation of the primary process and its time-extended truncated signature.

Implements equation (2.4) of "A joint framework for SPX, VIX and VXX" as a
joint Euler-Maruyama recursion: for every word I (in increasing length order)

    d<e_I, X_hat_t> = <Le_I, X_hat_t> dt + <e_{I'}, X_hat_t> * dW^{i_k}_t   (i_k in 1..d)
    d<e_I, X_hat_t> = <e_{I'}, X_hat_t> dt                                  (i_k = 0, time letter)

so that the signature components of every word in a chosen (L-closed) word
basis are simulated simultaneously with the primary factors themselves,
using the same driving Brownian increments. This is the "signature as a
driven SDE" viewpoint used throughout Section 2.2 and Section 4 of the paper,
and avoids a separate symbolic-to-numeric translation step.

Covariance libraries (arch, hmmlearn, ...) do not provide this signature-SDE
construction, so it is implemented here directly from the paper's equation
(2.4); only the elementary Gaussian/ chi-squared sampling uses numpy/scipy.
"""
from __future__ import annotations

import numpy as np

from src.signature import PolynomialDiffusionAlgebra, Word, close_words_under_L


def ou_cir_algebra(
    ou_kappa: tuple[float, ...],
    ou_theta: tuple[float, ...],
    ou_eta: tuple[float, ...],
    ou_x0: tuple[float, ...],
    correlation: np.ndarray,
    cir_kappa: float | None = None,
    cir_theta: float | None = None,
    cir_eta: float | None = None,
    cir_y0: float | None = None,
) -> PolynomialDiffusionAlgebra:
    """Build the dual-operator algebra for n_ou correlated OU factors plus,
    optionally, one additional CIR factor (letters: 0=time, 1..n_ou=OU,
    n_ou+1=CIR). The CIR driver is assumed independent of the OU drivers,
    matching Section 4.2.
    """
    n_ou = len(ou_kappa)
    has_cir = cir_kappa is not None
    n_factors = n_ou + (1 if has_cir else 0)

    drift: dict[int, dict[Word, float]] = {}
    for idx in range(n_ou):
        letter = idx + 1
        kappa, theta, x0 = ou_kappa[idx], ou_theta[idx], ou_x0[idx]
        drift[letter] = {(): kappa * theta - kappa * x0, (letter,): -kappa}
    if has_cir:
        cir_letter = n_ou + 1
        drift[cir_letter] = {
            (): cir_kappa * cir_theta - cir_kappa * cir_y0,
            (cir_letter,): -cir_kappa,
        }

    covariance: dict[tuple[int, int], dict[Word, float]] = {}
    for i in range(n_ou):
        for j in range(i, n_ou):
            a_ij = ou_eta[i] * ou_eta[j] * correlation[i, j]
            if a_ij != 0.0:
                covariance[(i + 1, j + 1)] = {(): a_ij}
    if has_cir:
        cir_letter = n_ou + 1
        # a_CIR,CIR(Y_t) = eta^2 * Y_t = eta^2 * Y0 + eta^2 * <e_CIR, X_hat_t>
        covariance[(cir_letter, cir_letter)] = {(): cir_eta ** 2 * cir_y0, (cir_letter,): cir_eta ** 2}

    return PolynomialDiffusionAlgebra(n_factors=n_factors, drift=drift, covariance=covariance)


def simulate_signature_paths(
    algebra: PolynomialDiffusionAlgebra,
    words: list[Word],
    n_paths: int,
    horizon_days: float,
    steps_per_day: int,
    seed: int,
    factor_x0: tuple[float, ...],
    correlation: np.ndarray,
    cir_letter: int | None = None,
    cir_eta: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Jointly simulate the primary factors and the signature components of
    every word in `words` (which must be closed under L, see
    `close_words_under_L`).

    Returns (times_years, signature_tensor, perp_increments) where
    signature_tensor has shape (n_steps + 1, n_paths, len(words)) with
    signature_tensor[..., k] = <e_{words[k]}, X_hat_t>, and perp_increments
    has shape (n_steps, n_paths): standard Gaussian increments of an
    auxiliary Brownian motion independent of the factor drivers, used later
    to build the leverage-saturated price driver B^L (equation 3.11).
    """
    n_factors = algebra.n_factors
    n_steps = int(round(horizon_days * steps_per_day))
    dt_years = (1.0 / 365.0) / steps_per_day
    rng = np.random.default_rng(seed)

    chol = np.linalg.cholesky(correlation)
    raw_normals = rng.normal(size=(n_steps, n_paths, n_factors))
    dW = np.einsum("ij,spj->spi", chol, raw_normals) * np.sqrt(dt_years)
    perp_increments = rng.normal(size=(n_steps, n_paths)) * np.sqrt(dt_years)

    index = {w: i for i, w in enumerate(words)}
    words_by_level: dict[int, list[Word]] = {}
    for w in words:
        words_by_level.setdefault(len(w), []).append(w)
    max_level = max(words_by_level)

    dim = len(words)
    sig = np.zeros((n_steps + 1, n_paths, dim))
    if () in index:
        sig[:, :, index[()]] = 1.0
    factor_paths = np.zeros((n_steps + 1, n_paths, n_factors))
    for j in range(n_factors):
        factor_paths[0, :, j] = factor_x0[j] if j < len(factor_x0) else 0.0

    # drift coefficient lookup, precomputed per word (list of (target_word, coeff))
    drift_terms: dict[Word, list[tuple[Word, float]]] = {
        w: list(algebra.apply_L(w).items()) for w in words
    }

    for step in range(n_steps):
        current = sig[step]
        nxt = np.zeros_like(current)
        if () in index:
            nxt[:, index[()]] = 1.0
        for level in sorted(words_by_level):
            if level == 0:
                continue
            for word in words_by_level[level]:
                k = index[word]
                drift_val = np.zeros(n_paths)
                for target_word, coeff in drift_terms[word]:
                    drift_val += coeff * current[:, index[target_word]]
                last_letter = word[-1]
                if last_letter == 0:
                    nxt[:, k] = current[:, k] + current[:, index[word[:-1]]] * dt_years
                else:
                    diffusion_letter = last_letter - 1  # 0-based index into dW
                    nxt[:, k] = (
                        current[:, k]
                        + drift_val * dt_years
                        + current[:, index[word[:-1]]] * dW[step, :, diffusion_letter]
                    )
        # CIR full-truncation: clip negative realized variance state to zero to
        # avoid a complex-valued diffusion coefficient on the next step. This is
        # the standard full-truncation Euler scheme (Lord, Koekkoek & Van Dijk,
        # 2010); the Feller ratio of 2 keeps negative excursions rare on this grid.
        if cir_letter is not None and (cir_letter,) in index:
            y_idx = index[(cir_letter,)]
            nxt[:, y_idx] = np.maximum(nxt[:, y_idx], -factor_x0[cir_letter - 1] + 1e-10)
        sig[step + 1] = nxt
        for j in range(n_factors):
            letter = j + 1
            if (letter,) in index:
                factor_paths[step + 1, :, j] = factor_x0[j] + sig[step + 1, :, index[(letter,)]]

    times_years = np.arange(n_steps + 1) * dt_years
    return times_years, sig, perp_increments
