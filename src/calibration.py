"""Joint SPX-VIX-VXX calibration loss (Appendix C) and option-surface helpers.

Custom implementation of the paper's joint loss function; scipy.optimize
supplies the constrained local optimizer (SLSQP), since no quant library in
this environment implements this specific multi-market calibration loss.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.pricing import black76_implied_vol, black76_vega


@dataclass
class OptionQuote:
    strike: float
    maturity: float
    bid_iv: float
    ask_iv: float
    forward: float


@dataclass
class MarketSurface:
    """Bid/ask implied-volatility quotes for one market (SPX, VIX or VXX)."""

    quotes: list[OptionQuote] = field(default_factory=list)

    def mid_ivs(self) -> np.ndarray:
        return np.array([0.5 * (q.bid_iv + q.ask_iv) for q in self.quotes])

    def spreads(self) -> np.ndarray:
        return np.array([q.ask_iv - q.bid_iv for q in self.quotes])


def loss_block(
    model_prices: np.ndarray,
    quotes: list[OptionQuote],
    rate: float,
    beta: float,
) -> float:
    """One block of the joint loss (equation C.1): a Huber-type distance
    between the model implied volatility and the bid-ask band, normalized by
    Black-76 vega times the quoted bid-ask spread, with shape parameter
    `beta` controlling the transition between quadratic and linear loss.
    """
    total = 0.0
    for price, quote in zip(model_prices, quotes):
        model_iv = black76_implied_vol(price, quote.forward, quote.strike, rate, quote.maturity)
        if not np.isfinite(model_iv):
            total += 10.0
            continue
        mid_iv = 0.5 * (quote.bid_iv + quote.ask_iv)
        spread = max(quote.ask_iv - quote.bid_iv, 1e-4)
        vega = black76_vega(quote.forward, quote.strike, mid_iv, rate, quote.maturity)
        scale = max(vega * spread, 1e-8)
        if quote.bid_iv <= model_iv <= quote.ask_iv:
            distance = 0.0
        else:
            distance = min(abs(model_iv - quote.bid_iv), abs(model_iv - quote.ask_iv))
        residual = distance / scale
        # Huber-type blending controlled by beta: quadratic near zero,
        # linear in the tail, matching the paper's "Huber loss" description
        # of equation (C.1) (Appendix C).
        if residual <= beta:
            total += 0.5 * residual ** 2
        else:
            total += beta * (residual - 0.5 * beta)
    return total / max(len(quotes), 1)


def joint_loss(
    spx_prices: np.ndarray,
    vix_prices: np.ndarray,
    vxx_prices: np.ndarray,
    spx_quotes: list[OptionQuote],
    vix_quotes: list[OptionQuote],
    vxx_quotes: list[OptionQuote],
    rate: float,
    lambdas: tuple[float, float, float],
    betas: tuple[float, float, float],
    block_scales: tuple[float, float, float],
) -> float:
    """L_joint = sum_A lambda_A * B_A / s_A, A in {SPX, VIX, VXX}."""
    b_spx = loss_block(spx_prices, spx_quotes, rate, betas[0])
    b_vix = loss_block(vix_prices, vix_quotes, rate, betas[1])
    b_vxx = loss_block(vxx_prices, vxx_quotes, rate, betas[2])
    lam_spx, lam_vix, lam_vxx = lambdas
    s_spx, s_vix, s_vxx = block_scales
    return (
        lam_spx * b_spx / s_spx
        + lam_vix * b_vix / s_vix
        + lam_vxx * b_vxx / s_vxx
    )


def rho_constraint_value(rho: np.ndarray, r_inv: np.ndarray) -> float:
    """rho^T R^{-1} rho, must stay below 1 (martingality constraint of
    Proposition 3.11 and Section 4)."""
    return float(rho @ r_inv @ rho)
