"""Synthetic SPX, VIX and VXX option-quote surfaces.

The calibration described in Section 4 uses historical bid/ask quotes for
SPX, VIX and VXX options obtained through Cboe DataShop under a commercial
license (Section 4, Definition 4.1). That licensed, contract-level dataset
cannot be obtained in this environment: Massive's options analytics endpoint
(`GET /options/v1/analytics`), which is the only data source in this project
that could serve historical option greeks and implied volatility, returned
HTTP 403 "You are not entitled to this data" for SPX, VIX and VXX index
options under the available subscription. This module therefore generates a
synthetic bid/ask implied-volatility surface with the strike grid, moneyness
range, maturities and interest rate stated in Section 4, and with the
qualitative smile shapes the paper reports (SPX downside skew, increasing VIX
smile, increasing VXX smile), so that the calibration pipeline can be
exercised end to end. It is not a substitute for the paper's reported
calibration outputs; see results/RESULTS.md.
"""
from __future__ import annotations

import numpy as np

from src.calibration import MarketSurface, OptionQuote


def _synthetic_iv_curve(
    moneyness: np.ndarray, atm_vol: float, skew: float, convexity: float, bid_ask_width: float
) -> tuple[np.ndarray, np.ndarray]:
    """A simple quadratic-in-log-moneyness smile: sigma(k) = atm_vol + skew*k +
    convexity*k^2, k = log(moneyness), with a constant relative bid-ask width.
    This is a smile parametrization (not a model), used only to label a
    synthetic surface; it is not part of the paper's methodology.
    """
    k = np.log(moneyness)
    mid = np.maximum(atm_vol + skew * k + convexity * k ** 2, 0.01)
    half_width = 0.5 * bid_ask_width * mid
    return mid - half_width, mid + half_width


def generate_surface(
    spot_or_level: float,
    rate: float,
    maturity_days: float,
    moneyness_range: tuple[float, float],
    n_strikes: int,
    atm_vol: float,
    skew: float,
    convexity: float,
    bid_ask_width: float,
) -> MarketSurface:
    """Build one synthetic bid/ask surface for a given market."""
    maturity_years = maturity_days / 365.0
    forward = spot_or_level * np.exp(rate * maturity_years)
    moneyness = np.linspace(moneyness_range[0], moneyness_range[1], n_strikes)
    strikes = moneyness * forward
    bid_iv, ask_iv = _synthetic_iv_curve(moneyness, atm_vol, skew, convexity, bid_ask_width)
    quotes = [
        OptionQuote(strike=float(k), maturity=maturity_years, bid_iv=float(b), ask_iv=float(a), forward=float(forward))
        for k, b, a in zip(strikes, bid_iv, ask_iv)
    ]
    return MarketSurface(quotes=quotes)


def generate_joint_surfaces(
    spx_spot: float,
    vix_level: float,
    vxx_spot: float,
    rate: float,
    spx_maturity_days: float,
    vix_maturity_days: float,
    vxx_maturity_days: float,
    spx_moneyness: tuple[float, float],
    vix_moneyness: tuple[float, float],
    vxx_moneyness: tuple[float, float],
    n_spx: int,
    n_vix: int,
    n_vxx: int,
    seed: int,
) -> tuple[MarketSurface, MarketSurface, MarketSurface]:
    """Synthetic SPX (downside skew), VIX (increasing smile) and VXX
    (increasing smile) surfaces with the strike counts, moneyness ranges,
    maturities and interest rate of Section 4's market-data description.
    """
    rng = np.random.default_rng(seed)
    spx = generate_surface(
        spx_spot, rate, spx_maturity_days, spx_moneyness, n_spx,
        atm_vol=0.17, skew=-0.12, convexity=0.05, bid_ask_width=0.08,
    )
    vix = generate_surface(
        vix_level, rate, vix_maturity_days, vix_moneyness, n_vix,
        atm_vol=0.95, skew=0.35, convexity=0.10, bid_ask_width=0.10,
    )
    vxx = generate_surface(
        vxx_spot, rate, vxx_maturity_days, vxx_moneyness, n_vxx,
        atm_vol=0.55, skew=0.30, convexity=0.08, bid_ask_width=0.12,
    )
    # Small, seeded jitter to avoid a perfectly smooth parametric curve.
    for surface in (spx, vix, vxx):
        jitter = rng.normal(scale=0.003, size=len(surface.quotes))
        for quote, eps in zip(surface.quotes, jitter):
            quote.bid_iv = max(quote.bid_iv + eps, 0.01)
            quote.ask_iv = max(quote.ask_iv + eps, quote.bid_iv + 0.005)
    return spx, vix, vxx
