"""Tests for src/vix_model_1d.py (Corollary 3.20, Example 3.21, Figure 1)."""
import numpy as np
import pytest

from src.vix_model_1d import (
    nonnegativity_condition_holds,
    simulate_brownian_paths,
    spx_instantaneous_variance,
    vix,
    vix_futures,
    vxx,
)

ELL_EMPTY, ELL_1, ELL_11 = 0.3, 0.0, 1.0
DELTA_YEARS = 30.0 / 365.0
TAU_YEARS = 28.0 / 365.0


def test_simulate_brownian_paths_shape_and_start():
    t, paths = simulate_brownian_paths(n_paths=50, horizon_days=30, steps_per_day=4, seed=0)
    assert paths.shape == (121, 50)
    assert np.allclose(paths[0], 0.0)
    assert np.isclose(t[-1], 30.0 / 365.0)


def test_vix_at_t0_equals_ell_empty():
    """At X_0 = 0, VIX_0 = ell_empty = 0.3 (Example 3.21)."""
    v0 = vix(np.zeros(5), ELL_EMPTY, ELL_1, ELL_11)
    assert np.allclose(v0, 0.3)


def test_nonnegativity_condition_holds_for_paper_parameters():
    """0.3 >= 0 + 7*(30/365)/16 ~= 0.03596 (equation 3.16)."""
    assert nonnegativity_condition_holds(ELL_EMPTY, ELL_1, ELL_11, DELTA_YEARS)
    rhs = ELL_1 ** 2 / (2 * ELL_11 ** 2) + 7 * DELTA_YEARS / 16.0
    assert rhs == pytest.approx(0.03596, abs=1e-4)


def test_nonnegativity_condition_rejects_non_positive_ell11():
    with pytest.raises(ValueError):
        nonnegativity_condition_holds(ELL_EMPTY, ELL_1, 0.0, DELTA_YEARS)


def test_vix_futures_reduces_to_vix_at_maturity():
    """F_t(T, ell) -> VIX_t as t -> T (Corollary 3.20)."""
    t = np.array([30.0 / 365.0])
    v = np.array([[0.31]])
    f = vix_futures(v, t, maturity_years=30.0 / 365.0, ell_11=ELL_11)
    assert np.allclose(f, v)


def test_vix_futures_slope_matches_ell11_over_2():
    """d F_t(T)/dT = ell_11 / 2 (Corollary 3.20)."""
    t = np.array([0.0])
    v = np.array([[0.3]])
    f1 = vix_futures(v, t, maturity_years=30.0 / 365.0, ell_11=ELL_11)
    f2 = vix_futures(v, t, maturity_years=60.0 / 365.0, ell_11=ELL_11)
    slope = (f2[0, 0] - f1[0, 0]) / (30.0 / 365.0)
    assert slope == pytest.approx(ELL_11 / 2.0)


def test_spx_variance_nonnegative_on_simulated_paths():
    """Under condition (3.16), sigma_t^S(ell)^2 >= 0 on all simulated paths."""
    _, paths = simulate_brownian_paths(n_paths=2000, horizon_days=30, steps_per_day=4, seed=1)
    v = vix(paths, ELL_EMPTY, ELL_1, ELL_11)
    variance = spx_instantaneous_variance(paths, v, DELTA_YEARS, ELL_EMPTY, ELL_1, ELL_11)
    assert nonnegativity_condition_holds(ELL_EMPTY, ELL_1, ELL_11, DELTA_YEARS)
    assert variance.min() >= -1e-10


def test_vix_nonnegative_on_simulated_paths():
    _, paths = simulate_brownian_paths(n_paths=2000, horizon_days=30, steps_per_day=4, seed=2)
    v = vix(paths, ELL_EMPTY, ELL_1, ELL_11)
    assert v.min() >= 0.0


def test_vxx_starts_at_vxx0():
    t, paths = simulate_brownian_paths(n_paths=10, horizon_days=30, steps_per_day=4, seed=3)
    v = vix(paths, ELL_EMPTY, ELL_1, ELL_11)
    dt_years = t[1] - t[0]
    vxx0 = 0.340
    path_vxx = vxx(paths, v, t, dt_years, vxx0, ELL_EMPTY, ELL_11, TAU_YEARS)
    assert np.allclose(path_vxx[0], vxx0)


def test_vxx_positive_and_finite():
    t, paths = simulate_brownian_paths(n_paths=500, horizon_days=30, steps_per_day=4, seed=4)
    v = vix(paths, ELL_EMPTY, ELL_1, ELL_11)
    dt_years = t[1] - t[0]
    path_vxx = vxx(paths, v, t, dt_years, 0.340, ELL_EMPTY, ELL_11, TAU_YEARS)
    assert np.isfinite(path_vxx).all()
    assert (path_vxx > 0).all()


def test_single_input_rejected():
    with pytest.raises(ValueError):
        simulate_brownian_paths(n_paths=0, horizon_days=30, steps_per_day=4, seed=0)
    with pytest.raises(ValueError):
        simulate_brownian_paths(n_paths=10, horizon_days=-1, steps_per_day=4, seed=0)
