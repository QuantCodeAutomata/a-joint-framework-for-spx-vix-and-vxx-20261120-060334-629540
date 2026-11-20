"""Unit tests for src/signature.py: shuffle product, dual operator G, and the
matrix A = integral_0^Delta exp(sG) ds (Section 2.2, Theorem 2.4, Theorem 3.14)."""
import numpy as np
import pytest

from src.signature import (
    PolynomialDiffusionAlgebra,
    build_G,
    close_words_under_L,
    integral_matrix_exp,
    nilpotent_powers,
    shuffle,
    word_to_vector,
)


def test_shuffle_identity_with_empty_word():
    """shuffle(e_I, e_empty) = e_I for any word (Definition 2.2)."""
    u = {(1, 2): 1.0}
    out = shuffle(u, {(): 1.0})
    assert out == {(1, 2): 1.0}


def test_shuffle_commutative():
    u = {(1,): 2.0}
    v = {(2,): 3.0}
    assert shuffle(u, v) == shuffle(v, u)


def test_shuffle_single_letters_gives_two_interleavings():
    """shuffle(e_1, e_2) = e_{12} + e_{21} (Definition 2.2)."""
    out = shuffle({(1,): 1.0}, {(2,): 1.0})
    assert out == {(1, 2): 1.0, (2, 1): 1.0}


def test_shuffle_square_doubles_length():
    """ell shuffle ell on a length-<=2 ell reaches length <=4 words only,
    matching the "shuffle square reaches level 4" remark (Section 4.1)."""
    ell = {(): 0.3, (1,): 0.1, (1, 1): 1.0}
    sq = shuffle(ell, ell)
    assert max(len(w) for w in sq) == 4


def test_word_to_vector_roundtrip_and_missing_word_raises():
    words = [(), (1,), (2,)]
    vec = word_to_vector({(1,): 2.0, (2,): -1.0}, words)
    assert np.allclose(vec, [0.0, 2.0, -1.0])
    with pytest.raises(KeyError):
        word_to_vector({(1, 1): 1.0}, words)


def _ou_algebra() -> PolynomialDiffusionAlgebra:
    """A single OU factor dX = kappa(theta - X) dt + eta dW (Section 4.1 table,
    factor 1 parameters), used as a minimal fixture for the dual operator."""
    kappa, theta, eta, x0 = 0.1, 0.1, 0.7, 1.0
    drift = {1: {(): kappa * theta - kappa * x0, (1,): -kappa}}
    covariance = {(1, 1): {(): eta ** 2}}
    return PolynomialDiffusionAlgebra(n_factors=1, drift=drift, covariance=covariance)


def test_close_words_under_L_is_finite_and_length_non_increasing():
    algebra = _ou_algebra()
    words = close_words_under_L({(1, 1)}, algebra)
    assert () in words
    # L never increases word length (Theorem 2.4).
    for w in words:
        for target in algebra.apply_L(w):
            assert len(target) <= len(w)


def test_G_is_block_triangular_by_word_length_for_mean_reverting_factor():
    """Theorem 2.4 only guarantees that L does not increase word length, i.e. G
    is block upper triangular when words are ordered by length and the row for
    the empty word is zero (L e_empty = 0). It does NOT make G nilpotent once
    the drift is state-dependent (mean reversion): e.g. L e_{11} for an OU
    factor picks up a term proportional to e_{11} itself (via e_1 shuffle
    (-kappa e_1) inside L's drift contribution), a diagonal self-loop with
    eigenvalue -2*kappa != 0. Nilpotency of G only holds in the driftless
    Brownian case of Corollary 3.20 (covered by test_signature_1d.py)."""
    algebra = _ou_algebra()
    words = close_words_under_L({(1, 1)}, algebra)
    G = build_G(words, algebra)
    by_length = sorted(range(len(words)), key=lambda i: len(words[i]))
    ordered = [words[i] for i in by_length]
    G_ordered = G[np.ix_(by_length, by_length)]
    for col, w in enumerate(ordered):
        for row, target in enumerate(ordered):
            if len(target) > len(w):
                assert G_ordered[row, col] == 0.0
    empty_idx = ordered.index(())
    assert np.allclose(G_ordered[:, empty_idx], 0.0)
    kappa = 0.1
    eleven_idx = ordered.index((1, 1))
    # The mean-reversion self-loop makes this diagonal entry nonzero, so G is
    # not nilpotent: G^k never vanishes identically for this OU factor.
    assert np.isclose(G_ordered[eleven_idx, eleven_idx], -2 * kappa)
    powers = nilpotent_powers(G, max_power=len(words) + 2)
    assert any(np.any(p) for p in powers)


def test_integral_matrix_exp_matches_closed_form_for_invertible_M():
    """A = integral_0^Delta exp(sM) ds = M^{-1}(exp(Delta M) - I) when M is
    invertible; check the block-matrix construction against this closed form."""
    rng = np.random.default_rng(0)
    M = rng.normal(size=(4, 4))
    # make M invertible (generic random matrix is a.s. invertible already)
    delta = 0.3
    from scipy.linalg import expm
    closed_form = np.linalg.solve(M, expm(delta * M) - np.eye(4))
    numeric = integral_matrix_exp(M, delta)
    assert np.allclose(closed_form, numeric, atol=1e-8)


def test_integral_matrix_exp_handles_singular_generator():
    """G built on an L-closed word basis is singular (the empty-word row is
    always zero since L e_empty = 0), even though it need not be nilpotent for
    a mean-reverting factor; A must still be computed (and be invertible) via
    the block-matrix identity of Lemma 3.17 / Theorem 3.14."""
    algebra = _ou_algebra()
    words = close_words_under_L({(1, 1)}, algebra)
    G = build_G(words, algebra)
    A = integral_matrix_exp(G, 30.0 / 365.0)
    assert np.isfinite(A).all()
    # A must be invertible (Lemma 3.17 / Theorem 3.14 precondition).
    np.linalg.inv(A)
