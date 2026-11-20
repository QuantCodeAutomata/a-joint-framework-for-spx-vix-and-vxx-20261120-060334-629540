"""Tensor-algebra machinery for the VIX-based signature model.

Implements, from first principles, the objects of Section 2.2 of "A joint
framework for SPX, VIX and VXX":

* words over the alphabet {0, 1, ..., d} (0 is the time letter);
* the shuffle product (Definition 2.2);
* the dual operator L of equation (2.5), for a polynomial diffusion with
  affine drift b and affine-or-constant covariance a;
* its matrix representative G (equation preceding Theorem 2.4) on any word
  set that is closed under L;
* the matrix A = integral_0^Delta exp(s G^T) ds needed in Theorem 3.14.

No library in this environment implements the signature dual-operator
construction for general polynomial diffusions, so this module is a direct,
custom translation of equations (2.2), (2.5) and (2.6); only elementary
linear algebra (numpy, scipy.linalg.expm) is reused.

A word-generator-invariant subspace (the span of a finite set of words closed
under repeated application of L) is built here by breadth-first search from a
seed set, rather than by the paper's SVD-based compression of the complete
level-<=2n basis (Remark 3.18). Both constructions give a subspace on which
G is a faithful finite-dimensional representative of L and on which Theorem
2.4 / Theorem 3.14 hold exactly; the BFS version need not have the same
dimension as the paper's reduced basis.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.linalg import expm

Word = tuple[int, ...]
SparseVec = dict[Word, float]


@lru_cache(maxsize=None)
def _word_shuffle(w1: Word, w2: Word) -> tuple[tuple[Word, float], ...]:
    """Shuffle product of two words (Definition 2.2), returned as a tuple of
    (word, coefficient) pairs. Recursive definition on the last letter:
    shuffle(u a, v b) = shuffle(u, v b) . a + shuffle(u a, v) . b.
    """
    if not w1:
        return ((w2, 1.0),)
    if not w2:
        return ((w1, 1.0),)
    out: dict[Word, float] = {}
    for word, coeff in _word_shuffle(w1[:-1], w2):
        key = word + (w1[-1],)
        out[key] = out.get(key, 0.0) + coeff
    for word, coeff in _word_shuffle(w1, w2[:-1]):
        key = word + (w2[-1],)
        out[key] = out.get(key, 0.0) + coeff
    return tuple(out.items())


def shuffle(u: SparseVec, v: SparseVec) -> SparseVec:
    """Shuffle product of two sparse linear combinations of words."""
    out: SparseVec = {}
    for w1, c1 in u.items():
        for w2, c2 in v.items():
            for word, coeff in _word_shuffle(w1, w2):
                out[word] = out.get(word, 0.0) + c1 * c2 * coeff
    return out


def word_to_vector(vec: SparseVec, words: list[Word]) -> np.ndarray:
    """Project a sparse linear combination onto an explicit, ordered word
    basis, dropping words not in the basis (which must be zero there, i.e.
    the basis must contain the support of `vec`)."""
    index = {w: i for i, w in enumerate(words)}
    out = np.zeros(len(words))
    for word, coeff in vec.items():
        if word not in index:
            raise KeyError(f"word {word} outside the provided basis")
        out[index[word]] = coeff
    return out


@dataclass
class PolynomialDiffusionAlgebra:
    """Drift b_j and covariance a_ij of an affine (polynomial-diffusion)
    primary process, represented as sparse elements of T^{<=1}(R^d), so that
    the dual operator L of equation (2.5) can be evaluated on any word."""

    n_factors: int
    drift: dict[int, SparseVec]  # b_j, keyed by factor letter j in 1..n_factors
    covariance: dict[tuple[int, int], SparseVec]  # a_ij, keyed by sorted (i, j)

    def apply_L(self, word: Word) -> SparseVec:
        """L e_I = e_{I'} shuffle b_{i_k} + 1/2 e_{I''} shuffle a_{i_{k-1} i_k}
        (equation 2.5). Words ending in the time letter 0 only carry the
        deterministic d/dt contribution handled by the Euler recursion in
        src.primary_process (time has zero drift/covariance row); this
        method is used to explore reachable words and to build G on
        factor-letter words.
        """
        if len(word) == 0:
            return {}
        last = word[-1]
        out: SparseVec = {}
        if last != 0:
            b = self.drift.get(last, {})
            for w, c in shuffle({word[:-1]: 1.0}, b).items():
                out[w] = out.get(w, 0.0) + c
        else:
            out[word[:-1]] = out.get(word[:-1], 0.0) + 1.0
        if len(word) >= 2:
            second_last = word[-2]
            if second_last != 0 and last != 0:
                key = (second_last, last) if second_last <= last else (last, second_last)
                a = self.covariance.get(key, {})
                for w, c in shuffle({word[:-2]: 1.0}, a).items():
                    out[w] = out.get(w, 0.0) + 0.5 * c
        return out


def close_words_under_L(seed_words: set[Word], algebra: PolynomialDiffusionAlgebra) -> list[Word]:
    """Breadth-first closure of `seed_words` under repeated application of L:
    the smallest set of words W such that L(span(W)) subseteq span(W).
    L never increases word length (Theorem 2.4), so the search always
    terminates on a finite alphabet."""
    seen = set(seed_words)
    seen.add(())
    queue: deque[Word] = deque(seen)
    while queue:
        word = queue.popleft()
        for target in algebra.apply_L(word):
            if target not in seen:
                seen.add(target)
                queue.append(target)
    return sorted(seen, key=lambda w: (len(w), w))


def build_G(words: list[Word], algebra: PolynomialDiffusionAlgebra) -> np.ndarray:
    """Matrix representative G of L on the word basis `words` (must be
    closed under L): G[L(I), L(J)] = eta_{JI} where L e_I = sum_J eta_{IJ} e_J.
    """
    index = {w: i for i, w in enumerate(words)}
    dim = len(words)
    G = np.zeros((dim, dim))
    for word in words:
        i = index[word]
        for target, coeff in algebra.apply_L(word).items():
            j = index[target]
            G[j, i] = coeff
    return G


def nilpotent_powers(G: np.ndarray, max_power: int | None = None) -> list[np.ndarray]:
    """Powers I, G, G^2, ... of a *strictly nilpotent* dual-operator matrix G.

    Theorem 2.4 states only that L never *increases* word length; on a word
    basis with a letter whose drift is affine and self-referencing (e.g. a
    mean-reverting Ornstein-Uhlenbeck or CIR factor, Section 4), L e_I can
    have a nonzero component along e_I itself (the "diagonal" mean-reversion
    term), so G need not be nilpotent in general. G is nilpotent only in
    special cases such as the driftless one-dimensional Brownian motion of
    Corollary 3.20 (`src/vix_model_1d.py`), which however has its own
    closed-form expressions and does not call this function. Callers that
    need exp(t G) for a general polynomial-diffusion primary process must use
    `spectral_propagator` below instead of this truncated power series.
    """
    n = G.shape[0]
    powers = [np.eye(n)]
    current = np.eye(n)
    limit = max_power if max_power is not None else n
    for _ in range(limit):
        current = current @ G
        if not np.any(current):
            break
        powers.append(current)
    return powers


@dataclass
class SpectralPropagator:
    """Cached eigendecomposition G = V diag(eigvals) V^{-1}, used to evaluate
    exp(t G) @ v for many different (t, v) at O(dim) cost per evaluation once
    the O(dim^2) change of basis `coeffs = V^{-1} v` has been done once per
    vector (Theorem 2.4's semigroup property <ell, P_{T-t} X_hat_t> =
    <e^{(T-t)G} ell, X_hat_t>, Theorem 3.14). Falls back to direct
    `scipy.linalg.expm` when G is defective (not diagonalizable) or
    ill-conditioned, since the paper does not restrict G to be diagonalizable
    in general (e.g. the CIR dual operator of Section 4.2)."""

    G: np.ndarray
    eigvals: np.ndarray
    eigvecs: np.ndarray
    eigvecs_inv: np.ndarray
    well_conditioned: bool

    def coeffs(self, vec: np.ndarray) -> np.ndarray:
        """Change `vec` to the eigenbasis once; reused across every t."""
        return self.eigvecs_inv @ vec.astype(complex)

    def propagate(self, coeffs: np.ndarray, t: float) -> np.ndarray:
        """exp(t G) @ vec from its pre-computed eigen-coefficients."""
        if not self.well_conditioned:
            return (expm(t * self.G) @ coeffs.real)
        value = self.eigvecs @ (coeffs * np.exp(t * self.eigvals))
        imag = np.abs(value.imag)
        scale = np.maximum(np.abs(value.real), 1e-12)
        assert np.all(imag <= 1e-6 * scale + 1e-8), (
            "exp(t G) @ vec must be real since G and vec are real; a large "
            "imaginary residual indicates a defective/ill-conditioned eigenbasis"
        )
        return value.real


def spectral_propagator(G: np.ndarray, cond_threshold: float = 1e10) -> SpectralPropagator:
    """Build the cached eigendecomposition of G used by `SpectralPropagator`.

    Falls back to treating G as ill-conditioned (every `propagate` call then
    uses `scipy.linalg.expm` directly) when the eigenvector matrix is close
    to singular, which signals that G is defective on this word basis.
    """
    if G.shape[0] == 0:
        return SpectralPropagator(G, np.zeros(0, dtype=complex), np.zeros((0, 0), dtype=complex),
                                   np.zeros((0, 0), dtype=complex), True)
    eigvals, eigvecs = np.linalg.eig(G)
    cond = np.linalg.cond(eigvecs)
    well_conditioned = np.isfinite(cond) and cond < cond_threshold
    eigvecs_inv = np.linalg.inv(eigvecs) if well_conditioned else np.zeros_like(eigvecs)
    return SpectralPropagator(G, eigvals, eigvecs, eigvecs_inv, well_conditioned)


def integral_matrix_exp(M: np.ndarray, delta: float) -> np.ndarray:
    """A = integral_0^delta exp(s M) ds, via the block-matrix identity
    exp([[M, I], [0, 0]] * delta) = [[exp(delta M), A], [0, I]], which
    remains valid when M is singular (unlike the closed form M^{-1}(e^{dM}-I))."""
    n = M.shape[0]
    block = np.zeros((2 * n, 2 * n))
    block[:n, :n] = M
    block[:n, n:] = np.eye(n)
    exp_block = expm(delta * block)
    return exp_block[:n, n:]
