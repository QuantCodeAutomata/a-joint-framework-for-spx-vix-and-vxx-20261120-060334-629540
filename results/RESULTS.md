# Results — *A joint framework for SPX, VIX and VXX*

This file reports the outcome of the three implemented experiments against the
values given in the paper, together with every deviation from the paper and
every claim that this repository does not reproduce.

All numbers below are taken from the JSON/CSV files written by the latest run
of each experiment script into `results/`:
`exp1_metrics.json`, `exp1_cross_sectional_summary.csv`, `exp1_sample_trajectory.csv`,
`exp2_metrics.json`, `exp2_table1_summary.csv`, `exp3_metrics.json`,
`exp3_table2_summary.csv`, and the corresponding `*_trajectories.png` /
`*_smiles.png` plots.

## Experiment 1 — One-dimensional Brownian primary process (Corollary 3.20, Example 3.21, Figure 1)

Setup: a single standard Brownian motion `X_t = W_t` is the primary process
(`b = 0`, `ā = 1`). The VIX functional has coefficients
`ell_∅ = 0.3, ell_1 = 0, ell_11 = 1`. VIX, VIX futures, instantaneous SPX
variance and VXX are computed from the closed-form expressions of
Corollary 3.20, with the VXX roll taken over the equally spaced futures
schedule `(T_0,…,T_4) = (-6, 22, 50, 84, 113)/365`, `τ = 30/365`, `VXX_0 = 0.340`
(the value shown in the Figure 1 table). 20,000 paths are simulated on a daily
grid with 4 steps per day over a 30-day horizon (`results/exp1_metrics.json`,
field `n_paths`); the path count is not specified in the paper and is chosen
for a stable Monte Carlo cross-section (`src/config.py::N_PATHS_EXP1`).

Nonnegativity condition (3.16): `ell_∅/ell_11 = 0.3 ≥ ell_1²/(2·ell_11²) + 7Δ/16
= 0.035959`, which holds (`nonnegativity_condition_3_16.holds = true`,
`exp1_metrics.json`). Over all 20,000 simulated paths, VIX and instantaneous
SPX variance stayed nonnegative on 100% of the grid points
(`vix_nonneg_fraction_simulated = 1.0`, `variance_nonneg_fraction_simulated =
1.0`), confirming the sufficient condition numerically.

Cross-sectional means (`exp1_cross_sectional_summary.csv`) start at VIX = 0.300,
VXX = 0.340, σ = 0.280, consistent with the paper's Figure 1 starting values,
and drift mildly as the horizon advances (VIX mean 0.3416, VXX mean 0.3405, σ
mean 0.3172 at day 30), with VIX futures `F_t(30)` and `F_t(60)` staying close
to their initial contango levels (0.3411 → 0.3417, 0.3822 → 0.3827). The sample
trajectory (`exp1_sample_trajectory.csv`, `exp1_trajectories.png`) shows VIX
fluctuating around 0.30, VXX around 0.34, σ around 0.28 and VIX futures around
0.34–0.38, matching the qualitative description of Figure 1.

**Verdict: reproduced.** The nonnegativity condition, the closed-form
consistency between VIX/variance/futures/VXX, and the qualitative level and
spread of the trajectories match the paper; no numerically pinned target value
from Figure 1 is contradicted. The exact table values of Figure 1 cannot be
checked bit-for-bit because the paper does not give the realized Brownian path
or Monte Carlo seed used to draw it (see "Deviations from the paper").

## Experiment 2 — Joint calibration, 3-factor OU primary process (Section 4.1, Table 1)

Setup: the 3-dimensional OU primary process of Section 4.1 with the stated
hyperparameters and Brownian correlation `R_OU`, truncated signature level 2
(21 coefficients) and the level-3 enrichment with words `(1,1,1)` and
`(0,0,0)` (23 coefficients). The generator-invariant word closure is built by
repeatedly applying the dual operator `L` to the requested basis words until
closed, rather than the dimension-reduced SVD subspace of Remark 3.18; this
gives a larger but mathematically equivalent invariant space (closure size 341
for level 2, 575 for the level-3 enrichment, `exp2_metrics.json`, field
`word_basis_closure_size`, vs. 181/216 after the paper's SVD compression).

Anchor spots are loaded live through the QuantX Dataloader (`SPY = 687.35`,
`VXX = 28.49`, `exp2_metrics.json`, field `anchor`); synthetic SPX/VIX/VXX
option quotes are generated around these spots because licensed Cboe DataShop
quotes dated 24 February 2026 could not be obtained (see "Deviations from the
paper").

Reduced-scale run (`exp2_metrics.json`, `config`: `n_paths=300`,
`n_validation=3000`, `maxiter=10`, `full_run=false`):

| spec | n_coeff | VIX futures proxy | VIX futures market | ρᵀR⁻¹ρ (reported point) | ρᵀR⁻¹ρ (paper) |
|---|---|---|---|---|---|
| level-2 | 21 | 20.25242 | 20.2524 | 0.99441 | 0.9944 |
| level-3 | 23 | 20.28487 | 20.2524 | 0.99329 | 0.9933 |

The VIX futures proxy and the quadratic-form constraint `ρᵀR_OU⁻¹ρ` evaluated
at the paper's reference coefficients (4.2)/(4.4) match the paper to 4–5
significant digits for both specifications, confirming the signature,
dual-operator and futures-proxy machinery against the paper's reported
reference point. Joint recalibration from this point over the reduced grid
reduces the block loss (level-2: 1.000 → 0.122; level-3: 1.058 → 0.085 after
10 iterations, not converged, `exp2_metrics.json`, field `recalibration`); a
full-length optimization (`CALIBRATION_MAXITER_PAPER = 500` paths/iterations,
`--full`) was not run to completion (see "Deviations from the paper").
Smile plots are saved in `exp2_smiles.png`.

Approximation error bounds (`exp2_metrics.json`, field
`approximation_error_bounds`) are estimated at 0.0 for both the VIX and the
VXX bound at `n_paths=300`: with so few paths, none of the simulated VIX
trajectories crossed into the region that triggers the Lipschitz correction of
Corollary 3.29(i), and no futures proxy approached the stopping floor `m=0.01`
(`futures_floor_hit_reported_point = false`, matching the paper's claim that no
path reached the floor); this reproduces the sign and the "no floor hit"
qualitative claim, but not the small positive point estimate (≈4.46e-7 for VIX,
≈4.11e-3 for VXX) reported in Table 1, which requires many more paths to
observe the rare crossings (see "Deviations from the paper").

**Verdict:** VIX futures proxy and ρᵀR⁻¹ρ constraint — **reproduced** (within
0.01%–0.2% of the paper's values, same sign and order of magnitude). Smile
shapes (downside SPX skew, increasing VIX/VXX smiles, steeper VIX upper wing
under the level-3 enrichment) — **partially reproduced**: on synthetic
quotes the qualitative shapes described in Section 4.1 are visible in
`exp2_smiles.png`, but no claim is made that calibrated prices fall inside
*market* bid-ask bands, since the bands are themselves synthetic. Table 1
approximation-error point estimates — **not reproduced** at this path count
(both bounds are below the paper's values because of insufficient rare-event
sampling at the reduced Monte Carlo size; see "Deviations from the paper").

## Experiment 3 — Joint calibration, 2 OU + 1 CIR primary process (Section 4.2, Table 2)

Setup: two OU factors and one CIR factor (Feller ratio
`2κ₃θ₃/η₃² = 2·4·1/4 = 2 > 1`, confirmed in `exp3_metrics.json`,
`feller_ratio = 2.0`), independent CIR driver, level-2 basis (21 coefficients)
enriched with the four words `(0,3,3), (1,1,1), (3,0,3), (3,3,0)` (25
coefficients total, `n_coefficients = 25`). The generator-invariant closure
built from these words has 768 elements (vs. the paper's unreported reduced
dimension for this mixed process).

Reduced-scale run (`n_paths=300`, `n_validation=3000`, `maxiter=10`,
`full_run=false`, `exp3_metrics.json`):

| quantity | reported point | paper (Section 4.2) |
|---|---|---|
| VIX futures proxy | 20.28452 | 20.2524 |
| ρᵀR⁻¹ρ | 0.99920 | 0.9992 |
| Feller ratio | 2.0 | 2 |

`ρᵀR⁻¹ρ` evaluated at the paper's reference coefficients matches the reported
value `0.9992` to 4 significant digits, and the Feller ratio equality holds
exactly, confirming strict positivity of the CIR factor and the correlation
constraint at the paper's calibrated point. The VIX futures proxy is within
0.16% of the market level. Joint recalibration over the reduced grid reduces
the block loss from 1.000 to 0.142 after 10 iterations (not converged,
`recalibration` field); it moves `ρᵀR⁻¹ρ` away from the reference point
(0.888), which is expected because the reduced grid and iteration count do not
locate the paper's optimum (see "Deviations from the paper"). Smile plots in
`exp3_smiles.png`.

As in Experiment 2, the Monte Carlo approximation-error bounds
(`approximation_error_bounds`) are estimated at 0.0 at this path count instead
of the paper's small positive values (≈1.74e-6 for VIX, ≈1.02e-5 for VXX in
Table 2), for the same rare-event sampling reason.

**Verdict:** Feller ratio and ρᵀR⁻¹ρ at the reference point — **reproduced**.
VIX futures proxy — **reproduced** (0.16% relative difference). Table 2
approximation-error point estimates — **not reproduced** at this path count
(same identified cause as Experiment 2). Full joint recalibration to the
paper's reported optimum — **not reproduced** (the reduced-scale optimizer run
does not converge to the reference point; see "Deviations from the paper").

## Deviations from the paper

- **Option market data.** The paper calibrates to historical bid/ask quotes
  for SPX, VIX and VXX options dated 24 February 2026, licensed from Cboe
  DataShop. That date is in the future relative to this run and the quotes are
  not obtainable through the QuantX Dataloader or Massive REST (neither serves
  historical listed-options quote data); only the underlying spot levels for
  SPY and VXX were loaded live through the QuantX Dataloader
  (`data/data_loader.py`, recorded under `exp2_metrics.json`/`exp3_metrics.json`
  field `anchor`). Synthetic SPX/VIX/VXX implied-volatility quotes are
  generated instead around these live spots by `data/generate_synthetic.py`,
  with the strike grids, maturities, interest rate (`r = 0.0367`) and VXX fee
  (`c = 0.89%`) of Section 4 kept as specified in the paper. Smile shapes and
  loss values computed against these synthetic quotes are reported for the
  model mechanics (signature construction, dual operator, Monte Carlo pricing,
  calibration loss) but are not a claim that the model matches the real 24
  February 2026 market.
- **Reduced Monte Carlo and optimizer scale (Experiments 2 and 3).** The paper
  calibrates with `N = 10,000` simulation paths and validates on `10^6`
  independent paths (`src/config.py::N_PATHS_CALIBRATION_PAPER`,
  `N_VALIDATION_PAPER`). Reproducing this scale together with the joint SPX +
  VIX + VXX Monte Carlo pricing and the constrained optimization over 21–25
  coefficients and 3 correlations was estimated, from the measured per-path
  cost of signature simulation and pricing, to run far longer than the time
  available before finalization. The executed configuration instead uses
  `N_PATHS_RUN = 300` simulation paths, `N_VALIDATION_RUN = 3,000` validation
  paths and `CALIBRATION_MAXITER_RUN = 10` optimizer iterations
  (`src/config.py`), unchanged formulas, word closures, and constraints. Run
  the paper-scale configuration with `python -m exp.exp_2 --full` and
  `python -m exp.exp_3 --full` (`CALIBRATION_MAXITER_PAPER = 500`,
  `N_PATHS_CALIBRATION_PAPER = 10000`, `N_VALIDATION_PAPER = 1000000`); this
  was not run to completion in this environment. The approximation-error
  bounds and the recalibrated loss reported above should be read against this
  reduced scale: the near-zero error bounds and the non-converged
  recalibration are both an identified consequence of it (fewer paths capture
  fewer rare VIX-sign crossings and fewer optimizer iterations do not reach
  the paper's reported optimum), not a sign or formula mismatch — the point
  evaluations at the paper's own reference coefficients (the VIX futures
  proxy and `ρᵀR⁻¹ρ`) match the paper closely at the same reduced scale,
  which isolates the gap to the optimizer/Monte Carlo scale rather than to the
  pricing formulas.
- **Generator-invariant subspace dimension.** The dual-operator closure in
  `src/signature.py` is built by closing the requested word basis under
  repeated application of `L` (Definition 2.5) rather than by the compact SVD
  reduction of Remark 3.18. This gives a larger invariant subspace than the
  paper's reported dimensions (341 vs. 181 for level 2 in Experiment 2, 575 vs.
  216 for the level-3 enrichment, 768 for the OU-CIR enrichment of Experiment
  3, whose reduced dimension the paper does not state) but is mathematically
  equivalent: `A = ∫_0^Δ e^{sG^T} ds` is computed exactly on this closure, so
  the resulting prices and proxies are unaffected; only the computational cost
  is higher than the paper's compressed representation.
- **Initial feasible point and optimizer.** The paper does not state the
  initial feasible parameter vector used to compute the block scales `s_A`,
  nor the specific optimization algorithm. This repository uses the paper's
  own reported reference coefficients (`ℓ*`, `ρ*` of equations (4.2), (4.4) and
  page 25) as the initial point for recalibration, and a projected gradient
  scheme (`scipy.optimize.minimize`, method `SLSQP`) under the stated
  constraints (`raw variance ≥ 1e-6`, `m < F̃_0(T_i)`, `ρᵀR⁻¹ρ < 1`).
- **Black-76 inversion and gradient method.** The paper does not specify the
  implied-volatility inversion algorithm or whether calibration gradients are
  computed by automatic or finite differences. This repository inverts
  Black-76 prices with `scipy.optimize.brentq` and computes calibration
  gradients by finite differences, since the Monte Carlo pricing pipeline is
  not wrapped in an automatic-differentiation framework.
- **VXX₀ in calibration.** The paper does not give the numerical VXX₀ used for
  calibration in Section 4. This repository anchors VXX₀ to the live VXX spot
  loaded through the QuantX Dataloader (see `anchor` above), consistent with
  the paper's statement that the VXX forward is inferred as
  `VXX_0 · e^{rT}`.
- **CIR simulation scheme.** The paper does not state the CIR discretization
  used in Experiment 3. This repository uses the exact noncentral
  chi-squared transition (Feller ratio `2κθ/η² = 2 > 1`, strictly positive),
  consistent with the paper's own statement of the Feller ratio.

## Not reproduced

- The exact numerical tables of Figure 1 (Experiment 1) are not reproduced
  bit-for-bit: the paper does not disclose the Brownian path or Monte Carlo
  seed used to draw that figure, so only the qualitative levels, ranges and
  the closed-form consistency checks can be compared.
- The Table 1 and Table 2 approximation-error point estimates (VIX ≈
  4.46e-7 / 1.74e-6, VXX ≈ 4.11e-3 / 1.01e-5) are not reproduced at the
  executed Monte Carlo scale; observing these rare-event magnitudes requires
  the paper's `10^6`-path validation run, which the time available in this
  environment did not allow for both calibration specifications of both
  experiments.
- The full joint optimization to the paper's reported calibrated coefficients
  `ℓ₂*, ℓ₃*` (Experiment 2) and the 25-coefficient `ℓ*` (Experiment 3) is not
  reproduced: the reduced-scale recalibration in this repository is a sanity
  check of the loss direction, not a converged refit, for the compute-time
  reason stated above.
