"""Module-level configuration for all experiments.

Every experiment parameter lives here as an UPPER_CASE module-level variable
with a comment stating its meaning, unit, and source (paper section/equation,
or "not specified in the paper"). No classes or dicts are used, per the
repository's parameter convention.
"""

# ---------------------------------------------------------------------------
# Experiment 1: one-dimensional Brownian primary process (Corollary 3.20)
# ---------------------------------------------------------------------------
EXP1_ELL_EMPTY = 0.3          # ell_empty VIX functional coefficient (Example 3.21 / Figure 1)
EXP1_ELL_1 = 0.0              # ell_1 VIX functional coefficient (Example 3.21)
EXP1_ELL_11 = 1.0             # ell_11 VIX functional coefficient (Example 3.21)
EXP1_HORIZON_DAYS = 30        # simulation horizon in days (Figure 1 table spans 0-30 days)
EXP1_STEPS_PER_DAY = 4        # Euler grid steps per day (implementation_notes: n_day=4)
EXP1_N_PATHS = 20_000         # Monte Carlo paths for sample trajectories (not specified in the paper)
EXP1_SEED = 20260220          # random seed for reproducibility (not specified in the paper)
EXP1_VXX0 = 0.340             # VXX_0 initial value, inferred from the Figure 1 table (Section 3, Example 3.21)
# VIX futures roll maturities (T0..T4) in years, Delta=30/365 (Corollary 3.20 / Section 3)
EXP1_FUTURES_MATURITIES_DAYS = (-6, 22, 50, 84, 113)   # days, as stated in data_requirements
EXP1_DELTA_DAYS = 30                                   # Delta, the VIX-window length in days (Definition 3.1)
EXP1_TAU_DAYS = EXP1_FUTURES_MATURITIES_DAYS[2] - EXP1_FUTURES_MATURITIES_DAYS[1]  # tau = T2-T1 = 28 days, equal spacing assumption (Corollary 3.20)
EXP1_REPORT_TIMES_DAYS = (0, 5, 10, 15, 20, 25, 30)    # sampling times for the Figure-1-style table (Figure 1)

# ---------------------------------------------------------------------------
# Experiment 2: 3-factor Ornstein-Uhlenbeck primary process (Section 4.1)
# ---------------------------------------------------------------------------
# OU factor hyperparameters (kappa: mean-reversion speed 1/yr, theta: long-run mean,
# eta: volatility, X0: initial state), Section 4.1 table.
EXP2_OU_KAPPA = (0.1, 25.0, 10.0)
EXP2_OU_THETA = (0.1, 4.0, 0.08)
EXP2_OU_ETA = (0.7, 10.0, 5.0)
EXP2_OU_X0 = (1.0, 0.08, 2.0)
# Brownian correlation matrix R_OU of the three OU drivers (Section 4.1)
EXP2_R_OU = (
    (1.0, 0.213, -0.576),
    (0.213, 1.0, -0.044),
    (-0.576, -0.044, 1.0),
)
EXP2_LEVERAGE_L = 100.0        # leverage-saturation cap L in equation (3.11)
EXP2_DELTA_DAYS = 30           # Delta, VIX window (Definition 3.1)
# VIX futures roll maturities in days (data_requirements / Section 4)
EXP2_FUTURES_MATURITIES_DAYS = (-6, 22, 50, 84, 113)
EXP2_VXX_FEE = 0.0089          # VXX annualized fee c (data_requirements)
EXP2_INTEREST_RATE = 0.0367    # risk-free rate r (data_requirements)
# VXX_0: not restated numerically in Section 4 ("VXX from VXX_0*e^{rT}"); taken
# as the VXX close on the 24 Feb 2026 quote date (QuantX Dataloader, see
# data/data_loader.py::load_underlying_spots). This is consistent with the
# strike/moneyness table: 25/0.8755 = 28.56 and 37/1.2958 = 28.55 both imply a
# VXX forward of about 28.55, matching 28.49*e^{0.0367*24/365} = 28.56.
EXP2_VXX0 = 28.49
# SPX and VIX forwards implied by the strike/moneyness endpoints of the
# quote table (data_requirements): 5610/0.8128 = 7575/1.0975 = 6902.1 (SPX),
# 16.5/0.8147 = 20.2528 and 28/1.3826 = 20.2524 (VIX futures quote, matching
# the paper's reported value of 20.2524 in Section 4.1's expected outcomes).
EXP2_SPX_FORWARD = 6902.1
EXP2_VIX_FUTURES_MARKET = 20.2524
# Monte Carlo and calibration settings (Section 4.1, "Calibration settings")
EXP2_N_PATHS_PAPER = 10_000    # N, Monte Carlo paths used by the paper
EXP2_N_PATHS_RUN = 2_000       # paths actually used in this run (reduced for runtime/memory; see results/RESULTS.md)
EXP2_N_VALIDATION_PAPER = 1_000_000  # independent validation paths used by the paper
EXP2_N_VALIDATION_RUN = 20_000       # validation paths actually used in this run
EXP2_STEPS_PER_DAY = 4          # n_day, Euler steps per day (Section 4.1)
EXP2_LOSS_LAMBDA = (0.35, 0.5, 0.15)   # (lambda_SPX, lambda_VIX, lambda_VXX) block weights (Section 4.1)
EXP2_LOSS_BETA = (0.98, 0.1, 0.98)     # (beta_SPX, beta_VIX, beta_VXX) loss-shape parameters (Section 4.1)
EXP2_VARIANCE_FLOOR = 1e-6     # epsilon_var, raw-variance constraint floor (Appendix C)
EXP2_FUTURES_FLOOR_M = 0.01    # m, VIX futures stopping floor (Appendix C)
EXP2_SPX_MATURITY_DAYS = 24    # SPX option maturity (data_requirements)
EXP2_VIX_MATURITY_DAYS = 22    # VIX option maturity (data_requirements)
EXP2_VXX_MATURITY_DAYS = 24    # VXX option maturity (data_requirements)
EXP2_SPX_MONEYNESS = (0.8128, 1.0975)  # SPX strike moneyness range (data_requirements)
EXP2_VIX_MONEYNESS = (0.8147, 1.3826)  # VIX strike moneyness range (data_requirements)
EXP2_VXX_MONEYNESS = (0.8755, 1.2958)  # VXX strike moneyness range (data_requirements)
EXP2_N_STRIKES_SPX = 29        # number of SPX strikes (data_requirements)
EXP2_N_STRIKES_VIX = 13        # number of VIX strikes (data_requirements)
EXP2_N_STRIKES_VXX = 13        # number of VXX strikes (data_requirements)
EXP2_ERROR_BOUND_ISTAR = 2     # i* truncation index used in the Monte Carlo error-bound estimate (methodology, step 13)

# ---------------------------------------------------------------------------
# Experiment 3: 2 OU factors + 1 CIR factor primary process (Section 4.2)
# ---------------------------------------------------------------------------
EXP3_OU_KAPPA = (0.1, 25.0)     # kappa for the slow/fast OU factors (Section 4.2 table)
EXP3_OU_THETA = (0.1, 4.0)
EXP3_OU_ETA = (0.7, 10.0)
EXP3_OU_X0 = (1.0, 0.08)
EXP3_CIR_KAPPA = 4.0            # CIR mean-reversion speed (Section 4.2 table)
EXP3_CIR_THETA = 1.0            # CIR long-run mean
EXP3_CIR_ETA = 2.0              # CIR volatility-of-volatility
EXP3_CIR_Y0 = 1.0               # CIR initial state
EXP3_FELLER_RATIO = 2 * EXP3_CIR_KAPPA * EXP3_CIR_THETA / EXP3_CIR_ETA ** 2  # 2*kappa*theta/eta^2, must exceed 1 for strict positivity (Section 4.2)
# Correlation matrix R: OU1-OU2 correlated, CIR independent of both (Section 4.2)
EXP3_R = (
    (1.0, 0.213, 0.0),
    (0.213, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
EXP3_LEVERAGE_L = 100.0
EXP3_DELTA_DAYS = 30
EXP3_FUTURES_MATURITIES_DAYS = (-6, 22, 50, 84, 113)
EXP3_VXX_FEE = 0.0089
EXP3_INTEREST_RATE = 0.0367
EXP3_VXX0 = 28.49                 # same quote-date VXX close as exp_2 (Section 4.2 reuses Section 4's market data)
EXP3_SPX_FORWARD = 6902.1          # same SPX forward as exp_2 (Section 4.2 reuses Section 4's market data)
EXP3_VIX_FUTURES_MARKET = 20.2524  # same VIX futures quote as exp_2
EXP3_N_PATHS_PAPER = 10_000
EXP3_N_PATHS_RUN = 2_000
EXP3_N_VALIDATION_PAPER = 1_000_000
EXP3_N_VALIDATION_RUN = 20_000
EXP3_STEPS_PER_DAY = 4
EXP3_LOSS_LAMBDA = (0.25, 0.45, 0.3)    # (lambda_SPX, lambda_VIX, lambda_VXX) (Section 4.2)
EXP3_LOSS_BETA = (0.98, 0.1, 0.5)       # (beta_SPX, beta_VIX, beta_VXX) (Section 4.2)
EXP3_VARIANCE_FLOOR = 1e-6
EXP3_FUTURES_FLOOR_M = 0.01
EXP3_SPX_MATURITY_DAYS = 24
EXP3_VIX_MATURITY_DAYS = 22
EXP3_VXX_MATURITY_DAYS = 24
EXP3_SPX_MONEYNESS = (0.8128, 1.0975)
EXP3_VIX_MONEYNESS = (0.8147, 1.3826)
EXP3_VXX_MONEYNESS = (0.8755, 1.2958)
EXP3_N_STRIKES_SPX = 29
EXP3_N_STRIKES_VIX = 13
EXP3_N_STRIKES_VXX = 13
EXP3_ERROR_BOUND_ISTAR = 2

# ---------------------------------------------------------------------------
# Shared calibration/optimization settings (Appendix C)
# ---------------------------------------------------------------------------
CALIBRATION_MAXITER_PAPER = 500    # optimizer iterations for a full calibration (not specified in the paper; conservative default)
CALIBRATION_MAXITER_RUN = 80       # iterations actually used in this run (reduced for runtime; see results/RESULTS.md)
RANDOM_SEED = 20260220             # shared Monte Carlo seed, not specified in the paper
