# AGENTS.md

Guide for coding agents and contributors working in this repository.

## What this repository implements

A replication of three experiments from *A joint framework for SPX, VIX and
VXX*: the VIX-based signature model built on a time-extended signature of a
primary Markov process, with closed-form VIX/VXX/variance expressions in the
one-dimensional Brownian case (Corollary 3.20) and Monte Carlo calibration to
joint SPX/VIX/VXX option quotes for 3-factor OU and 2-factor-OU-plus-CIR
primary processes (Section 4). See `results/RESULTS.md` for the comparison
against the paper's reported values, including every deviation and every claim
not reproduced.

## Layout

- `src/` — flat library modules: `config.py` (all experiment parameters, no
  classes), `signature.py` (shuffle product, dual operator, generator-invariant
  word closure), `primary_process.py` (Brownian/OU/CIR simulation),
  `vix_model_1d.py` (closed-form Experiment 1 model), `model.py` and
  `pricing.py` (Monte Carlo pricing of SPX/VIX/VXX options and VIX futures),
  `calibration.py` / `calibrate.py` (joint calibration loss and optimizer).
- `exp/` — experiment entry points: `exp_1.py`, `exp_2.py`, `exp_3.py`. Each
  runs as `python -m exp.<name>` from the repository root.
- `data/` — `data_loader.py` (QuantX Dataloader OHLCV/spot loads) and
  `generate_synthetic.py` (synthetic SPX/VIX/VXX option surfaces used in place
  of the unavailable licensed Cboe DataShop quotes; see "Deviations from the
  paper" in `results/RESULTS.md`). No other data code lives outside `data/`.
- `tests/` — pytest unit tests for `src/` (synthetic/closed-form fixtures) and
  market-data tests marked `market_data` (skipped without
  `DATALOADER_API_KEY`).
- `results/` — every generated table, plot and metrics file, plus
  `RESULTS.md`. Rerunning an experiment overwrites its files here; update the
  numbers and verdicts in `RESULTS.md` by hand after checking them against the
  new files.

## Commands

```bash
pip install -r requirements.txt
cp .env.example .env            # only needed to rerun data/data_loader.py

python -m pytest -m "not market_data"   # unit tests (default)
python -m pytest                         # full suite, needs DATALOADER_API_KEY

python -m exp.exp_1              # closed-form 1D model (Corollary 3.20)
python -m exp.exp_2               # reduced-scale OU calibration
python -m exp.exp_2 --full        # paper-scale OU calibration (N=10,000 paths,
                                   # 10^6 validation paths; long-running)
python -m exp.exp_3               # reduced-scale OU+CIR calibration
python -m exp.exp_3 --full        # paper-scale OU+CIR calibration
```

`reproduce.sh` runs the test suite followed by the three experiments in their
default (reduced-scale) configuration.

## Conventions

- All experiment parameters live in `src/config.py` as module-level
  `UPPER_CASE` variables, grouped by experiment, each with a one-line comment
  giving its meaning, unit and source (a paper section/equation, or "not
  specified in the paper"). No parameter classes or dicts.
- Market data is loaded only through `data/data_loader.py`
  (QuantX Dataloader, `https://quantx-api.limex.pro/apis/dataloader`); no
  module outside `data/` calls a market-data API directly.
- `DATALOADER_API_KEY` and `MASSIVE_API_KEY` are read from the environment
  (`.env`, never committed) and used only for Dataloader/Massive requests.
- Unit tests use small fixtures or synthetic/closed-form inputs; market-data
  tests use a session-scoped cached slice loaded through
  `data/data_loader.py` and are skipped when no API key is set. Run
  `python -m pytest -m "not market_data"` for routine checks.
- Deviations from the paper (data source, reduced Monte Carlo/optimizer scale,
  generator-invariant subspace construction, unspecified details) are recorded
  only in `results/RESULTS.md`, not in code comments or config files.
