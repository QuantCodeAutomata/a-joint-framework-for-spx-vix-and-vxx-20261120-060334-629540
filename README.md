# A joint framework for SPX, VIX and VXX

Replication of selected experiments from *A joint framework for SPX, VIX and VXX*
(working paper). This repository is a work in progress; see `results/RESULTS.md`
for the current state of each experiment.

## Layout

- `src/` — signature algebra, primary-process simulation, pricing, calibration, configuration.
- `exp/` — experiment entry points (`python -m exp.exp_1`, `exp.exp_2`, `exp.exp_3`).
- `data/` — `data_loader.py` (QuantX Dataloader market data) and `generate_synthetic.py`
  (synthetic SPX/VIX/VXX option surfaces, used because licensed option quotes could
  not be obtained in this environment; see `results/RESULTS.md`).
- `tests/` — pytest unit and market-data tests.
- `results/` — generated tables, plots and `RESULTS.md`.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in API keys only if re-running data/data_loader.py
```

## Running

```bash
python -m pytest -m "not market_data"
python -m exp.exp_1
python -m exp.exp_2
python -m exp.exp_3
```

See `results/RESULTS.md` for the full run log and paper comparison.
