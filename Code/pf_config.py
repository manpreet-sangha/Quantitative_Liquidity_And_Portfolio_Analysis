"""Settings for Part 2 (portfolio analysis).

Reads the shared config.json for paths and the fixed model numbers the brief gives
us (information coefficient, risk aversion, look-back window). Kept separate from the
Part 1 config so nothing from the liquidity pipeline runs when we work on Part 2.
"""

from pathlib import Path
import json

BASE = Path(__file__).resolve().parent
with open(BASE / "config.json", encoding="utf-8") as f:
    _C = json.load(f)

_p2 = _C["part2"]

# --- Paths ---
DATA_XLSX = (BASE / _p2["data_xlsx"]).resolve()
OUTPUT_DIR = (BASE / _C["output_dir"] / "part2").resolve()
FIGURE_DIR = (OUTPUT_DIR / "figures").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# --- Fixed numbers from the brief ---
INFO_COEFF = _p2["information_coefficient"]   # how well the signal predicts returns
RISK_AVERSION = _p2["risk_aversion"]          # how strongly we penalise risk
WINDOW = _p2["window_months"]                 # look-back length for cov / vol (months)
N_PORTFOLIOS = _p2["n_portfolios"]            # number of momentum-sorted groups
MOM_LOOKBACK = _p2["momentum_lookback"]       # months in the momentum look-back
MOM_SKIP = _p2["momentum_skip"]               # most-recent months left out of the signal

# --- Conventions ---
MONTHS_PER_YEAR = 12          # for turning monthly numbers into yearly ones
WORLD = "World"               # column name for the average-across-countries return
RISK_FREE = 0.0               # assumed risk-free rate for Sharpe ratios
