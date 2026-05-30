"""Central configuration for the SMM921 Part 1 code.

Single source of truth: reads config.json and resolves all paths relative to this
Code/ folder, so no paths or parameters are hard-coded in the individual modules.

Stock selection is mode-switchable (set "selection_mode" in config.json):
  - "fixed"  -> use the explicit list part1.stocks (default: ULVR/EXPN/KGF)
  - "random" -> draw one stock per market-cap bucket from the stock universe
                (logic lives in select_stocks.py), reproducible via random_seed.
Either way, the rest of the pipeline simply reads PART1_STOCKS / STOCK_NAMES.
"""

from pathlib import Path
import json

BASE = Path(__file__).resolve().parent
with open(BASE / "config.json", encoding="utf-8") as f:
    CONFIG = json.load(f)

# --- Raw-data column names ---
COLS = CONFIG["columns"]

# --- Filters / parameters ---
FILTERS = CONFIG["filters"]
PRICE_DIVISOR = CONFIG["data"]["price_to_gbp_divisor"]

# --- Resolved paths ---
RAW_CSV = (BASE / CONFIG["data"]["raw_csv"]).resolve()
OUTPUT_DIR = (BASE / CONFIG["output_dir"]).resolve()
FIGURE_DIR = (BASE / CONFIG["figure_dir"]).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# --- Part 1 stock selection ---
_p1 = CONFIG["part1"]
SELECTION_MODE = _p1.get("selection_mode", "fixed")
RANDOM_SEED = _p1.get("random_seed", 42)
BUCKETS = _p1.get("buckets", ["Large", "Mid", "Small"])
UNIVERSE_CSV = (BASE / _p1.get("universe_csv", "output/stock_universe.csv")).resolve()

if SELECTION_MODE == "random" and UNIVERSE_CSV.exists():
    import select_stocks
    PART1_STOCKS, STOCK_NAMES = select_stocks.select_one_per_bucket(
        UNIVERSE_CSV, BUCKETS, RANDOM_SEED)
else:
    if SELECTION_MODE == "random":
        print(f"[config] selection_mode='random' but {UNIVERSE_CSV.name} not found "
              f"- falling back to fixed stocks. Run 01_identify_stocks.py first.")
    PART1_STOCKS = _p1["stocks"]
    STOCK_NAMES = _p1["stock_names"]


def session_bounds_min():
    """Trading-session window as (start, end) in minutes-of-day."""
    def to_min(hhmm):
        h, m = hhmm.split(":")
        return int(h) * 60 + int(m)
    return to_min(FILTERS["session_start"]), to_min(FILTERS["session_end"])
