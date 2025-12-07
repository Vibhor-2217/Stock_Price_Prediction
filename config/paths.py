# config/paths.py

from pathlib import Path

# Project root = one level above this file's directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

def get_paths(symbol: str):
    """
    Central place for all paths related to a given ticker.
    Call this from anywhere instead of hard-coding AAPL or C:... paths.
    """
    symbol = symbol.upper()

    return {
        "raw_csv": RAW_DIR / f"{symbol}.csv",
        "processed_csv": PROCESSED_DIR / f"{symbol}_processed.csv",
        "regime_csv": PROCESSED_DIR / f"{symbol}_regime.csv",

        "twohead_ckpt": MODELS_DIR / f"{symbol}_twohead_best.pt",
        "regime_ckpt": MODELS_DIR / f"{symbol}_regime_encoder.pt",
        "grpo_ckpt": MODELS_DIR / f"{symbol}_grpo_block_regime.pt",

        "results_dir": RESULTS_DIR,
    }
