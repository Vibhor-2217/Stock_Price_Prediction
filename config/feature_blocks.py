# config/feature_blocks.py

from typing import Dict, List


# --- YOU MUST CUSTOMIZE THIS FOR YOUR COLUMNS --- #
#
# Put the actual column names from AAPL_regime.csv here.
# Only include FEATURE columns (not 'date', 'direction', 'return', etc.).
#
# Example mapping – adjust to match your processed CSV exactly:
FEATURE_BLOCKS: Dict[str, List[str]] = {
    "price": [
        "open",
        "high",
        "low",
        "close",
        "adj close",
        "log_return",
    ],
    "volume": [
        "volume",
        "obv",
        "volume_sma_20",
    ],
    "volatility": [
         "atr_14",
        "bb_width",
    ],
    "technical": [
"rsi_14",
        "stoch_k",
        "stoch_d",
        "willr",
        "roc",
        "ema_10",
        "ema_20",
        "ema_50",
        "sma_20",
        "sma_50",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_high",
        "bb_low",
    ],
    "regime": [
        "regime_0", "regime_1", "regime_2", "regime_3",
    ],
}


def build_block_index_map(columns) -> Dict[str, List[int]]:
    """
    Given a list of column names, return a mapping:
      block_name -> list of column indices in X
    Only columns present in `columns` are kept.
    """
    name_to_idx = {c: i for i, c in enumerate(columns)}
    block_indices: Dict[str, List[int]] = {}

    for block_name, col_names in FEATURE_BLOCKS.items():
        idxs = []
        for cname in col_names:
            if cname in name_to_idx:
                idxs.append(name_to_idx[cname])
        if idxs:
            block_indices[block_name] = idxs

    return block_indices
