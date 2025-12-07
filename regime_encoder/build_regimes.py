from pathlib import Path
import argparse
import pandas as pd

from regime_encoder.hmm_regime_encoder import HMMRegimeEncoder
from config.paths import get_paths  # ✅ use centralised paths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--symbol",
        type=str,
        default="AAPL",
        help="Ticker symbol, e.g. AAPL / MSFT / TSLA",
    )
    parser.add_argument(
        "--n_states",
        type=int,
        default=4,
        help="Number of HMM regimes (states) to fit",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    symbol = args.symbol.upper()
    n_states = args.n_states

    paths = get_paths(symbol)
    input_path = paths["processed_csv"]   # e.g. data/processed/AAPL_processed.csv
    output_path = paths["regime_csv"]     # e.g. data/processed/AAPL_regime.csv

    print(f"[INFO] Building regimes for {symbol}...")
    print(f"[INFO] Loading {input_path} ...")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Processed file not found for {symbol}: {input_path}\n"
            f"Run data/loaders/save_processed.py --symbol {symbol} first."
        )

    df = pd.read_csv(input_path)

    has_date = "date" in df.columns
    date_series = df["date"] if has_date else None

    if "return" not in df.columns:
        raise ValueError("Column 'return' not found in processed CSV. "
                         "Ensure process_data.py computed next-day return.")

    feature_cols = ["return"]
    print(f"[INFO] Fitting {n_states}-state HMM on features: {feature_cols}")

    encoder = HMMRegimeEncoder(n_states=n_states)
    regimes = encoder.fit_predict(df, feature_cols)
    K = encoder.n_states

    # Add hard regime label
    df["regime"] = regimes

    # Add one-hot regime probabilities (regime_0 ... regime_{K-1})
    for k in range(K):
        df[f"regime_{k}"] = (regimes == k).astype(float)

    # Restore date at the end (if present)
    if has_date:
        df["date"] = date_series

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"[INFO] Saved regime-annotated file to {output_path}")
    print(f"[INFO] Final shape: {df.shape}")


if __name__ == "__main__":
    main()
