# data/loaders/save_processed.py

import argparse
from pathlib import Path

import pandas as pd

# Import your existing processing pipeline
from data.loaders.process_data import process_data
from config.paths import get_paths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--symbol",
        type=str,
        default="AAPL",
        help="Ticker symbol (e.g. AAPL, MSFT, TSLA)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    symbol = args.symbol.upper()

    paths = get_paths(symbol)
    raw_file = paths["raw_csv"]
    processed_file = paths["processed_csv"]

    print(f"[STEP] Processing dataset for {symbol}...")
    print(f"[INFO] Raw CSV: {raw_file}")

    if not raw_file.exists():
        raise FileNotFoundError(
            f"Raw file not found for {symbol}: {raw_file}\n"
            f"Please put {symbol}.csv inside data/raw/"
        )

    # --- Run your existing processing pipeline ---
    df_processed = process_data(raw_file)

    # Ensure processed directory exists
    processed_file.parent.mkdir(parents=True, exist_ok=True)

    # Save
    df_processed.to_csv(processed_file, index=False)
    print(f"[OK] Saved processed dataset to: {processed_file}")
    print(f"[INFO] Processed shape: {df_processed.shape}")


if __name__ == "__main__":
    main()
