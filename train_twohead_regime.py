# train_twohead_regime.py
#
# Train the Two-Head LSTM using regime-annotated data (AAPL_regime.csv).
# Paths are fully relative, so this works on any machine / repo location.

from pathlib import Path
import pandas as pd

from data.loaders.dataset_builder import DatasetBuilder
from training.train_twohead import train_twohead


# repo root (same folder where this file lives)
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def main():
    csv_path = DATA_DIR / "AAPL_regime.csv"
    print(f"[INFO] Loading regime-annotated CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Drop date column from features if present
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    print("[INFO] Building sequences with regime features...")
    builder = DatasetBuilder(lookback=60)
    X, y_return, y_dir = builder.create_sequences(df)

    print(f"[INFO] X shape: {X.shape}")
    print(f"[INFO] y_return shape: {y_return.shape}")
    print(f"[INFO] y_dir shape: {y_dir.shape}")

    print("[INFO] Training TwoHead LSTM with HMM regimes...")
    _model = train_twohead(
        X,
        y_return,
        y_dir,
        epochs=15,
        batch_size=32,
        lr=1e-3,
    )

    print("[DONE] Training completed.")


if __name__ == "__main__":
    main()
