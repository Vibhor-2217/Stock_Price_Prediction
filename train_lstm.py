import pandas as pd
import torch

from data.loaders.dataset_builder import DatasetBuilder
from training.train_baseline import train_baseline

def main():

    print("[INFO] Loading processed CSV...")
    df = pd.read_csv("data/processed/AAPL_processed.csv")

    # Remove non-feature columns (DatasetBuilder will handle the rest)
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    print("[INFO] Building sequences...")
    builder = DatasetBuilder(lookback=60)
    X, y_return, y_dir = builder.create_sequences(df)

    print(f"[INFO] X shape: {X.shape}")
    print(f"[INFO] y_return shape: {y_return.shape}")
    print(f"[INFO] y_dir shape: {y_dir.shape}")

    print("[INFO] Starting training...")
    model = train_baseline(X, y_return, y_dir, epochs=15, batch_size=32, lr=1e-3)

    print("[DONE] Training completed.")

if __name__ == "__main__":
    main()
