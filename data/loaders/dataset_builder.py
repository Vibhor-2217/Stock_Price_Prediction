import numpy as np
import torch

class DatasetBuilder:
    def __init__(self, lookback=60):
        self.lookback = lookback

    def create_sequences(self, df):
        X, y_return, y_dir = [], [], []

        # Remove raw close, date, return, direction from features
        feature_df = df.drop(columns=["close", "return", "direction", "date"], errors="ignore")

        X_values = feature_df.copy().values
        returns = df["return"].values
        directions = df["direction"].values

        for i in range(len(df) - self.lookback):
            X.append(X_values[i:i + self.lookback])
            y_return.append(returns[i + self.lookback])
            y_dir.append(directions[i + self.lookback])

        return (
            torch.tensor(np.array(X), dtype=torch.float32),
            torch.tensor(np.array(y_return), dtype=torch.float32),
            torch.tensor(np.array(y_dir), dtype=torch.long)
        )
