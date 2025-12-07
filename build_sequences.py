import pandas as pd
from data.loaders.dataset_builder import DatasetBuilder

# Load processed CSV
df = pd.read_csv("data/processed/AAPL_processed.csv")

# Remove non-numeric columns before feeding into LSTM
if "date" in df.columns:
    df = df.drop(columns=["date"])

# Build sequences
builder = DatasetBuilder(lookback=60)
X, y_price, y_dir = builder.create_sequences(df)

print("X shape:", X.shape)          # Expected: (samples, 60, features)
print("y_price shape:", y_price.shape)
print("y_dir shape:", y_dir.shape)

print("\nSample X[0] window shape:", X[0].shape)
print("Sample direction label:", y_dir[0])
print("Sample next-day price:", y_price[0])
