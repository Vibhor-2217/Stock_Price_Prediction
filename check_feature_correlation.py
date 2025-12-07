import pandas as pd

df = pd.read_csv("data/processed/AAPL_processed.csv")

df = df.select_dtypes(include=['float64', 'int64'])

# Drop known leaks
for col in ["return", "direction", "log_return"]:
    if col in df.columns:
        df = df.drop(columns=[col])

# Create shifted direction target
df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

print(df.corr()["target"].sort_values(ascending=False))
