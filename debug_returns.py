import pandas as pd

df = pd.read_csv("data/processed/AAPL_processed.csv")

print("Direction counts:")
print(df["direction"].value_counts())

print("\nFirst 20 rows:")
print(df[["close", "direction", "return"]].head(20))

print("\nLast 20 rows:")
print(df[["close", "direction", "return"]].tail(20))
