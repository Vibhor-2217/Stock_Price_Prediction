import pandas as pd
from features.technical_indicators import add_technical_indicators

df = pd.read_csv("data/raw/AAPL.csv")
df.columns = [c.lower() for c in df.columns]

df2 = add_technical_indicators(df)

print(df2.head())
print(df2.shape)
print(df2.columns)
