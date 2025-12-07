import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/AAPL_processed.csv")

# Keep numeric
df = df.select_dtypes(include=[np.number])

# Build direction target
direction = (df["return"].shift(-1) > 0).astype(int)

print("Direction value counts:")
print(direction.value_counts(normalize=True))

print("\nRaw counts:")
print(direction.value_counts())
