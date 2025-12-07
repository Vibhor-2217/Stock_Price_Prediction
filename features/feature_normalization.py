from sklearn.preprocessing import StandardScaler
import pandas as pd

class FeatureNormalizer:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, df):
        scaled = self.scaler.fit_transform(df)
        return pd.DataFrame(scaled, columns=df.columns)
