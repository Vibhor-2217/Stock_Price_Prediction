import pandas as pd
import numpy as np
from features.technical_indicators import add_technical_indicators
from features.feature_normalization import FeatureNormalizer


def process_data(raw_csv_path: str):
    print(f"[INFO] Loading raw file: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path)
    df.columns = [c.lower() for c in df.columns]

    # Store date if it exists
    dates = df["date"] if "date" in df.columns else None
    if dates is not None:
        df = df.drop(columns=["date"])

    # ---------------------------------------------
    # ADD TECHNICAL INDICATORS FIRST
    # ---------------------------------------------
    df = add_technical_indicators(df)

    # ---------------------------------------------
    # COMPUTE NEXT-DAY RETURN
    # ---------------------------------------------
    df["return"] = np.log(df["close"].shift(-1) / df["close"])

    # ---------------------------------------------
    # DROP NaNs — THIS REMOVES BAD INDICATOR ROWS
    # ---------------------------------------------
    df = df.dropna().reset_index(drop=True)

    # ---------------------------------------------
    #COMPUTE DIRECTION **AFTER** DROPPING NAs
    # Direction is perfectly aligned if derived from return.
    # ---------------------------------------------
    df["direction"] = (df["return"] > 0).astype(int)

    # ---------------------------------------------
    # NORMALIZE FEATURE COLUMNS (NOT TARGETS)
    # ---------------------------------------------
    feature_cols = df.columns.drop(["close", "return", "direction"])

    normalizer = FeatureNormalizer()
    df_scaled = normalizer.fit_transform(df[feature_cols])

    # ---------------------------------------------
    # ADD BACK TARGETS + CLOSE (UNSCALED)
    # ---------------------------------------------
    df_scaled["return"] = df["return"].values
    df_scaled["direction"] = df["direction"].values
    df_scaled["close"] = df["close"].values

    # ---------------------------------------------
    # ADD DATE BACK IF NEEDED
    # ---------------------------------------------
    if dates is not None:
        df_scaled["date"] = dates.iloc[-len(df_scaled):].values

    return df_scaled
