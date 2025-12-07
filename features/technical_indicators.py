import pandas as pd
import numpy as np
import ta

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 25–30 technical indicators to the dataframe.
    df MUST contain columns: open, high, low, close, volume
    """

    df = df.copy()

    # ====================
    # PRICE MOMENTUM
    # ====================
    df["rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["stoch_k"] = ta.momentum.StochasticOscillator(
        high=df["high"], low=df["low"], close=df["close"]
    ).stoch()
    df["stoch_d"] = ta.momentum.StochasticOscillator(
        high=df["high"], low=df["low"], close=df["close"]
    ).stoch_signal()
    df["willr"] = ta.momentum.WilliamsRIndicator(
        high=df["high"], low=df["low"], close=df["close"]
    ).williams_r()
    df["roc"] = ta.momentum.ROCIndicator(df["close"], window=12).roc()

    # ====================
    # TREND INDICATORS
    # ====================
    df["ema_10"] = ta.trend.EMAIndicator(df["close"], window=10).ema_indicator()
    df["ema_20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()

    df["sma_20"] = ta.trend.SMAIndicator(df["close"], window=20).sma_indicator()
    df["sma_50"] = ta.trend.SMAIndicator(df["close"], window=50).sma_indicator()

    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # ====================
    # VOLATILITY INDICATORS
    # ====================
    bb = ta.volatility.BollingerBands(df["close"], window=20)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()

    df["atr_14"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=14
    ).average_true_range()

    # ====================
    # VOLUME INDICATORS
    # ====================
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(
        df["close"], df["volume"]
    ).on_balance_volume()

    df["volume_sma_20"] = (
        df["volume"].rolling(window=20).mean()
    )

    # ====================
    # RETURNS
    # ====================
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Drop any rows with NaN after indicator creation
    df = df.dropna().reset_index(drop=True)

    return df
