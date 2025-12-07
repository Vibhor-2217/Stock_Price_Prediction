import yfinance as yf
import pandas as pd
import os

class YFinanceLoader:
    def __init__(self, save_path="data/raw"):
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def download(self, ticker, period="10y", interval="1d"):
        """
        Most reliable way to fetch data via yfinance without timeout issues.
        """
        print(f"Downloading {ticker} from yfinance...")

        df = yf.Ticker(ticker).history(
            period=period,
            interval=interval,
            auto_adjust=False,
            actions=False,
            rounding=False
        )

        if df.empty:
            raise ValueError(f"No data returned for {ticker}")

        # Reset index & rename columns
        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]

        file_path = os.path.join(self.save_path, f"{ticker}.csv")
        df.to_csv(file_path, index=False)

        print(f"Saved: {file_path} ({len(df)} rows)")
        return df
