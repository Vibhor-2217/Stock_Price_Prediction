import requests
import pandas as pd
import os
import time

class AlphaVantageLoader:
    def __init__(self, api_key, save_path="data/raw"):
        self.api_key = api_key
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def download(self, ticker, outputsize="full"):
        """
        Downloads historical daily OHLCV data using free AlphaVantage API.
        """
        print(f"\nDownloading {ticker} ...")

        url = (
            "https://www.alphavantage.co/query?"
            "function=TIME_SERIES_DAILY&"
            f"symbol={ticker}&outputsize={outputsize}&apikey={self.api_key}"
        )

        response = requests.get(url)

        if response.status_code != 200:
            raise ValueError(f"HTTP error: {response.status_code}")

        data = response.json()

        # Detect API limit or key issues
        if "Note" in data:
            print("API limit reached. Waiting 60 seconds...")
            time.sleep(60)
            return self.download(ticker, outputsize)

        if "Error Message" in data:
            raise ValueError(f"Invalid API request: {data['Error Message']}")

        if "Time Series (Daily)" not in data:
            print("⚠ Raw JSON from API:", data)
            raise ValueError("API returned no time series data. Check key or limits.")

        ts = data["Time Series (Daily)"]

        df = pd.DataFrame.from_dict(ts, orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        df = df.rename(columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume"
        })

        df = df.apply(pd.to_numeric)

        file_path = os.path.join(self.save_path, f"{ticker}.csv")
        df.to_csv(file_path, index=True)

        print(f"✔ Saved: {file_path} ({len(df)} rows)")
        return df
