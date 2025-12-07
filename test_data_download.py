from data.loaders.yfinance_loader import YFinanceLoader

loader = YFinanceLoader()
df = loader.download("AAPL", period="10y")
print(df.tail())
print(len(df))
