from data.loaders.alphavantage_loader import AlphaVantageLoader

loader = AlphaVantageLoader(api_key="XSZDC9UJWJ1C52HR")

df = loader.download("AAPL")
print(df.tail())
print(len(df))
