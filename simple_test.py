print("Script started!")

import yfinance as yf
print("yfinance imported")

import pandas as pd
print("pandas imported")

# Try downloading
print("Downloading AAPL...")
df = yf.download('AAPL', period='5d', progress=False)
print(f"Downloaded {len(df)} rows")
print(df.head())

print("Done!")