import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def download_stock_data(ticker, period='2y'):
    """
    Download historical stock data
    ticker: Stock symbol (e.g., 'AAPL', 'RELIANCE.NS')
    period: Time period ('1y', '2y', '5y', 'max')
    """
    print(f"Downloading data for {ticker}...")
    
    # Download data
    stock_data = yf.download(ticker, period=period)
    
    # Save to CSV
    stock_data.to_csv(f'data/{ticker}_data.csv')
    print(f"Data saved to data/{ticker}_data.csv")
    print(f"Downloaded {len(stock_data)} days of data\n")
    
    return stock_data

def get_stock_info(ticker):
    """Get basic stock information"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        print(f"Stock: {info.get('longName', ticker)}")
        print(f"Sector: {info.get('sector', 'N/A')}")
        print(f"Current Price: ${info.get('currentPrice', 'N/A')}")
        print()
    except Exception as e:
        print(f"Could not fetch info for {ticker}: {e}\n")

# Create data directory
os.makedirs('data', exist_ok=True)

print("="*60)
print("STOCK DATA COLLECTION")
print("="*60)

# Download data for multiple stocks
stocks = ['AAPL', 'GOOGL', 'MSFT']

for ticker in stocks:
    try:
        get_stock_info(ticker)
        data = download_stock_data(ticker, period='2y')
    except Exception as e:
        print(f"Error downloading {ticker}: {e}\n")

print("="*60)
print("✅ DATA COLLECTION COMPLETE!")
print("="*60)
print(f"\nCheck the 'data' folder for CSV files")