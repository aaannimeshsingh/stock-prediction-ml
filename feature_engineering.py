import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import os

def add_technical_indicators(df):
    """Add technical indicators as features"""
    df = df.copy()
    
    print("Adding technical indicators...")
    
    # Simple Moving Averages
    df['SMA_10'] = SMAIndicator(close=df['Close'], window=10).sma_indicator()
    df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    
    # Exponential Moving Averages
    df['EMA_10'] = EMAIndicator(close=df['Close'], window=10).ema_indicator()
    df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
    
    # RSI (Relative Strength Index)
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    
    # MACD
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()
    
    # Bollinger Bands
    bb = BollingerBands(close=df['Close'])
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['BB_Mid'] = bb.bollinger_mavg()
    
    # Price momentum features
    df['Price_Change'] = df['Close'].pct_change()
    df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
    df['Price_Change_10d'] = df['Close'].pct_change(periods=10)
    
    # Volume features
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean()
    
    # Daily returns
    df['Daily_Return'] = (df['Close'] - df['Open']) / df['Open']
    
    # High-Low range
    df['HL_Range'] = df['High'] - df['Low']
    df['HL_Pct'] = (df['High'] - df['Low']) / df['Close']
    
    print(f"✅ Added {len([col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']])} technical indicators")
    
    return df

def prepare_data_for_ml(df, target_days=1):
    """
    Prepare data for machine learning
    target_days: predict price N days ahead
    """
    df = df.copy()
    
    # Create target variable (future price)
    df['Target'] = df['Close'].shift(-target_days)
    
    # Create binary classification target (Up/Down)
    df['Target_Direction'] = (df['Target'] > df['Close']).astype(int)
    
    # Drop rows with NaN values
    df_clean = df.dropna()
    
    print(f"\n📊 Dataset Info:")
    print(f"Total samples: {len(df_clean)}")
    print(f"Features: {len(df_clean.columns) - 2}")  # Excluding Target columns
    print(f"Date range: {df_clean.index.min().date()} to {df_clean.index.max().date()}")
    
    return df_clean

def split_data(df, train_size=0.8):
    """Split data into train and test sets (time-series split)"""
    split_idx = int(len(df) * train_size)
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"\n📦 Data Split:")
    print(f"Training set: {len(train_df)} samples ({train_df.index.min().date()} to {train_df.index.max().date()})")
    print(f"Test set: {len(test_df)} samples ({test_df.index.min().date()} to {test_df.index.max().date()})")
    
    return train_df, test_df

if __name__ == "__main__":
    print("="*60)
    print("FEATURE ENGINEERING")
    print("="*60)
    
    # Process each stock
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    
    for ticker in tickers:
        print(f"\n📊 Processing {ticker}...")
        
        # Load data
        df = pd.read_csv(f'data/{ticker}_data.csv', index_col='Date', parse_dates=True)
        
        # Add technical indicators
        df_features = add_technical_indicators(df)
        
        # Prepare for ML
        df_ml = prepare_data_for_ml(df_features, target_days=1)
        
        # Split data
        train_df, test_df = split_data(df_ml)
        
        # Save processed data
        train_df.to_csv(f'data/{ticker}_train.csv')
        test_df.to_csv(f'data/{ticker}_test.csv')
        df_features.to_csv(f'data/{ticker}_features.csv')
        
        print(f"✅ Saved processed files for {ticker}")
    
    print("\n" + "="*60)
    print("✅ FEATURE ENGINEERING COMPLETE!")
    print("="*60)
    print(f"\n📁 Files saved:")
    print(f"   - *_train.csv (training data)")
    print(f"   - *_test.csv (test data)")
    print(f"   - *_features.csv (all features)")