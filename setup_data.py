"""
Setup script to download data and train models on first run.
This runs automatically when the app starts on Streamlit Cloud.
"""

import os
import pandas as pd
import yfinance as yf
from pathlib import Path
import joblib

def setup_directories():
    """Create necessary directories"""
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)
    print("✅ Directories created")

def download_data():
    """Download stock data if not exists"""
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    
    for ticker in tickers:
        data_file = f'data/{ticker}_data.csv'
        
        if not os.path.exists(data_file):
            print(f"📥 Downloading {ticker} data...")
            df = yf.download(ticker, period='2y', progress=False)
            
            # Fix multi-level columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df.to_csv(data_file)
            print(f"✅ {ticker} data saved")
        else:
            print(f"✅ {ticker} data already exists")

def train_models():
    """Train models if not exists"""
    from feature_engineering import add_technical_indicators, prepare_data_for_ml
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    
    for ticker in tickers:
        model_file = f'models/{ticker}_best_model.pkl'
        
        if not os.path.exists(model_file):
            print(f"🔧 Training model for {ticker}...")
            
            # Load data
            df = pd.read_csv(f'data/{ticker}_data.csv', index_col='Date', parse_dates=True)
            
            # Add features
            df_features = add_technical_indicators(df)
            df_ml = prepare_data_for_ml(df_features, target_days=1)
            
            # Split
            split_idx = int(len(df_ml) * 0.8)
            train_df = df_ml.iloc[:split_idx]
            test_df = df_ml.iloc[split_idx:]
            
            # Prepare features
            exclude_cols = ['Target', 'Target_Direction', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            feature_columns = [col for col in train_df.columns if col not in exclude_cols]
            
            X_train = train_df[feature_columns]
            y_train = train_df['Target']
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Train
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            
            # Save
            joblib.dump(model, f'models/{ticker}_best_model.pkl')
            joblib.dump(scaler, f'models/{ticker}_scaler.pkl')
            joblib.dump(feature_columns, f'models/{ticker}_features.pkl')
            
            print(f"✅ {ticker} model trained and saved")
        else:
            print(f"✅ {ticker} model already exists")

def setup():
    """Main setup function"""
    print("🚀 Starting setup...")
    setup_directories()
    download_data()
    train_models()
    print("✅ Setup complete!")

if __name__ == "__main__":
    setup()