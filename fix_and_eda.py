import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create directories
os.makedirs('data', exist_ok=True)
os.makedirs('plots', exist_ok=True)

print("="*60)
print("FIXING AND DOWNLOADING FRESH DATA")
print("="*60)

tickers = ['AAPL', 'GOOGL', 'MSFT']
all_dfs = {}

for ticker in tickers:
    print(f"\n📥 Downloading {ticker}...")
    
    # Download data
    df = yf.download(ticker, period='2y', progress=False)
    
    # Fix multi-level columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Reset index to make Date a regular column
    df = df.reset_index()
    
    # Clean up - remove any weird rows
    df = df[df['Date'].notna()]
    
    # Ensure proper column names
    expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    if set(df.columns) != set(expected_columns):
        print(f"   ⚠️  Unexpected columns: {df.columns.tolist()}")
        print(f"   Expected: {expected_columns}")
    
    # Save with proper format
    df.to_csv(f'data/{ticker}_data.csv', index=False)
    
    all_dfs[ticker] = df
    
    print(f"   ✅ {len(df)} days downloaded")
    print(f"   📅 Range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"   💰 Latest close: ${df['Close'].iloc[-1]:.2f}")

# Verify the files are correct
print("\n" + "="*60)
print("VERIFYING FILES")
print("="*60)

for ticker in tickers:
    print(f"\n📄 Checking {ticker}_data.csv...")
    df_check = pd.read_csv(f'data/{ticker}_data.csv', nrows=3)
    print(f"   Columns: {df_check.columns.tolist()}")
    print(f"   First row Date: {df_check['Date'].iloc[0]}")
    print(f"   First row Close: ${df_check['Close'].iloc[0]:.2f}")

print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

# Now create all the plots
for ticker, df in all_dfs.items():
    print(f"\n📊 Creating plots for {ticker}...")
    
    # Convert Date to datetime for plotting
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 1. Price trend
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'], df['Close'], linewidth=2, color='#2E86AB')
    plt.fill_between(df['Date'], df['Close'], alpha=0.3, color='#2E86AB')
    plt.title(f'{ticker} Stock Price Over Time', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price ($)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'plots/{ticker}_price_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Price trend saved")
    
    # 2. Volume
    plt.figure(figsize=(14, 6))
    colors = ['#06D6A0' if close >= open_ else '#EF476F' 
              for close, open_ in zip(df['Close'], df['Open'])]
    plt.bar(df['Date'], df['Volume'], alpha=0.7, color=colors, width=1)
    plt.title(f'{ticker} Trading Volume', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Volume', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'plots/{ticker}_volume.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Volume chart saved")
    
    # 3. Candlestick (last 90 days)
    df_recent = df.tail(90).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(16, 8))
    colors = ['#06D6A0' if close >= open_ else '#EF476F' 
              for close, open_ in zip(df_recent['Close'], df_recent['Open'])]
    
    for i, row in df_recent.iterrows():
        ax.plot([i, i], [row['Low'], row['High']], color='black', linewidth=1, alpha=0.8)
        ax.plot([i, i], [row['Open'], row['Close']], color=colors[i], linewidth=5)
    
    ax.set_title(f'{ticker} Candlestick Chart (Last 90 Days)', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Days', fontsize=14)
    ax.set_ylabel('Price ($)', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'plots/{ticker}_candlestick.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Candlestick chart saved")
    
    # 4. Correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation = df[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
    sns.heatmap(correlation, annot=True, cmap='RdYlGn', center=0, 
                square=True, linewidths=2, fmt='.3f', cbar_kws={"shrink": 0.8})
    plt.title(f'{ticker} - Feature Correlation', fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'plots/{ticker}_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Correlation heatmap saved")
    
    # 5. Daily returns
    daily_returns = df['Close'].pct_change().dropna() * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    axes[0].plot(df['Date'].iloc[1:], daily_returns, linewidth=1, alpha=0.7, color='#2E86AB')
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_title(f'{ticker} Daily Returns', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Date', fontsize=12)
    axes[0].set_ylabel('Daily Return (%)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(daily_returns, bins=50, color='#06D6A0', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=daily_returns.mean(), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {daily_returns.mean():.2f}%')
    axes[1].set_title(f'{ticker} Returns Distribution', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Daily Return (%)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'plots/{ticker}_daily_returns.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Daily returns saved")

# 6. Comparison chart
print(f"\n📊 Creating stock comparison...")
plt.figure(figsize=(14, 7))
colors = ['#2E86AB', '#A23B72', '#F18F01']

for i, (ticker, df) in enumerate(all_dfs.items()):
    normalized = (df['Close'] / df['Close'].iloc[0]) * 100
    plt.plot(df['Date'], normalized, label=ticker, linewidth=2, color=colors[i])

plt.title('Stock Price Comparison (Normalized to 100)', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Normalized Price', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/stock_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Comparison chart saved")

print("\n" + "="*60)
print("✅ ALL DONE!")
print("="*60)
print(f"\n📁 Check the 'plots' folder for {len(tickers) * 5 + 1} visualizations!")
print(f"📁 Data files are properly formatted in 'data' folder")
print(f"\nNext steps:")
print(f"1. Run: python feature_engineering.py")
print(f"2. Run: python train_models.py")
print(f"3. Run: streamlit run streamlit_app.py")