import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def load_data(ticker):
    """Load stock data from CSV"""
    # Read CSV without specifying index initially
    df = pd.read_csv(f'data/{ticker}_data.csv')
    
    # The first column should be the date
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    
    return df

def basic_statistics(df, ticker):
    """Print basic statistics"""
    print("="*60)
    print(f"BASIC STATISTICS - {ticker}")
    print("="*60)
    print(f"\nDataset Shape: {df.shape}")
    print(f"Date Range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nPrice Statistics:")
    print(f"  Highest Close: ${df['Close'].max():.2f}")
    print(f"  Lowest Close: ${df['Close'].min():.2f}")
    print(f"  Average Close: ${df['Close'].mean():.2f}")
    print(f"  Current Close: ${df['Close'].iloc[-1]:.2f}")
    
def plot_stock_price(df, ticker):
    """Plot closing price over time"""
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Close'], label='Close Price', linewidth=2, color='#2E86AB')
    plt.fill_between(df.index, df['Close'], alpha=0.3, color='#2E86AB')
    plt.title(f'{ticker} Stock Price Over Time', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=14, fontweight='bold')
    plt.ylabel('Price ($)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'plots/{ticker}_price_trend.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: plots/{ticker}_price_trend.png")
    plt.close()

def plot_volume(df, ticker):
    """Plot trading volume"""
    plt.figure(figsize=(14, 6))
    colors = ['#06D6A0' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#EF476F' 
              for i in range(len(df))]
    plt.bar(df.index, df['Volume'], alpha=0.7, color=colors, width=1)
    plt.title(f'{ticker} Trading Volume', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=14, fontweight='bold')
    plt.ylabel('Volume', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(f'plots/{ticker}_volume.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: plots/{ticker}_volume.png")
    plt.close()

def plot_candlestick_simple(df, ticker, days=90):
    """Simple candlestick representation"""
    df_recent = df.tail(days).copy()
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Color based on price movement
    colors = ['#06D6A0' if close >= open_ else '#EF476F' 
              for close, open_ in zip(df_recent['Close'], df_recent['Open'])]
    
    # Plot high-low lines and open-close bars
    for i, (idx, row) in enumerate(df_recent.iterrows()):
        ax.plot([i, i], [row['Low'], row['High']], color='black', linewidth=1, alpha=0.8)
        ax.plot([i, i], [row['Open'], row['Close']], color=colors[i], linewidth=5, solid_capstyle='round')
    
    ax.set_title(f'{ticker} Candlestick Chart (Last {days} Days)', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Days from Start', fontsize=14, fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'plots/{ticker}_candlestick.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: plots/{ticker}_candlestick.png")
    plt.close()

def plot_price_comparison(dfs, tickers):
    """Compare multiple stock prices (normalized)"""
    plt.figure(figsize=(14, 7))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for i, (ticker, df) in enumerate(zip(tickers, dfs)):
        # Normalize prices to start at 100
        normalized = (df['Close'] / df['Close'].iloc[0]) * 100
        plt.plot(df.index, normalized, label=ticker, linewidth=2, color=colors[i])
    
    plt.title('Stock Price Comparison (Normalized to 100)', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=14, fontweight='bold')
    plt.ylabel('Normalized Price', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('plots/stock_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: plots/stock_comparison.png")
    plt.close()

def correlation_analysis(df, ticker):
    """Analyze correlations between features"""
    plt.figure(figsize=(10, 8))
    correlation = df[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
    sns.heatmap(correlation, annot=True, cmap='RdYlGn', center=0, 
                square=True, linewidths=2, fmt='.3f', 
                cbar_kws={"shrink": 0.8}, annot_kws={"size": 12, "weight": "bold"})
    plt.title(f'{ticker} - Feature Correlation Matrix', fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'plots/{ticker}_correlation.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: plots/{ticker}_correlation.png")
    plt.close()

def plot_daily_returns(df, ticker):
    """Plot daily returns distribution"""
    daily_returns = df['Close'].pct_change().dropna() * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Line plot
    axes[0].plot(df.index[1:], daily_returns, linewidth=1, alpha=0.7, color='#2E86AB')
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_title(f'{ticker} Daily Returns (%)', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Date', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Daily Return (%)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # Histogram
    axes[1].hist(daily_returns, bins=50, color='#06D6A0', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=daily_returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {daily_returns.mean():.2f}%')
    axes[1].set_title(f'{ticker} Returns Distribution', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Daily Return (%)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'plots/{ticker}_daily_returns.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: plots/{ticker}_daily_returns.png")
    plt.close()

# Create plots directory
os.makedirs('plots', exist_ok=True)

print("\n" + "="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

# Stocks to analyze
tickers = ['AAPL', 'GOOGL', 'MSFT']

# Load all data
dfs = []
for ticker in tickers:
    print(f"\n📊 Analyzing {ticker}...")
    df = load_data(ticker)
    dfs.append(df)
    
    # Run analysis
    basic_statistics(df, ticker)
    plot_stock_price(df, ticker)
    plot_volume(df, ticker)
    plot_candlestick_simple(df, ticker, days=90)
    correlation_analysis(df, ticker)
    plot_daily_returns(df, ticker)

# Compare all stocks
print(f"\n📊 Creating comparison charts...")
plot_price_comparison(dfs, tickers)

print("\n" + "="*60)
print("✅ EDA COMPLETE!")
print("="*60)
print(f"\n📁 All visualizations saved in 'plots/' folder")
print(f"   Total plots created: {len(tickers) * 5 + 1}")