import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

def create_summary_dashboard():
    """Create a comprehensive dashboard of all results"""
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # Row 1: Recent price trends
    for idx, ticker in enumerate(tickers):
        ax = fig.add_subplot(gs[0, idx])
        
        # Load data
        df = pd.read_csv(f'data/{ticker}_data.csv', index_col='Date', parse_dates=True)
        recent = df.tail(60)  # Last 60 days
        
        ax.plot(recent.index, recent['Close'], color=colors[idx], linewidth=2)
        ax.fill_between(recent.index, recent['Close'], alpha=0.3, color=colors[idx])
        ax.set_title(f'{ticker} - Last 60 Days', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Price ($)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
    
    # Row 2: Model performance comparison
    ax_perf = fig.add_subplot(gs[1, :])
    
    # Load model results from saved files
    model_data = []
    for ticker in tickers:
        test_df = pd.read_csv(f'data/{ticker}_test.csv', index_col='Date', parse_dates=True)
        model = joblib.load(f'models/{ticker}_best_model.pkl')
        scaler = joblib.load(f'models/{ticker}_scaler.pkl')
        features = joblib.load(f'models/{ticker}_features.pkl')
        
        X_test = test_df[features]
        y_test = test_df['Target']
        
        X_scaled = scaler.transform(X_test)
        predictions = model.predict(X_scaled)
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((predictions - y_test)**2))
        mae = np.mean(np.abs(predictions - y_test))
        
        model_data.append({'Ticker': ticker, 'RMSE': rmse, 'MAE': mae})
    
    model_df = pd.DataFrame(model_data)
    
    x = np.arange(len(tickers))
    width = 0.35
    
    bars1 = ax_perf.bar(x - width/2, model_df['RMSE'], width, label='RMSE', color='steelblue')
    bars2 = ax_perf.bar(x + width/2, model_df['MAE'], width, label='MAE', color='coral')
    
    ax_perf.set_xlabel('Stock', fontsize=12, fontweight='bold')
    ax_perf.set_ylabel('Error ($)', fontsize=12, fontweight='bold')
    ax_perf.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax_perf.set_xticks(x)
    ax_perf.set_xticklabels(tickers)
    ax_perf.legend(fontsize=11)
    ax_perf.grid(True, alpha=0.3, axis='y')
    
    # Row 3: Predictions vs Actual (last 30 days)
    for idx, ticker in enumerate(tickers):
        ax = fig.add_subplot(gs[2, idx])
        
        # Load test data and predictions
        test_df = pd.read_csv(f'data/{ticker}_test.csv', index_col='Date', parse_dates=True)
        model = joblib.load(f'models/{ticker}_best_model.pkl')
        scaler = joblib.load(f'models/{ticker}_scaler.pkl')
        features = joblib.load(f'models/{ticker}_features.pkl')
        
        X_test = test_df[features]
        y_test = test_df['Target']
        
        X_scaled = scaler.transform(X_test)
        predictions = model.predict(X_scaled)
        
        # Plot last 30 predictions
        last_30 = min(30, len(y_test))
        dates = test_df.index[-last_30:]
        actual = y_test.values[-last_30:]
        pred = predictions[-last_30:]
        
        ax.plot(dates, actual, label='Actual', color=colors[idx], linewidth=2, marker='o', markersize=4)
        ax.plot(dates, pred, label='Predicted', color='red', linewidth=2, linestyle='--', marker='s', markersize=4)
        
        ax.set_title(f'{ticker} - Predictions vs Actual', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Price ($)', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
    
    plt.suptitle('Stock Prediction ML - Complete Dashboard', fontsize=22, fontweight='bold', y=0.995)
    
    # Save
    plt.savefig('plots/complete_dashboard.png', dpi=300, bbox_inches='tight')
    print("✅ Dashboard saved: plots/complete_dashboard.png")
    plt.close()

if __name__ == "__main__":
    print("="*60)
    print("CREATING COMPREHENSIVE DASHBOARD")
    print("="*60)
    
    create_summary_dashboard()
    
    print("\n🎉 Dashboard created successfully!")
    print("📁 Open 'plots/complete_dashboard.png' to view")