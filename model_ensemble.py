import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

class EnsemblePredictor:
    """Ensemble of multiple ML models for improved predictions"""
    
    def __init__(self, ticker):
        self.ticker = ticker
        self.scaler = None
        self.features = None
        
    def create_ensemble_prediction(self):
        """Combine predictions from multiple approaches"""
        print(f"\n{'='*60}")
        print(f"ENSEMBLE MODEL EVALUATION - {self.ticker}")
        print(f"{'='*60}")
        
        # Load test data
        test_df = pd.read_csv(f'data/{self.ticker}_test.csv', index_col='Date', parse_dates=True)
        train_df = pd.read_csv(f'data/{self.ticker}_train.csv', index_col='Date', parse_dates=True)
        
        # Load model components
        model = joblib.load(f'models/{self.ticker}_best_model.pkl')
        self.scaler = joblib.load(f'models/{self.ticker}_scaler.pkl')
        self.features = joblib.load(f'models/{self.ticker}_features.pkl')
        
        X_test = test_df[self.features]
        y_test = test_df['Target']
        X_scaled = self.scaler.transform(X_test)
        
        # Method 1: Base model prediction
        pred_ml = model.predict(X_scaled)
        
        # Method 2: Simple Moving Average based prediction
        pred_sma = test_df['Close'].rolling(window=10).mean().shift(1)
        pred_sma = pred_sma.fillna(method='bfill')
        
        # Method 3: Exponential Moving Average based prediction
        pred_ema = test_df['Close'].ewm(span=10, adjust=False).mean().shift(1)
        pred_ema = pred_ema.fillna(method='bfill')
        
        # Method 4: Momentum based prediction (last price + average change)
        momentum = test_df['Close'].diff().rolling(window=5).mean()
        pred_momentum = test_df['Close'] + momentum
        pred_momentum = pred_momentum.fillna(method='bfill')
        
        # Ensure all predictions are numpy arrays with same length
        pred_sma = pred_sma.values
        pred_ema = pred_ema.values
        pred_momentum = pred_momentum.values
        
        # Ensemble: Weighted average
        # Give more weight to ML model
        weights = {
            'ML Model': 0.50,
            'SMA': 0.20,
            'EMA': 0.20,
            'Momentum': 0.10
        }
        
        ensemble_pred = (
            weights['ML Model'] * pred_ml +
            weights['SMA'] * pred_sma +
            weights['EMA'] * pred_ema +
            weights['Momentum'] * pred_momentum
        )
        
        # Calculate metrics for each method
        methods = {
            'ML Model': pred_ml,
            'SMA Prediction': pred_sma,
            'EMA Prediction': pred_ema,
            'Momentum Prediction': pred_momentum,
            'Ensemble': ensemble_pred
        }
        
        print(f"\n📊 Performance Comparison:\n")
        
        results = []
        for name, predictions in methods.items():
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
            
            results.append({
                'Method': name,
                'RMSE': rmse,
                'MAE': mae,
                'R² Score': r2,
                'MAPE (%)': mape
            })
            
            print(f"{name}:")
            print(f"  RMSE: ${rmse:.2f}")
            print(f"  MAE: ${mae:.2f}")
            print(f"  R² Score: {r2:.4f}")
            print(f"  MAPE: {mape:.2f}%")
            print()
        
        # Find best method
        results_df = pd.DataFrame(results)
        best_method = results_df.loc[results_df['RMSE'].idxmin()]
        
        print(f"{'='*60}")
        print(f"🏆 Best Method: {best_method['Method']}")
        print(f"   RMSE: ${best_method['RMSE']:.2f}")
        print(f"   R² Score: {best_method['R² Score']:.4f}")
        print(f"{'='*60}")
        
        # Plot comparison
        self.plot_ensemble_comparison(test_df.index, y_test, methods)
        
        # Plot metrics comparison
        self.plot_metrics_comparison(results_df)
        
        return results_df
    
    def plot_ensemble_comparison(self, dates, actual, predictions_dict):
        """Plot all predictions vs actual"""
        fig, axes = plt.subplots(3, 2, figsize=(18, 12))
        axes = axes.flatten()
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#06D6A0', '#EF476F']
        
        for idx, (name, pred) in enumerate(predictions_dict.items()):
            ax = axes[idx]
            
            ax.plot(dates, actual, label='Actual', linewidth=2, color='black', alpha=0.7)
            ax.plot(dates, pred, label=name, linewidth=2, color=colors[idx], linestyle='--', alpha=0.8)
            
            ax.set_title(f'{name}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=10)
            ax.set_ylabel('Price ($)', fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
        
        # Hide last subplot if odd number
        if len(predictions_dict) < 6:
            axes[-1].axis('off')
        
        plt.suptitle(f'{self.ticker} - Ensemble Model Comparison', fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f'plots/{self.ticker}_ensemble_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ Ensemble comparison saved: plots/{self.ticker}_ensemble_comparison.png")
        plt.close()
    
    def plot_metrics_comparison(self, results_df):
        """Plot bar chart of metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = ['RMSE', 'MAE', 'R² Score', 'MAPE (%)']
        colors = ['steelblue', 'coral', 'seagreen', 'purple']
        
        for idx, (metric, color) in enumerate(zip(metrics, colors)):
            ax = axes[idx // 2, idx % 2]
            
            values = results_df[metric].values
            methods = results_df['Method'].values
            
            bars = ax.barh(methods, values, color=color, alpha=0.7)
            
            # Highlight best performer
            if metric == 'R² Score':
                best_idx = np.argmax(values)
            else:
                best_idx = np.argmin(values)
            bars[best_idx].set_color('gold')
            bars[best_idx].set_alpha(1.0)
            
            ax.set_xlabel(metric, fontsize=12, fontweight='bold')
            ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, v in enumerate(values):
                if metric in ['R² Score']:
                    ax.text(v, i, f' {v:.4f}', va='center', fontsize=9)
                elif metric in ['MAPE (%)']:
                    ax.text(v, i, f' {v:.2f}%', va='center', fontsize=9)
                else:
                    ax.text(v, i, f' ${v:.2f}', va='center', fontsize=9)
        
        plt.suptitle(f'{self.ticker} - Ensemble Metrics Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'plots/{self.ticker}_ensemble_metrics.png', dpi=300, bbox_inches='tight')
        print(f"✅ Ensemble metrics saved: plots/{self.ticker}_ensemble_metrics.png")
        plt.close()

if __name__ == "__main__":
    print("="*60)
    print("ENSEMBLE MODEL EVALUATION")
    print("="*60)
    
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    all_results = []
    
    for ticker in tickers:
        ensemble = EnsemblePredictor(ticker)
        results = ensemble.create_ensemble_prediction()
        
        # Add ticker column
        results['Ticker'] = ticker
        all_results.append(results)
    
    # Combined summary
    print("\n\n" + "="*60)
    print("ENSEMBLE SUMMARY - ALL STOCKS")
    print("="*60)
    
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Show best method for each stock
    for ticker in tickers:
        ticker_data = combined_df[combined_df['Ticker'] == ticker]
        best = ticker_data.loc[ticker_data['RMSE'].idxmin()]
        print(f"\n{ticker}: Best method is '{best['Method']}' with RMSE ${best['RMSE']:.2f}")
    
    print("\n" + "="*60)
    print("🎉 ENSEMBLE EVALUATION COMPLETE!")
    print("="*60)