import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class StockPredictor:
    def __init__(self, ticker='AAPL'):
        self.ticker = ticker
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_data(self):
        """Load preprocessed train and test data"""
        self.train_df = pd.read_csv(f'data/{self.ticker}_train.csv', index_col='Date', parse_dates=True)
        self.test_df = pd.read_csv(f'data/{self.ticker}_test.csv', index_col='Date', parse_dates=True)
        
        # Define features (exclude target and price columns)
        exclude_cols = ['Target', 'Target_Direction', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        self.feature_columns = [col for col in self.train_df.columns if col not in exclude_cols]
        
        print(f"✅ Loaded data for {self.ticker}")
        print(f"📊 Features: {len(self.feature_columns)}")
        print(f"   Using: {', '.join(self.feature_columns[:5])}... (showing first 5)")
        
    def prepare_features(self):
        """Prepare X and y for training"""
        # Training data
        X_train = self.train_df[self.feature_columns]
        y_train = self.train_df['Target']
        
        # Test data
        X_test = self.test_df[self.feature_columns]
        y_test = self.test_df['Target']
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self):
        """Train multiple ML models"""
        X_train, X_test, y_train, y_test = self.prepare_features()
        
        # Define models
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
        }
        
        results = {}
        
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        for name, model in models_to_train.items():
            print(f"\n🔧 Training {name}...")
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_r2 = r2_score(y_test, y_pred_test)
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
            
            results[name] = {
                'model': model,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'mape': mape,
                'predictions': y_pred_test
            }
            
            print(f"   Train RMSE: ${train_rmse:.2f}")
            print(f"   Test RMSE: ${test_rmse:.2f}")
            print(f"   Test MAE: ${test_mae:.2f}")
            print(f"   Test R²: {test_r2:.4f}")
            print(f"   MAPE: {mape:.2f}%")
        
        self.models = results
        self.y_test = y_test
        
        return results
    
    def plot_results(self):
        """Visualize model performance"""
        # Create comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        test_dates = self.test_df.index
        
        for idx, (name, result) in enumerate(self.models.items()):
            ax = axes[idx]
            
            # Plot actual vs predicted
            ax.plot(test_dates, self.y_test.values, label='Actual', linewidth=2, alpha=0.8, color='#2E86AB')
            ax.plot(test_dates, result['predictions'], label='Predicted', linewidth=2, alpha=0.8, linestyle='--', color='#F18F01')
            
            ax.set_title(f'{name}\nRMSE: ${result["test_rmse"]:.2f} | R²: {result["test_r2"]:.3f}', 
                        fontweight='bold', fontsize=12)
            ax.set_xlabel('Date', fontsize=10)
            ax.set_ylabel('Stock Price ($)', fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
        
        # Hide extra subplot
        axes[-1].axis('off')
        
        plt.suptitle(f'{self.ticker} - Model Predictions Comparison', fontsize=20, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f'plots/{self.ticker}_model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved: plots/{self.ticker}_model_comparison.png")
        plt.close()
        
    def plot_metrics_comparison(self):
        """Compare model metrics"""
        metrics_df = pd.DataFrame({
            'Model': list(self.models.keys()),
            'RMSE': [r['test_rmse'] for r in self.models.values()],
            'MAE': [r['test_mae'] for r in self.models.values()],
            'R² Score': [r['test_r2'] for r in self.models.values()],
            'MAPE (%)': [r['mape'] for r in self.models.values()]
        })
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # RMSE
        axes[0, 0].barh(metrics_df['Model'], metrics_df['RMSE'], color='steelblue')
        axes[0, 0].set_xlabel('RMSE ($)', fontweight='bold')
        axes[0, 0].set_title('Root Mean Squared Error (Lower is Better)', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        # MAE
        axes[0, 1].barh(metrics_df['Model'], metrics_df['MAE'], color='coral')
        axes[0, 1].set_xlabel('MAE ($)', fontweight='bold')
        axes[0, 1].set_title('Mean Absolute Error (Lower is Better)', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # R² Score
        axes[1, 0].barh(metrics_df['Model'], metrics_df['R² Score'], color='seagreen')
        axes[1, 0].set_xlabel('R² Score', fontweight='bold')
        axes[1, 0].set_title('R² Score (Higher is Better)', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # MAPE
        axes[1, 1].barh(metrics_df['Model'], metrics_df['MAPE (%)'], color='purple')
        axes[1, 1].set_xlabel('MAPE (%)', fontweight='bold')
        axes[1, 1].set_title('Mean Absolute Percentage Error (Lower is Better)', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        plt.suptitle(f'{self.ticker} - Model Performance Metrics', fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f'plots/{self.ticker}_metrics_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: plots/{self.ticker}_metrics_comparison.png")
        plt.close()
        
        print("\n" + "="*60)
        print(f"{self.ticker} - MODEL PERFORMANCE SUMMARY")
        print("="*60)
        print(metrics_df.to_string(index=False))
        
        return metrics_df
        
    def save_best_model(self):
        """Save the best performing model"""
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Find best model based on R² score
        best_model_name = max(self.models.items(), key=lambda x: x[1]['test_r2'])[0]
        best_model = self.models[best_model_name]['model']
        
        # Save model and scaler
        joblib.dump(best_model, f'models/{self.ticker}_best_model.pkl')
        joblib.dump(self.scaler, f'models/{self.ticker}_scaler.pkl')
        joblib.dump(self.feature_columns, f'models/{self.ticker}_features.pkl')
        
        print(f"\n✅ Best model saved: {best_model_name}")
        print(f"   R² Score: {self.models[best_model_name]['test_r2']:.4f}")
        print(f"   RMSE: ${self.models[best_model_name]['test_rmse']:.2f}")
        
        return best_model_name

if __name__ == "__main__":
    print("="*60)
    print("MACHINE LEARNING MODEL TRAINING")
    print("="*60)
    
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    all_metrics = []
    
    for ticker in tickers:
        print(f"\n\n{'='*60}")
        print(f"TRAINING MODELS FOR {ticker}")
        print(f"{'='*60}")
        
        # Initialize predictor
        predictor = StockPredictor(ticker=ticker)
        
        # Load data
        predictor.load_data()
        
        # Train models
        results = predictor.train_models()
        
        # Visualize results
        predictor.plot_results()
        metrics = predictor.plot_metrics_comparison()
        metrics['Ticker'] = ticker
        all_metrics.append(metrics)
        
        # Save best model
        best_model = predictor.save_best_model()
    
    # Create summary comparison
    print("\n\n" + "="*60)
    print("FINAL SUMMARY - BEST MODELS")
    print("="*60)
    
    summary_df = pd.concat(all_metrics, ignore_index=True)
    best_per_stock = summary_df.loc[summary_df.groupby('Ticker')['R² Score'].idxmax()]
    
    print(best_per_stock[['Ticker', 'Model', 'RMSE', 'MAE', 'R² Score', 'MAPE (%)']].to_string(index=False))
    
    print("\n" + "="*60)
    print("🎉 ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*60)
    print(f"\n📁 Outputs:")
    print(f"   - Model comparison plots in 'plots/' folder")
    print(f"   - Best models saved in 'models/' folder")