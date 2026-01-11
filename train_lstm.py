import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

class LSTMStockPredictor:
    def __init__(self, ticker='AAPL', lookback=60):
        self.ticker = ticker
        self.lookback = lookback  # Use 60 days of history to predict next day
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        
    def load_and_prepare_data(self):
        """Load and prepare data for LSTM"""
        print(f"\n📊 Loading data for {self.ticker}...")
        
        # Load the full dataset with features
        df = pd.read_csv(f'data/{self.ticker}_features.csv', index_col='Date', parse_dates=True)
        
        # Use only Close price for simplicity (univariate LSTM)
        data = df[['Close']].values
        
        # Split into train/test (80/20)
        split_idx = int(len(data) * 0.8)
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        # Scale the data
        train_scaled = self.scaler.fit_transform(train_data)
        test_scaled = self.scaler.transform(test_data)
        
        # Create sequences
        X_train, y_train = self.create_sequences(train_scaled)
        X_test, y_test = self.create_sequences(test_scaled)
        
        print(f"✅ Data prepared")
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Test samples: {X_test.shape[0]}")
        print(f"   Sequence length: {self.lookback} days")
        
        return X_train, y_train, X_test, y_test, test_data
    
    def create_sequences(self, data):
        """Create sequences for LSTM"""
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback:i, 0])
            y.append(data[i, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def build_model(self):
        """Build LSTM model"""
        print("\n🏗️  Building LSTM model...")
        
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(self.lookback, 1)),
            Dropout(0.2),
            
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            
            LSTM(units=50),
            Dropout(0.2),
            
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        print("✅ Model built")
        print(f"   Total parameters: {model.count_params():,}")
        
        self.model = model
        return model
    
    def train_model(self, X_train, y_train, X_test, y_test):
        """Train the LSTM model"""
        print("\n🚀 Training LSTM model...")
        print("   This may take 2-3 minutes...")
        
        # Early stopping to prevent overfitting
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train
        history = self.model.fit(
        X_train, y_train,
        epochs=20,        # Changed from 50 to 20
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1         # Changed from 0 to 1 - THIS IS CRITICAL
        )
        
        print(f"✅ Training complete!")
        print(f"   Epochs trained: {len(history.history['loss'])}")
        print(f"   Final train loss: {history.history['loss'][-1]:.6f}")
        print(f"   Final val loss: {history.history['val_loss'][-1]:.6f}")
        
        return history
    
    def evaluate_model(self, X_test, y_test, test_data):
        """Evaluate model performance"""
        print("\n📈 Evaluating model...")
        
        # Make predictions
        predictions_scaled = self.model.predict(X_test, verbose=0)
        
        # Inverse transform to get actual prices
        predictions = self.scaler.inverse_transform(predictions_scaled)
        actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        mae = mean_absolute_error(actual, predictions)
        r2 = r2_score(actual, predictions)
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        
        print(f"\n📊 LSTM Model Performance:")
        print(f"   RMSE: ${rmse:.2f}")
        print(f"   MAE: ${mae:.2f}")
        print(f"   R² Score: {r2:.4f}")
        print(f"   MAPE: {mape:.2f}%")
        
        return predictions, actual, rmse, mae, r2, mape
    
    def plot_results(self, predictions, actual, train_size):
        """Plot predictions vs actual"""
        plt.figure(figsize=(16, 7))
        
        # Create date range for x-axis
        dates = pd.date_range(start=pd.Timestamp.now() - pd.Timedelta(days=len(actual)), 
                              periods=len(actual), freq='D')
        
        plt.plot(dates, actual, label='Actual Price', linewidth=2, color='#2E86AB', alpha=0.8)
        plt.plot(dates, predictions, label='LSTM Predicted', linewidth=2, color='#F18F01', 
                linestyle='--', alpha=0.8)
        
        plt.title(f'{self.ticker} - LSTM Stock Price Prediction', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Date', fontsize=14, fontweight='bold')
        plt.ylabel('Stock Price ($)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(f'plots/{self.ticker}_lstm_prediction.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved: plots/{self.ticker}_lstm_prediction.png")
        plt.close()
    
    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 5))
        
        plt.plot(history.history['loss'], label='Training Loss', linewidth=2, color='#2E86AB')
        plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='#F18F01')
        
        plt.title(f'{self.ticker} - LSTM Training History', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(f'plots/{self.ticker}_lstm_training.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: plots/{self.ticker}_lstm_training.png")
        plt.close()
    
    def save_model(self):
        """Save LSTM model"""
        os.makedirs('models', exist_ok=True)
        
        self.model.save(f'models/{self.ticker}_lstm_model.keras')
        joblib.dump(self.scaler, f'models/{self.ticker}_lstm_scaler.pkl')
        
        print(f"\n✅ LSTM model saved:")
        print(f"   - models/{self.ticker}_lstm_model.keras")
        print(f"   - models/{self.ticker}_lstm_scaler.pkl")

if __name__ == "__main__":
    print("="*60)
    print("LSTM DEEP LEARNING MODEL TRAINING")
    print("="*60)
    
    tickers = ['AAPL']
    lstm_results = []
    
    for ticker in tickers:
        print(f"\n\n{'='*60}")
        print(f"TRAINING LSTM FOR {ticker}")
        print(f"{'='*60}")
        
        # Initialize predictor
        predictor = LSTMStockPredictor(ticker=ticker, lookback=60)
        
        # Load and prepare data
        X_train, y_train, X_test, y_test, test_data = predictor.load_and_prepare_data()
        
        # Build model
        model = predictor.build_model()
        
        # Train model
        history = predictor.train_model(X_train, y_train, X_test, y_test)
        
        # Evaluate
        predictions, actual, rmse, mae, r2, mape = predictor.evaluate_model(X_test, y_test, test_data)
        
        # Plot results
        predictor.plot_results(predictions, actual, len(X_train))
        predictor.plot_training_history(history)
        
        # Save model
        predictor.save_model()
        
        # Store results
        lstm_results.append({
            'Ticker': ticker,
            'RMSE': rmse,
            'MAE': mae,
            'R² Score': r2,
            'MAPE (%)': mape
        })
    
    # Summary
    print("\n\n" + "="*60)
    print("LSTM MODELS - FINAL SUMMARY")
    print("="*60)
    
    results_df = pd.DataFrame(lstm_results)
    print(results_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("🎉 ALL LSTM MODELS TRAINED SUCCESSFULLY!")
    print("="*60)
    print(f"\n📁 Outputs:")
    print(f"   - LSTM prediction plots in 'plots/' folder")
    print(f"   - LSTM models saved in 'models/' folder")