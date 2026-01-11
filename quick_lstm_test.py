import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

print("Starting quick LSTM test...")

# Load data
print("Loading AAPL data...")
df = pd.read_csv('data/AAPL_features.csv', index_col='Date', parse_dates=True)
data = df[['Close']].values

# Scale
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create sequences (simplified)
lookback = 30  # Reduced from 60
X, y = [], []
for i in range(lookback, len(scaled_data)):
    X.append(scaled_data[i-lookback:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X).reshape(-1, lookback, 1)
y = np.array(y)

# Split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Simpler model
print("\nBuilding simple LSTM...")
model = Sequential([
    LSTM(25, input_shape=(lookback, 1)),  # Much smaller
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
print(f"Parameters: {model.count_params()}")

# Train with progress
print("\nTraining (this should take 2-5 minutes)...")
history = model.fit(
    X_train, y_train,
    epochs=10,  # Just 10 epochs
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1  # Show progress!
)

print("\n✅ Training complete!")

# Make predictions
predictions = model.predict(X_test, verbose=0)
predictions = scaler.inverse_transform(predictions)
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE
rmse = np.sqrt(np.mean((predictions - actual)**2))
print(f"\nRMSE: ${rmse:.2f}")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(actual, label='Actual', linewidth=2)
plt.plot(predictions, label='Predicted', linewidth=2, linestyle='--')
plt.title('Quick LSTM Test - AAPL')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('plots/quick_lstm_test.png', dpi=150)
print("✅ Plot saved: plots/quick_lstm_test.png")
plt.close()

print("\n🎉 Done! If this worked, your setup is fine.")