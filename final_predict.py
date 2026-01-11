import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

print("="*60)
print("STOCK PRICE PREDICTIONS FOR TOMORROW")
print("="*60)
print(f"📅 Today: {datetime.now().strftime('%B %d, %Y')}")
print(f"🔮 Predicting for: {(datetime.now() + timedelta(days=1)).strftime('%B %d, %Y')}")
print("="*60)

tickers = ['AAPL', 'GOOGL', 'MSFT']
predictions = []

for ticker in tickers:
    # Load latest data
    df = pd.read_csv(f'data/{ticker}_test.csv', index_col='Date', parse_dates=True)
    
    # Load model
    model = joblib.load(f'models/{ticker}_best_model.pkl')
    scaler = joblib.load(f'models/{ticker}_scaler.pkl')
    features = joblib.load(f'models/{ticker}_features.pkl')
    
    # Get latest features
    latest = df[features].iloc[-1:]
    latest_scaled = scaler.transform(latest)
    
    # Predict
    predicted_price = model.predict(latest_scaled)[0]
    current_price = df['Close'].iloc[-1]
    change = predicted_price - current_price
    pct_change = (change / current_price) * 100
    
    predictions.append({
        'Stock': ticker,
        'Current Price': f'${current_price:.2f}',
        'Predicted Price': f'${predicted_price:.2f}',
        'Change': f'${change:+.2f}',
        '% Change': f'{pct_change:+.2f}%',
        'Signal': '📈 BUY' if change > 0 else '📉 SELL'
    })
    
    print(f"\n{ticker}:")
    print(f"  Current: ${current_price:.2f}")
    print(f"  Predicted: ${predicted_price:.2f}")
    print(f"  Change: ${change:+.2f} ({pct_change:+.2f}%)")
    print(f"  Signal: {'📈 BUY' if change > 0 else '📉 SELL'}")

# Summary table
print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
df_pred = pd.DataFrame(predictions)
print(df_pred.to_string(index=False))

print("\n" + "="*60)
print("⚠️  DISCLAIMER")
print("="*60)
print("These are ML predictions based on historical patterns.")
print("Not financial advice. Do your own research!")
print("="*60)