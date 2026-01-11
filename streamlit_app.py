import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Check if data exists, if not, run setup
def check_and_setup():
    """Check if data exists, if not run setup"""
    if not os.path.exists('data') or not os.path.exists('data/AAPL_data.csv'):
        st.info("🚀 **First time setup detected!** Downloading data and training models...")
        st.info("⏳ This will take 2-3 minutes. Please wait...")
        
        with st.spinner("Setting up your dashboard..."):
            try:
                run_complete_setup()
                st.success("✅ Setup complete! Refreshing...")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Setup failed: {str(e)}")
                st.info("Please check your internet connection and refresh the page.")
                st.stop()

def run_complete_setup():
    """Run complete data collection and model training"""
    import yfinance as yf
    from ta.trend import SMAIndicator, EMAIndicator, MACD
    from ta.momentum import RSIIndicator
    from ta.volatility import BollingerBands
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    import joblib
    
    # Create directories
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)
    
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    
    # Download data
    for ticker in tickers:
        st.write(f"📥 Downloading {ticker} data...")
        df = yf.download(ticker, period='2y', progress=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df.index = pd.to_datetime(df.index)
        df.to_csv(f'data/{ticker}_data.csv')
    
    # Feature engineering and training
    for ticker in tickers:
        st.write(f"🔧 Processing {ticker}...")
        
        df = pd.read_csv(f'data/{ticker}_data.csv', index_col='Date', parse_dates=True)
        
        # Add ALL technical indicators
        df['SMA_10'] = SMAIndicator(close=df['Close'], window=10).sma_indicator()
        df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
        df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
        df['EMA_10'] = EMAIndicator(close=df['Close'], window=10).ema_indicator()
        df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Diff'] = macd.macd_diff()
        
        bb = BollingerBands(close=df['Close'])
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        df['BB_Mid'] = bb.bollinger_mavg()
        
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
        df['Price_Change_10d'] = df['Close'].pct_change(periods=10)
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean()
        df['Daily_Return'] = (df['Close'] - df['Open']) / df['Open']
        df['HL_Range'] = df['High'] - df['Low']
        df['HL_Pct'] = (df['High'] - df['Low']) / df['Close']
        
        # Save features
        df.to_csv(f'data/{ticker}_features.csv')
        
        # Create target
        df['Target'] = df['Close'].shift(-1)
        df['Target_Direction'] = (df['Target'] > df['Close']).astype(int)
        df = df.dropna()
        
        # Split
        split_idx = int(len(df) * 0.8)
        train_df = df[:split_idx]
        test_df = df[split_idx:]
        
        # Save splits
        train_df.to_csv(f'data/{ticker}_train.csv')
        test_df.to_csv(f'data/{ticker}_test.csv')
        
        # Train model
        st.write(f"🤖 Training model for {ticker}...")
        
        exclude_cols = ['Target', 'Target_Direction', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        feature_columns = [col for col in train_df.columns if col not in exclude_cols]
        
        X_train = train_df[feature_columns]
        y_train = train_df['Target']
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Try Ridge, fall back to Linear
        try:
            model = Ridge(alpha=1.0)
            model.fit(X_train_scaled, y_train)
        except:
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
        
        # Save
        joblib.dump(model, f'models/{ticker}_best_model.pkl')
        joblib.dump(scaler, f'models/{ticker}_scaler.pkl')
        joblib.dump(feature_columns, f'models/{ticker}_features.pkl')

# Run setup check
check_and_setup()

# Now import everything else
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Page config
st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")

# Title
st.title("📈 AI Stock Market Prediction Dashboard")
st.markdown("### Machine Learning-Powered Stock Price Forecasting")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Settings")
ticker = st.sidebar.selectbox("Select Stock", ['AAPL', 'GOOGL', 'MSFT'], index=0)
time_range = st.sidebar.selectbox("Time Range", ['1 Month', '3 Months', '6 Months', '1 Year', '2 Years'], index=2)

range_map = {'1 Month': 30, '3 Months': 90, '6 Months': 180, '1 Year': 365, '2 Years': 730}
days = range_map[time_range]

# Load data
@st.cache_data
def load_stock_data(ticker):
    return pd.read_csv(f'data/{ticker}_data.csv', index_col='Date', parse_dates=True)

@st.cache_resource
def load_model(ticker):
    model = joblib.load(f'models/{ticker}_best_model.pkl')
    scaler = joblib.load(f'models/{ticker}_scaler.pkl')
    features = joblib.load(f'models/{ticker}_features.pkl')
    return model, scaler, features

try:
    df = load_stock_data(ticker)
    model, scaler, features = load_model(ticker)
    
    # Display current info
    col1, col2, col3, col4 = st.columns(4)
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2]
    change = current_price - prev_price
    pct_change = (change / prev_price) * 100
    
    col1.metric("Current Price", f"${current_price:.2f}", f"{pct_change:+.2f}%")
    col2.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
    col3.metric("High (Today)", f"${df['High'].iloc[-1]:.2f}")
    col4.metric("Low (Today)", f"${df['Low'].iloc[-1]:.2f}")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Price Chart", "🔮 Predictions", "📈 Technical Analysis", "🤖 Model Performance"])
    
    with tab1:
        st.subheader(f"{ticker} Price History")
        
        fig = go.Figure()
        recent_df = df.tail(days)
        
        fig.add_trace(go.Candlestick(
            x=recent_df.index, open=recent_df['Open'], high=recent_df['High'],
            low=recent_df['Low'], close=recent_df['Close'], name='Price'
        ))
        
        fig.update_layout(title=f'{ticker} Stock Price', yaxis_title='Price ($)',
                         xaxis_title='Date', height=500, template='plotly_white',
                         xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume
        fig_vol = go.Figure()
        colors = ['green' if recent_df['Close'].iloc[i] >= recent_df['Open'].iloc[i] else 'red' 
                  for i in range(len(recent_df))]
        fig_vol.add_trace(go.Bar(x=recent_df.index, y=recent_df['Volume'], name='Volume', marker_color=colors))
        fig_vol.update_layout(title='Trading Volume', yaxis_title='Volume', xaxis_title='Date',
                             height=300, template='plotly_white')
        st.plotly_chart(fig_vol, use_container_width=True)
    
    with tab2:
        st.subheader("🔮 Next Day Price Prediction")
        
        test_df = pd.read_csv(f'data/{ticker}_test.csv', index_col='Date', parse_dates=True)
        latest_features = test_df[features].iloc[-1:]
        latest_scaled = scaler.transform(latest_features)
        
        predicted_price = model.predict(latest_scaled)[0]
        current = test_df['Close'].iloc[-1]
        pred_change = predicted_price - current
        pred_pct = (pred_change / current) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Price", f"${predicted_price:.2f}")
        col2.metric("Expected Change", f"${pred_change:+.2f}")
        col3.metric("% Change", f"{pred_pct:+.2f}%")
        
        if pred_change > 0:
            st.success("📈 **SIGNAL: BUY** - Price expected to increase")
        else:
            st.error("📉 **SIGNAL: SELL** - Price expected to decrease")
        
        st.info(f"📅 Prediction for: {(datetime.now() + timedelta(days=1)).strftime('%B %d, %Y')}")
        st.markdown("---")
        
        st.subheader("Historical Predictions vs Actual Prices")
        X_test = test_df[features]
        y_test = test_df['Target']
        X_test_scaled = scaler.transform(X_test)
        predictions = model.predict(X_test_scaled)
        
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=test_df.index, y=y_test, mode='lines',
                                      name='Actual Price', line=dict(color='blue', width=2)))
        fig_pred.add_trace(go.Scatter(x=test_df.index, y=predictions, mode='lines',
                                      name='Predicted Price', line=dict(color='red', width=2, dash='dash')))
        fig_pred.update_layout(title='Model Predictions vs Actual Prices (Test Set)',
                              yaxis_title='Price ($)', xaxis_title='Date',
                              height=450, template='plotly_white')
        st.plotly_chart(fig_pred, use_container_width=True)
    
    with tab3:
        st.subheader("📈 Technical Indicators")
        
        features_df = pd.read_csv(f'data/{ticker}_features.csv', index_col='Date', parse_dates=True)
        recent_features = features_df.tail(days)
        
        # Moving Averages
        st.markdown("#### Moving Averages")
        fig_ma = go.Figure()
        ma_data = recent_features[['Close', 'SMA_10', 'SMA_20', 'SMA_50']].dropna()
        
        if len(ma_data) > 0:
            fig_ma.add_trace(go.Scatter(x=ma_data.index, y=ma_data['Close'], 
                                        name='Close Price', line=dict(color='black', width=2)))
            fig_ma.add_trace(go.Scatter(x=ma_data.index, y=ma_data['SMA_10'], 
                                        name='SMA 10', line=dict(color='orange', width=1.5)))
            fig_ma.add_trace(go.Scatter(x=ma_data.index, y=ma_data['SMA_20'], 
                                        name='SMA 20', line=dict(color='green', width=1.5)))
            fig_ma.add_trace(go.Scatter(x=ma_data.index, y=ma_data['SMA_50'], 
                                        name='SMA 50', line=dict(color='red', width=1.5)))
        
        fig_ma.update_layout(height=400, template='plotly_white', title='Price with Moving Averages')
        st.plotly_chart(fig_ma, use_container_width=True)
        
        # RSI and MACD
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### RSI")
            rsi_data = recent_features[['RSI']].dropna()
            
            if len(rsi_data) > 0:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=rsi_data.index, y=rsi_data['RSI'], 
                                             mode='lines', name='RSI', line=dict(color='purple', width=2)))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                fig_rsi.update_layout(height=300, template='plotly_white')
                st.plotly_chart(fig_rsi, use_container_width=True)
                
                current_rsi = rsi_data['RSI'].iloc[-1]
                if current_rsi > 70:
                    st.warning(f"⚠️ RSI: {current_rsi:.1f} - Overbought")
                elif current_rsi < 30:
                    st.success(f"✅ RSI: {current_rsi:.1f} - Oversold")
                else:
                    st.info(f"ℹ️ RSI: {current_rsi:.1f} - Neutral")
        
        with col2:
            st.markdown("#### MACD")
            macd_data = recent_features[['MACD', 'MACD_Signal', 'MACD_Diff']].dropna()
            
            if len(macd_data) > 0:
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=macd_data.index, y=macd_data['MACD'], 
                                              name='MACD', line=dict(color='blue', width=2)))
                fig_macd.add_trace(go.Scatter(x=macd_data.index, y=macd_data['MACD_Signal'], 
                                              name='Signal', line=dict(color='orange', width=2)))
                fig_macd.add_trace(go.Bar(x=macd_data.index, y=macd_data['MACD_Diff'], 
                                          name='Histogram', marker_color='gray', opacity=0.3))
                fig_macd.update_layout(height=300, template='plotly_white')
                st.plotly_chart(fig_macd, use_container_width=True)
                
                current_macd = macd_data['MACD'].iloc[-1]
                current_signal = macd_data['MACD_Signal'].iloc[-1]
                if current_macd > current_signal:
                    st.success("✅ MACD: Bullish")
                else:
                    st.error("⚠️ MACD: Bearish")
        
        # Bollinger Bands
        st.markdown("#### Bollinger Bands")
        bb_data = recent_features[['Close', 'BB_High', 'BB_Mid', 'BB_Low']].dropna()
        
        if len(bb_data) > 0:
            fig_bb = go.Figure()
            fig_bb.add_trace(go.Scatter(x=bb_data.index, y=bb_data['BB_High'], 
                                        name='Upper Band', line=dict(color='red', width=1, dash='dash')))
            fig_bb.add_trace(go.Scatter(x=bb_data.index, y=bb_data['BB_Mid'], 
                                        name='Middle Band', line=dict(color='blue', width=1)))
            fig_bb.add_trace(go.Scatter(x=bb_data.index, y=bb_data['BB_Low'], 
                                        name='Lower Band', line=dict(color='green', width=1, dash='dash')))
            fig_bb.add_trace(go.Scatter(x=bb_data.index, y=bb_data['Close'], 
                                        name='Close', line=dict(color='black', width=2)))
            fig_bb.update_layout(height=400, template='plotly_white')
            st.plotly_chart(fig_bb, use_container_width=True)
    
    with tab4:
        st.subheader("🤖 Model Performance Metrics")
        
        test_df = pd.read_csv(f'data/{ticker}_test.csv', index_col='Date', parse_dates=True)
        X_test = test_df[features]
        y_test = test_df['Target']
        X_test_scaled = scaler.transform(X_test)
        predictions = model.predict(X_test_scaled)
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("RMSE", f"${rmse:.2f}")
        col2.metric("MAE", f"${mae:.2f}")
        col3.metric("R² Score", f"{r2:.4f}")
        col4.metric("MAPE", f"{mape:.2f}%")
        
        if r2 > 0.9:
            st.success(f"✅ Excellent! Model explains {r2*100:.1f}% of variance.")
        elif r2 > 0.8:
            st.info(f"✔️ Good performance. {r2*100:.1f}% variance explained.")
        else:
            st.warning(f"⚠️ Moderate performance. {r2*100:.1f}% variance explained.")

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    if st.button("🔄 Reset Data"):
        import shutil
        shutil.rmtree('data', ignore_errors=True)
        shutil.rmtree('models', ignore_errors=True)
        st.rerun()

# Footer
st.markdown("---")
st.markdown("### ⚠️ Disclaimer")
st.warning("Educational project only. Not financial advice.")
st.markdown("**Built with ❤️ by Animesh Singh** | MIT Manipal | [GitHub](https://github.com/aaannimeshsingh)")