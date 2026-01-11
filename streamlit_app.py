import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
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

# Map time range
range_map = {
    '1 Month': 30,
    '3 Months': 90,
    '6 Months': 180,
    '1 Year': 365,
    '2 Years': 730
}
days = range_map[time_range]

# Load data
@st.cache_data
def load_stock_data(ticker):
    df = pd.read_csv(f'data/{ticker}_data.csv', index_col='Date', parse_dates=True)
    return df

@st.cache_resource
def load_model(ticker):
    model = joblib.load(f'models/{ticker}_best_model.pkl')
    scaler = joblib.load(f'models/{ticker}_scaler.pkl')
    features = joblib.load(f'models/{ticker}_features.pkl')
    return model, scaler, features

# Load
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
        
        # Interactive Plotly chart
        fig = go.Figure()
        
        recent_df = df.tail(days)
        
        fig.add_trace(go.Candlestick(
            x=recent_df.index,
            open=recent_df['Open'],
            high=recent_df['High'],
            low=recent_df['Low'],
            close=recent_df['Close'],
            name='Price'
        ))
        
        fig.update_layout(
            title=f'{ticker} Stock Price',
            yaxis_title='Price ($)',
            xaxis_title='Date',
            height=500,
            template='plotly_white',
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume chart
        fig_vol = go.Figure()
        colors = ['green' if recent_df['Close'].iloc[i] >= recent_df['Open'].iloc[i] else 'red' 
                  for i in range(len(recent_df))]
        
        fig_vol.add_trace(go.Bar(
            x=recent_df.index,
            y=recent_df['Volume'],
            name='Volume',
            marker_color=colors
        ))
        
        fig_vol.update_layout(
            title='Trading Volume',
            yaxis_title='Volume',
            xaxis_title='Date',
            height=300,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_vol, use_container_width=True)
    
    with tab2:
        st.subheader("🔮 Next Day Price Prediction")
        
        # Load test data for prediction
        test_df = pd.read_csv(f'data/{ticker}_test.csv', index_col='Date', parse_dates=True)
        
        # Get latest features
        latest_features = test_df[features].iloc[-1:]
        latest_scaled = scaler.transform(latest_features)
        
        # Predict
        predicted_price = model.predict(latest_scaled)[0]
        current = test_df['Close'].iloc[-1]
        pred_change = predicted_price - current
        pred_pct = (pred_change / current) * 100
        
        # Display prediction
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Predicted Price", f"${predicted_price:.2f}")
        col2.metric("Expected Change", f"${pred_change:+.2f}")
        col3.metric("% Change", f"{pred_pct:+.2f}%")
        
        # Signal
        if pred_change > 0:
            st.success("📈 **SIGNAL: BUY** - Price expected to increase")
        else:
            st.error("📉 **SIGNAL: SELL** - Price expected to decrease")
        
        st.info(f"📅 Prediction for: {(datetime.now() + timedelta(days=1)).strftime('%B %d, %Y')}")
        
        st.markdown("---")
        
        # Historical predictions vs actual
        st.subheader("Historical Predictions vs Actual Prices")
        
        X_test = test_df[features]
        y_test = test_df['Target']
        X_test_scaled = scaler.transform(X_test)
        predictions = model.predict(X_test_scaled)
        
        fig_pred = go.Figure()
        
        fig_pred.add_trace(go.Scatter(
            x=test_df.index,
            y=y_test,
            mode='lines',
            name='Actual Price',
            line=dict(color='blue', width=2)
        ))
        
        fig_pred.add_trace(go.Scatter(
            x=test_df.index,
            y=predictions,
            mode='lines',
            name='Predicted Price',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig_pred.update_layout(
            title='Model Predictions vs Actual Prices (Test Set)',
            yaxis_title='Price ($)',
            xaxis_title='Date',
            height=450,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_pred, use_container_width=True)
    
    with tab3:
        st.subheader("📈 Technical Indicators")
        
        features_df = pd.read_csv(f'data/{ticker}_features.csv', index_col='Date', parse_dates=True)
        recent_features = features_df.tail(days)
        
        # Moving Averages
        st.markdown("#### Moving Averages")
        fig_ma = go.Figure()
        
        fig_ma.add_trace(go.Scatter(x=recent_features.index, y=recent_features['Close'], 
                                    name='Close Price', line=dict(color='black', width=2)))
        fig_ma.add_trace(go.Scatter(x=recent_features.index, y=recent_features['SMA_10'], 
                                    name='SMA 10', line=dict(color='orange', width=1.5)))
        fig_ma.add_trace(go.Scatter(x=recent_features.index, y=recent_features['SMA_20'], 
                                    name='SMA 20', line=dict(color='green', width=1.5)))
        fig_ma.add_trace(go.Scatter(x=recent_features.index, y=recent_features['SMA_50'], 
                                    name='SMA 50', line=dict(color='red', width=1.5)))
        
        fig_ma.update_layout(height=400, template='plotly_white', 
                            title='Price with Moving Averages')
        st.plotly_chart(fig_ma, use_container_width=True)
        
        # RSI and MACD side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### RSI (Relative Strength Index)")
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=recent_features.index, y=recent_features['RSI'], 
                                         mode='lines', name='RSI', 
                                         line=dict(color='purple', width=2)))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", 
                             annotation_text="Overbought (70)")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", 
                             annotation_text="Oversold (30)")
            
            fig_rsi.update_layout(height=300, template='plotly_white')
            st.plotly_chart(fig_rsi, use_container_width=True)
            
            # RSI interpretation
            current_rsi = recent_features['RSI'].iloc[-1]
            if current_rsi > 70:
                st.warning(f"⚠️ RSI: {current_rsi:.1f} - Overbought")
            elif current_rsi < 30:
                st.success(f"✅ RSI: {current_rsi:.1f} - Oversold")
            else:
                st.info(f"ℹ️ RSI: {current_rsi:.1f} - Neutral")
        
        with col2:
            st.markdown("#### MACD")
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=recent_features.index, y=recent_features['MACD'], 
                                          name='MACD', line=dict(color='blue', width=2)))
            fig_macd.add_trace(go.Scatter(x=recent_features.index, y=recent_features['MACD_Signal'], 
                                          name='Signal Line', line=dict(color='orange', width=2)))
            fig_macd.add_trace(go.Bar(x=recent_features.index, y=recent_features['MACD_Diff'], 
                                      name='Histogram', marker_color='gray', opacity=0.3))
            
            fig_macd.update_layout(height=300, template='plotly_white')
            st.plotly_chart(fig_macd, use_container_width=True)
            
            # MACD interpretation
            current_macd = recent_features['MACD'].iloc[-1]
            current_signal = recent_features['MACD_Signal'].iloc[-1]
            if current_macd > current_signal:
                st.success("✅ MACD: Bullish Signal")
            else:
                st.error("⚠️ MACD: Bearish Signal")
        
        # Bollinger Bands
        st.markdown("#### Bollinger Bands")
        fig_bb = go.Figure()
        
        fig_bb.add_trace(go.Scatter(x=recent_features.index, y=recent_features['BB_High'], 
                                    name='Upper Band', line=dict(color='red', width=1, dash='dash')))
        fig_bb.add_trace(go.Scatter(x=recent_features.index, y=recent_features['BB_Mid'], 
                                    name='Middle Band', line=dict(color='blue', width=1)))
        fig_bb.add_trace(go.Scatter(x=recent_features.index, y=recent_features['BB_Low'], 
                                    name='Lower Band', line=dict(color='green', width=1, dash='dash')))
        fig_bb.add_trace(go.Scatter(x=recent_features.index, y=recent_features['Close'], 
                                    name='Close Price', line=dict(color='black', width=2)))
        
        fig_bb.update_layout(height=400, template='plotly_white')
        st.plotly_chart(fig_bb, use_container_width=True)
    
    with tab4:
        st.subheader("🤖 Model Performance Metrics")
        
        # Calculate metrics
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
        col1.metric("RMSE", f"${rmse:.2f}", help="Root Mean Squared Error - Lower is better")
        col2.metric("MAE", f"${mae:.2f}", help="Mean Absolute Error - Lower is better")
        col3.metric("R² Score", f"{r2:.4f}", help="Coefficient of Determination - Higher is better (max 1.0)")
        col4.metric("MAPE", f"{mape:.2f}%", help="Mean Absolute Percentage Error - Lower is better")
        
        # Accuracy interpretation
        if r2 > 0.9:
            st.success(f"✅ **Excellent Model Performance!** The model explains {r2*100:.1f}% of price variance.")
        elif r2 > 0.8:
            st.info(f"✔️ **Good Model Performance.** The model explains {r2*100:.1f}% of price variance.")
        else:
            st.warning(f"⚠️ **Moderate Model Performance.** The model explains {r2*100:.1f}% of price variance.")
        
        st.markdown("---")
        
        # Error distribution
        st.subheader("📊 Prediction Error Distribution")
        errors = predictions - y_test
        
        fig_errors = go.Figure()
        fig_errors.add_trace(go.Histogram(x=errors, nbinsx=30, 
                                          marker_color='steelblue', name='Error Distribution'))
        fig_errors.update_layout(
            title='Distribution of Prediction Errors',
            xaxis_title='Error ($)',
            yaxis_title='Frequency',
            height=350,
            template='plotly_white'
        )
        st.plotly_chart(fig_errors, use_container_width=True)
        
        st.markdown("---")
        
        # Model information
        st.subheader("ℹ️ Model Information")
        
        train_df = pd.read_csv(f'data/{ticker}_train.csv', index_col='Date', parse_dates=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_info = {
                'Model Type': type(model).__name__,
                'Number of Features': len(features),
                'Training Samples': len(train_df),
                'Test Samples': len(test_df)
            }
            
            for key, value in model_info.items():
                st.metric(key, value)
        
        with col2:
            data_info = {
                'Data Start Date': df.index.min().strftime('%Y-%m-%d'),
                'Data End Date': df.index.max().strftime('%Y-%m-%d'),
                'Total Days': len(df),
                'Train/Test Split': '80/20'
            }
            
            for key, value in data_info.items():
                st.metric(key, value)

except FileNotFoundError as e:
    st.error(f"❌ Error loading data for {ticker}. Please make sure you've run the training scripts first.")
    st.info("Run these commands in order: `python data_collection.py`, `python feature_engineering.py`, `python train_models.py`")

# Footer
st.markdown("---")
st.markdown("### ⚠️ Disclaimer")
st.warning("This is an educational project for learning machine learning and data science. Stock price predictions are based on historical patterns and technical indicators. This is NOT financial advice. Always do your own research and consult with financial advisors before making investment decisions.")

st.markdown("---")
st.markdown("**Built with ❤️ by Animesh Singh** | MIT Manipal | B.Tech Computer Science & Financial Technology")
st.markdown("📧 animeshsinghmanu@gmail.com | [GitHub](https://github.com/aaannimeshsingh)")