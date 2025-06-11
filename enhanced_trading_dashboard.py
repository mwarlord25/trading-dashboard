import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to fetch data
@st.cache_data
def get_data(symbol, interval, period='60d'):
    try:
        data = yf.download(symbol, interval=interval, period=period, progress=False)
        if data.empty:
            return None
        data.dropna(inplace=True)
        return data
    except Exception as e:
        return None

# Strategy functions
def sma_strategy(data):
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['Signal_SMA'] = np.where(data['SMA20'] > data['SMA50'], 'Buy', 'Sell')
    return data

def rsi_strategy(data, period=14):
    delta = data['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['Signal_RSI'] = np.where(data['RSI'] < 30, 'Buy', np.where(data['RSI'] > 70, 'Sell', 'Hold'))
    return data

def macd_strategy(data):
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['Signal_MACD'] = np.where(data['MACD'] > data['Signal_Line'], 'Buy', 'Sell')
    return data

def bollinger_strategy(data):
    sma = data['Close'].rolling(window=20).mean()
    std = data['Close'].rolling(window=20).std()
    data['Upper'] = sma + (2 * std)
    data['Lower'] = sma - (2 * std)
    data['Signal_BB'] = np.where(data['Close'] < data['Lower'], 'Buy', np.where(data['Close'] > data['Upper'], 'Sell', 'Hold'))
    return data

# Streamlit layout
st.set_page_config(layout="wide")
st.sidebar.title("Trading Dashboard")

# Inputs
symbol = st.sidebar.selectbox("Select Symbol", ['GBPUSD=X', 'EURUSD=X', 'AAPL', 'GOOG', 'MSFT'])
interval = st.sidebar.selectbox("Select Interval", ['1m', '5m', '10m', '15m', '30m', '1h', '4h', '1d', '1wk', '1mo'])
strategies = st.sidebar.multiselect("Select Strategies", ['SMA Crossover', 'RSI', 'MACD', 'Bollinger Bands'], default=['SMA Crossover'])

# Fetch data
data = get_data(symbol, interval)

# Main area
col1, col2 = st.columns([1, 3])

with col1:
    st.write("### Selected Symbol:", symbol)
    st.write("### Interval:", interval)
    if data is None:
        st.error("Failed to retrieve data. Please try a different symbol or interval.")
    else:
        st.success("Live data loaded successfully.")
        st.write("### Latest Price:", data['Close'].iloc[-1])

with col2:
    if data is not None:
        if 'SMA Crossover' in strategies:
            data = sma_strategy(data)
        if 'RSI' in strategies:
            data = rsi_strategy(data)
        if 'MACD' in strategies:
            data = macd_strategy(data)
        if 'Bollinger Bands' in strategies:
            data = bollinger_strategy(data)

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data['Close'], label='Close Price')
        if 'SMA Crossover' in strategies:
            ax.plot(data['SMA20'], label='SMA 20')
            ax.plot(data['SMA50'], label='SMA 50')
        if 'Bollinger Bands' in strategies:
            ax.plot(data['Upper'], linestyle='--', color='gray', label='Upper Band')
            ax.plot(data['Lower'], linestyle='--', color='gray', label='Lower Band')
        ax.set_title(f"{symbol} Price Chart")
        ax.legend()
        st.pyplot(fig)

        # Alerts
        st.subheader("Trade Alerts")
        alerts = []
        if 'SMA Crossover' in strategies:
            if data['Signal_SMA'].iloc[-1] == 'Buy':
                alerts.append("SMA Crossover: Buy Signal")
            elif data['Signal_SMA'].iloc[-1] == 'Sell':
                alerts.append("SMA Crossover: Sell Signal")
        if 'RSI' in strategies:
            if data['Signal_RSI'].iloc[-1] == 'Buy':
                alerts.append("RSI: Buy Signal (Oversold)")
            elif data['Signal_RSI'].iloc[-1] == 'Sell':
                alerts.append("RSI: Sell Signal (Overbought)")
        if 'MACD' in strategies:
            if data['Signal_MACD'].iloc[-1] == 'Buy':
                alerts.append("MACD: Buy Signal")
            elif data['Signal_MACD'].iloc[-1] == 'Sell':
                alerts.append("MACD: Sell Signal")
        if 'Bollinger Bands' in strategies:
            if data['Signal_BB'].iloc[-1] == 'Buy':
                alerts.append("Bollinger Bands: Buy Signal (Below Lower Band)")
            elif data['Signal_BB'].iloc[-1] == 'Sell':
                alerts.append("Bollinger Bands: Sell Signal (Above Upper Band)")

        if alerts:
            for alert in alerts:
                st.warning(alert)
        else:
            st.info("No trade signals at the moment.")




# --- Predictive Signal with Entry and Exit Time (GMT+0) ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def generate_features(data):
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Upper_BB'] = data['Close'].rolling(window=20).mean() + 2 * data['Close'].rolling(window=20).std()
    data['Lower_BB'] = data['Close'].rolling(window=20).mean() - 2 * data['Close'].rolling(window=20).std()
    data = data.dropna()
    return data

def train_predictive_model(data):
    data = generate_features(data.copy())
    data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    features = ['SMA_10', 'SMA_50', 'RSI', 'MACD', 'Upper_BB', 'Lower_BB']
    data = data.dropna()
    if len(data) < 10:
        return None, None, None
    X = data[features]
    y = data['Target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    latest = X.iloc[[-1]]
    latest_scaled = scaler.transform(latest)
    prediction = model.predict(latest_scaled)[0]
    signal = 'Buy' if prediction == 1 else 'Sell'
    entry_time = data.index[-1].tz_localize('UTC')
    exit_time = entry_time + timedelta(minutes=30)
    return signal, entry_time.strftime('%Y-%m-%d %H:%M GMT+0'), exit_time.strftime('%Y-%m-%d %H:%M GMT+0')

# Example usage in Streamlit
import streamlit as st
if 'data' in locals() and not data.empty:
    signal, entry_time, exit_time = train_predictive_model(data)
    if signal:
        st.subheader("📈 Predictive Trade Signal")
        st.markdown(f"**Signal**: {signal}")
        st.markdown(f"**Entry Time**: {entry_time}")
        st.markdown(f"**Suggested Exit Time**: {exit_time}")
