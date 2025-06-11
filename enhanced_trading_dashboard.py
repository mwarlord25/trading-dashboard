import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Technical Indicator Functions
# -----------------------------
def add_indicators(df):
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Upper_BB'] = df['Close'].rolling(window=20).mean() + 2 * df['Close'].rolling(window=20).std()
    df['Lower_BB'] = df['Close'].rolling(window=20).mean() - 2 * df['Close'].rolling(window=20).std()
    return df

# -----------------------------
# Predictive Model Integration
# -----------------------------
def train_predictive_model():
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(1200)) + 100
    dates = pd.date_range(end=pd.Timestamp.today(), periods=1200)
    df = pd.DataFrame({'Date': dates, 'Close': prices})
    df['Open'] = df['Close'] + np.random.randn(1200)
    df['High'] = df[['Open', 'Close']].max(axis=1) + np.random.rand(1200)
    df['Low'] = df[['Open', 'Close']].min(axis=1) - np.random.rand(1200)
    df['Volume'] = np.random.randint(1000000, 5000000, size=1200)
    df.set_index('Date', inplace=True)
    df = add_indicators(df)
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df.dropna(inplace=True)
    features = ['SMA_10', 'SMA_50', 'RSI', 'MACD', 'Signal_Line', 'Upper_BB', 'Lower_BB']
    X = df[features]
    y = df['Target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler, features

model, scaler, feature_cols = train_predictive_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(layout="wide")
st.title("📈 Enhanced Trading Dashboard with Predictive Signals")

symbols = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Tesla (TSLA)": "TSLA",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD"
}

symbol_name = st.sidebar.selectbox("Select Symbol", list(symbols.keys()))
symbol = symbols[symbol_name]
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

strategies = st.sidebar.multiselect("Select Strategies", ["SMA Crossover", "RSI", "MACD", "Bollinger Bands"])

@st.cache_data
def load_data(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

data = load_data(symbol, start_date, end_date)

if data.empty:
    st.warning("No data available for the selected symbol and date range.")
else:
    data = add_indicators(data)

    # Apply strategies
    if "SMA Crossover" in strategies:
        data['SMA_Signal'] = np.where(data['SMA_10'] > data['SMA_50'], 1, 0)
    if "RSI" in strategies:
        data['RSI_Signal'] = np.where(data['RSI'] < 30, 1, np.where(data['RSI'] > 70, -1, 0))
    if "MACD" in strategies:
        data['MACD_Signal'] = np.where(data['MACD'] > data['Signal_Line'], 1, 0)
    if "Bollinger Bands" in strategies:
        data['BB_Signal'] = np.where(data['Close'] < data['Lower_BB'], 1, np.where(data['Close'] > data['Upper_BB'], -1, 0))

    # Predictive Signal
    latest = data.dropna().iloc[-1:]
    if not latest.empty:
        X_pred = scaler.transform(latest[feature_cols])
        prediction = model.predict(X_pred)[0]
        signal = "Buy" if prediction == 1 else "Sell"
        st.subheader("📊 Predicted Signal")
        st.markdown(f"**Signal**: `{signal}`")
        st.markdown(f"**Time**: `{latest.index[0].strftime('%Y-%m-%d')}`")

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))
    fig.update_layout(title=f"{symbol} Price Chart", xaxis_rangeslider_visible=False, template="plotly_dark")

    if "SMA Crossover" in strategies:
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_10'], mode='lines', name='SMA 10'))
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='SMA 50'))
    if "Bollinger Bands" in strategies:
        fig.add_trace(go.Scatter(x=data.index, y=data['Upper_BB'], line=dict(dash='dot'), name='Upper BB'))
        fig.add_trace(go.Scatter(x=data.index, y=data['Lower_BB'], line=dict(dash='dot'), name='Lower BB'))

    st.plotly_chart(fig, use_container_width=True)

