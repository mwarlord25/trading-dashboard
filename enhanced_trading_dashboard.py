
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("📈 Trading Dashboard with Strategy Overlays")

# Sidebar inputs
symbol = st.sidebar.text_input("Enter Symbol (e.g., AAPL, EURUSD=X, GBPUSD=X)", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=180))
end_date = st.sidebar.date_input("End Date", datetime.now())
strategies = st.sidebar.multiselect("Select Strategies", ["SMA Crossover", "RSI", "MACD", "Bollinger Bands"])

# Fetch data
@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    return df

data = load_data(symbol, start_date, end_date)

# Strategy functions
def apply_sma(data):
    data["SMA20"] = data["Close"].rolling(window=20).mean()
    data["SMA50"] = data["Close"].rolling(window=50).mean()
    return data

def apply_rsi(data, window=14):
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))
    return data

def apply_macd(data):
    exp1 = data["Close"].ewm(span=12, adjust=False).mean()
    exp2 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = exp1 - exp2
    data["Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    return data

def apply_bollinger(data):
    sma = data["Close"].rolling(window=20).mean()
    std = data["Close"].rolling(window=20).std()
    data["Upper"] = sma + 2 * std
    data["Lower"] = sma - 2 * std
    return data

# Apply selected strategies
if not data.empty:
    if "SMA Crossover" in strategies:
        data = apply_sma(data)
    if "RSI" in strategies:
        data = apply_rsi(data)
    if "MACD" in strategies:
        data = apply_macd(data)
    if "Bollinger Bands" in strategies:
        data = apply_bollinger(data)

    # Plotting
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name="Candlestick"
    ))

    if "SMA Crossover" in strategies:
        fig.add_trace(go.Scatter(x=data.index, y=data["SMA20"], line=dict(color="blue", width=1), name="SMA20"))
        fig.add_trace(go.Scatter(x=data.index, y=data["SMA50"], line=dict(color="orange", width=1), name="SMA50"))

    if "Bollinger Bands" in strategies:
        fig.add_trace(go.Scatter(x=data.index, y=data["Upper"], line=dict(color="gray", width=1), name="Upper Band"))
        fig.add_trace(go.Scatter(x=data.index, y=data["Lower"], line=dict(color="gray", width=1), name="Lower Band"))

    fig.update_layout(
        title=f"{symbol} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    # Additional plots
    if "RSI" in strategies:
        st.subheader("RSI Indicator")
        st.line_chart(data["RSI"])

    if "MACD" in strategies:
        st.subheader("MACD Indicator")
        st.line_chart(data[["MACD", "Signal"]])
else:
    st.warning("No data available for the selected symbol and date range.")
