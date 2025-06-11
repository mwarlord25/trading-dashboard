
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("📈 Enhanced Trading Dashboard")

# Predefined symbols
symbol_map = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Tesla (TSLA)": "TSLA",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD"
}

# Sidebar controls
symbol_name = st.sidebar.selectbox("Select Symbol", list(symbol_map.keys()))
symbol = symbol_map[symbol_name]
start_date = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=180))
end_date = st.sidebar.date_input("End Date", datetime.today())
strategies = st.sidebar.multiselect("Select Strategies", ["SMA Crossover", "RSI", "MACD", "Bollinger Bands"])

# Fetch data
@st.cache_data
def get_data(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

data = get_data(symbol, start_date, end_date)

# Strategy functions
def sma_strategy(df):
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()
    return df

def rsi_strategy(df, period=14):
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def macd_strategy(df):
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df

def bollinger_strategy(df):
    sma = df["Close"].rolling(window=20).mean()
    std = df["Close"].rolling(window=20).std()
    df["Upper_Band"] = sma + (2 * std)
    df["Lower_Band"] = sma - (2 * std)
    return df

# Apply selected strategies
if not data.empty:
    for strategy in strategies:
        if strategy == "SMA Crossover":
            data = sma_strategy(data)
        elif strategy == "RSI":
            data = rsi_strategy(data)
        elif strategy == "MACD":
            data = macd_strategy(data)
        elif strategy == "Bollinger Bands":
            data = bollinger_strategy(data)

    # Plotting
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name="Price"
    ))

    if "SMA20" in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data["SMA20"], mode="lines", name="SMA20"))
    if "SMA50" in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data["SMA50"], mode="lines", name="SMA50"))
    if "Upper_Band" in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data["Upper_Band"], mode="lines", name="Upper Band"))
        fig.add_trace(go.Scatter(x=data.index, y=data["Lower_Band"], mode="lines", name="Lower Band"))

    fig.update_layout(
        title=f"{symbol_name} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No data available for the selected symbol and date range.")
