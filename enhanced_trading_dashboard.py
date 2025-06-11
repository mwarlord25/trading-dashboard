
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide")
st.title("📈 Trading Agent Dashboard v2")

@st.cache_data
def load_data(symbol, start, end, interval):
    try:
        df = yf.download(symbol, start=start, end=end, interval=interval)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def sma_strategy(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    return df

def rsi_strategy(df, period=14):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def macd_strategy(df):
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

def bollinger_strategy(df):
    sma = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['Upper'] = sma + (2 * std)
    df['Lower'] = sma - (2 * std)
    return df

def plot_chart(df, symbol, strategies):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'],
                                 name='Candlestick'))

    if 'SMA Crossover' in strategies:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='SMA 20'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='SMA 50'))

    if 'Bollinger Bands' in strategies:
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='gray', width=1), name='Upper Band'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='gray', width=1), name='Lower Band'))

    fig.update_layout(title=f"{symbol} Price Chart", xaxis_title="Date", yaxis_title="Price", height=600)
    st.plotly_chart(fig, use_container_width=True)

    if 'RSI' in strategies:
        st.subheader("RSI Indicator")
        st.line_chart(df['RSI'])

    if 'MACD' in strategies:
        st.subheader("MACD Indicator")
        st.line_chart(df[['MACD', 'Signal']])

# Sidebar inputs
symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAPL")
interval = st.sidebar.selectbox("Select Interval", ['1d', '1h', '15m'])
start_date = st.sidebar.date_input("Start Date", value=datetime(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.today())
strategies = st.sidebar.multiselect("Select Strategies", ['SMA Crossover', 'RSI', 'MACD', 'Bollinger Bands'])

# Load and process data
data = load_data(symbol, start_date, end_date, interval)

if not data.empty:
    strategy_functions = {
        'SMA Crossover': sma_strategy,
        'RSI': rsi_strategy,
        'MACD': macd_strategy,
        'Bollinger Bands': bollinger_strategy
    }

    for strategy in strategies:
        data = strategy_functions[strategy](data)

    plot_chart(data, symbol, strategies)

    # Download button
    csv = data.to_csv().encode('utf-8')
    st.download_button("Download Data as CSV", csv, f"{symbol}_data.csv", "text/csv")
else:
    st.warning("No data available for the selected parameters.")
