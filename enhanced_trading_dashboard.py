import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Define a list of predefined symbols
predefined_symbols = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Tesla (TSLA)": "TSLA",
    "Euro/USD (EURUSD=X)": "EURUSD=X",
    "GBP/USD (GBPUSD=X)": "GBPUSD=X",
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD"
}

# Function to fetch data from Yahoo Finance
@st.cache_data
def fetch_data_yahoo(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end)
        if not data.empty:
            return data
    except Exception as e:
        st.warning(f"Yahoo Finance error: {e}")
    return pd.DataFrame()

# Placeholder for additional data sources (e.g., Alpha Vantage)
def fetch_data_fallback(symbol, start, end):
    # Placeholder for future implementation
    return pd.DataFrame()

# Unified function to fetch data with fallback
def get_financial_data(symbol, start, end):
    data = fetch_data_yahoo(symbol, start, end)
    if data.empty:
        data = fetch_data_fallback(symbol, start, end)
    return data

# Streamlit UI
st.title("Enhanced Trading Dashboard")

# Sidebar for symbol selection
symbol_label = st.sidebar.selectbox("Select Symbol", list(predefined_symbols.keys()))
symbol = predefined_symbols[symbol_label]

# Date range selection
end_date = st.sidebar.date_input("End Date", datetime.today())
start_date = st.sidebar.date_input("Start Date", end_date - timedelta(days=90))

# Fetch data
data = get_financial_data(symbol, start_date, end_date)

# Display chart
if not data.empty:
    fig = go.Figure()

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))

    # Volume bar chart
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume',
        marker_color='lightblue',
        yaxis='y2',
        opacity=0.3
    ))

    # Layout adjustments
    fig.update_layout(
        title=f"{symbol} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        yaxis2=dict(
            overlaying='y',
            side='right',
            showgrid=False,
            title='Volume'
        ),
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("No data available for the selected symbol and date range.")

