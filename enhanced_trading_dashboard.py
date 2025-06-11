
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

# Streamlit UI
st.title("Enhanced Trading Dashboard with Backtesting and Alerts")

# User inputs
symbol = st.selectbox("Select Symbol", ['AAPL', 'MSFT', 'GOOGL', 'EURUSD=X', 'GBPUSD=X'])
start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))
email_alert = st.checkbox("Enable Email Alerts")
user_email = st.text_input("Enter your email for alerts") if email_alert else None

# Fetch data
@st.cache_data
def load_data(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

data = load_data(symbol, start_date, end_date)

if data.empty:
    st.warning("No data available for the selected symbol and date range.")
    st.stop()

# Strategy: Simple Moving Average Crossover
def sma_strategy(data):
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['Signal'] = 0
    data.loc[data['SMA20'] > data['SMA50'], 'Signal'] = 1
    data.loc[data['SMA20'] < data['SMA50'], 'Signal'] = -1
    return data

# Apply strategy
data = sma_strategy(data)

# Plotting
def plot_chart(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close'))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], name='SMA20'))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], name='SMA50'))
    st.plotly_chart(fig)

plot_chart(data)

# Backtesting
def backtest(data):
    data['Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Signal'].shift(1) * data['Returns']
    data['Cumulative_Strategy'] = (1 + data['Strategy_Returns']).cumprod()
    data['Cumulative_Market'] = (1 + data['Returns']).cumprod()
    win_rate = (data['Strategy_Returns'] > 0).sum() / data['Strategy_Returns'].count()
    return data, win_rate

data, win_rate = backtest(data)
st.subheader("Backtesting Results")
st.write(f"Strategy Win Rate: {win_rate:.2%}")

# Trade Signal
if not data['Signal'].dropna().empty:
    latest_signal = data['Signal'].iloc[-1]
    signal_time = data.index[-1]
    if latest_signal == 1:
        st.success(f"Buy Signal at {signal_time}")
        trade_action = "Buy"
    elif latest_signal == -1:
        st.error(f"Sell Signal at {signal_time}")
        trade_action = "Sell"
    else:
        st.info(f"Hold Signal at {signal_time}")
        trade_action = "Hold"
else:
    st.warning("No signal data available.")
    trade_action = "None"

# Email Alert
def send_email(recipient, subject, body):
    try:
        smtp_server = "smtp.gmail.com"
        smtp_port = 465
        sender_email = "your_email@gmail.com"
        sender_password = "your_app_password"

        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = recipient

        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient, msg.as_string())
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

if email_alert and user_email and trade_action in ["Buy", "Sell"]:
    subject = f"Trade Alert: {trade_action} {symbol}"
    body = f"A {trade_action} signal was generated for {symbol} at {signal_time}."
    if send_email(user_email, subject, body):
        st.success("Email alert sent successfully.")
