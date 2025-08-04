# ==========================
# Bitcoin Price Prediction - Streamlit App
# ==========================

import streamlit as st
import numpy as np
import joblib

# Configure Streamlit page
st.set_page_config(page_title="Bitcoin Price Predictor", layout="centered")
st.title("📈 Bitcoin Price Prediction Tool")
st.markdown("""
Enter Bitcoin market statistics to predict the **next day's closing price**.
""")

# Load trained model and scaler
model = joblib.load("btc_price_model.pkl")
scaler = joblib.load("btc_scaler.pkl")

# Sidebar inputs for features
st.sidebar.header("Input Features")
open_price = st.sidebar.number_input("Opening Price (USD)", value=50000.0, format="%.2f")
high_price = st.sidebar.number_input("Highest Price (USD)", value=50500.0, format="%.2f")
low_price = st.sidebar.number_input("Lowest Price (USD)", value=49500.0, format="%.2f")
volume = st.sidebar.number_input("Trading Volume", value=10000.0, format="%.2f")
ma7 = st.sidebar.number_input("7-Day Moving Average", value=50200.0, format="%.2f")
ma21 = st.sidebar.number_input("21-Day Moving Average", value=49800.0, format="%.2f")
volatility = st.sidebar.number_input("Price Volatility (21-day STD)", value=300.0, format="%.2f")
returns = st.sidebar.number_input("Daily Returns", value=0.002, format="%.4f")

# Predict button
if st.sidebar.button("Predict Price"):
    # Prepare data
    features = np.array([[open_price, high_price, low_price, volume,
                          ma7, ma21, volatility, returns]])
    features_scaled = scaler.transform(features)
    
    # Make prediction
    predicted_price = model.predict(features_scaled)[0]
    
    # Display result
    st.subheader("Predicted Next-Day Closing Price")
    st.success(f"${predicted_price:,.2f} USD")

    st.caption("Prediction is based on historical data patterns and may not reflect sudden market events.")
