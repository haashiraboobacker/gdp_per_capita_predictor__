import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model and scaler
model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🌍 Global GDP Growth Predictor")
st.write("Enter economic indicators to predict annual GDP growth.")

# To match the 16 features identified in your scaler.pkl 
col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Year", value=2026)
    inflation = st.number_input("Inflation (CPI %)", value=2.0)
    gdp_usd = st.number_input("GDP (Current USD)", value=1e10)
    unemployment = st.number_input("Unemployment Rate (%)", value=5.0)
    interest_rate = st.number_input("Interest Rate (Real, %)", value=3.0)
    inf_deflator = st.number_input("Inflation (GDP Deflator, %)", value=2.0)
    gdp_growth_prev = st.number_input("GDP Growth (% Annual) - Prev Year", value=2.5)
    current_account = st.number_input("Current Account Balance (% GDP)", value=0.0)

with col2:
    gov_expense = st.number_input("Government Expense (% of GDP)", value=20.0)
    gov_revenue = st.number_input("Government Revenue (% of GDP)", value=20.0)
    tax_revenue = st.number_input("Tax Revenue (% of GDP)", value=15.0)
    gni = st.number_input("Gross National Income (USD)", value=1e10)
    public_debt = st.number_input("Public Debt (% of GDP)", value=50.0)
    # Adding placeholders for the remaining 3 columns to reach 16
    feat_14 = st.hidden = 0.0
    feat_15 = st.hidden = 0.0
    feat_16 = st.hidden = 0.0

if st.button("Predict Growth"):
    # Create the array with all 16 features in the EXACT order from scaler.pkl 
    features = np.array([[
        year, inflation, gdp_usd, unemployment, interest_rate, 
        inf_deflator, gdp_growth_prev, current_account, gov_expense, 
        gov_revenue, tax_revenue, gni, public_debt, feat_14, feat_15, feat_16
    ]]) 
    
    # Scale and Predict
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    st.success(f"Predicted GDP Growth: {prediction[0]:.2f}%")
