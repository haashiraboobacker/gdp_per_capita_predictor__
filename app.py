import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Loading the model and scaler
model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🌍 Economic Growth Predictor")
st.write("Enter the numerical indicators below to generate a prediction.")

# Create two columns for a cleaner numerical input layout
col1, col2 = st.columns(2)

with col1:
    # Feature 1-7
    f1 = st.number_input("Year", value=2024, step=1)
    f2 = st.number_input("Inflation (CPI %)", value=2.0)
    f3 = st.number_input("GDP (Current USD)", value=1.0e10)
    f4 = st.number_input("Unemployment Rate (%)", value=5.0)
    f5 = st.number_input("Interest Rate (Real, %)", value=3.0)
    f6 = st.number_input("Inflation (GDP Deflator, %)", value=2.0)
    f7 = st.number_input("GDP Growth (% Annual)", value=2.5)

with col2:
    # Feature 8-13
    f8 = st.number_input("Current Account Balance (% GDP)", value=0.0)
    f9 = st.number_input("Government Expense (% of GDP)", value=20.0)
    f10 = st.number_input("Government Revenue (% of GDP)", value=20.0)
    f11 = st.number_input("Tax Revenue (% of GDP)", value=15.0)
    f12 = st.number_input("Gross National Income (USD)", value=1.0e10)
    f13 = st.number_input("Public Debt (% of GDP)", value=50.0)

if st.button("Predict"):
    # The array must have exactly 13 numerical elements in this order
    input_array = np.array([[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13]])
    
    try:
        # Scale the numerical inputs
        scaled_data = scaler.transform(input_array)
        
        # Run prediction
        result = model.predict(scaled_data)
        st.success(f"Predicted Annual GDP Growth: {result[0]:.2f}%")
        
    except Exception as e:
        st.error(f"Error: {e}")
