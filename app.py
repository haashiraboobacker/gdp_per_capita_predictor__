import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🌍 Global GDP Growth Predictor")
st.write("Enter economic indicators to predict annual GDP growth.")

# Create input fields for your 16 columns
# (Example for a few columns; add the rest of your list here)
inflation = st.number_input("Inflation (CPI %)", value=2.0)
debt = st.number_input("Public Debt (% of GDP)", value=50.0)
unemployment = st.number_input("Unemployment Rate (%)", value=5.0)

# Prediction Logic
if st.button("Predict Growth"):
    # 1. Arrange inputs into a 2D array (must match the order of your 16 columns)
    # Note: Remember to log-transform currency inputs if you used that in training!
    features = np.array([[inflation, debt, unemployment, ...]]) 
    
    # 2. Scale the features
    scaled_features = scaler.transform(features)
    
    # 3. Predict
    prediction = model.predict(scaled_features)
    st.success(f"Predicted GDP Growth: {prediction[0]:.2f}%")