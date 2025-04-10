
import streamlit as st
import joblib
import numpy as np

st.title("🌱 Sustainability Score Prediction")

model = joblib.load("sustainability_model.pkl")
scaler = joblib.load("sustainability_scaler.pkl")

soil_pH = st.slider("Soil pH", 0.0, 14.0, 6.5)
soil_moisture = st.slider("Soil Moisture (%)", 0.0, 100.0, 50.0)
temperature = st.slider("Temperature (°C)", -10.0, 50.0, 25.0)
fertilizer = st.number_input("Fertilizer Usage (kg)", 0.0, 500.0, 50.0)
pesticide = st.number_input("Pesticide Usage (kg)", 0.0, 500.0, 10.0)

if st.button("Predict Sustainability"):
    X = np.array([[soil_pH, soil_moisture, temperature, fertilizer, pesticide]])
    X_scaled = scaler.transform(X)
    result = model.predict(X_scaled)[0]
    st.success(f"🌍 Sustainability Score: {result:.2f}")
