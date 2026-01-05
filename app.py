import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page config
st.set_page_config(page_title="Earnings Manipulation Predictor", layout="centered")

st.title("📊 Earnings Manipulation Risk Predictor")
st.write("Enter financial ratios to assess manipulation risk.")

# Load model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# User inputs
dsri = st.number_input("DSRI (Days Sales Receivable Index)", value=1.0)
gmi  = st.number_input("GMI (Gross Margin Index)", value=1.0)
aqi  = st.number_input("AQI (Asset Quality Index)", value=1.0)
sgi  = st.number_input("SGI (Sales Growth Index)", value=1.0)
depi = st.number_input("DEPI (Depreciation Index)", value=1.0)
sgai = st.number_input("SGAI (SG&A Index)", value=1.0)
tata = st.number_input("TATA (Total Accruals to Total Assets)", value=0.0)
lvgi = st.number_input("LVGI (Leverage Index)", value=1.0)

# Predict button
if st.button("Predict Manipulation Risk"):
    input_data = np.array([[dsri, gmi, aqi, sgi, depi, sgai, tata, lvgi]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Manipulation (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low Risk of Manipulation (Probability: {probability:.2f})")
