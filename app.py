import streamlit as st
import pickle
import pandas as pd
import joblib

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = joblib.load(f)

# ✅ These are the only features your model was trained on
selected_features = ['bank_type_Private', 'year', 'month', 'credit_growth_rate', 'repo_rate', 'crr', 'deposit_growth_rate', 'gdp_growth', 'inflation_rate', 'consumer_confidence_index', 'investment_sentiment', 'npa_ratio', 'liquidity_ratio', 'global_trade_index', 'bank_size']

# Streamlit UI
st.title("📈 Lending Rate Prediction based on Bank Growth Rate")

st.markdown("""
This app predicts the **Lending Rate** based on bank growth regionally and bank-wise.
Please enter the feature values below:
""")

# Create input fields for each selected feature
input_data = {}
for feature in selected_features:
    input_data[feature] = st.number_input(f"{feature}", value=0.0)

# Convert input into DataFrame
input_df = pd.DataFrame([input_data])

# Make prediction when button is clicked
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        st.success(f"📊 Predicted Lending Rate: **{prediction[0]:.2f}%**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
