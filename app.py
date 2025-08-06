import streamlit as st
import pandas as pd
import joblib

# Load model and RFE selector
model = joblib.load("model.pkl")
rfe_selector = joblib.load("rfe_selector.pkl")

# Your full list of input features (before RFE)
all_input_features = ['feature_1', 'feature_2', 'feature_3', ..., 'feature_n']

# Take input from user
input_data = {}
for feature in all_input_features:
    input_data[feature] = st.number_input(f"{feature}", value=0.0)

input_df = pd.DataFrame([input_data])

# Transform and predict
if st.button("Predict"):
    try:
        transformed_input = rfe_selector.transform(input_df)
        prediction = model.predict(transformed_input)
        st.success(f"Predicted value: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
