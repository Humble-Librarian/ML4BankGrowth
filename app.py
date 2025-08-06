
import streamlit as st
import pickle
import pandas as pd
import joblib

with open('model.pkl', 'rb') as f:
    model = joblib.load(f)

# Define the features used in the model
selected_features = ['region_Andaman & Nicobar', 'region_Andhra Pradesh', 'region_Arunachal Pradesh', 'region_Assam', 'region_Bihar', 'region_Chandigarh', 'region_Chhattisgarh', 'region_Dadra & Nagar Haveli', 'region_Delhi', 'region_Goa', 'region_Gujarat', 'region_Haryana', 'region_Himachal Pradesh', 'region_Jammu & Kashmir', 'region_Jharkhand', 'region_Karnataka', 'region_Kerala', 'region_Ladakh', 'region_Lakshadweep', 'region_Madhya Pradesh', 'region_Maharashtra', 'region_Manipur', 'region_Meghalaya', 'region_Mizoram', 'region_Nagaland', 'region_Odisha', 'region_Puducherry', 'region_Punjab', 'region_Rajasthan', 'region_Sikkim', 'region_Tamil Nadu', 'region_Telangana', 'region_Tripura', 'region_Uttar Pradesh', 'region_Uttarakhand', 'region_West Bengal', 'bank_Axis', 'lending_rate']

# Streamlit App UI
st.title("📈 Lending Rate Predictions on Bank growth rate")

st.markdown(
    """
    This app predicts the **Lending Rate** based on Bank growth rate
    """
)

# Collect input data from user
input_data = {}
for feature in selected_features:
    input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)

# Convert input into DataFrame
input_df = pd.DataFrame([input_data])

# Predict when button is pressed
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.success(f"📊 Predicted Lending rate: **{prediction[0]:.2f}%**")
