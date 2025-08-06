import streamlit as st
import pickle
import pandas as pd
import joblib

# Load trained model and RFE selector
with open('model.pkl', 'rb') as f:
    model = joblib.load(f)

with open('rfe_selector.pkl', 'rb') as f:
    rfe_selector = joblib.load(f)

# Define the full input features (before RFE)
selected_features = ['region_Andaman & Nicobar', 'region_Andhra Pradesh', 'region_Arunachal Pradesh',
                     'region_Assam', 'region_Bihar', 'region_Chandigarh', 'region_Chhattisgarh',
                     'region_Dadra & Nagar Haveli', 'region_Delhi', 'region_Goa', 'region_Gujarat',
                     'region_Haryana', 'region_Himachal Pradesh', 'region_Jammu & Kashmir',
                     'region_Jharkhand', 'region_Karnataka', 'region_Kerala', 'region_Ladakh',
                     'region_Lakshadweep', 'region_Madhya Pradesh', 'region_Maharashtra', 'region_Manipur',
                     'region_Meghalaya', 'region_Mizoram', 'region_Nagaland', 'region_Odisha',
                     'region_Puducherry', 'region_Punjab', 'region_Rajasthan', 'region_Sikkim',
                     'region_Tamil Nadu', 'region_Telangana', 'region_Tripura', 'region_Uttar Pradesh',
                     'region_Uttarakhand', 'region_West Bengal']

# Streamlit UI
st.title("📈 Lending Rate Predictions on Bank Growth Rate")

st.markdown("""
This app predicts the **Lending Rate** based on regional bank growth indicators.
""")

# Collect user input
input_data = {}
for feature in selected_features:
    input_data[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Predict
if st.button("Predict"):
    try:
        # Transform the input using the same RFE selector used during training
        input_transformed = rfe_selector.transform(input_df)
        prediction = model.predict(input_transformed)
        st.success(f"📊 Predicted Lending Rate: **{prediction[0]:.2f}%**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
