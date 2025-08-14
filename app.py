import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib # Or import pickle if you used that

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Lending Rate Predictor",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Model & Feature Configuration
# -----------------------------------------------------------------------------

# --- 1. UPDATE THE MODEL PATH ---
MODEL_PATH = 'model.pkl' # IMPORTANT: Replace with the actual path to your model file

# --- 2. DEFINE FEATURE ORDER ---
# This list MUST be in the exact same order as the features your model was trained on.
FEATURE_ORDER = [
    'region_Andaman & Nicobar', 'region_Assam', 'region_Bihar', 'region_Chandigarh',
    'region_Dadra & Nagar Haveli', 'region_Goa', 'region_Gujarat', 'region_Haryana',
    'region_Himachal Pradesh', 'region_Jammu & Kashmir', 'region_Karnataka',
    'region_Kerala', 'region_Ladakh', 'region_Madhya Pradesh', 'region_Meghalaya',
    'region_Mizoram', 'region_Rajasthan', 'region_Sikkim', 'region_Tamil Nadu',
    'region_Tripura', 'region_Uttarakhand', 'bank_IndusInd', 'bank_UCO Bank',
    'loan_type_Agriculture', 'repo_rate'
]

# --- 3. DEFINE USER-FRIENDLY OPTIONS FOR DROPDOWNS ---
# These are extracted from your feature list to populate the select boxes.
REGION_OPTIONS = [
    'Andaman & Nicobar', 'Assam', 'Bihar', 'Chandigarh', 'Dadra & Nagar Haveli',
    'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu & Kashmir',
    'Karnataka', 'Kerala', 'Ladakh', 'Madhya Pradesh', 'Meghalaya', 'Mizoram',
    'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Tripura', 'Uttarakhand'
]
BANK_OPTIONS = ['IndusInd', 'UCO Bank']
LOAN_TYPE_OPTIONS = ['Agriculture'] # Add more if your model supports them

# -----------------------------------------------------------------------------
# Load Model
# -----------------------------------------------------------------------------

@st.cache_resource
def load_model(model_path):
    """Loads the pickled machine learning model with error handling."""
    try:
        with open(model_path, 'rb') as f:
            model = joblib.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at '{model_path}'. Please check the path.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

# Load the model
model = load_model(MODEL_PATH)

# -----------------------------------------------------------------------------
# Sidebar - User Inputs
# -----------------------------------------------------------------------------
st.sidebar.title("Prediction Inputs")
st.sidebar.markdown("Select the parameters to see the real-time prediction.")

# --- Input Groups ---
st.sidebar.header("Loan & Bank Details")
selected_region = st.sidebar.selectbox("Region", REGION_OPTIONS)
selected_bank = st.sidebar.selectbox("Bank", BANK_OPTIONS)
selected_loan_type = st.sidebar.selectbox("Loan Type", LOAN_TYPE_OPTIONS)

st.sidebar.header("Economic Factors")
repo_rate_input = st.sidebar.slider("Repo Rate (%)", 3.0, 9.0, 6.5, 0.25)


# -----------------------------------------------------------------------------
# Main Panel - Title, Prediction, and Visualizations
# -----------------------------------------------------------------------------
st.title("🏦 Multi-Feature Lending Rate Predictor")
st.markdown("This dashboard uses a detailed model including region, bank, and loan type.")

if model:
    # --- Create the feature dictionary for the model ---
    input_features = {feature: 0 for feature in FEATURE_ORDER}

    # Set the continuous feature
    input_features['repo_rate'] = repo_rate_input

    # Set the one-hot encoded features based on user selection
    region_feature = f"region_{selected_region}"
    if region_feature in input_features:
        input_features[region_feature] = 1

    bank_feature = f"bank_{selected_bank}"
    if bank_feature in input_features:
        input_features[bank_feature] = 1

    loan_type_feature = f"loan_type_{selected_loan_type}"
    if loan_type_feature in input_features:
        input_features[loan_type_feature] = 1

    # --- Prediction ---
    # Convert input dictionary to a DataFrame in the correct feature order
    input_df = pd.DataFrame([input_features])[FEATURE_ORDER]

    try:
        prediction = model.predict(input_df)[0]

        # --- Display Results ---
        st.markdown("---")
        st.header("Prediction Result")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric(
                label="Predicted Lending Rate",
                value=f"{prediction:.2f}%",
                help="This is the lending rate predicted by the model based on the inputs from the sidebar."
            )

        with col2:
            # Create a gauge chart for visual context
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={'text': "Lending Rate Gauge"},
                gauge={'axis': {'range': [5, 15]}, # Adjust range as needed
                       'bar': {'color': "#2a9d8f"},
                       'steps': [
                           {'range': [5, 8], 'color': "#e9c46a"},
                           {'range': [8, 11], 'color': "#f4a261"},
                           {'range': [11, 15], 'color': "#e76f51"},
                       ]}
            ))
            fig.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

        # --- Model Transparency ---
        st.markdown("---")
        with st.expander("🔍 Click to see the feature vector used for this prediction"):
            st.dataframe(input_df)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

else:
    st.warning("Model is not loaded. Please check the configuration and file path.")
