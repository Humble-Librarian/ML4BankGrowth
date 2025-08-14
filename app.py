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
    'bank_type_Private', 'year', 'month', 'credit_growth_rate', 'repo_rate', 'crr',
    'deposit_growth_rate', 'gdp_growth', 'inflation_rate', 'consumer_confidence_index',
    'investment_sentiment', 'npa_ratio', 'liquidity_ratio', 'global_trade_index', 'bank_size'
]

# -----------------------------------------------------------------------------
# Load Model
# -----------------------------------------------------------------------------

@st.cache_resource
def load_model(model_path):
    """Loads the pickled machine learning model with error handling."""
    try:
        with open(model_path, 'rb') as f:
            # Use joblib.load if you saved with joblib, otherwise use pickle.load
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
st.sidebar.markdown("Adjust the values below to see the real-time prediction.")

# Create a dictionary to hold user inputs
input_data = {}

# --- Input Groups ---
st.sidebar.header("Bank Details")
bank_type_selection = st.sidebar.selectbox("Bank Type", ["Private", "Public"])
input_data['bank_type_Private'] = 1 if bank_type_selection == "Private" else 0
input_data['npa_ratio'] = st.sidebar.slider("NPA Ratio (%)", 0.0, 15.0, 2.5, 0.1, help="Non-Performing Asset Ratio")
input_data['liquidity_ratio'] = st.sidebar.slider("Liquidity Ratio (%)", 10.0, 50.0, 20.0, 0.5)
input_data['bank_size'] = st.sidebar.slider("Bank Size (in Billion USD)", 10.0, 500.0, 100.0, 10.0)

st.sidebar.header("Economic Factors")
input_data['repo_rate'] = st.sidebar.slider("Repo Rate (%)", 3.0, 9.0, 6.5, 0.25)
input_data['crr'] = st.sidebar.slider("Cash Reserve Ratio (CRR %)", 3.0, 6.0, 4.5, 0.1)
input_data['credit_growth_rate'] = st.sidebar.slider("Credit Growth Rate (%)", -5.0, 20.0, 10.0, 0.5)
input_data['deposit_growth_rate'] = st.sidebar.slider("Deposit Growth Rate (%)", -5.0, 20.0, 12.0, 0.5)
input_data['gdp_growth'] = st.sidebar.slider("GDP Growth (%)", -10.0, 10.0, 7.0, 0.1)
input_data['inflation_rate'] = st.sidebar.slider("Inflation Rate (%)", 1.0, 10.0, 5.0, 0.1)
input_data['consumer_confidence_index'] = st.sidebar.slider("Consumer Confidence Index", 80.0, 120.0, 100.0, 0.5)
input_data['investment_sentiment'] = st.sidebar.slider("Investment Sentiment Index", 80.0, 120.0, 105.0, 0.5)
input_data['global_trade_index'] = st.sidebar.slider("Global Trade Index", 80.0, 120.0, 102.0, 0.5)

st.sidebar.header("Time Period")
input_data['year'] = st.sidebar.number_input("Year", min_value=2020, max_value=2030, value=2024)
input_data['month'] = st.sidebar.slider("Month", 1, 12, 6)

# -----------------------------------------------------------------------------
# Main Panel - Title, Prediction, and Visualizations
# -----------------------------------------------------------------------------
st.title("🏦 Advanced Lending Rate Predictor")
st.markdown("This dashboard predicts the lending rate using a machine learning model based on 15 key financial and economic indicators.")

if model:
    # --- Prediction ---
    # Convert input dictionary to a DataFrame in the correct feature order
    input_df = pd.DataFrame([input_data])[FEATURE_ORDER]

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
        with st.expander("🔍 Click to see the feature values used for this prediction"):
            st.dataframe(input_df)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

else:
    st.warning("Model is not loaded. Please check the configuration and file path.")
