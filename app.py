import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Lending Rate Predictor",
    page_icon="💸",
    layout="wide",
)

# --- Load Model ---
MODEL_PATH = "model.pkl"

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model file not found at '{path}'. Make sure it’s in the same directory as app.py.")
        return None
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model(MODEL_PATH)

# --- Load Feature Selector if needed ---
SELECTOR_PATH = "rfe_selector.pkl"
feature_names = None
if model:
    if os.path.exists(SELECTOR_PATH):
        try:
            selector = joblib.load(SELECTOR_PATH)
            feature_names = selector.get_feature_names_out()
        except Exception as e:
            st.warning(f"Could not load feature selector: {e}")
    else:
        st.warning(f"Feature selector not found at '{SELECTOR_PATH}', will use default features.")

# Fallback if selector unavailable: match features embedded logic
if feature_names is None:
    st.error("Feature names not available. Please ensure 'rfe_selector.pkl' is present.")
    st.stop()

# --- Sidebar Inputs ---
st.sidebar.title("🔧 Input Configuration")
with st.sidebar.expander("Loan & Bank Details", expanded=True):
    region = st.selectbox("Region", options=[])  # populate your region list
    bank = st.selectbox("Bank", options=[])      # populate your bank list
    bank_type = st.radio("Bank Type", ["Public", "Private"], horizontal=True)
    loan_type = st.radio("Loan Type", ["Agriculture", "MSME"], horizontal=True)

with st.sidebar.expander("Time & Season", expanded=False):
    season = st.selectbox("Season", ["Winter", "Summer", "Monsoon", "Autumn"])
    month = st.slider("Month", 1, 12, 6)
    quarter = st.slider("Quarter", 1, 4, 2)

with st.sidebar.expander("Economic Factors", expanded=False):
    repo_rate = st.slider("Repo Rate (%)", 3.0, 9.0, 6.5, 0.25)
    gdp_growth = st.slider("GDP Growth (%)", -10.0, 10.0, 7.0, 0.1)

if st.sidebar.button("🔄 Reset Inputs"):
    st.experimental_rerun()

# --- Main Title ---
st.title("Lending Rate Predictor")
st.markdown("Predict lending rates using a Ridge Regression model trained on RFE-selected features.")

if not model:
    st.stop()

# --- Prepare Input DataFrame ---
input_dict = {feat: 0 for feat in feature_names}

# Populate input_dict based on your one-hot or feature logic
# Example: input_dict['repo_rate'] = repo_rate
# (Ensure all your feature_name keys appear here correctly)

input_df = pd.DataFrame([input_dict])

# --- Prediction Section ---
try:
    prediction = model.predict(input_df)[0]
    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction < 6:
        st.success(f"✅ Predicted Lending Rate: **{prediction:.2f}%** (Low)")
    elif prediction <= 9:
        st.warning(f"⚠️ Predicted Lending Rate: **{prediction:.2f}%** (Moderate)")
    else:
        st.error(f"🚨 Predicted Lending Rate: **{prediction:.2f}%** (High)")

    # Top 5 Features Chart
    if hasattr(model, "coef_"):
        coefs = model.coef_.flatten()
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": np.abs(coefs)
        }).sort_values(by="Importance", ascending=False).head(5)

        st.subheader("Top 5 Important Features")
        fig, ax = plt.subplots()
        ax.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
        ax.set_xlabel("Coefficient Magnitude")
        ax.set_title("Feature Importance")
        plt.gca().invert_yaxis()
        st.pyplot(fig)

except Exception as e:
    st.error(f"Prediction error: {e}")

# --- About Section ---
st.markdown("---")
st.subheader("About this Project")
st.markdown("""
- **Model:** Ridge Regression with R² ≈ 0.8  
- **Features:** RFE-selected features from `rfe_selector.pkl`  
- **Deployment:** Streamlit App  
""")
