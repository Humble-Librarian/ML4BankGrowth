import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# ---------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Lending Rate Predictor",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Model & Feature Configuration
# ---------------------------------------------------------------------
MODEL_PATH = 'model.pkl'

FEATURE_ORDER = [
    'region_Andaman & Nicobar', 'region_Arunachal Pradesh', 'region_Assam', 'region_Bihar',
    'region_Chandigarh', 'region_Chhattisgarh', 'region_Dadra & Nagar Haveli', 'region_Delhi',
    'region_Goa', 'region_Gujarat', 'region_Haryana', 'region_Himachal Pradesh',
    'region_Jammu & Kashmir', 'region_Karnataka', 'region_Kerala', 'region_Ladakh',
    'region_Madhya Pradesh', 'region_Maharashtra', 'region_Meghalaya', 'region_Mizoram',
    'region_Punjab', 'region_Rajasthan', 'region_Sikkim', 'region_Tamil Nadu',
    'region_Tripura', 'region_Uttarakhand', 'region_West Bengal', 'bank_Canara',
    'bank_IndusInd', 'bank_Kotak', 'bank_UCO Bank', 'bank_Yes Bank', 'bank_type_Public',
    'loan_type_Agriculture', 'loan_type_MSME', 'season_Winter', 'month', 'repo_rate',
    'gdp_growth', 'monthly_quarter'
]

REGION_OPTIONS = [
    'Andaman & Nicobar', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chandigarh',
    'Chhattisgarh', 'Dadra & Nagar Haveli', 'Delhi', 'Goa', 'Gujarat', 'Haryana',
    'Himachal Pradesh', 'Jammu & Kashmir', 'Karnataka', 'Kerala', 'Ladakh',
    'Madhya Pradesh', 'Maharashtra', 'Meghalaya', 'Mizoram', 'Punjab', 'Rajasthan',
    'Sikkim', 'Tamil Nadu', 'Tripura', 'Uttarakhand', 'West Bengal'
]
BANK_OPTIONS = ['Canara', 'IndusInd', 'Kotak', 'UCO Bank', 'Yes Bank']
BANK_TYPE_OPTIONS = ['Public', 'Private']
LOAN_TYPE_OPTIONS = ['Agriculture', 'MSME']
SEASON_OPTIONS = ['Winter', 'Summer', 'Monsoon', 'Autumn']

# ---------------------------------------------------------------------
# Load Model
# ---------------------------------------------------------------------
@st.cache_resource
def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            return joblib.load(f)
    except Exception as e:
        st.error(f"⚠️ Could not load model: {e}")
        return None

model = load_model(MODEL_PATH)

# ---------------------------------------------------------------------
# Sidebar Inputs
# ---------------------------------------------------------------------
st.sidebar.title("⚙️ Prediction Inputs")
st.sidebar.caption("Adjust values to see real-time lending rate prediction")

with st.sidebar.expander("🏦 Loan & Bank Details", expanded=True):
    selected_region = st.selectbox("Region", REGION_OPTIONS)
    selected_bank = st.selectbox("Bank", BANK_OPTIONS)
    selected_bank_type = st.radio("Bank Type", BANK_TYPE_OPTIONS, horizontal=True)
    selected_loan_type = st.radio("Loan Type", LOAN_TYPE_OPTIONS, horizontal=True)

with st.sidebar.expander("📅 Time & Season", expanded=False):
    selected_season = st.selectbox("Season", SEASON_OPTIONS)
    month_input = st.slider("Month", 1, 12, 6)
    monthly_quarter_input = st.slider("Quarter", 1, 4, 2)

with st.sidebar.expander("📊 Economic Factors", expanded=False):
    repo_rate_input = st.slider("Repo Rate (%)", 3.0, 9.0, 6.5, 0.25)
    gdp_growth_input = st.slider("GDP Growth (%)", -10.0, 10.0, 7.0, 0.1)

# Reset button
if st.sidebar.button("🔄 Reset Inputs"):
    st.experimental_rerun()

# ---------------------------------------------------------------------
# Main Layout
# ---------------------------------------------------------------------
st.title("💸 Lending Rate Predictor")
st.markdown(
    "This app predicts the **lending rate of banks** using a machine learning model trained on multiple economic and financial factors."
)

if model:
    # Prepare feature dict
    input_features = {feature: 0 for feature in FEATURE_ORDER}
    input_features.update({
        'repo_rate': repo_rate_input,
        'gdp_growth': gdp_growth_input,
        'month': month_input,
        'monthly_quarter': monthly_quarter_input
    })

    def set_one_hot(prefix, selection, features_dict):
        name = f"{prefix}_{selection}"
        if name in features_dict:
            features_dict[name] = 1

    set_one_hot('region', selected_region, input_features)
    set_one_hot('bank', selected_bank, input_features)
    set_one_hot('loan_type', selected_loan_type, input_features)
    set_one_hot('season', selected_season, input_features)
    if selected_bank_type == 'Public':
        input_features['bank_type_Public'] = 1

    input_df = pd.DataFrame([input_features])[FEATURE_ORDER]

    # Prediction
    try:
        prediction = model.predict(input_df)[0]

        st.markdown("---")
        st.subheader("📈 Prediction Result")

        # Color coding logic
        if prediction < 6:
            st.success(f"✅ Predicted Lending Rate: {prediction:.2f}% (Low)")
        elif 6 <= prediction <= 9:
            st.warning(f"⚠️ Predicted Lending Rate: {prediction:.2f}% (Moderate)")
        else:
            st.error(f"🚨 Predicted Lending Rate: {prediction:.2f}% (High)")

        # Feature importance (if available)
        if hasattr(model, "coef_"):
            st.markdown("### 🔑 Feature Importance (Top 8)")
            coef_df = pd.DataFrame({
                "Feature": FEATURE_ORDER,
                "Coefficient": model.coef_.flatten()
            }).sort_values("Coefficient", key=abs, ascending=False).head(8)

            fig = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h",
                         title="Top Feature Influences on Lending Rate")
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("📂 Model Input Data"):
            st.dataframe(input_df)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

else:
    st.warning("Model is not loaded. Please check the model path.")

# ---------------------------------------------------------------------
# About Section
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("ℹ️ About this Project")
st.markdown("""
- **Goal**: Predict lending rates for banks using regional, seasonal, and macroeconomic indicators.  
- **Model Used**: Ridge Regression (R² ≈ 0.8).  
- **Features**: 40 input features covering geography, loan types, repo rate, GDP growth, etc.  
- **Deployment**: Built with Streamlit for interactive exploration.  

""")
