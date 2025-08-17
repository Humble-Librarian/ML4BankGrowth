import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# Load Model and RFE Selector
# ==============================
@st.cache_resource
def load_components():
    model = joblib.load("model.pkl")
    try:
        selector = joblib.load("rfe_selector.pkl")
    except:
        selector = None
    return model, selector

model, selector = load_components()

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="ML for Bank Loan Growth", layout="wide")

st.title("🏦 Machine Learning on Lending Rate for Bank Loan Growth")
st.write("This app predicts **bank loan growth** using a Ridge Regression model "
         "trained on regional and macroeconomic factors.")

# ==============================
# Sidebar Inputs
# ==============================
st.sidebar.header("Input Features")

# Example input features (adjust to match your dataset)
input_data = {
    "lending_rate": st.sidebar.number_input("Lending Rate (%)", 5.0, 20.0, 10.0),
    "gdp_growth": st.sidebar.number_input("GDP Growth (%)", -5.0, 15.0, 5.0),
    "inflation": st.sidebar.number_input("Inflation (%)", 0.0, 15.0, 4.0),
    "unemployment": st.sidebar.number_input("Unemployment (%)", 0.0, 15.0, 6.0),
}

df = pd.DataFrame([input_data])

# Apply feature selection if available
if selector is not None:
    try:
        feature_names = np.array(df.columns)[selector.support_]
        df = df[feature_names]
    except Exception as e:
        st.warning(f"Feature selection could not be applied: {e}")

# ==============================
# Make Prediction
# ==============================
if st.sidebar.button("Predict Loan Growth"):
    prediction = model.predict(df)[0]
    st.subheader("📊 Predicted Bank Loan Growth")
    st.success(f"Estimated Loan Growth: **{prediction:.2f}%**")

# ==============================
# Feature Importance (Top 5)
# ==============================
if st.checkbox("Show Feature Importance"):
    if hasattr(model, "coef_"):
        try:
            if selector is not None:
                feature_names = np.array(input_data.keys())[selector.support_]
            else:
                feature_names = np.array(list(input_data.keys()))

            importances = model.coef_
            importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance": np.abs(importances)
            }).sort_values(by="importance", ascending=False).head(5)

            st.subheader("🔑 Top 5 Important Features")
            fig, ax = plt.subplots()
            ax.barh(importance_df["feature"], importance_df["importance"])
            ax.set_xlabel("Importance (absolute coefficient)")
            st.pyplot(fig)

        except Exception as e:
            st.warning(f"Could not compute feature importance: {e}")
    else:
        st.info("Model does not provide coefficients for importance.")
